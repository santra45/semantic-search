import hashlib
import hmac
import json
import time
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text


class DomainAuthorizer:
    """
    Secure domain authorization with multiple layers of validation.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def validate_request(
        self, 
        request: Request, 
        license_data: Dict[str, Any],
        api_key: Optional[str] = None
    ) -> bool:
        """
        Multi-layer domain authorization validation.
        
        Args:
            request: FastAPI Request object
            license_data: License information from validate_license_key
            api_key: Optional API key for additional validation
            
        Returns:
            True if request is authorized
            
        Raises:
            HTTPException with appropriate status code and detail
        """
        allowed_domain = license_data.get("domain")
        client_id = license_data["client_id"]
        
        if not allowed_domain:
            # If no domain restriction, allow but log for security
            self._log_security_event(
                client_id, 
                "no_domain_restriction", 
                "License has no domain restriction"
            )
            return True
        
        # Layer 1: IP-based validation (most reliable)
        self._validate_ip_address(client_id, allowed_domain, request)
        
        # Layer 2: Enhanced header validation with multiple checks
        self._validate_request_headers(client_id, allowed_domain, request)
        
        # Layer 3: API key validation if provided
        if api_key:
            self._validate_api_key(client_id, api_key)
        
        # Layer 4: Rate limiting check
        self._check_rate_limit(client_id, request)
        
        # Log successful authorization
        self._log_security_event(client_id, "authorized", "Request authorized successfully")
        
        return True
    
    def _validate_ip_address(self, client_id: str, allowed_domain: str, request: Request):
        """Validate client IP against allowed domain."""
        client_ip = self._get_client_ip(request)
        
        # Check if IP is in whitelist for this domain
        if not self._is_ip_allowed_for_domain(client_ip, allowed_domain):
            self._log_security_event(
                client_id, 
                "ip_blocked", 
                f"IP {client_ip} not allowed for domain {allowed_domain}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"IP address not authorized for domain: {allowed_domain}"
            )
    
    def _validate_request_headers(self, client_id: str, allowed_domain: str, request: Request):
        """Enhanced header validation with multiple checks."""
        client_ip = self._get_client_ip(request)
        if client_ip in ["127.0.0.1", "localhost"] or self._is_private_ip(client_ip):
            return True
        headers = request.headers
        
        # Check multiple headers that are harder to spoof together
        origin = headers.get("origin")
        referer = headers.get("referer")
        host = headers.get("host")
        x_forwarded_host = headers.get("x-forwarded-host")
        x_forwarded_for = headers.get("x-forwarded-for")
        
        valid_domains = self._get_all_valid_domains(allowed_domain)
        
        # Validate Origin header
        if origin:
            origin_domain = urlparse(origin).hostname
            if origin_domain not in valid_domains:
                self._log_security_event(
                    client_id, 
                    "origin_invalid", 
                    f"Origin {origin} not allowed for domain {allowed_domain}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Origin not authorized. Expected: {allowed_domain}"
                )
        
        # Validate Referer header
        if referer:
            referer_domain = urlparse(referer).hostname
            if referer_domain not in valid_domains:
                self._log_security_event(
                    client_id, 
                    "referer_invalid", 
                    f"Referer {referer} not allowed for domain {allowed_domain}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Referer not authorized. Expected: {allowed_domain}"
                )
        
        # Validate Host header
        if host:
            if host not in valid_domains and not any(host.endswith(f".{domain}") for domain in valid_domains):
                self._log_security_event(
                    client_id, 
                    "host_invalid", 
                    f"Host {host} not allowed for domain {allowed_domain}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Host not authorized. Expected: {allowed_domain}"
                )
        
        # Validate X-Forwarded-Host header (for proxy setups)
        if x_forwarded_host:
            if x_forwarded_host not in valid_domains:
                self._log_security_event(
                    client_id, 
                    "x_forwarded_host_invalid", 
                    f"X-Forwarded-Host {x_forwarded_host} not allowed for domain {allowed_domain}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"X-Forwarded-Host not authorized. Expected: {allowed_domain}"
                )
        
        # At least one header must match the allowed domain
        headers_to_check = [origin, referer, host, x_forwarded_host]
        matching_headers = []
        
        for header_value in headers_to_check:
            if header_value:
                header_domain = urlparse(header_value).hostname or header_value.split(':')[0]
                if header_domain in valid_domains or any(header_domain.endswith(f".{domain}") for domain in valid_domains):
                    matching_headers.append(header_value)
        
        if not matching_headers:
            self._log_security_event(
                client_id, 
                "no_matching_headers", 
                f"No headers matched allowed domain {allowed_domain}"
            )
            raise HTTPException(
                status_code=403,
                detail=f"No valid domain headers found. Expected: {allowed_domain}"
            )
    
    def _validate_api_key(self, client_id: str, api_key: str):
        """Validate API key if provided."""
        # Store and validate API keys per client
        stored_keys = self._get_client_api_keys(client_id)
        
        if not stored_keys or api_key not in stored_keys:
            self._log_security_event(
                client_id, 
                "invalid_api_key", 
                "Invalid API key provided"
            )
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
    
    def _check_rate_limit(self, client_id: str, request: Request):
        """Check rate limiting per client."""
        client_ip = self._get_client_ip(request)
        current_time = int(time.time())
        
        # Get current usage for this client/IP
        usage = self.db.execute(text("""
            SELECT request_count, window_start
            FROM rate_limits
            WHERE client_id = :client_id AND ip_address = :ip
            AND window_start > :window_start
        """), {
            "client_id": client_id,
            "ip": client_ip,
            "window_start": current_time - 3600  # 1 hour window
        }).fetchone()
        
        max_requests_per_hour = 1000  # Configurable per plan
        
        if usage and usage.request_count >= max_requests_per_hour:
            self._log_security_event(
                client_id, 
                "rate_limit_exceeded", 
                f"Rate limit exceeded for IP {client_ip}"
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Update rate limit counter
        if usage:
            self.db.execute(text("""
                UPDATE rate_limits
                SET request_count = request_count + 1
                WHERE client_id = :client_id AND ip_address = :ip
            """), {"client_id": client_id, "ip": client_ip})
        else:
            self.db.execute(text("""
                INSERT INTO rate_limits (client_id, ip_address, request_count, window_start)
                VALUES (:client_id, :ip, 1, :window_start)
            """), {
                "client_id": client_id,
                "ip": client_ip,
                "window_start": current_time
            })
        
        self.db.commit()
    
    def _get_client_ip(self, request: Request) -> str:
        """Get real client IP address."""
        # Check various headers for real IP
        ip_headers = [
            "x-forwarded-for",
            "x-real-ip", 
            "cf-connecting-ip",  # Cloudflare
            "x-client-ip",
            "x-forwarded",
            "forwarded-for",
            "forwarded"
        ]
        
        for header in ip_headers:
            ip = request.headers.get(header)
            if ip:
                # X-Forwarded-For can contain multiple IPs, take the first one
                return ip.split(",")[0].strip()
        
        # Fallback to remote address
        return request.client.host if request.client else "unknown"
    
    def _get_all_valid_domains(self, allowed_domain: str) -> List[str]:
        """Get list of all valid domains including subdomains."""
        domains = [allowed_domain]
        
        # Add common subdomains
        subdomains = ["www", "api", "app", "admin"]
        for subdomain in subdomains:
            domains.append(f"{subdomain}.{allowed_domain}")
        
        return domains
    
    def _is_ip_allowed_for_domain(self, ip: str, domain: str) -> bool:
        """Check if IP is allowed for the given domain."""
        # For now, allow localhost and private networks for development
        # In production, this should be configured per domain
        allowed_ips = [
            "127.0.0.1", "localhost", "::1",
            # Add production IPs here per domain
        ]
        
        # Check if it's a private network IP
        if self._is_private_ip(ip):
            return True
        
        return ip in allowed_ips
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private network range."""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return ip_obj.is_private
        except ValueError:
            return False
    
    def _get_client_api_keys(self, client_id: str) -> List[str]:
        """Get stored API keys for client."""
        result = self.db.execute(text("""
            SELECT api_key
            FROM client_api_keys
            WHERE client_id = :client_id AND is_active = 1
        """), {"client_id": client_id}).fetchall()
        
        return [row.api_key for row in result] if result else []
    
    def _log_security_event(self, client_id: str, event_type: str, details: str):
        """Log security events for monitoring."""
        try:
            self.db.execute(text("""
                INSERT INTO security_logs (id, client_id, event_type, details, timestamp)
                VALUES (:id, :client_id, :event_type, :details, :timestamp)
            """), {
                "id": f"sec_{int(time.time())}_{client_id[:8]}",
                "client_id": client_id,
                "event_type": event_type,
                "details": details,
                "timestamp": int(time.time())
            })
            self.db.commit()
        except Exception:
            # Don't fail the request if logging fails
            pass


def create_request_signature(payload: Dict[str, Any], secret: str) -> str:
    """
    Create HMAC signature for request payload.
    
    Args:
        payload: Request data to sign
        secret: Client secret key
        
    Returns:
        Hex signature string
    """
    # Sort keys for consistent signature
    sorted_payload = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    
    return hmac.new(
        secret.encode('utf-8'),
        sorted_payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def verify_request_signature(payload: Dict[str, Any], signature: str, secret: str) -> bool:
    """
    Verify HMAC signature of request payload.
    
    Args:
        payload: Request data to verify
        signature: Received signature
        secret: Client secret key
        
    Returns:
        True if signature is valid
    """
    expected_signature = create_request_signature(payload, secret)
    return hmac.compare_digest(expected_signature, signature)
