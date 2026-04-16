"""
Token Usage Tracking Service

This service provides functionality to track token usage and costs for different LLM operations
including embedding search, embedding documents, and product reranking.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.services.database import get_db

logger = logging.getLogger(__name__)

# Query types that can be tracked
QUERY_TYPES = {
    'embed_search': 'embed_search',
    'embed_document': 'embed_document', 
    'product_rerank': 'product_rerank',
    'content_rerank': 'content_rerank',
    'chat_answer': 'chat_answer',
    'chat_context': 'chat_context',
    'chat_rewrite': 'chat_rewrite',
}

class TokenUsageTracker:
    """Service for tracking token usage and costs per client request."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db or next(get_db())
    
    def create_usage_record(
        self,
        client_id: str,
        query_type: str,
        llm_provider: str,
        llm_model: str,
        input_tokens: int,
        output_tokens: int = 0,
        input_cost: float = 0.0,
        output_cost: float = 0.0,
        request_text_length: int = 0,
        response_text_length: int = 0,
        request_id: Optional[str] = None
    ) -> str:
        """
        Create a new token usage record.
        
        Args:
            client_id: Client identifier
            query_type: Type of query (embed_search, embed_document, product_rerank)
            llm_provider: Provider name (google, openai, anthropic)
            llm_model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens (default 0 for embeddings)
            input_cost: Cost for input tokens
            output_cost: Cost for output tokens
            request_text_length: Length of request text in characters
            response_text_length: Length of response text in characters
            request_id: Optional unique request ID (generated if not provided)
            
        Returns:
            request_id: The unique request ID for this record
        """
        
        if query_type not in QUERY_TYPES.values():
            raise ValueError(f"Invalid query_type: {query_type}. Must be one of: {list(QUERY_TYPES.values())}")
        
        # Generate request ID if not provided
        request_id = request_id or str(uuid.uuid4())
        
        # Calculate totals
        total_tokens = input_tokens + output_tokens
        total_cost = input_cost + output_cost
        
        insert_sql = """
        INSERT INTO token_usage_tracking (
            client_id, request_id, query_type, llm_provider, llm_model,
            input_tokens, output_tokens, total_tokens,
            input_cost, output_cost, total_cost,
            request_text_length, response_text_length
        ) VALUES (
            :client_id, :request_id, :query_type, :llm_provider, :llm_model,
            :input_tokens, :output_tokens, :total_tokens,
            :input_cost, :output_cost, :total_cost,
            :request_text_length, :response_text_length
        )
        """
        
        try:
            self.db.execute(text(insert_sql), {
                'client_id': client_id,
                'request_id': request_id,
                'query_type': query_type,
                'llm_provider': llm_provider,
                'llm_model': llm_model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'request_text_length': request_text_length,
                'response_text_length': response_text_length
            })
            self.db.commit()
            
            logger.debug(f"✅ Tracked token usage: {query_type} for client {client_id} - {total_tokens} tokens, ${total_cost:.8f}")
            return request_id
            
        except Exception as e:
            logger.error(f"❌ Failed to track token usage: {e}")
            self.db.rollback()
            raise
    
    def get_client_usage_stats(
        self, 
        client_id: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get usage statistics for a specific client.
        
        Args:
            client_id: Client identifier
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing usage statistics
        """
        
        where_clause = "WHERE client_id = :client_id"
        params = {'client_id': client_id}
        
        if start_date:
            where_clause += " AND created_at >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            where_clause += " AND created_at <= :end_date"
            params['end_date'] = end_date
        
        sql = f"""
        SELECT 
            query_type,
            llm_provider,
            llm_model,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_request,
            MIN(created_at) as first_request,
            MAX(created_at) as last_request
        FROM token_usage_tracking
        {where_clause}
        GROUP BY query_type, llm_provider, llm_model
        ORDER BY total_cost DESC
        """
        
        try:
            result = self.db.execute(text(sql), params)
            rows = result.fetchall()
            
            stats = {
                'client_id': client_id,
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'usage_by_type': []
            }
            
            for row in rows:
                stats['usage_by_type'].append({
                    'query_type': row.query_type,
                    'llm_provider': row.llm_provider,
                    'llm_model': row.llm_model,
                    'request_count': row.request_count,
                    'total_input_tokens': row.total_input_tokens,
                    'total_output_tokens': row.total_output_tokens,
                    'total_tokens': row.total_tokens,
                    'total_input_cost': float(row.total_input_cost),
                    'total_output_cost': float(row.total_output_cost),
                    'total_cost': float(row.total_cost),
                    'avg_cost_per_request': float(row.avg_cost_per_request),
                    'first_request': row.first_request,
                    'last_request': row.last_request
                })
            
            # Calculate totals across all types
            total_stats = {
                'total_requests': sum(row['request_count'] for row in stats['usage_by_type']),
                'total_tokens': sum(row['total_tokens'] for row in stats['usage_by_type']),
                'total_cost': sum(row['total_cost'] for row in stats['usage_by_type'])
            }
            stats['totals'] = total_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to get client usage stats: {e}")
            raise
    
    def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Get usage summary across all clients.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            Dictionary containing usage summary
        """
        
        where_clause = "WHERE 1=1"
        params = {}
        
        if start_date:
            where_clause += " AND created_at >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            where_clause += " AND created_at <= :end_date"
            params['end_date'] = end_date
        
        sql = f"""
        SELECT 
            COUNT(DISTINCT client_id) as unique_clients,
            COUNT(*) as total_requests,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(input_cost) as total_input_cost,
            SUM(output_cost) as total_output_cost,
            SUM(total_cost) as total_cost,
            AVG(total_cost) as avg_cost_per_request
        FROM token_usage_tracking
        {where_clause}
        """
        
        try:
            result = self.db.execute(text(sql), params)
            row = result.fetchone()
            
            return {
                'period': {
                    'start_date': start_date,
                    'end_date': end_date
                },
                'unique_clients': row.unique_clients,
                'total_requests': row.total_requests,
                'total_input_tokens': row.total_input_tokens or 0,
                'total_output_tokens': row.total_output_tokens or 0,
                'total_tokens': row.total_tokens or 0,
                'total_input_cost': float(row.total_input_cost or 0),
                'total_output_cost': float(row.total_output_cost or 0),
                'total_cost': float(row.total_cost or 0),
                'avg_cost_per_request': float(row.avg_cost_per_request or 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get usage summary: {e}")
            raise

# Convenience function for easy tracking
def track_usage(
    client_id: str,
    query_type: str,
    llm_provider: str,
    llm_model: str,
    input_tokens: int,
    output_tokens: int = 0,
    input_cost: float = 0.0,
    output_cost: float = 0.0,
    request_text_length: int = 0,
    response_text_length: int = 0,
    request_id: Optional[str] = None
) -> str:
    """
    Convenience function to track token usage without needing to manage database sessions.
    """
    tracker = TokenUsageTracker()
    try:
        return tracker.create_usage_record(
            client_id=client_id,
            query_type=query_type,
            llm_provider=llm_provider,
            llm_model=llm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            request_text_length=request_text_length,
            response_text_length=response_text_length,
            request_id=request_id
        )
    finally:
        tracker.db.close()
