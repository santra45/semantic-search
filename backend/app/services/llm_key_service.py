import base64
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from hashlib import sha256

def decrypt_key(encrypted_blob: str, license_key: str) -> str:
    try:
        # 1. Handle the dot-suffix if it exists
        # Your log shows: [Base64].[UUID]
        if "." in encrypted_blob:
            encrypted_blob = encrypted_blob.split(".")[0]
        
        # 2. Decode Base64
        data = base64.b64decode(encrypted_blob)
        iv = data[:16]
        payload = data[16:]
        
        # 3. Derive Key - Ensure license_key is a clean string
        # IMPORTANT: If PHP uses the 'raw' license key, Python must too.
        key = sha256(license_key.strip().encode('utf-8')).digest()
        
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        # 4. Decrypt and Unpad
        decrypted_raw = cipher.decrypt(payload)
        return unpad(decrypted_raw, AES.block_size).decode('utf-8')
        
    except Exception as e:
        print(f"❌ Decryption failed: {e}")
        return None