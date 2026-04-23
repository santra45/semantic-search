"""
Create token_usage_tracking table for monitoring API costs and usage per client.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from backend.app.services.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_token_usage_table():
    """Create the token_usage_tracking table with all necessary fields."""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS token_usage_tracking (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        client_id VARCHAR(36) NOT NULL,
        request_id VARCHAR(255) NOT NULL UNIQUE,
        query_type ENUM('embed_search', 'embed_document', 'embed_query', 'product_rerank', 'content_rerank', 'chat_answer', 'chat_context', 'chat_rewrite') NOT NULL,
        llm_provider VARCHAR(50) NOT NULL,
        llm_model VARCHAR(100) NOT NULL,
        
        -- Token counts
        input_tokens INT NOT NULL DEFAULT 0,
        output_tokens INT NOT NULL DEFAULT 0,
        total_tokens INT NOT NULL DEFAULT 0,
        
        -- Cost tracking
        input_cost DECIMAL(10, 8) NOT NULL DEFAULT 0.00000000,
        output_cost DECIMAL(10, 8) NOT NULL DEFAULT 0.00000000,
        total_cost DECIMAL(10, 8) NOT NULL DEFAULT 0.00000000,
        
        -- Request metadata
        request_text_length INT NOT NULL DEFAULT 0,
        response_text_length INT NOT NULL DEFAULT 0,
        
        -- Timestamps
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Indexes for performance
        INDEX idx_client_id (client_id),
        INDEX idx_query_type (query_type),
        INDEX idx_llm_provider (llm_provider),
        INDEX idx_llm_model (llm_model),
        INDEX idx_created_at (created_at),
        INDEX idx_client_created (client_id, created_at),
        
        -- Constraints
        CONSTRAINT chk_tokens_non_negative CHECK (input_tokens >= 0 AND output_tokens >= 0 AND total_tokens >= 0),
        CONSTRAINT chk_costs_non_negative CHECK (input_cost >= 0 AND output_cost >= 0 AND total_cost >= 0),
        CONSTRAINT chk_total_tokens_match CHECK (total_tokens = input_tokens + output_tokens),
        CONSTRAINT chk_total_cost_match CHECK (total_cost = input_cost + output_cost)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    
    try:
        with engine.connect() as conn:
            # Drop table if it exists for clean recreation
            conn.execute(text("DROP TABLE IF EXISTS token_usage_tracking"))
            logger.info("Dropped existing token_usage_tracking table")
            
            # Create the new table without foreign key constraint
            conn.execute(text(create_table_sql))
            conn.commit()
            logger.info("✅ Successfully created token_usage_tracking table")
            
            # Show table structure
            result = conn.execute(text("DESCRIBE token_usage_tracking"))
            logger.info("Table structure:")
            for row in result:
                logger.info(f"  {row}")
                
    except Exception as e:
        logger.error(f"❌ Failed to create token_usage_tracking table: {e}")
        raise

def test_table_operations():
    """Test basic CRUD operations on the token_usage_tracking table."""
    
    test_insert_sql = """
    INSERT INTO token_usage_tracking (
        client_id, request_id, query_type, llm_provider, llm_model,
        input_tokens, output_tokens, total_tokens,
        input_cost, output_cost, total_cost,
        request_text_length, response_text_length
    ) VALUES (
        'test_client_123', 'test_req_001', 'embed_search', 'google', 'gemini-embedding-001',
        100, 0, 100,
        0.00001490, 0.00000000, 0.00001490,
        500, 0
    );
    """
    
    test_select_sql = """
    SELECT * FROM token_usage_tracking WHERE client_id = 'test_client_123';
    """
    
    try:
        with engine.connect() as conn:
            # Insert test data
            conn.execute(text(test_insert_sql))
            conn.commit()
            logger.info("✅ Successfully inserted test data")
            
            # Select and verify
            result = conn.execute(text(test_select_sql))
            rows = result.fetchall()
            logger.info(f"✅ Found {len(rows)} test records")
            
            # Clean up test data
            conn.execute(text("DELETE FROM token_usage_tracking WHERE client_id = 'test_client_123'"))
            conn.commit()
            logger.info("✅ Cleaned up test data")
            
    except Exception as e:
        logger.error(f"❌ Table operations test failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Creating token_usage_tracking table...")
    create_token_usage_table()
    
    logger.info("Testing table operations...")
    test_table_operations()
    
    logger.info("🎉 Token usage tracking table setup complete!")
