#!/usr/bin/env python3
"""
Stage 08: Embeddings Generation
Generates vector embeddings from Stage 7 summarized content for semantic search.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import re
from collections import defaultdict

import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
from openai import OpenAI
import tiktoken
import numpy as np
from tqdm import tqdm
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage_08_embeddings.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingsGenerator:
    """Generates embeddings from Stage 7 summarized content."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the embeddings generator."""
        self.config = self._load_config(config_path)
        self.db_conn = None
        self.db_cursor = None
        self.openai_client = None
        self.oauth_token = None
        self.ssl_cert_path = None
        self.tokenizer = None
        self.stats = defaultdict(int)
        
        # Embedding configuration
        self.embedding_model = "text-embedding-3-large"
        self.embedding_dimensions = 3072  # Full dimensions with halfvec support
        self.use_halfvec = True  # Use half-precision vectors for 3072 dimensions
        self.chunk_threshold = 1000  # Tokens threshold for chunking
        self.target_chunk_size = 500  # Target size for chunks
        self.min_final_chunk = 300  # Minimum size for final chunk
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = Path(__file__).parent / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for SSL certificate path
        if 'ssl_cert_path' in config:
            cert_path = Path(__file__).parent.parent / config['ssl_cert_path']
            if cert_path.exists():
                self.ssl_cert_path = str(cert_path)
        
        return config
    
    def connect_db(self):
        """Establish database connection."""
        try:
            self.db_conn = psycopg2.connect(
                host=self.config['database']['host'],
                port=self.config['database']['port'],
                database=self.config['database']['name'],
                user=self.config['database']['user'],
                password=self.config['database']['password']
            )
            self.db_cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def get_oauth_token(self) -> Optional[str]:
        """Obtain OAuth token for LLM API access."""
        try:
            token_endpoint = self.config['stage_08_embeddings_generation']['llm_config']['token_endpoint']
            
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': os.getenv("LLM_CLIENT_ID"),
                'client_secret': os.getenv("LLM_CLIENT_SECRET")
            }
            
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            # Set up SSL context if certificate available
            verify_ssl = True
            if self.ssl_cert_path and os.path.exists(self.ssl_cert_path):
                verify_ssl = self.ssl_cert_path
            
            response = requests.post(
                token_endpoint,
                data=auth_data,
                headers=headers,
                verify=verify_ssl,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return token_data.get('access_token')
            else:
                logger.error(f"OAuth token request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OAuth token acquisition failed: {e}")
            return None
    
    def setup_openai(self):
        """Initialize OpenAI client with OAuth and tokenizer."""
        try:
            # Get OAuth token
            self.oauth_token = self.get_oauth_token()
            if not self.oauth_token:
                raise RuntimeError("Failed to obtain OAuth token")
            
            # Setup OpenAI client with OAuth token
            self.openai_client = OpenAI(
                api_key=self.oauth_token,
                base_url=self.config['stage_08_embeddings_generation']['llm_config']['base_url'],
                timeout=self.config['stage_08_embeddings_generation']['llm_config']['timeout']
            )
            
            # Setup tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            logger.info("OpenAI client with OAuth and tokenizer initialized")
        except Exception as e:
            logger.error(f"Failed to setup OpenAI: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def find_sentence_boundary(self, text: str, target_pos: int, window: int = 50) -> int:
        """Find the nearest sentence boundary to target position."""
        # Look for sentence endings within window
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        
        # Find all sentence endings in window
        endings = []
        for match in re.finditer(r'[.!?]\s+', text[start:end]):
            endings.append(start + match.end())
        
        if endings:
            # Find closest to target
            return min(endings, key=lambda x: abs(x - target_pos))
        
        # Fallback to space if no sentence ending found
        space_pos = text.find(' ', target_pos)
        if space_pos != -1:
            return space_pos
        
        return target_pos
    
    def chunk_text(self, text: str, max_tokens: int = 500) -> List[Tuple[str, int]]:
        """
        Intelligently chunk text into smaller pieces.
        Returns list of (chunk_text, token_count) tuples.
        """
        total_tokens = self.count_tokens(text)
        
        # If already under threshold, return as single chunk
        if total_tokens <= self.chunk_threshold:
            return [(text, total_tokens)]
        
        chunks = []
        remaining_text = text
        
        while remaining_text:
            # Estimate character position for target chunk size
            chars_per_token = len(remaining_text) / self.count_tokens(remaining_text)
            target_chars = int(max_tokens * chars_per_token)
            
            if len(remaining_text) <= target_chars:
                # Last chunk
                chunk_tokens = self.count_tokens(remaining_text)
                
                # Check if final chunk is too small
                if chunks and chunk_tokens < self.min_final_chunk:
                    # Merge with previous chunk
                    prev_chunk, prev_tokens = chunks[-1]
                    merged_chunk = prev_chunk + " " + remaining_text
                    merged_tokens = self.count_tokens(merged_chunk)
                    chunks[-1] = (merged_chunk, merged_tokens)
                else:
                    chunks.append((remaining_text, chunk_tokens))
                break
            
            # Find good breaking point
            break_pos = self.find_sentence_boundary(remaining_text, target_chars)
            chunk = remaining_text[:break_pos].strip()
            
            # Validate chunk size
            chunk_tokens = self.count_tokens(chunk)
            
            # Adjust if chunk is too large
            while chunk_tokens > max_tokens * 1.2:  # Allow 20% overflow
                target_chars = int(target_chars * 0.9)
                break_pos = self.find_sentence_boundary(remaining_text, target_chars)
                chunk = remaining_text[:break_pos].strip()
                chunk_tokens = self.count_tokens(chunk)
            
            chunks.append((chunk, chunk_tokens))
            remaining_text = remaining_text[break_pos:].strip()
        
        return chunks
    
    def generate_embedding(self, text: str, retry_count: int = 3) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI API."""
        for attempt in range(retry_count):
            try:
                # Note: When using full 3072 dimensions, we don't specify dimensions parameter
                # as that's the default for text-embedding-3-large
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                    # dimensions parameter omitted to get full 3072 dimensions
                )
                return response.data[0].embedding
            except Exception as e:
                logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to generate embedding after {retry_count} attempts")
                    return None
    
    def process_stage_7_output(self):
        """Process Stage 7 output to add tokens and generate embeddings."""
        try:
            # Get Stage 7 data
            query = """
                SELECT 
                    id,
                    event_id,
                    file_name,
                    section_type,
                    speaker_id,
                    speaker_name,
                    speaker_title,
                    qa_id,
                    question_speaker_name,
                    answer_speaker_name,
                    paragraph_id,
                    paragraph_sequence,
                    paragraph_text,
                    paragraph_summary,
                    processed_at_stage_7
                FROM stage_07_llm_summarization
                WHERE processed_at_stage_7 IS NOT NULL
                ORDER BY event_id, paragraph_sequence
            """
            
            self.db_cursor.execute(query)
            records = self.db_cursor.fetchall()
            
            logger.info(f"Processing {len(records)} records from Stage 7")
            
            # Group records for block token calculation
            event_blocks = defaultdict(lambda: defaultdict(list))
            
            for record in records:
                event_id = record['event_id']
                
                if record['section_type'] == 'MD':
                    block_key = f"speaker_{record['speaker_id']}"
                else:  # Q&A
                    block_key = f"qa_{record['qa_id']}"
                
                event_blocks[event_id][block_key].append(record)
            
            # Process each record
            processed_records = []
            
            for record in tqdm(records, desc="Processing paragraphs"):
                event_id = record['event_id']
                
                # Calculate paragraph tokens
                paragraph_text = record['paragraph_summary'] or record['paragraph_text']
                paragraph_tokens = self.count_tokens(paragraph_text)
                
                # Calculate block tokens
                if record['section_type'] == 'MD':
                    block_key = f"speaker_{record['speaker_id']}"
                else:
                    block_key = f"qa_{record['qa_id']}"
                
                block_records = event_blocks[event_id][block_key]
                block_tokens = sum(
                    self.count_tokens(r['paragraph_summary'] or r['paragraph_text']) 
                    for r in block_records
                )
                
                # Chunk if necessary
                chunks = self.chunk_text(paragraph_text, self.target_chunk_size)
                
                for chunk_id, (chunk_text, chunk_tokens) in enumerate(chunks, 1):
                    # Generate embedding
                    embedding = self.generate_embedding(chunk_text)
                    
                    if embedding:
                        processed_record = {
                            **dict(record),
                            'paragraph_tokens': paragraph_tokens,
                            'block_tokens': block_tokens,
                            'chunk_id': chunk_id,
                            'chunk_text': chunk_text,
                            'chunk_tokens': chunk_tokens,
                            'embedding': embedding,
                            'embedding_model': self.embedding_model,
                            'embedding_dimensions': self.embedding_dimensions,
                            'processed_at_stage_8': datetime.now()
                        }
                        processed_records.append(processed_record)
                        
                        self.stats['chunks_created'] += 1
                        self.stats['embeddings_generated'] += 1
                    else:
                        self.stats['embedding_failures'] += 1
                
                self.stats['paragraphs_processed'] += 1
                
                # Rate limiting and OAuth refresh
                if self.stats['embeddings_generated'] % 100 == 0:
                    time.sleep(1)  # Brief pause every 100 embeddings
                    
                # Refresh OAuth token every 500 embeddings to prevent expiration
                if self.stats['embeddings_generated'] % 500 == 0 and self.stats['embeddings_generated'] > 0:
                    logger.info("Refreshing OAuth token...")
                    self.setup_openai()
            
            return processed_records
            
        except Exception as e:
            logger.error(f"Error processing Stage 7 output: {e}")
            raise
    
    def save_to_database(self, records: List[Dict[str, Any]]):
        """Save processed records with embeddings to database."""
        try:
            # Ensure pgvector extension is installed
            self.db_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            self.db_conn.commit()
            
            # Create table with halfvec for 3072 dimensions
            create_table_query = """
                CREATE TABLE IF NOT EXISTS stage_08_embeddings (
                    id SERIAL PRIMARY KEY,
                    event_id VARCHAR(50),
                    file_name TEXT,
                    section_type VARCHAR(10),
                    speaker_id INTEGER,
                    speaker_name TEXT,
                    speaker_title TEXT,
                    qa_id INTEGER,
                    question_speaker_name TEXT,
                    answer_speaker_name TEXT,
                    paragraph_id VARCHAR(100),
                    paragraph_sequence INTEGER,
                    paragraph_text TEXT,
                    paragraph_summary TEXT,
                    paragraph_tokens INTEGER,
                    block_tokens INTEGER,
                    chunk_id INTEGER,
                    chunk_text TEXT,
                    chunk_tokens INTEGER,
                    embedding halfvec(3072),  -- Using halfvec for full 3072 dimensions
                    embedding_model VARCHAR(100),
                    embedding_dimensions INTEGER,
                    processed_at_stage_7 TIMESTAMP,
                    processed_at_stage_8 TIMESTAMP,
                    UNIQUE(paragraph_id, chunk_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_stage_08_event_id ON stage_08_embeddings(event_id);
                -- Using halfvec_cosine_ops for halfvec type with HNSW index (better performance)
                CREATE INDEX IF NOT EXISTS idx_stage_08_embedding ON stage_08_embeddings 
                    USING hnsw (embedding halfvec_cosine_ops);
            """
            
            self.db_cursor.execute(create_table_query)
            self.db_conn.commit()
            
            # Insert records
            insert_query = """
                INSERT INTO stage_08_embeddings (
                    event_id, file_name, section_type, speaker_id, speaker_name,
                    speaker_title, qa_id, question_speaker_name, answer_speaker_name,
                    paragraph_id, paragraph_sequence, paragraph_text, paragraph_summary,
                    paragraph_tokens, block_tokens, chunk_id, chunk_text, chunk_tokens,
                    embedding, embedding_model, embedding_dimensions,
                    processed_at_stage_7, processed_at_stage_8
                ) VALUES (
                    %(event_id)s, %(file_name)s, %(section_type)s, %(speaker_id)s, %(speaker_name)s,
                    %(speaker_title)s, %(qa_id)s, %(question_speaker_name)s, %(answer_speaker_name)s,
                    %(paragraph_id)s, %(paragraph_sequence)s, %(paragraph_text)s, %(paragraph_summary)s,
                    %(paragraph_tokens)s, %(block_tokens)s, %(chunk_id)s, %(chunk_text)s, %(chunk_tokens)s,
                    %(embedding)s, %(embedding_model)s, %(embedding_dimensions)s,
                    %(processed_at_stage_7)s, %(processed_at_stage_8)s
                )
                ON CONFLICT (paragraph_id, chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    chunk_tokens = EXCLUDED.chunk_tokens,
                    processed_at_stage_8 = EXCLUDED.processed_at_stage_8
            """
            
            for record in tqdm(records, desc="Saving to database"):
                # Embedding is already in list format from OpenAI API
                # PostgreSQL will handle conversion to halfvec automatically
                self.db_cursor.execute(insert_query, record)
            
            self.db_conn.commit()
            logger.info(f"Saved {len(records)} records to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            self.db_conn.rollback()
            raise
    
    def print_statistics(self):
        """Print processing statistics."""
        logger.info("\n" + "="*50)
        logger.info("STAGE 8 PROCESSING STATISTICS")
        logger.info("="*50)
        logger.info(f"Paragraphs processed: {self.stats['paragraphs_processed']}")
        logger.info(f"Chunks created: {self.stats['chunks_created']}")
        logger.info(f"Embeddings generated: {self.stats['embeddings_generated']}")
        logger.info(f"Embedding failures: {self.stats['embedding_failures']}")
        logger.info(f"Average chunks per paragraph: {self.stats['chunks_created'] / max(1, self.stats['paragraphs_processed']):.2f}")
        logger.info("="*50)
    
    def run(self, limit: Optional[int] = None):
        """Main execution method."""
        try:
            logger.info("Starting Stage 8: Embeddings Generation")
            
            # Initialize connections
            self.connect_db()
            self.setup_openai()
            
            # Process Stage 7 output
            processed_records = self.process_stage_7_output()
            
            if processed_records:
                # Save to database
                self.save_to_database(processed_records)
                
                # Print statistics
                self.print_statistics()
                
                logger.info("Stage 8 completed successfully")
            else:
                logger.warning("No records processed")
            
        except Exception as e:
            logger.error(f"Stage 8 failed: {e}")
            raise
        finally:
            if self.db_cursor:
                self.db_cursor.close()
            if self.db_conn:
                self.db_conn.close()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stage 8: Generate embeddings from summarized content")
    parser.add_argument('--config', default='../config.yaml', help='Path to config file')
    parser.add_argument('--limit', type=int, help='Limit number of events to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = EmbeddingsGenerator(args.config)
        generator.run(limit=args.limit)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()