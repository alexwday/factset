"""
Stage 8: Embeddings Generation
Processes Stage 7 summarized content to create embeddings for each chunk.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 7 pattern with embeddings generation logic.
Uses per-transcript OAuth refresh, incremental saving, and enhanced error logging.
"""

import os
import tempfile
import logging
import json
import time
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
import io
import re
import requests

import yaml
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
import tiktoken

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []  # Detailed execution log entries
error_log = []  # Error log entries (only if errors occur)

# LLM-specific globals
llm_client = None
oauth_token = None
ssl_cert_path = None
tokenizer = None


def setup_logging() -> logging.Logger:
    """Set up minimal console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def log_console(message: str, level: str = "INFO"):
    """Log minimal message to console."""
    global logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


def log_execution(message: str, details: Dict[str, Any] = None):
    """Log detailed execution information for main log file."""
    global execution_log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "details": details or {},
    }
    execution_log.append(log_entry)


def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information for error log file."""
    global error_log
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    error_log.append(error_entry)


class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""

    def __init__(self):
        self.embedding_errors = []
        self.chunking_errors = []
        self.authentication_errors = []
        self.processing_errors = []
        self.validation_errors = []
        self.total_embeddings = 0
        self.total_chunks = 0
        self.using_fallback_tokenizer = False

    def log_embedding_error(self, transcript_id: str, paragraph_id: str, error: str):
        """Log embedding generation errors."""
        self.embedding_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "paragraph_id": paragraph_id,
            "error": error,
            "action_required": "Review embedding generation logic"
        })

    def log_chunking_error(self, transcript_id: str, paragraph_id: str, error: str):
        """Log text chunking errors."""
        self.chunking_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "paragraph_id": paragraph_id,
            "error": error,
            "action_required": "Review chunking logic"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review transcript data and processing logic"
        })

    def log_validation_error(self, transcript_id: str, validation_issue: str):
        """Log validation errors."""
        self.validation_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "validation_issue": validation_issue,
            "action_required": "Review embedding validation logic"
        })

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_path = config["stage_08_embeddings_generation"]["output_logs_path"]
        
        error_types = {
            "embedding": self.embedding_errors,
            "chunking": self.chunking_errors,
            "authentication": self.authentication_errors,
            "processing": self.processing_errors,
            "validation": self.validation_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                error_file_path = nas_path_join(logs_path, f"stage_08_{error_type}_errors_{timestamp}.json")
                
                try:
                    error_data = json.dumps(errors, indent=2, default=str)
                    file_obj = io.BytesIO(error_data.encode('utf-8'))
                    
                    if nas_upload_file(nas_conn, file_obj, error_file_path):
                        log_console(f"Saved {len(errors)} {error_type} errors")
                    else:
                        log_console(f"Failed to save {error_type} errors", "WARNING")
                except Exception as e:
                    log_console(f"Error saving {error_type} errors: {e}", "ERROR")


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any], enhanced_error_logger: EnhancedErrorLogger):
    """Save execution logs, error logs, and summary to NAS."""
    global execution_log, error_log
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_08_embeddings_generation"]["output_logs_path"]
    
    # Save execution log
    if execution_log:
        exec_log_path = nas_path_join(logs_path, f"stage_08_execution_{timestamp}.json")
        exec_log_data = json.dumps(execution_log, indent=2, default=str)
        exec_file_obj = io.BytesIO(exec_log_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, exec_file_obj, exec_log_path):
            log_console(f"Execution log saved: {len(execution_log)} entries")
        else:
            log_console("Failed to save execution log", "WARNING")
    
    # Save error log (if errors occurred)
    if error_log:
        error_log_path = nas_path_join(logs_path, f"stage_08_errors_{timestamp}.json")
        error_log_data = json.dumps(error_log, indent=2, default=str)
        error_file_obj = io.BytesIO(error_log_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, error_file_obj, error_log_path):
            log_console(f"Error log saved: {len(error_log)} entries")
        else:
            log_console("Failed to save error log", "WARNING")
    
    # Save stage summary
    summary_path = nas_path_join(logs_path, f"stage_08_summary_{timestamp}.json")
    summary_data = json.dumps(stage_summary, indent=2, default=str)
    summary_file_obj = io.BytesIO(summary_data.encode('utf-8'))
    
    if nas_upload_file(nas_conn, summary_file_obj, summary_path):
        log_console("Stage summary saved")
    else:
        log_console("Failed to save stage summary", "WARNING")
    
    # Save enhanced error logs
    enhanced_error_logger.save_error_logs(nas_conn)


def nas_path_join(*parts) -> str:
    """Join NAS path parts using forward slashes."""
    return "/".join(parts)


def validate_nas_path(path: str) -> bool:
    """Validate NAS path format."""
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
    for char in invalid_chars:
        if char in path:
            return False
    
    return True


def get_nas_share_name() -> str:
    """Get NAS share name from environment variable."""
    share_name = os.getenv("NAS_SHARE_NAME")
    if not share_name:
        log_error("NAS_SHARE_NAME environment variable not set", "configuration")
        raise ValueError("NAS_SHARE_NAME environment variable not set")
    return share_name


def get_nas_connection() -> Optional[SMBConnection]:
    """Establish connection to NAS using environment variables."""
    try:
        nas_share = get_nas_share_name()
        conn = SMBConnection(
            os.getenv("NAS_USER", ""),
            os.getenv("NAS_PASSWORD", ""),
            "script_client",
            nas_share,
            domain=os.getenv("NAS_DOMAIN", ""),
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if conn.connect(nas_share, 445):
            log_execution("NAS connection established", {"share": nas_share})
            return conn
        else:
            log_error("Failed to connect to NAS", "nas_connection", {"share": nas_share})
            return None
    
    except Exception as e:
        log_error(f"NAS connection error: {str(e)}", "nas_connection", {"error": str(e)})
        return None


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on NAS."""
    try:
        conn.getAttributes(get_nas_share_name(), file_path)
        return True
    except:
        return False


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            conn.retrieveFile(get_nas_share_name(), nas_file_path, temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            os.unlink(temp_file.name)
            return content
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download", {"error": str(e)})
        return None


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory recursively on NAS."""
    try:
        path_parts = dir_path.split('/')
        current_path = ""
        
        for part in path_parts:
            if part:
                current_path = nas_path_join(current_path, part) if current_path else part
                try:
                    conn.createDirectory(get_nas_share_name(), current_path)
                except:
                    pass
        
        return True
    except Exception as e:
        log_error(f"Failed to create directory: {dir_path}", "nas_mkdir", {"error": str(e)})
        return False


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return False
    
    try:
        dir_path = "/".join(nas_file_path.split("/")[:-1])
        if dir_path and not nas_file_exists(conn, nas_file_path):
            nas_create_directory_recursive(conn, dir_path)
        
        local_file_obj.seek(0)
        conn.storeFile(get_nas_share_name(), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload", {"error": str(e)})
        return False


def load_stage_config(nas_conn: SMBConnection) -> Dict:
    """Load and validate Stage 8 configuration from NAS."""
    try:
        config_path = "Finance Data and Analytics/DSA/Earnings Call Transcripts/Scripts/database_refresh/config.yaml"
        config_data = nas_download_file(nas_conn, config_path)
        
        if not config_data:
            raise ValueError("Failed to download config file from NAS")
        
        config = yaml.safe_load(config_data.decode('utf-8'))
        
        required_keys = ["stage_08_embeddings_generation"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        log_execution("Configuration loaded successfully", {
            "config_path": config_path
        })
        
        return config
    
    except Exception as e:
        log_error(f"Failed to load configuration: {str(e)}", "config_load")
        raise


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download SSL certificate from NAS and return temp file path."""
    try:
        ssl_cert_path = config.get("ssl_cert_path")
        if not ssl_cert_path:
            log_console("No SSL certificate path configured")
            return None
        
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
        if not cert_data:
            log_console("Failed to download SSL certificate", "WARNING")
            return None
        
        with tempfile.NamedTemporaryFile(suffix=".cer", delete=False) as temp_cert:
            temp_cert.write(cert_data)
            temp_cert_path = temp_cert.name
        
        log_execution("SSL certificate downloaded", {"temp_path": temp_cert_path})
        return temp_cert_path
    
    except Exception as e:
        log_error(f"SSL certificate setup failed: {e}", "ssl_setup")
        return None


def get_oauth_token() -> Optional[str]:
    """Obtain OAuth token for LLM API access."""
    try:
        token_endpoint = config["stage_08_embeddings_generation"]["llm_config"]["token_endpoint"]
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': os.getenv("LLM_CLIENT_ID"),
            'client_secret': os.getenv("LLM_CLIENT_SECRET")
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        # Set up SSL context if certificate available
        verify_ssl = True
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            verify_ssl = ssl_cert_path
        
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
            error_msg = f"OAuth token request failed: {response.status_code}"
            log_error(error_msg, "oauth_token", {"status_code": response.status_code})
            return None
            
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {e}"
        log_error(error_msg, "oauth_token", {"error": str(e)})
        return None


def setup_llm_client(token: str) -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    try:
        client = OpenAI(
            api_key=token,
            base_url=config["stage_08_embeddings_generation"]["llm_config"]["base_url"],
            timeout=config["stage_08_embeddings_generation"]["llm_config"]["timeout"]
        )
        
        log_execution("LLM client setup completed")
        return client
    except Exception as e:
        log_error(f"Failed to setup LLM client: {e}", "llm_setup")
        return None


def refresh_oauth_token_for_transcript(transcript_info: Dict[str, Any]) -> None:
    """Refresh OAuth token per transcript to prevent expiration."""
    global oauth_token, llm_client
    
    log_execution(f"Refreshing OAuth token for transcript", {
        "transcript_id": transcript_info.get("transcript_id", "unknown")
    })
    
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client(oauth_token)
        log_execution("OAuth token refreshed successfully")
    else:
        raise RuntimeError("Failed to refresh OAuth token")


def save_results_incrementally(results: List[Dict], output_path: str, is_first_batch: bool = False):
    """Save results incrementally after each transcript (following Stage 7 array pattern)."""
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        # Append records using Stage 7's pattern
        if not append_records_to_json_array(nas_conn, results, output_path, is_first_batch):
            raise RuntimeError("Failed to append records to output file")
        
        log_console(f"Appended {len(results)} records to output file")
        log_execution(f"Incrementally saved {len(results)} records", {
            "output_path": output_path,
            "is_first_batch": is_first_batch
        })
        
        nas_conn.close()
        
    except Exception as e:
        error_msg = f"Failed to save results incrementally: {e}"
        log_error(error_msg, "incremental_save", {"error": str(e)})
        raise


def append_records_to_json_array(nas_conn: SMBConnection, records: List[Dict], file_path: str, is_first: bool = False) -> bool:
    """Append records to a JSON array file (Stage 7 pattern)."""
    try:
        # Prepare JSON content
        if not is_first:
            # Add comma before new records if not the first entry
            content = ","
        else:
            # Start the JSON array
            content = "["
        
        # Add records
        records_json = [json.dumps(record, indent=2, default=str) for record in records]
        content += ",\n".join(records_json)
        
        # Check if file exists and download if appending
        if not is_first and nas_file_exists(nas_conn, file_path):
            existing_content = nas_download_file(nas_conn, file_path)
            if existing_content:
                content = existing_content.decode('utf-8') + "," + ",\n".join(records_json)
        
        # Upload the updated content
        file_obj = io.BytesIO(content.encode('utf-8'))
        return nas_upload_file(nas_conn, file_obj, file_path)
        
    except Exception as e:
        log_error(f"Failed to append records: {e}", "json_append")
        return False


def close_json_array(nas_conn: SMBConnection, file_path: str) -> bool:
    """Close a JSON array file by appending the closing bracket (Stage 7 pattern)."""
    try:
        # Download existing content
        existing_content = nas_download_file(nas_conn, file_path)
        if existing_content is None:
            log_error("No existing file to close", "json_close")
            return False
        
        # Add closing bracket
        content = existing_content.decode('utf-8') + "\n]"
        
        # Upload the updated content
        file_obj = io.BytesIO(content.encode('utf-8'))
        return nas_upload_file(nas_conn, file_obj, file_path)
        
    except Exception as e:
        log_error(f"Failed to close JSON array: {e}", "json_close", {"error": str(e)})
        return False


def setup_tokenizer():
    """Setup tiktoken tokenizer for token counting with fallback support."""
    global tokenizer
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        log_execution("Tokenizer initialized with tiktoken")
        log_console("Tokenizer initialized successfully with tiktoken")
        return True
    except Exception as e:
        log_error(f"Failed to setup tiktoken tokenizer: {e}", "tokenizer_setup")
        log_console(f"WARNING: tiktoken initialization failed: {e}", "WARNING")
        log_console("Using fallback token estimation method", "WARNING")
        tokenizer = None  # Explicitly set to None to trigger fallback
        return False


def estimate_tokens_fallback(text: str) -> int:
    """
    Fallback token estimation when tiktoken is not available.
    
    Uses a hybrid approach combining character and word-based estimation.
    Based on empirical observations:
    - Average ~4 characters per token for English text
    - Average ~1.3 tokens per word
    - Adjusts for punctuation and special characters
    """
    if not text:
        return 0
    
    # Method 1: Character-based estimation (tends to underestimate)
    char_estimate = len(text) / 4.0
    
    # Method 2: Word-based estimation (tends to overestimate for technical text)
    words = text.split()
    word_estimate = len(words) * 1.3
    
    # Method 3: Refined character estimate accounting for whitespace
    # Remove extra whitespace for more accurate count
    compressed_text = ' '.join(text.split())
    refined_char_estimate = len(compressed_text) / 3.5
    
    # Take weighted average favoring refined character method
    # This tends to be most accurate for financial transcripts
    final_estimate = (
        refined_char_estimate * 0.5 +  # 50% weight
        char_estimate * 0.3 +           # 30% weight
        word_estimate * 0.2             # 20% weight
    )
    
    # Add 10% buffer for safety in chunking decisions
    # Better to overestimate and chunk more than underestimate and fail
    final_estimate = int(final_estimate * 1.1)
    
    return final_estimate


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken with fallback estimation.
    
    Attempts to use tiktoken for accurate counting, but falls back to
    estimation if tiktoken is not available or fails.
    """
    global tokenizer
    
    # Try to initialize tokenizer if not already done
    if tokenizer is None and not hasattr(count_tokens, '_fallback_warned'):
        setup_tokenizer()
    
    # Try tiktoken if available
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            # Tiktoken failed on this specific text
            log_error(f"tiktoken encoding failed for text: {e}", "token_encoding")
            log_console(f"WARNING: tiktoken encoding failed, using fallback", "WARNING")
            
            # Set flag to avoid repeated warnings
            count_tokens._fallback_warned = True
            
            # Fall through to fallback method
    
    # Use fallback estimation
    if not hasattr(count_tokens, '_fallback_warned'):
        log_console("Using fallback token estimation (tiktoken unavailable)", "WARNING")
        log_execution("Using fallback token estimation method", {
            "reason": "tiktoken unavailable or failed",
            "text_length": len(text)
        })
        count_tokens._fallback_warned = True
    
    return estimate_tokens_fallback(text)


def find_sentence_boundary(text: str, target_pos: int, window: int = 50) -> int:
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


def chunk_text(text: str, max_tokens: int = 500, chunk_threshold: int = 1000, min_final_chunk: int = 300) -> List[Tuple[str, int]]:
    """
    Intelligently chunk text into smaller pieces.
    Returns list of (chunk_text, token_count) tuples.
    """
    total_tokens = count_tokens(text)
    
    # If already under threshold, return as single chunk
    if total_tokens <= chunk_threshold:
        return [(text, total_tokens)]
    
    chunks = []
    remaining_text = text
    
    while remaining_text:
        # Estimate character position for target chunk size
        chars_per_token = len(remaining_text) / count_tokens(remaining_text)
        target_chars = int(max_tokens * chars_per_token)
        
        if len(remaining_text) <= target_chars:
            # Last chunk
            chunk_tokens = count_tokens(remaining_text)
            
            # Check if final chunk is too small
            if chunks and chunk_tokens < min_final_chunk:
                # Merge with previous chunk
                prev_chunk, prev_tokens = chunks[-1]
                merged_chunk = prev_chunk + " " + remaining_text
                merged_tokens = count_tokens(merged_chunk)
                chunks[-1] = (merged_chunk, merged_tokens)
            else:
                chunks.append((remaining_text, chunk_tokens))
            break
        
        # Find good breaking point
        break_pos = find_sentence_boundary(remaining_text, target_chars)
        chunk = remaining_text[:break_pos].strip()
        
        # Validate chunk size
        chunk_tokens = count_tokens(chunk)
        
        # Adjust if chunk is too large
        while chunk_tokens > max_tokens * 1.2:  # Allow 20% overflow
            target_chars = int(target_chars * 0.9)
            break_pos = find_sentence_boundary(remaining_text, target_chars)
            chunk = remaining_text[:break_pos].strip()
            chunk_tokens = count_tokens(chunk)
        
        chunks.append((chunk, chunk_tokens))
        remaining_text = remaining_text[break_pos:].strip()
    
    return chunks


def generate_embedding(text: str, retry_count: int = 3) -> Optional[List[float]]:
    """Generate embedding for text using OpenAI API."""
    global llm_client
    
    embed_config = config["stage_08_embeddings_generation"].get("embedding_config", {})
    model = embed_config.get("model", "text-embedding-3-large")
    
    for attempt in range(retry_count):
        try:
            response = llm_client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            log_console(f"Embedding attempt {attempt + 1} failed: {e}", "WARNING")
            if attempt < retry_count - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    
    return None


def load_stage7_data(nas_conn: SMBConnection) -> List[Dict]:
    """Load Stage 7 summarized content data from NAS."""
    try:
        # Input path is already a full path from config
        nas_file_path = config["stage_08_embeddings_generation"]["input_data_path"]
        
        input_data = nas_download_file(nas_conn, nas_file_path)
        
        if not input_data:
            raise ValueError(f"Failed to download Stage 7 data from: {nas_file_path}")
        
        # Stage 7 outputs JSON array format
        stage7_records = json.loads(input_data.decode('utf-8'))
        
        log_execution(f"Loaded {len(stage7_records)} records from Stage 7")
        return stage7_records
    
    except Exception as e:
        log_error(f"Failed to load Stage 7 data: {str(e)}", "data_load")
        raise


def process_transcript(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process all records for a single transcript to generate embeddings."""
    
    enhanced_records = []
    
    # Get embedding config
    embed_config = config["stage_08_embeddings_generation"].get("embedding_config", {})
    target_chunk_size = embed_config.get("target_chunk_size", 500)
    chunk_threshold = embed_config.get("chunk_threshold", 1000)
    min_final_chunk = embed_config.get("min_final_chunk", 300)
    
    # Group records for block token calculation
    event_blocks = defaultdict(lambda: defaultdict(list))
    
    for record in transcript_records:
        event_id = record.get('event_id', transcript_id)
        
        if record.get('section_type') == 'MD':
            block_key = f"speaker_{record.get('speaker_id')}"
        else:  # Q&A
            block_key = f"qa_{record.get('qa_id')}"
        
        event_blocks[event_id][block_key].append(record)
    
    # Process each record
    for record in transcript_records:
        try:
            event_id = record.get('event_id', transcript_id)
            paragraph_id = record.get('paragraph_id', '')
            
            # Use summary if available, otherwise use original text
            paragraph_text = record.get('paragraph_summary') or record.get('paragraph_text', '')
            if not paragraph_text:
                log_console(f"No text found for paragraph {paragraph_id}", "WARNING")
                continue
            
            # Calculate paragraph tokens
            paragraph_tokens = count_tokens(paragraph_text)
            
            # Calculate block tokens
            if record.get('section_type') == 'MD':
                block_key = f"speaker_{record.get('speaker_id')}"
            else:
                block_key = f"qa_{record.get('qa_id')}"
            
            block_records = event_blocks[event_id][block_key]
            block_tokens = sum(
                count_tokens(r.get('paragraph_summary') or r.get('paragraph_text', ''))
                for r in block_records
            )
            
            # Chunk if necessary
            chunks = chunk_text(
                paragraph_text, 
                max_tokens=target_chunk_size,
                chunk_threshold=chunk_threshold,
                min_final_chunk=min_final_chunk
            )
            
            # Generate embedding for each chunk
            for chunk_id, (chunk_text, chunk_tokens) in enumerate(chunks, 1):
                try:
                    embedding = generate_embedding(chunk_text)
                    
                    if embedding:
                        # Create new record with all original fields plus new ones
                        enhanced_record = {
                            **record,  # Include all original fields
                            'paragraph_tokens': paragraph_tokens,
                            'block_tokens': block_tokens,
                            'chunk_id': chunk_id,
                            'total_chunks': len(chunks),
                            'chunk_text': chunk_text,
                            'chunk_tokens': chunk_tokens,
                            'embedding': embedding
                        }
                        enhanced_records.append(enhanced_record)
                        enhanced_error_logger.total_embeddings += 1
                        enhanced_error_logger.total_chunks += 1
                    else:
                        enhanced_error_logger.log_embedding_error(
                            transcript_id,
                            paragraph_id,
                            f"Failed to generate embedding for chunk {chunk_id}"
                        )
                        
                except Exception as e:
                    enhanced_error_logger.log_embedding_error(
                        transcript_id,
                        paragraph_id,
                        f"Embedding generation error: {str(e)}"
                    )
            
        except Exception as e:
            enhanced_error_logger.log_processing_error(
                transcript_id,
                f"Error processing paragraph {record.get('paragraph_id')}: {str(e)}"
            )
    
    log_execution(f"Processed transcript {transcript_id}", {
        "records_in": len(transcript_records),
        "records_out": len(enhanced_records),
        "chunks_created": enhanced_error_logger.total_chunks
    })
    
    return enhanced_records


def save_failed_transcripts(failed_transcripts: List[Dict], nas_conn: SMBConnection):
    """Save list of failed transcripts to a separate JSON file."""
    if not failed_transcripts:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    failed_summary = {
        "stage": "08_embeddings_generation",
        "timestamp": timestamp,
        "total_failed": len(failed_transcripts),
        "failed_transcripts": failed_transcripts
    }
    
    failed_path = nas_path_join(
        config["stage_08_embeddings_generation"]["output_logs_path"],
        f"stage_08_failed_transcripts_{timestamp}.json"
    )
    
    try:
        failed_data = json.dumps(failed_summary, indent=2, default=str)
        file_obj = io.BytesIO(failed_data.encode('utf-8'))
        
        if nas_upload_file(nas_conn, file_obj, failed_path):
            log_console(f"Saved {len(failed_transcripts)} failed transcripts")
            log_execution(f"Failed transcripts saved", {"count": len(failed_transcripts), "path": failed_path})
    except Exception as e:
        log_error(f"Failed to save failed transcripts: {e}", "save_failed")


def main():
    """Main execution function."""
    global config, logger, ssl_cert_path, llm_client, oauth_token
    
    # Initialize logging
    logger = setup_logging()
    log_console("=" * 50)
    log_console("Stage 8: Embeddings Generation")
    log_console("=" * 50)
    
    start_time = datetime.now()
    
    try:
        # Connect to NAS
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS")
        
        # Load configuration
        config = load_stage_config(nas_conn)
        
        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Setup tokenizer with fallback support
        tokenizer_success = setup_tokenizer()
        if not tokenizer_success:
            log_console("=" * 50)
            log_console("WARNING: Running with fallback token estimation")
            log_console("Token counts will be approximate")
            log_console("Consider installing tiktoken for accurate counts")
            log_console("=" * 50)
            log_execution("Stage 8 running with fallback token estimation", {
                "reason": "tiktoken unavailable",
                "impact": "Token counts will be approximate"
            })
        
        # Load Stage 7 data
        stage7_data = load_stage7_data(nas_conn)
        
        # Group by transcript (matching Stage 7's grouping logic)
        transcripts = defaultdict(list)
        for record in stage7_data:
            # Use filename as the transcript key (from Stage 7 output)
            filename = record.get("filename", "")
            if filename:
                transcript_id = filename
            else:
                # Fallback to ticker_event_id format
                ticker = record.get("ticker", "UNK")
                event_id = record.get("event_id", "unknown")
                transcript_id = f"{ticker}_{event_id}"
            transcripts[transcript_id].append(record)
        
        log_console(f"Loaded {len(stage7_data)} records from {len(transcripts)} transcripts")
        
        # Check for dev mode
        if config["stage_08_embeddings_generation"].get("dev_mode", False):
            max_transcripts = config["stage_08_embeddings_generation"].get("dev_max_transcripts", 10)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            log_console(f"Development mode: limited to {len(transcripts)} transcripts")
        
        # Initialize enhanced error logger
        enhanced_error_logger = EnhancedErrorLogger()
        enhanced_error_logger.using_fallback_tokenizer = (tokenizer is None)
        
        # Process each transcript
        all_enhanced_records = []
        failed_transcripts = []
        
        output_path = nas_path_join(
            config["stage_08_embeddings_generation"]["output_data_path"],
            "stage_08_embeddings.json"
        )
        
        for i, (transcript_id, transcript_records) in enumerate(transcripts.items(), 1):
            transcript_start = datetime.now()
            
            try:
                log_console(f"Processing transcript {i}/{len(transcripts)}: {transcript_id}")
                
                # Refresh OAuth token per transcript
                sample_record = transcript_records[0] if transcript_records else {}
                refresh_oauth_token_for_transcript(sample_record)
                
                # Process transcript
                enhanced_records = process_transcript(transcript_records, transcript_id, enhanced_error_logger)
                
                # Save results incrementally
                save_results_incrementally(enhanced_records, output_path, is_first_batch=(i == 1))
                
                all_enhanced_records.extend(enhanced_records)
                
                transcript_time = datetime.now() - transcript_start
                log_console(f"Completed transcript {transcript_id} in {transcript_time}")
                
                # Rate limiting
                time.sleep(1)
            
            except Exception as e:
                transcript_time = datetime.now() - transcript_start
                enhanced_error_logger.log_processing_error(transcript_id, f"Failed after {transcript_time}: {e}")
                
                failed_transcripts.append({
                    "transcript": transcript_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "processing_time": str(transcript_time)
                })
                
                log_console(f"Failed to process transcript {transcript_id}: {e}", "ERROR")
        
        # Close JSON array
        nas_conn_final = get_nas_connection()
        if nas_conn_final:
            close_json_array(nas_conn_final, output_path)
            nas_conn_final.close()
        
        # Save failed transcripts
        save_failed_transcripts(failed_transcripts, nas_conn)
        
        # Calculate final statistics
        end_time = datetime.now()
        total_time = end_time - start_time
        
        stage_summary = {
            "stage": "08_embeddings_generation",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time": str(total_time),
            "transcripts_processed": len(transcripts),
            "successful_transcripts": len(transcripts) - len(failed_transcripts),
            "failed_transcripts": len(failed_transcripts),
            "total_input_records": len(stage7_data),
            "total_output_records": len(all_enhanced_records),
            "total_embeddings": enhanced_error_logger.total_embeddings,
            "total_chunks": enhanced_error_logger.total_chunks,
            "embedding_errors": len(enhanced_error_logger.embedding_errors),
            "chunking_errors": len(enhanced_error_logger.chunking_errors),
            "processing_errors": len(enhanced_error_logger.processing_errors),
            "tokenizer_method": "fallback_estimation" if enhanced_error_logger.using_fallback_tokenizer else "tiktoken",
            "tokenizer_warning": "Token counts are approximate" if enhanced_error_logger.using_fallback_tokenizer else "Accurate token counts"
        }
        
        # Save logs
        save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)
        
        # Print summary
        log_console("=" * 50)
        log_console(f"Stage 8 completed in {total_time}")
        log_console(f"Transcripts: {len(transcripts)} total, {len(failed_transcripts)} failed")
        log_console(f"Records: {len(stage7_data)} input → {len(all_enhanced_records)} output")
        log_console(f"Embeddings generated: {enhanced_error_logger.total_embeddings}")
        log_console("=" * 50)
        
        # Close NAS connection
        nas_conn.close()
        
        # Clean up SSL certificate
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            os.unlink(ssl_cert_path)
        
        return 0 if len(failed_transcripts) == 0 else 1
    
    except Exception as e:
        log_console(f"Stage 8 failed: {e}", "ERROR")
        log_error(f"Stage 8 fatal error: {e}", "fatal", {"traceback": str(e)})
        return 1


if __name__ == "__main__":
    exit(main())