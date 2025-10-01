"""
Stage 7: LLM-Based Content Summarization
Processes Stage 6 classified content to create block-level summaries for reranking.
Self-contained standalone script that loads config from NAS at runtime.

Creates condensed summaries at the block level:
- Q&A Groups: One summary per complete Q&A conversation
- Management Discussion: One summary per speaker block
- Focus on speaker attribution, key points, and retrieval relevance

Architecture based on Stage 5/6 patterns with Stage 7 summarization logic.
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
        self.summarization_errors = []
        self.authentication_errors = []
        self.processing_errors = []
        self.validation_errors = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def log_summarization_error(self, transcript_id: str, section_type: str, error: str):
        """Log summarization-specific errors."""
        self.summarization_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "section_type": section_type,
            "error": error,
            "action_required": "Review transcript content and summarization logic"
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
            "action_required": "Review summary validation logic"
        })

    def accumulate_costs(self, token_usage: dict = None):
        """Accumulate total cost and token usage for final summary."""
        if token_usage:
            if "cost" in token_usage:
                self.total_cost += token_usage["cost"]["total_cost"]
            if "total_tokens" in token_usage:
                self.total_tokens += token_usage["total_tokens"]

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_path = config["stage_07_llm_summarization"]["output_logs_path"]
        
        error_types = {
            "summarization": self.summarization_errors,
            "authentication": self.authentication_errors,
            "processing": self.processing_errors,
            "validation": self.validation_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                filename = f"stage_07_llm_summarization_{error_type}_errors_{timestamp}.json"
                nas_path = nas_path_join(logs_path, "Errors", filename)
                
                try:
                    error_data = {
                        "error_type": error_type,
                        "timestamp": timestamp,
                        "total_errors": len(errors),
                        "errors": errors
                    }
                    
                    error_json = json.dumps(error_data, indent=2)
                    error_bytes = io.BytesIO(error_json.encode('utf-8'))
                    
                    if nas_upload_file(nas_conn, error_bytes, nas_path):
                        log_execution(f"Uploaded {error_type} error log: {filename}")
                    
                except Exception as e:
                    log_error(f"Failed to upload {error_type} error log: {e}", "error_logging", {})


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any], enhanced_error_logger: EnhancedErrorLogger):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_07_llm_summarization"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_07_llm_summarization",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_07_llm_summarization_{timestamp}.json"
    main_log_path = nas_path_join(logs_path, main_log_filename)
    main_log_json = json.dumps(main_log_content, indent=2)
    main_log_obj = io.BytesIO(main_log_json.encode("utf-8"))

    if nas_upload_file(nas_conn, main_log_obj, main_log_path):
        log_console(f"Execution log saved: {main_log_filename}")

    # Save enhanced error logs
    enhanced_error_logger.save_error_logs(nas_conn)

    # Save basic error log only if errors exist
    if error_log:
        errors_path = nas_path_join(logs_path, "Errors")
        nas_create_directory_recursive(nas_conn, errors_path)

        error_log_content = {
            "stage": "stage_07_llm_summarization",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_07_llm_summarization_errors_{timestamp}.json"
        error_log_path = nas_path_join(errors_path, error_log_filename)
        error_log_json = json.dumps(error_log_content, indent=2)
        error_log_obj = io.BytesIO(error_log_json.encode("utf-8"))

        if nas_upload_file(nas_conn, error_log_obj, error_log_path):
            log_console(f"Error log saved: {error_log_filename}", "WARNING")


def validate_environment_variables() -> None:
    """Validate all required environment variables are present."""

    required_env_vars = [
        "API_USERNAME",
        "API_PASSWORD", 
        "PROXY_USER",
        "PROXY_PASSWORD",
        "PROXY_URL",
        "NAS_USERNAME",
        "NAS_PASSWORD",
        "NAS_SERVER_IP",
        "NAS_SERVER_NAME",
        "NAS_SHARE_NAME",
        "NAS_BASE_PATH",
        "NAS_PORT",
        "CONFIG_PATH",
        "CLIENT_MACHINE_NAME",
        "LLM_CLIENT_ID",
        "LLM_CLIENT_SECRET",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log_error(
            error_msg,
            "environment_validation",
            {
                "missing_variables": missing_vars,
                "total_required": len(required_env_vars),
            },
        )
        raise ValueError(error_msg)

    log_execution(
        "Environment variables validated successfully",
        {
            "total_variables": len(required_env_vars),
            "variables_checked": required_env_vars,
        },
    )


def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""

    try:
        conn = SMBConnection(
            username=os.getenv("NAS_USERNAME"),
            password=os.getenv("NAS_PASSWORD"),
            my_name=os.getenv("CLIENT_MACHINE_NAME"),
            remote_name=os.getenv("NAS_SERVER_NAME"),
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )

        nas_port = int(os.getenv("NAS_PORT", 445))
        if conn.connect(os.getenv("NAS_SERVER_IP"), nas_port):
            log_execution(
                "NAS connection established successfully",
                {
                    "connection_type": "SMB/CIFS",
                    "server": os.getenv("NAS_SERVER_NAME"),
                    "port": nas_port,
                },
            )
            return conn
        else:
            log_error(
                "Failed to establish NAS connection",
                "nas_connection",
                {
                    "server": os.getenv("NAS_SERVER_NAME"),
                    "port": nas_port,
                },
            )
            return None

    except Exception as e:
        log_error(
            f"NAS connection error: {e}",
            "nas_connection",
            {
                "error": str(e),
                "server": os.getenv("NAS_SERVER_NAME"),
            },
        )
        return None


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = [str(part).strip("/") for part in parts if part]
    return "/".join(clean_parts)


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception:
        return False


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "security_validation", {})
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download", {"error": str(e)})
        return None


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory structure recursively on NAS."""
    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        return True

    path_parts = [part for part in normalized_path.split("/") if part]
    current_path = ""
    
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part
        try:
            conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
        except Exception:
            # Directory might already exist
            pass
    
    return True


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "security_validation", {})
        return False

    try:
        # Create parent directories
        path_parts = nas_file_path.split("/")[:-1]
        parent_dir = "/".join(path_parts)
        if parent_dir:
            nas_create_directory_recursive(conn, parent_dir)

        local_file_obj.seek(0)
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload", {"error": str(e)})
        return False


def load_stage_config(nas_conn: SMBConnection) -> Dict:
    """Load and validate Stage 7 configuration from NAS."""
    try:
        log_execution("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))

        if config_data:
            full_config = yaml.safe_load(config_data.decode("utf-8"))
            log_execution("Successfully loaded shared configuration from NAS")

            if "stage_07_llm_summarization" not in full_config:
                raise ValueError("Stage 7 configuration not found in config file")
                
            stage_config = full_config["stage_07_llm_summarization"]
            
            # Validate required configuration sections
            required_sections = ["llm_config"]
            for section in required_sections:
                if section not in stage_config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return full_config
        else:
            raise FileNotFoundError(f"Configuration file not found at {os.getenv('CONFIG_PATH')}")

    except json.JSONDecodeError as e:
        log_error(f"Invalid JSON in config file: {e}", "config_loading", {})
        raise
    except ValueError as e:
        log_error(f"Configuration validation failed: {e}", "config_validation", {})
        raise
    except Exception as e:
        log_error(f"Error loading config from NAS: {e}", "config_loading", {})
        raise


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download SSL certificate from NAS to temporary file."""
    try:
        log_execution("Setting up SSL certificate", {"ssl_cert_path": config["ssl_cert_path"]})
        
        cert_data = nas_download_file(nas_conn, config["ssl_cert_path"])
        if not cert_data:
            error_msg = f"SSL certificate not found at {config['ssl_cert_path']}"
            log_error(error_msg, "ssl_setup", {"path": config["ssl_cert_path"]})
            raise FileNotFoundError(error_msg)
        
        # Create temporary certificate file
        temp_cert_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".cer", delete=False
        )
        temp_cert_file.write(cert_data)
        temp_cert_file.close()
        
        # Set SSL environment variable for requests
        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert_file.name
        os.environ["SSL_CERT_FILE"] = temp_cert_file.name
        
        log_execution("SSL certificate setup complete", {"temp_cert_path": temp_cert_file.name})
        return temp_cert_file.name
        
    except Exception as e:
        error_msg = f"Error setting up SSL certificate: {e}"
        log_error(error_msg, "ssl_setup", {"exception_type": type(e).__name__})
        raise


def get_oauth_token() -> Optional[str]:
    """Obtain OAuth token for LLM API access."""
    try:
        token_endpoint = config["stage_07_llm_summarization"]["llm_config"]["token_endpoint"]
        
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
            base_url=config["stage_07_llm_summarization"]["llm_config"]["base_url"],
            timeout=config["stage_07_llm_summarization"]["llm_config"]["timeout"]
        )
        
        log_execution("LLM client setup completed")
        return client
        
    except Exception as e:
        error_msg = f"LLM client setup failed: {e}"
        log_error(error_msg, "llm_client_setup", {"error": str(e)})
        return None


def refresh_oauth_token_for_transcript(transcript_info: Dict[str, Any]) -> None:
    """Refresh OAuth token per transcript to prevent expiration."""
    global oauth_token, llm_client
    
    log_execution(f"Refreshing OAuth token for transcript", {
        "transcript": f"{transcript_info.get('ticker')}_{transcript_info.get('fiscal_year')}_{transcript_info.get('fiscal_quarter')}"
    })
    
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client(oauth_token)
        if not llm_client:
            raise RuntimeError("Failed to setup LLM client after token refresh")
    else:
        raise RuntimeError("Failed to refresh OAuth token")


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> Dict:
    """Calculate token costs based on configuration."""
    global config
    
    prompt_cost_per_1k = config["stage_07_llm_summarization"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_07_llm_summarization"]["llm_config"]["cost_per_1k_completion_tokens"]
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": round(total_cost, 6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }


def save_results_incrementally(results: List[Dict], output_path: str, is_first_batch: bool = False):
    """Save results incrementally after each transcript (following Stage 6 array pattern)."""
    global logger
    
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        if is_first_batch:
            # Initialize with opening bracket like Stage 6
            output_bytes = io.BytesIO(b"[\n")
            if not nas_upload_file(nas_conn, output_bytes, output_path):
                raise RuntimeError("Failed to initialize output file")
            log_console(f"Initialized output file for incremental saving")
        
        # Append records using Stage 6's pattern
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
    """Append records to a JSON array file (Stage 6 pattern)."""
    try:
        # Prepare JSON content
        if not is_first:
            # Add comma before new records if not the first entry
            records_json = ",\n" + json.dumps(records, indent=2)[1:-1]  # Remove [ and ]
        else:
            # First entry, no leading comma
            records_json = json.dumps(records, indent=2)[1:-1]  # Remove [ and ]
        
        # Download existing file to append
        existing_content = nas_download_file(nas_conn, file_path)
        if existing_content is None:
            log_error(f"Failed to download file for append: {file_path}", "output_save", {"path": file_path})
            return False
        
        # Append new records
        new_content = existing_content + records_json.encode('utf-8')
        content_bytes = io.BytesIO(new_content)
        content_bytes.seek(0)
        
        return nas_upload_file(nas_conn, content_bytes, file_path)
        
    except Exception as e:
        log_error(f"Failed to append records: {e}", "output_save", {"path": file_path, "error": str(e)})
        return False


def close_json_array(nas_conn: SMBConnection, file_path: str) -> bool:
    """Close a JSON array file by appending the closing bracket (Stage 6 pattern)."""
    try:
        # Download existing content
        existing_content = nas_download_file(nas_conn, file_path)
        if existing_content is None:
            return False
        
        # Add closing bracket
        new_content = existing_content + b"\n]"
        
        content_bytes = io.BytesIO(new_content)
        content_bytes.seek(0)
        
        return nas_upload_file(nas_conn, content_bytes, file_path)
        
    except Exception as e:
        log_error(f"Failed to close JSON array: {e}", "output_save", {"path": file_path, "error": str(e)})
        return False


# Security validation functions (from Stage 5/6)
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    try:
        normalized = os.path.normpath(path)
        return not ('..' in normalized or normalized.startswith('/'))
    except Exception:
        return False


def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    if not path or not isinstance(path, str):
        return False
    
    normalized = path.strip("/")
    if not normalized:
        return False
        
    parts = normalized.split("/")
    for part in parts:
        if not part or part in [".", ".."]:
            return False
        if not validate_file_path(part):
            return False
    
    return True


def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    try:
        if 'token=' in url:
            return url.split('token=')[0] + 'token=***'
        if 'api_key=' in url:
            return url.split('api_key=')[0] + 'api_key=***'
        return url
    except Exception:
        return "URL_SANITIZATION_ERROR"


def load_existing_output(nas_conn: SMBConnection, output_path: str) -> set:
    """
    Load existing output file and extract completed transcript IDs.
    Returns a set of transcript IDs (filenames) that have already been processed.
    Adapted from Stage 5/6 pattern for Stage 7 resume capability.
    """
    try:
        output_filename = "stage_07_summarized_content.json"
        output_file_path = nas_path_join(output_path, output_filename)

        # Check if file exists
        if not nas_file_exists(nas_conn, output_file_path):
            log_execution("No existing output file found, starting fresh")
            return set()

        # Download existing file
        log_console("Loading existing output file for resume...")
        existing_content = nas_download_file(nas_conn, output_file_path)
        if existing_content is None:
            log_execution("Could not load existing output file, starting fresh")
            return set()

        # Parse JSON content
        try:
            # Handle incomplete JSON arrays (missing closing bracket)
            content_str = existing_content.decode('utf-8').strip()
            if content_str and not content_str.endswith(']'):
                content_str += ']'  # Add closing bracket for parsing

            if not content_str or content_str == '[\n]' or content_str == '[]':
                log_execution("Existing output file is empty, starting fresh")
                return set()

            records = json.loads(content_str)

            # Extract unique transcript IDs (filenames)
            completed_ids = set()
            for record in records:
                # Stage 7 uses filename as transcript key
                filename = record.get('filename', '')
                if filename:
                    completed_ids.add(filename)

            # Get unique count
            unique_completed = len(completed_ids)
            log_console(f"Found {unique_completed} completed transcripts in existing output")
            log_execution(f"Loaded existing output with {len(records)} records from {unique_completed} transcripts")

            return completed_ids

        except json.JSONDecodeError as e:
            log_error(f"Failed to parse existing output file: {e}", "resume_load", {"error": str(e)})
            log_console("WARNING: Could not parse existing output, starting fresh", "WARNING")
            return set()

    except Exception as e:
        log_error(f"Error loading existing output: {e}", "resume_load", {"error": str(e)})
        return set()


def load_stage6_data(nas_conn: SMBConnection) -> List[Dict]:
    """Load Stage 6 classified content data from NAS."""
    try:
        # Don't prepend NAS_BASE_PATH - it's already included in the config path
        input_path = config["stage_07_llm_summarization"]["input_data_path"]
        
        log_execution(f"Loading Stage 6 data from: {input_path}")
        
        if not validate_nas_path(input_path):
            raise ValueError(f"Invalid input file path: {input_path}")
            
        input_data = nas_download_file(nas_conn, input_path)
        if not input_data:
            raise RuntimeError(f"Failed to download Stage 6 data from: {input_path}")
        
        # Decode the data
        input_str = input_data.decode('utf-8')
        
        # Check if the JSON array is properly closed
        # Stage 6 uses incremental writing and may not have closed the array properly
        input_str_stripped = input_str.rstrip()
        if input_str_stripped and not input_str_stripped.endswith(']'):
            log_console("Stage 6 output file not properly closed, attempting to fix...", "WARNING")
            
            # Check if we need to add a closing bracket
            # Remove any trailing comma and whitespace
            if input_str_stripped.endswith(','):
                input_str_stripped = input_str_stripped[:-1].rstrip()
            
            # Add closing bracket
            input_str = input_str_stripped + '\n]'
            
            log_execution("Fixed unclosed JSON array from Stage 6")
        
        # Stage 6 outputs JSON array format
        try:
            stage6_records = json.loads(input_str)
        except json.JSONDecodeError as e:
            # If still failing, try to diagnose the issue
            log_error(f"JSON decode error at position {e.pos}: {e.msg}", "json_parsing", {
                "line": e.lineno if hasattr(e, 'lineno') else None,
                "column": e.colno if hasattr(e, 'colno') else None,
                "position": e.pos,
                "message": e.msg
            })
            
            # Try to show context around the error
            if hasattr(e, 'pos') and e.pos:
                start = max(0, e.pos - 100)
                end = min(len(input_str), e.pos + 100)
                context = input_str[start:end]
                log_console(f"Context around error: ...{context}...", "ERROR")
            
            raise
        
        log_execution(f"Loaded {len(stage6_records)} records from Stage 6")
        return stage6_records
        
    except Exception as e:
        error_msg = f"Failed to load Stage 6 data: {e}"
        log_error(error_msg, "data_loading", {"error": str(e)})
        raise


def group_records_by_transcript(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by transcript using filename format (ticker_event_id)."""
    transcripts = defaultdict(list)
    
    for record in records:
        # Use filename as the transcript key (from Stage 7 logic)
        filename = record.get("filename", "")
        if filename:
            transcript_key = filename
        else:
            # Fallback to ticker_event_id format
            ticker = record.get("ticker", "unknown")
            event_id = record.get("event_id", "unknown")
            transcript_key = f"{ticker}_{event_id}"
        
        transcripts[transcript_key].append(record)
    
    log_execution(f"Grouped {len(records)} records into {len(transcripts)} transcripts")
    return dict(transcripts)


# Summarization tools and prompts (from current Stage 7)
def create_qa_group_summary_tools() -> List[Dict]:
    """Function calling schema for QA group summarization."""
    return [{
        "type": "function",
        "function": {
            "name": "summarize_qa_group",
            "description": "Create a condensed block summary for Q&A conversation retrieval and reranking",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string", 
                        "description": "Brief, condensed summary stating: (1) Speaker names involved, (2) Core topics discussed, (3) Retrieval relevance - what queries this block would answer"
                    }
                },
                "required": ["summary"]
            }
        }
    }]


def create_management_summary_tools() -> List[Dict]:
    """Function calling schema for Management Discussion summarization."""
    return [{
        "type": "function",
        "function": {
            "name": "summarize_management_speaker_block",
            "description": "Create a condensed block summary for management discussion retrieval and reranking",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief, condensed summary stating: (1) Executive speaker name, (2) Key points covered, (3) Retrieval relevance - what financial/business queries this block addresses"
                    }
                },
                "required": ["summary"]
            }
        }
    }]


def create_qa_group_summary_prompt(company_name: str, fiscal_info: str, 
                                  qa_group_records: List[Dict], 
                                  financial_categories: List[str]) -> str:
    """Create optimized prompt for QA group summarization (from current Stage 7)."""
    
    categories_str = ', '.join(financial_categories) if financial_categories else 'General Business'
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <call_type>Earnings Call Q&A Session</call_type>
  <qa_group_size>{len(qa_group_records)} paragraphs</qa_group_size>
  <financial_focus>{categories_str}</financial_focus>
</context>

<objective>
Create a BRIEF, CONDENSED summary for reranking and retrieval relevance assessment.
This summary will be used to determine if retrieved chunks match user queries.

Your summary must be structured in THREE parts:
1. SPEAKERS: List the analyst and executive names involved
2. CONTENT: 2-3 sentence condensed summary of what was discussed
3. RELEVANCE: What types of queries this block would be useful for

CRITICAL REQUIREMENTS:
- Keep the summary BRIEF and CONDENSED (aim for 3-4 sentences total)
- Always start with speaker names
- Focus on the core topic/question and key answer points
- End with retrieval relevance (e.g., "Useful for queries about...")
- Include specific metrics, numbers, or guidance mentioned
</objective>

<format_example>
"[Analyst: John Smith] asks about Q3 margin compression. [CFO: Jane Doe] explains 200bps decline due to supply chain costs and expects normalization by Q1. Useful for queries about margins, profitability, supply chain impact, or financial guidance."
</format_example>

<style>
Ultra-concise block summary optimized for reranking decisions.
</style>

<tone>
Direct and factual. No fluff, just core information for retrieval matching.
</tone>

<audience>
A reranking model that needs to quickly assess if this block matches a user's query.
</audience>

<response_format>
Use the summarize_qa_group function.
- One brief block summary following the three-part structure
- Keep it condensed and retrieval-focused
</response_format>
"""


def create_management_speaker_block_prompt(company_name: str, fiscal_info: str,
                                          speaker_block_records: List[Dict],
                                          speaker_name: str,
                                          financial_categories: List[str],
                                          block_position: str) -> str:
    """Create prompt for Management speaker block summarization."""
    
    categories_str = ', '.join(financial_categories) if financial_categories else 'General Business'
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <call_type>Earnings Call Management Discussion</call_type>
  <speaker>{speaker_name}</speaker>
  <block_size>{len(speaker_block_records)} paragraphs</block_size>
  <block_position>{block_position}</block_position>
  <financial_focus>{categories_str}</financial_focus>
</context>

<objective>
Create a BRIEF, CONDENSED summary for reranking and retrieval relevance assessment.
This summary will be used to determine if retrieved chunks match user queries.

Your summary must be structured in THREE parts:
1. SPEAKER: Executive name and role (if mentioned)
2. CONTENT: 2-3 sentence condensed summary of key points discussed
3. RELEVANCE: What types of queries this block would answer

CRITICAL REQUIREMENTS:
- Keep the summary BRIEF and CONDENSED (aim for 3-4 sentences total)
- Always start with "[Executive: {speaker_name}]"
- Focus on the 2-3 most important points or metrics
- End with retrieval relevance (e.g., "Useful for queries about...")
- Include specific numbers, guidance, or strategic decisions mentioned
- Make this block distinguishable from other speaker blocks

Block position context:
- {block_position} blocks typically contain {'overview and key results' if block_position == 'opening' else 'detailed analysis' if block_position == 'middle' else 'forward guidance and outlook'}
</objective>

<format_example>
"[CEO: John Smith] reports record Q3 revenue of $2.3B (+15% YoY) driven by cloud services growth. Announces $500M buyback program and raises FY guidance to $9.2B. Useful for queries about quarterly performance, revenue growth, cloud business, capital allocation, or financial guidance."
</format_example>

<style>
Ultra-concise block summary optimized for reranking decisions.
</style>

<tone>
Direct and factual. Focus on key financial/business points for retrieval matching.
</tone>

<audience>
A reranking model that needs to quickly assess if this block matches a user's query.
</audience>

<response_format>
Use the summarize_management_speaker_block function.
- One brief block summary following the three-part structure
- Keep it condensed and retrieval-focused
- Make it distinctive from other blocks
</response_format>
"""


def process_transcript_qa_groups(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process Q&A groups in a transcript for summarization."""
    global llm_client, config
    
    # Group Q&A records by qa_group_id
    qa_records = [r for r in transcript_records if r.get("section_name") == "Q&A" and r.get("qa_group_id")]
    if not qa_records:
        return []
    
    qa_groups = defaultdict(list)
    for record in qa_records:
        qa_groups[record["qa_group_id"]].append(record)
    
    log_execution(f"Processing {len(qa_groups)} Q&A groups for transcript {transcript_id}")
    
    enhanced_records = []
    
    for qa_group_id in sorted(qa_groups.keys()):
        qa_group_records = qa_groups[qa_group_id]
        
        try:
            # Build context for Q&A group with company information
            context_parts = []
            
            # Add company and earnings call context
            sample_record = qa_group_records[0]
            company_name = sample_record.get("company_name", "Unknown")
            fiscal_year = sample_record.get("fiscal_year", "Unknown")
            fiscal_quarter = sample_record.get("fiscal_quarter", "Unknown")
            
            context_parts.append(f"=== {company_name} {fiscal_quarter} {fiscal_year} EARNINGS CALL TRANSCRIPT ===")
            context_parts.append("=== Q&A SECTION ===")
            context_parts.append("")
            context_parts.append("=== Q&A CONVERSATION TO SUMMARIZE ===")
            
            for record in qa_group_records:
                role = "[ANALYST]" if "analyst" in record.get("speaker", "").lower() else "[EXECUTIVE]"
                context_parts.append(f"{role} {record.get('speaker', 'Unknown')}:")
                context_parts.append(record.get("paragraph_content", ""))
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Create QA group summary prompt
            system_prompt = create_qa_group_summary_prompt(
                company_name=qa_group_records[0].get("company_name", "Unknown"),
                fiscal_info=f"{qa_group_records[0].get('fiscal_year')} {qa_group_records[0].get('fiscal_quarter')}",
                qa_group_records=qa_group_records,
                financial_categories=qa_group_records[0].get("category_type", [])
            )
            
            # LLM call for Q&A group summarization
            response = llm_client.chat.completions.create(
                model=config["stage_07_llm_summarization"]["llm_config"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                tools=create_qa_group_summary_tools(),
                tool_choice="required",
                temperature=config["stage_07_llm_summarization"]["llm_config"]["temperature"],
                max_tokens=config["stage_07_llm_summarization"]["llm_config"]["max_tokens"],
                timeout=config["stage_07_llm_summarization"]["llm_config"]["timeout"]
            )
            
            # Process LLM response
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                response_data = json.loads(tool_call.function.arguments)
                
                summary = response_data.get("summary", "")
                
                # Apply same summary to ALL paragraphs in group
                for record in qa_group_records:
                    record["block_summary"] = summary
                
                enhanced_records.extend(qa_group_records)
                
                # Track costs
                if hasattr(response, 'usage') and response.usage:
                    cost_data = calculate_token_cost(
                        response.usage.prompt_tokens, 
                        response.usage.completion_tokens
                    )
                    enhanced_error_logger.accumulate_costs({
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_data
                    })
                
                log_execution(f"Summarized Q&A group {qa_group_id} with {len(qa_group_records)} paragraphs")
            
            else:
                enhanced_error_logger.log_summarization_error(
                    transcript_id, 
                    "Q&A",
                    f"No tool calls in Q&A group {qa_group_id} LLM response"
                )
                
                # Set null values for failed group
                for record in qa_group_records:
                    record["block_summary"] = None
                
                enhanced_records.extend(qa_group_records)
        
        except Exception as e:
            enhanced_error_logger.log_summarization_error(
                transcript_id,
                "Q&A", 
                f"Q&A group {qa_group_id} processing failed: {e}"
            )
            
            # Set null values for failed group
            for record in qa_group_records:
                record["block_summary"] = None
            
            enhanced_records.extend(qa_group_records)
    
    return enhanced_records


def process_transcript_management(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process Management Discussion section for summarization at speaker block level."""
    global llm_client, config
    
    # Get Management Discussion records
    mgmt_records = [r for r in transcript_records if r.get("section_name") == "MANAGEMENT DISCUSSION SECTION"]
    if not mgmt_records:
        return []
    
    # Group by speaker blocks
    speaker_blocks = defaultdict(list)
    for record in mgmt_records:
        block_id = record.get("speaker_block_id")
        speaker_blocks[block_id].append(record)
    
    # Sort blocks by ID
    sorted_block_ids = sorted(speaker_blocks.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    log_execution(f"Processing Management Discussion with {len(speaker_blocks)} speaker blocks for transcript {transcript_id}")
    
    enhanced_records = []
    
    for block_index, block_id in enumerate(sorted_block_ids):
        block_records = speaker_blocks[block_id]
        
        try:
            # Determine block position
            if block_index == 0:
                block_position = "opening"
            elif block_index == len(sorted_block_ids) - 1:
                block_position = "closing"
            else:
                block_position = "middle"
            
            # Build context with sliding window (current + previous context)
            context_parts = []
            
            # Add company and earnings call context
            sample_record = block_records[0]
            company_name = sample_record.get("company_name", "Unknown")
            fiscal_year = sample_record.get("fiscal_year", "Unknown")
            fiscal_quarter = sample_record.get("fiscal_quarter", "Unknown")
            
            context_parts.append(f"=== {company_name} {fiscal_quarter} {fiscal_year} EARNINGS CALL TRANSCRIPT ===")
            context_parts.append("=== MANAGEMENT DISCUSSION SECTION ===")
            context_parts.append("")
            
            # Add previous context using previously generated summaries (up to 2 previous speaker blocks)
            context_window_size = 2
            start_index = max(0, block_index - context_window_size)
            
            if start_index < block_index:
                context_parts.append("--- PREVIOUS SPEAKER BLOCK SUMMARIES FOR CONTEXT ---")
                for ctx_idx in range(start_index, block_index):
                    ctx_block_id = sorted_block_ids[ctx_idx]
                    ctx_records = speaker_blocks[ctx_block_id]
                    ctx_speaker = ctx_records[0].get("speaker", "Management")
                    
                    # Check if we already have a summary for this previous block
                    ctx_summary = None
                    for enhanced_record in enhanced_records:
                        if enhanced_record.get("speaker_block_id") == ctx_block_id:
                            ctx_summary = enhanced_record.get("block_summary")
                            break
                    
                    if ctx_summary:
                        context_parts.append(f"[PREVIOUS BLOCK {ctx_block_id}] {ctx_speaker}:")
                        context_parts.append(f"SUMMARY: {ctx_summary}")
                    else:
                        # Fallback to truncated content if no summary yet (shouldn't happen in normal flow)
                        context_parts.append(f"[PREVIOUS BLOCK {ctx_block_id}] {ctx_speaker}:")
                        ctx_content = " ".join([r.get("paragraph_content", "") for r in ctx_records])
                        if len(ctx_content) > 300:
                            ctx_content = ctx_content[:300] + "..."
                        context_parts.append(f"CONTENT: {ctx_content}")
                    context_parts.append("")
                context_parts.append("--- END PREVIOUS CONTEXT ---")
                context_parts.append("")
            
            # Add current speaker block to summarize
            context_parts.append("--- CURRENT SPEAKER BLOCK TO SUMMARIZE ---")
            speaker_name = block_records[0].get("speaker", "Management")
            context_parts.append(f"[CURRENT] {speaker_name}:")
            for record in block_records:
                context_parts.append(record.get("paragraph_content", ""))
                context_parts.append("")
            
            context = "\n".join(context_parts)
            
            # Create speaker block summary prompt
            system_prompt = create_management_speaker_block_prompt(
                company_name=block_records[0].get("company_name", "Unknown"),
                fiscal_info=f"{block_records[0].get('fiscal_year')} {block_records[0].get('fiscal_quarter')}",
                speaker_block_records=block_records,
                speaker_name=speaker_name,
                financial_categories=block_records[0].get("category_type", []),
                block_position=block_position
            )
            
            # LLM call for speaker block summarization
            response = llm_client.chat.completions.create(
                model=config["stage_07_llm_summarization"]["llm_config"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                tools=create_management_summary_tools(),
                tool_choice="required",
                temperature=config["stage_07_llm_summarization"]["llm_config"]["temperature"],
                max_tokens=config["stage_07_llm_summarization"]["llm_config"]["max_tokens"],
                timeout=config["stage_07_llm_summarization"]["llm_config"]["timeout"]
            )
            
            # Process LLM response
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                response_data = json.loads(tool_call.function.arguments)
                
                summary = response_data.get("summary", "")
                
                # Apply summary to all paragraphs in this speaker block
                for record in block_records:
                    record["block_summary"] = summary
                
                enhanced_records.extend(block_records)
                
                # Track costs
                if hasattr(response, 'usage') and response.usage:
                    cost_data = calculate_token_cost(
                        response.usage.prompt_tokens, 
                        response.usage.completion_tokens
                    )
                    enhanced_error_logger.accumulate_costs({
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_data
                    })
                
                log_execution(f"Summarized Management speaker block {block_id} with {len(block_records)} paragraphs")
            
            else:
                enhanced_error_logger.log_summarization_error(
                    transcript_id, 
                    "Management Discussion",
                    f"No tool calls in speaker block {block_id} LLM response"
                )
                
                # Set null values for failed block
                for record in block_records:
                    record["block_summary"] = None
                
                enhanced_records.extend(block_records)
        
        except Exception as e:
            enhanced_error_logger.log_summarization_error(
                transcript_id,
                "Management Discussion", 
                f"Speaker block {block_id} processing failed: {e}"
            )
            
            # Set null values for failed block
            for record in block_records:
                record["block_summary"] = None
            
            enhanced_records.extend(block_records)
    
    return enhanced_records


def process_other_records(transcript_records: List[Dict], transcript_id: str) -> List[Dict]:
    """Process records that are not Management Discussion or Q&A."""
    other_records = [r for r in transcript_records if r.get("section_name") not in ["MANAGEMENT DISCUSSION SECTION", "Q&A"]]
    
    if not other_records:
        return []
    
    log_execution(f"Processing {len(other_records)} other records for transcript {transcript_id}")
    
    for record in other_records:
        # Create basic summary for other content
        content = record.get("paragraph_content", "")
        section_name = record.get("section_name", "Unknown")
        
        if len(content) > 100:
            summary = f"[{section_name.upper()}] {content[:100]}..."
        else:
            summary = f"[{section_name.upper()}] {content}"
        
        record["block_summary"] = summary
    
    return other_records


def process_transcript(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process single transcript for summarization."""
    try:
        log_console(f"Processing transcript: {transcript_id}")
        
        # Process different sections
        enhanced_mgmt = process_transcript_management(transcript_records, transcript_id, enhanced_error_logger)
        enhanced_qa = process_transcript_qa_groups(transcript_records, transcript_id, enhanced_error_logger)
        enhanced_other = process_other_records(transcript_records, transcript_id)
        
        # Combine all enhanced records
        all_enhanced = enhanced_mgmt + enhanced_qa + enhanced_other
        
        log_execution(f"Completed transcript {transcript_id}: {len(all_enhanced)} records enhanced")
        return all_enhanced
        
    except Exception as e:
        enhanced_error_logger.log_processing_error(transcript_id, f"Transcript processing failed: {e}")
        
        # Return original records with null summary fields
        for record in transcript_records:
            record["block_summary"] = None
        
        return transcript_records


def save_failed_transcripts(failed_transcripts: List[Dict], nas_conn: SMBConnection):
    """Save failed transcripts to separate file."""
    if not failed_transcripts:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        failed_data = {
            "timestamp": datetime.now().isoformat(),
            "total_failed": len(failed_transcripts),
            "failed_transcripts": failed_transcripts
        }
        
        failed_path = nas_path_join(
            config["stage_07_llm_summarization"]["output_data_path"],
            f"stage_07_failed_transcripts_{timestamp}.json"
        )
        
        failed_json = json.dumps(failed_data, indent=2)
        failed_bytes = io.BytesIO(failed_json.encode('utf-8'))
        
        if nas_upload_file(nas_conn, failed_bytes, failed_path):
            log_console(f"Saved {len(failed_transcripts)} failed transcripts")
            log_execution(f"Failed transcripts saved", {"count": len(failed_transcripts), "path": failed_path})
    
    except Exception as e:
        log_error(f"Failed to save failed transcripts: {e}", "failed_transcript_save", {"error": str(e)})


def main():
    """Main function orchestrating Stage 7 LLM summarization."""
    global config, logger, ssl_cert_path
    
    # Setup
    logger = setup_logging()
    log_console("STAGE 7: LLM-BASED CONTENT SUMMARIZATION")
    log_execution("Stage 7 execution started")
    
    try:
        # Validate environment
        validate_environment_variables()
        
        # Connect to NAS
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS")
        
        # Load configuration
        config = load_stage_config(nas_conn)
        log_execution("Configuration loaded successfully")
        log_console(f"Development mode: {config['stage_07_llm_summarization'].get('dev_mode', False)}")
        
        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Load Stage 6 data
        stage6_records = load_stage6_data(nas_conn)
        
        # Group by transcript
        transcripts = group_records_by_transcript(stage6_records)
        
        # Apply development mode limits
        if config["stage_07_llm_summarization"].get("dev_mode", False):
            max_transcripts = config["stage_07_llm_summarization"].get("dev_max_transcripts", 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            log_console(f"Development mode: limited to {len(transcripts)} transcripts")

        # Check for resume capability (default: enabled)
        enable_resume = config["stage_07_llm_summarization"].get("enable_resume", True)
        completed_transcript_ids = set()

        if enable_resume:
            # Load existing output to identify completed transcripts
            output_path_base = config["stage_07_llm_summarization"]["output_data_path"]
            completed_transcript_ids = load_existing_output(nas_conn, output_path_base)

            if completed_transcript_ids:
                # Filter out already-processed transcripts
                original_count = len(transcripts)
                transcripts = {tid: records for tid, records in transcripts.items()
                              if tid not in completed_transcript_ids}
                resumed_count = original_count - len(transcripts)

                if resumed_count > 0:
                    log_console(f"RESUME MODE: Skipping {resumed_count} already-processed transcripts")
                    log_console(f"Remaining transcripts to process: {len(transcripts)}")

                if len(transcripts) == 0:
                    log_console("All transcripts already processed. Nothing to do.")
                    # Close JSON array and exit
                    output_path = nas_path_join(output_path_base, "stage_07_summarized_content.json")
                    if not close_json_array(nas_conn, output_path):
                        log_console("Warning: Failed to properly close existing output file", "WARNING")
                    return

        # Initialize enhanced error logger
        enhanced_error_logger = EnhancedErrorLogger()

        # Process each transcript
        all_enhanced_records = []
        failed_transcripts = []

        output_path = nas_path_join(
            config["stage_07_llm_summarization"]["output_data_path"],
            "stage_07_summarized_content.json"
        )

        start_time = datetime.now()

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
                # First batch only if: first transcript AND no existing output
                is_first_batch = (i == 1 and not completed_transcript_ids)
                save_results_incrementally(enhanced_records, output_path, is_first_batch=is_first_batch)
                
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
                    "reason": f"Processing failed - see error logs: {e}"
                })
                
                log_console(f"Failed transcript {transcript_id}: {e}", "ERROR")
        
        # Close JSON array
        nas_conn_final = get_nas_connection()
        if nas_conn_final:
            close_json_array(nas_conn_final, output_path)
            nas_conn_final.close()
        
        # Save failed transcripts
        save_failed_transcripts(failed_transcripts, nas_conn)
        
        # Calculate final metrics
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Calculate summary statistics
        records_with_summaries = len([r for r in all_enhanced_records if r.get("block_summary")])
        records_without_summaries = len(all_enhanced_records) - records_with_summaries
        
        stage_summary = {
            "total_transcripts": len(transcripts),
            "successful_transcripts": len(transcripts) - len(failed_transcripts),
            "failed_transcripts": len(failed_transcripts),
            "total_records": len(all_enhanced_records),
            "records_with_summaries": records_with_summaries,
            "execution_time": str(execution_time),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens,
            "resumed_from_existing": bool(completed_transcript_ids),
            "skipped_completed_transcripts": len(completed_transcript_ids)
        }
        
        # Final summary
        log_console("="*60)
        log_console("STAGE 7 SUMMARIZATION COMPLETE")
        log_console("="*60)
        log_console(f"Transcripts processed: {stage_summary['successful_transcripts']}/{stage_summary['total_transcripts']}")
        log_console(f"Records with summaries: {records_with_summaries}/{stage_summary['total_records']}")
        log_console(f"Records without summaries: {records_without_summaries}")
        log_console(f"Total cost: ${stage_summary['total_cost']:.4f}")
        log_console(f"Total tokens: {stage_summary['total_tokens']:,}")
        log_console(f"Execution time: {stage_summary['execution_time']}")
        log_console("="*60)
        
        # Save logs
        save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)

        # Cleanup
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            os.unlink(ssl_cert_path)

        nas_conn.close()

    except Exception as e:
        error_msg = f"Stage 7 execution failed: {e}"
        log_error(error_msg, "stage_execution", {"error": str(e)})
        log_console(error_msg, "ERROR")

        # Save whatever logs we have
        try:
            if 'nas_conn' in locals() and nas_conn:
                if 'enhanced_error_logger' not in locals():
                    enhanced_error_logger = EnhancedErrorLogger()
                stage_summary = {
                    "status": "failed",
                    "error": str(e),
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds() if 'start_time' in locals() else 0,
                    "resumed_from_existing": bool(completed_transcript_ids) if 'completed_transcript_ids' in locals() else False,
                    "skipped_completed_transcripts": len(completed_transcript_ids) if 'completed_transcript_ids' in locals() else 0
                }
                save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)
                nas_conn.close()
        except:
            pass

        raise

    finally:
        # Cleanup SSL certificate in all cases
        if 'ssl_cert_path' in locals() and ssl_cert_path and os.path.exists(ssl_cert_path):
            try:
                os.unlink(ssl_cert_path)
                log_console("SSL certificate cleaned up")
            except:
                pass


if __name__ == "__main__":
    main()