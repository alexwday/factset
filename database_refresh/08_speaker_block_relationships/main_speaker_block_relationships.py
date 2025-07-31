"""
Stage 8: Speaker Block Relationship Scoring
Processes Stage 7 summarized content to generate forward and backward importance scores
for Management Discussion speaker blocks to support retrieval relationship scoring.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 7 patterns with Stage 8 relationship scoring logic.
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
        logs_path = config["stage_08_speaker_block_relationships"]["output_logs_path"]
        
        error_types = {
            "summarization": self.summarization_errors,
            "authentication": self.authentication_errors,
            "processing": self.processing_errors,
            "validation": self.validation_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                filename = f"stage_08_speaker_block_relationships_{error_type}_errors_{timestamp}.json"
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
    logs_path = config["stage_08_speaker_block_relationships"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_08_speaker_block_relationships",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_08_speaker_block_relationships_{timestamp}.json"
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
            "stage": "stage_08_speaker_block_relationships",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_08_speaker_block_relationships_errors_{timestamp}.json"
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

            if "stage_08_speaker_block_relationships" not in full_config:
                raise ValueError("Stage 8 configuration not found in config file")
                
            stage_config = full_config["stage_08_speaker_block_relationships"]
            
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
        token_endpoint = config["stage_08_speaker_block_relationships"]["llm_config"]["token_endpoint"]
        
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
            base_url=config["stage_08_speaker_block_relationships"]["llm_config"]["base_url"],
            timeout=config["stage_08_speaker_block_relationships"]["llm_config"]["timeout"]
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
    
    prompt_cost_per_1k = config["stage_08_speaker_block_relationships"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_08_speaker_block_relationships"]["llm_config"]["cost_per_1k_completion_tokens"]
    
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


def load_stage7_data(nas_conn: SMBConnection) -> List[Dict]:
    """Load Stage 7 summarized content data from NAS."""
    try:
        # Don't prepend NAS_BASE_PATH - it's already included in the config path
        input_path = config["stage_08_speaker_block_relationships"]["input_data_path"]
        
        log_execution(f"Loading Stage 7 data from: {input_path}")
        
        if not validate_nas_path(input_path):
            raise ValueError(f"Invalid input file path: {input_path}")
            
        input_data = nas_download_file(nas_conn, input_path)
        if not input_data:
            raise RuntimeError(f"Failed to download Stage 7 data from: {input_path}")
        
        # Stage 7 outputs JSON array format
        stage7_records = json.loads(input_data.decode('utf-8'))
        
        log_execution(f"Loaded {len(stage7_records)} records from Stage 7")
        return stage7_records
        
    except Exception as e:
        error_msg = f"Failed to load Stage 7 data: {e}"
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


# Relationship context tools (Stage 8)
def create_speaker_block_relationship_tools() -> List[Dict]:
    """Function calling schema for speaker block context requirement assessment."""
    return [{
        "type": "function",
        "function": {
            "name": "assess_speaker_block_context",
            "description": "Determine if neighboring speaker blocks are required for proper understanding of current block",
            "parameters": {
                "type": "object",
                "properties": {
                    "backward_context_required": {
                        "type": "boolean",
                        "description": "True if the previous speaker block provides essential context for understanding the current block; False if current block is self-contained"
                    },
                    "forward_context_required": {
                        "type": "boolean", 
                        "description": "True if the next speaker block provides essential context for understanding the current block; False if current block is self-contained"
                    }
                },
                "required": ["backward_context_required", "forward_context_required"]
            }
        }
    }]


def create_speaker_block_relationship_prompt(company_name: str, fiscal_info: str,
                                           current_speaker: str, current_summary: str,
                                           previous_speaker: str, previous_summary: str,
                                           next_speaker: str, next_summary: str,
                                           financial_categories: List[str],
                                           block_position: str) -> str:
    """Create prompt for speaker block relationship scoring."""
    
    categories_str = ', '.join(financial_categories) if financial_categories else 'General Business'
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <call_type>Earnings Call Management Discussion</call_type>
  <current_speaker>{current_speaker}</current_speaker>
  <block_position>{block_position}</block_position>
  <financial_focus>{categories_str}</financial_focus>
</context>

<objective>
You are assessing context dependencies between speaker blocks in a {fiscal_info} earnings call transcript.
Your decision will determine whether neighboring blocks are retrieved together during search.

YOUR TASK:
Determine if neighboring speaker blocks are REQUIRED for proper understanding of the CURRENT block.

RETRIEVAL SCENARIO:
When a user searches and the current block matches their query, your flags determine:
- backward_context_required = True → Previous block WILL be included in results
- backward_context_required = False → Previous block will NOT be included
- forward_context_required = True → Next block WILL be included in results  
- forward_context_required = False → Next block will NOT be included

DECISION CRITERIA for "True" (context IS required):
1. INCOMPLETE INFORMATION: Current block references something explained in neighboring block
   - "As I mentioned earlier..." (backward = True)
   - "Let me elaborate on that..." (forward = True)
   - Numbers/metrics defined in neighboring block but used in current

2. LOGICAL DEPENDENCIES: Current block's meaning changes without neighboring context
   - Current block answers a question posed in previous block
   - Current block sets up information completed in next block
   - Multi-part explanations split across blocks

3. CONVERSATIONAL FLOW: Breaking the connection loses critical meaning
   - Speaker handoffs where context carries over
   - Progressive building of an argument
   - Cause-and-effect relationships across blocks

DECISION CRITERIA for "False" (context NOT required):
1. SELF-CONTAINED: Current block has complete information on its topic
   - Full explanation of a metric or initiative
   - Complete answer to a question
   - Standalone financial results

2. TOPIC CHANGE: Clear transition to unrelated subject
   - "Moving on to a different topic..."
   - "Let me now discuss..."
   - New speaker introducing new theme

3. REDUNDANCY: Neighboring blocks would add no new information
   - Repetition of same points
   - Procedural transitions
   - Thank you/acknowledgment blocks

CRITICAL: Think from a user's perspective - if they searched and found the current block,
would they be confused or missing key information without the neighboring block?
</objective>

<current_block>
SPEAKER: {current_speaker}
SUMMARY: {current_summary}
</current_block>

<previous_block>
SPEAKER: {previous_speaker if previous_speaker else "N/A (First block)"}
SUMMARY: {previous_summary if previous_summary else "N/A"}
</previous_block>

<next_block>
SPEAKER: {next_speaker if next_speaker else "N/A (Last block)"}
SUMMARY: {next_summary if next_summary else "N/A"}
</next_block>

<style>
Analytical and precise. Focus on information relationships and retrieval utility.
</style>

<tone>
Objective and systematic. Consider the practical retrieval scenario.
</tone>

<audience>
A retrieval system that needs to decide whether to include neighboring speaker blocks.
</audience>

<response_format>
Use the assess_speaker_block_context function.
- backward_context_required: True/False - Is previous block REQUIRED for understanding?
- forward_context_required: True/False - Is next block REQUIRED for understanding?
- Be decisive: either the context is required (True) or it's not (False)
- Consider user confusion: Would they miss critical information without the neighboring block?
- Default to False if uncertain - avoid over-including irrelevant content
</response_format>
"""


def process_transcript_non_management(transcript_records: List[Dict], transcript_id: str) -> List[Dict]:
    """Process non-Management Discussion records (Q&A and other sections) - pass through unchanged."""
    non_mgmt_records = [r for r in transcript_records if r.get("section_name") != "MANAGEMENT DISCUSSION SECTION"]
    
    if not non_mgmt_records:
        return []
    
    log_execution(f"Passing through {len(non_mgmt_records)} non-Management Discussion records for transcript {transcript_id}")
    
    # Stage 8 only processes Management Discussion blocks for relationship scoring
    # All other records (Q&A, etc.) are passed through unchanged from Stage 7
    return non_mgmt_records


def process_transcript_management(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process Management Discussion section for speaker block relationship scoring."""
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
    
    log_execution(f"Processing Management Discussion with {len(speaker_blocks)} speaker blocks for relationship scoring in transcript {transcript_id}")
    
    enhanced_records = []
    
    for block_index, block_id in enumerate(sorted_block_ids):
        block_records = speaker_blocks[block_id]
        
        try:
            # Determine block position and get neighboring blocks
            if block_index == 0:
                block_position = "opening"
            elif block_index == len(sorted_block_ids) - 1:
                block_position = "closing"
            else:
                block_position = "middle"
            
            # Get current, previous, and next block info
            current_speaker = block_records[0].get("speaker", "Management")
            current_summary = block_records[0].get("paragraph_summary", "")
            
            # Previous block (backward relationship)
            previous_speaker = None
            previous_summary = None
            if block_index > 0:
                prev_block_id = sorted_block_ids[block_index - 1]
                prev_records = speaker_blocks[prev_block_id]
                previous_speaker = prev_records[0].get("speaker", "Management")
                previous_summary = prev_records[0].get("paragraph_summary", "")
            
            # Next block (forward relationship)
            next_speaker = None
            next_summary = None
            if block_index < len(sorted_block_ids) - 1:
                next_block_id = sorted_block_ids[block_index + 1]
                next_records = speaker_blocks[next_block_id]
                next_speaker = next_records[0].get("speaker", "Management")
                next_summary = next_records[0].get("paragraph_summary", "")
            
            # Skip scoring if we don't have current summary (shouldn't happen with Stage 7 input)
            if not current_summary:
                log_execution(f"Skipping relationship scoring for block {block_id} - no summary found")
                enhanced_records.extend(block_records)
                continue
            
            # Create relationship scoring prompt
            system_prompt = create_speaker_block_relationship_prompt(
                company_name=block_records[0].get("company_name", "Unknown"),
                fiscal_info=f"{block_records[0].get('fiscal_year')} {block_records[0].get('fiscal_quarter')}",
                current_speaker=current_speaker,
                current_summary=current_summary,
                previous_speaker=previous_speaker,
                previous_summary=previous_summary,
                next_speaker=next_speaker,
                next_summary=next_summary,
                financial_categories=block_records[0].get("category_type", []),
                block_position=block_position
            )
            
            # Build simple context for LLM (summaries are in the prompt)
            context = f"Score the relationship importance between speaker blocks for retrieval purposes."
            
            # LLM call for relationship scoring
            response = llm_client.chat.completions.create(
                model=config["stage_08_speaker_block_relationships"]["llm_config"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                tools=create_speaker_block_relationship_tools(),
                tool_choice="required",
                temperature=config["stage_08_speaker_block_relationships"]["llm_config"]["temperature"],
                max_tokens=config["stage_08_speaker_block_relationships"]["llm_config"]["max_tokens"],
                timeout=config["stage_08_speaker_block_relationships"]["llm_config"]["timeout"]
            )
            
            # Process LLM response
            if response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                response_data = json.loads(tool_call.function.arguments)
                
                backward_context_required = response_data.get("backward_context_required", False)
                forward_context_required = response_data.get("forward_context_required", False)
                
                # Apply context requirement flags to all paragraphs in this speaker block
                for record in block_records:
                    record["backward_context_required"] = backward_context_required
                    record["forward_context_required"] = forward_context_required
                
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
                
                log_execution(f"Assessed context for Management speaker block {block_id}: backward={backward_context_required}, forward={forward_context_required}")
            
            else:
                enhanced_error_logger.log_summarization_error(
                    transcript_id, 
                    "Management Discussion",
                    f"No tool calls in speaker block {block_id} relationship scoring response"
                )
                
                # Set null values for failed block
                for record in block_records:
                    record["backward_context_required"] = None
                    record["forward_context_required"] = None
                
                enhanced_records.extend(block_records)
        
        except Exception as e:
            enhanced_error_logger.log_summarization_error(
                transcript_id,
                "Management Discussion", 
                f"Speaker block {block_id} relationship scoring failed: {e}"
            )
            
            # Set null values for failed block
            for record in block_records:
                record["backward_context_required"] = None
                record["forward_context_required"] = None
            
            enhanced_records.extend(block_records)
    
    return enhanced_records


def process_transcript(transcript_records: List[Dict], transcript_id: str, enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process single transcript for relationship scoring."""
    try:
        log_console(f"Processing transcript: {transcript_id}")
        
        # Process Management Discussion sections for relationship scoring
        enhanced_mgmt = process_transcript_management(transcript_records, transcript_id, enhanced_error_logger)
        
        # Pass through all non-Management Discussion records unchanged
        enhanced_non_mgmt = process_transcript_non_management(transcript_records, transcript_id)
        
        # Combine all enhanced records
        all_enhanced = enhanced_mgmt + enhanced_non_mgmt
        
        log_execution(f"Completed transcript {transcript_id}: {len(all_enhanced)} records processed")
        return all_enhanced
        
    except Exception as e:
        enhanced_error_logger.log_processing_error(transcript_id, f"Transcript processing failed: {e}")
        
        # Return original records with null context requirement fields
        for record in transcript_records:
            if record.get("section_name") == "MANAGEMENT DISCUSSION SECTION":
                record["backward_context_required"] = None
                record["forward_context_required"] = None
        
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
            config["stage_08_speaker_block_relationships"]["output_data_path"],
            f"stage_08_failed_transcripts_{timestamp}.json"
        )
        
        failed_json = json.dumps(failed_data, indent=2)
        failed_bytes = io.BytesIO(failed_json.encode('utf-8'))
        
        if nas_upload_file(nas_conn, failed_bytes, failed_path):
            log_console(f"Saved {len(failed_transcripts)} failed transcripts")
            log_execution(f"Failed transcripts saved", {"count": len(failed_transcripts), "path": failed_path})
    
    except Exception as e:
        log_error(f"Failed to save failed transcripts: {e}", "failed_transcript_save", {"error": str(e)})


def main():
    """Main function orchestrating Stage 8 speaker block relationship scoring."""
    global config, logger, ssl_cert_path
    
    # Setup
    logger = setup_logging()
    log_console("STAGE 8: SPEAKER BLOCK RELATIONSHIP SCORING")
    log_execution("Stage 8 execution started")
    
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
        log_console(f"Development mode: {config['stage_08_speaker_block_relationships'].get('dev_mode', False)}")
        
        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Load Stage 7 data
        stage7_records = load_stage7_data(nas_conn)
        
        # Group by transcript
        transcripts = group_records_by_transcript(stage7_records)
        
        # Apply development mode limits
        if config["stage_08_speaker_block_relationships"].get("dev_mode", False):
            max_transcripts = config["stage_08_speaker_block_relationships"].get("dev_max_transcripts", 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            log_console(f"Development mode: limited to {len(transcripts)} transcripts")
        
        # Initialize enhanced error logger
        enhanced_error_logger = EnhancedErrorLogger()
        
        # Process each transcript
        all_enhanced_records = []
        failed_transcripts = []
        
        output_path = nas_path_join(
            config["stage_08_speaker_block_relationships"]["output_data_path"],
            "stage_08_relationship_scored_content.json"
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
        
        # Calculate context assessment statistics
        mgmt_records_with_flags = len([r for r in all_enhanced_records 
                                     if r.get("section_name") == "MANAGEMENT DISCUSSION SECTION" 
                                     and r.get("backward_context_required") is not None 
                                     and r.get("forward_context_required") is not None])
        mgmt_records_total = len([r for r in all_enhanced_records if r.get("section_name") == "MANAGEMENT DISCUSSION SECTION"])
        non_mgmt_records = len(all_enhanced_records) - mgmt_records_total
        
        # Calculate flag statistics
        backward_true_count = len([r for r in all_enhanced_records 
                                 if r.get("section_name") == "MANAGEMENT DISCUSSION SECTION" 
                                 and r.get("backward_context_required") is True])
        forward_true_count = len([r for r in all_enhanced_records 
                                if r.get("section_name") == "MANAGEMENT DISCUSSION SECTION" 
                                and r.get("forward_context_required") is True])
        
        stage_summary = {
            "total_transcripts": len(transcripts),
            "successful_transcripts": len(transcripts) - len(failed_transcripts),
            "failed_transcripts": len(failed_transcripts),
            "total_records": len(all_enhanced_records),
            "mgmt_records_with_flags": mgmt_records_with_flags,
            "mgmt_records_total": mgmt_records_total,
            "non_mgmt_records_passthrough": non_mgmt_records,
            "backward_context_required_count": backward_true_count,
            "forward_context_required_count": forward_true_count,
            "execution_time": str(execution_time),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens
        }
        
        # Final summary
        log_console("="*60)
        log_console("STAGE 8 CONTEXT ASSESSMENT COMPLETE")
        log_console("="*60)
        log_console(f"Transcripts processed: {stage_summary['successful_transcripts']}/{stage_summary['total_transcripts']}")
        log_console(f"Management records with flags: {mgmt_records_with_flags}/{mgmt_records_total}")
        log_console(f"Backward context required: {backward_true_count}")
        log_console(f"Forward context required: {forward_true_count}")
        log_console(f"Non-management records (passthrough): {non_mgmt_records}")
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
        log_console(f"Stage 8 failed: {e}", "ERROR")
        log_error(f"Stage 8 execution failed: {e}", "main_execution", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()