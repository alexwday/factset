"""
Stage 7: LLM-Based Content Summarization
Processes Stage 6 classified content to create summaries for each transcript.
Self-contained standalone script that loads config from NAS at runtime.

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
        
        # Stage 6 outputs JSON array format
        stage6_records = json.loads(input_data.decode('utf-8'))
        
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
            "description": "Summarize complete Q&A group conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string", 
                        "description": "Comprehensive summary of the complete Q&A conversation for reranking"
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
            "description": "Summarize a single management speaker block",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Comprehensive summary of this speaker block for reranking"
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
You are creating a single summary for this complete Q&A conversation that will be assigned to ALL paragraphs in the group.
This summary will be used for post-retrieval reranking to filter relevant content from earnings call transcripts.

The summary should capture:
- Analyst question and management response
- Key financial metrics, guidance, or business developments mentioned
- Strategic insights or forward-looking statements
- Any quantitative data or performance indicators

RERANKING OPTIMIZATION:
- Include speaker attribution: [ANALYST] question about [topic] and [EXECUTIVE] response covering [key points]
- Preserve financial terminology and quantitative data for semantic search
- Focus on searchable concepts that users might query
- Write for relevance filtering by smaller models
</objective>

<style>
Professional financial analysis. Summary should read like an executive briefing point optimized for relevance filtering.
</style>

<tone>
Analytical and concise. Focus on financial substance and business insights.
</tone>

<audience>
A smaller model that will judge relevance to user queries about earnings calls and filter results accordingly.
</audience>

<response_format>
Use the summarize_qa_group function.
- Provide one comprehensive summary for the entire Q&A conversation
- Include speaker attribution and financial context for reranking
- Preserve quantitative data and forward-looking statements
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
You are creating a summary for this specific management speaker block from a {fiscal_info} earnings call transcript.
This summary will be used for post-retrieval reranking to filter relevant content from earnings call transcripts.

The summary should capture:
- Key points made by this specific executive
- Financial metrics or guidance mentioned  
- Strategic initiatives or decisions discussed
- Forward-looking statements or outlook
- Operational updates or challenges addressed

RERANKING OPTIMIZATION:
- Start with speaker attribution: [EXECUTIVE: {speaker_name}]
- Preserve financial terminology and quantitative data for semantic search
- Focus on the main topics covered in THIS speaker block
- Make summary distinctive from other speaker blocks
- Include context about {company_name}'s {fiscal_info} performance
- Consider how this block builds on or relates to the previous speaker block summaries provided

Consider the block's position in the earnings call:
- Opening blocks often contain key financial highlights and quarterly results
- Middle blocks typically provide detailed segment analysis and operational updates  
- Closing blocks may contain forward guidance and strategic outlook
- Use previous block summaries to understand the conversation flow and avoid repetition
</objective>

<style>
Professional financial analysis. Summary should be concise yet comprehensive, optimized for distinguishing this block from others.
</style>

<tone>
Analytical and focused. Capture the essence of what THIS executive said in THIS block.
</tone>

<audience>
A smaller model that will judge relevance to user queries and needs to distinguish between different speaker blocks.
</audience>

<response_format>
Use the summarize_management_speaker_block function.
- Provide one summary for this specific speaker block  
- Include speaker name and key topics
- Make summary distinctive from other blocks
- Focus on what makes this block unique and searchable
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
                    record["paragraph_summary"] = summary
                
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
                    record["paragraph_summary"] = None
                
                enhanced_records.extend(qa_group_records)
        
        except Exception as e:
            enhanced_error_logger.log_summarization_error(
                transcript_id,
                "Q&A", 
                f"Q&A group {qa_group_id} processing failed: {e}"
            )
            
            # Set null values for failed group
            for record in qa_group_records:
                record["paragraph_summary"] = None
            
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
                            ctx_summary = enhanced_record.get("paragraph_summary")
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
                    record["paragraph_summary"] = summary
                
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
                    record["paragraph_summary"] = None
                
                enhanced_records.extend(block_records)
        
        except Exception as e:
            enhanced_error_logger.log_summarization_error(
                transcript_id,
                "Management Discussion", 
                f"Speaker block {block_id} processing failed: {e}"
            )
            
            # Set null values for failed block
            for record in block_records:
                record["paragraph_summary"] = None
            
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
        
        record["paragraph_summary"] = summary
    
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
            record["paragraph_summary"] = None
        
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
        
        stage_summary = {
            "total_transcripts": len(transcripts),
            "successful_transcripts": len(transcripts) - len(failed_transcripts),
            "failed_transcripts": len(failed_transcripts),
            "total_records": len(all_enhanced_records),
            "records_with_summaries": records_with_summaries,
            "execution_time": str(execution_time),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens
        }
        
        # Calculate summary statistics
        records_with_summaries = len([r for r in all_enhanced_records if r.get("paragraph_summary")])
        records_without_summaries = len(all_enhanced_records) - records_with_summaries
        
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
        log_console(f"Stage 7 failed: {e}", "ERROR")
        log_error(f"Stage 7 execution failed: {e}", "main_execution", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()