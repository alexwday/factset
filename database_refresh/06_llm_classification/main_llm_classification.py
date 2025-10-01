"""
Stage 6: LLM-Based Financial Content Classification
Single-pass classification system returning all applicable categories.
Management Discussion: Speaker block level classification with 2-block context.
Q&A: Conversation level classification (entire Q&A group) with no context.
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
from collections import defaultdict, deque

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []
error_log = []

# LLM-specific globals
llm_client = None
oauth_token = None
ssl_cert_path = None

# Category registry - maps ID to category info
CATEGORY_REGISTRY = {}


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
        self.classification_errors = []
        self.authentication_errors = []
        self.validation_errors = []
        self.processing_errors = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def log_classification_error(self, transcript_id: str, pass_type: str, error: str):
        """Log classification-specific errors."""
        self.classification_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "pass_type": pass_type,
            "error": error,
            "action_required": "Review transcript structure and classification logic"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_validation_error(self, transcript_id: str, validation_issue: str):
        """Log category validation errors."""
        self.validation_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "validation_issue": validation_issue,
            "action_required": "Review category assignments"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review transcript data and processing logic"
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
        logs_path = config["stage_06_llm_classification"]["output_logs_path"]
        
        error_types = {
            "classification": self.classification_errors,
            "authentication": self.authentication_errors,
            "validation": self.validation_errors,
            "processing": self.processing_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                filename = f"stage_06_llm_classification_{error_type}_errors_{timestamp}.json"
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
    logs_path = config["stage_06_llm_classification"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_06_llm_classification",
        "execution_start": (
            execution_log[0]["timestamp"]
            if execution_log
            else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_06_llm_classification_{timestamp}.json"
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
            "stage": "stage_06_llm_classification",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "error_summary": stage_summary.get("errors", {}),
            "errors": error_log,
        }

        error_log_filename = f"stage_06_llm_classification_errors_{timestamp}.json"
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
                    "port": nas_port,
                    "share_name": os.getenv("NAS_SHARE_NAME"),
                },
            )
            return conn
        else:
            log_error(
                "Failed to establish NAS connection",
                "nas_connection",
                {"server_ip": os.getenv("NAS_SERVER_IP"), "port": nas_port},
            )
            return None

    except Exception as e:
        log_error(
            f"Error creating NAS connection: {e}",
            "nas_connection",
            {"exception_type": type(e).__name__, "server_ip": os.getenv("NAS_SERVER_IP")},
        )
        return None


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""

    try:
        log_execution("Loading configuration from NAS", {"config_path": os.getenv("CONFIG_PATH")})

        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            error_msg = f"Configuration file not found at {os.getenv('CONFIG_PATH')}"
            log_error(error_msg, "config_load", {"path": os.getenv("CONFIG_PATH")})
            raise FileNotFoundError(error_msg)

        # Parse YAML configuration
        stage_config = yaml.safe_load(config_data.decode("utf-8"))
        log_execution("Configuration loaded successfully", {"sections": list(stage_config.keys())})

        # Load monitored institutions from separate file
        config_path = os.getenv("CONFIG_PATH")
        config_dir = os.path.dirname(config_path) if config_path else ""
        institutions_path = os.path.join(config_dir, "monitored_institutions.yaml")
        log_execution("Loading monitored institutions from separate file", {"institutions_path": institutions_path})
        
        institutions_data = nas_download_file(nas_conn, institutions_path)
        if institutions_data:
            try:
                stage_config["monitored_institutions"] = yaml.safe_load(institutions_data.decode("utf-8"))
                log_execution("Loaded monitored institutions successfully", {"count": len(stage_config["monitored_institutions"])})
            except yaml.YAMLError as e:
                log_error("Failed to load monitored_institutions.yaml, falling back to config.yaml", "config_parse", {"yaml_error": str(e)})
        else:
            log_execution("monitored_institutions.yaml not found, using main config file")

        # Load financial categories from NAS Config folder
        financial_categories_path = os.path.join(config_dir, "financial_categories.yaml")
        log_execution("Loading financial categories from NAS", {"categories_path": financial_categories_path})
        
        categories_data = nas_download_file(nas_conn, financial_categories_path)
        if categories_data:
            try:
                categories_yaml = yaml.safe_load(categories_data.decode("utf-8"))
                # Update the financial_categories in stage_06_llm_classification section
                if "stage_06_llm_classification" not in stage_config:
                    stage_config["stage_06_llm_classification"] = {}
                stage_config["stage_06_llm_classification"]["financial_categories"] = categories_yaml
                log_execution("Loaded financial categories successfully", {"count": len(categories_yaml)})
            except yaml.YAMLError as e:
                log_error("Failed to load financial_categories.yaml, falling back to config.yaml", "config_parse", {"yaml_error": str(e)})
                # Keep financial_categories from main config if separate file fails
        else:
            log_execution("financial_categories.yaml not found, using main config file")

        # Validate configuration
        validate_config_structure(stage_config)
        return stage_config

    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML in configuration file: {e}"
        log_error(error_msg, "config_parse", {"yaml_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading configuration from NAS: {e}"
        log_error(error_msg, "config_load", {"exception_type": type(e).__name__})
        raise


def validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate configuration structure and build category registry."""
    global CATEGORY_REGISTRY

    required_sections = [
        "ssl_cert_path",
        "api_settings",
        "stage_06_llm_classification",
        "monitored_institutions"
    ]

    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {"missing_section": section})
            raise ValueError(error_msg)

    # Validate stage_06_llm_classification specific parameters
    stage_06_config = config["stage_06_llm_classification"]
    required_stage_06_params = [
        "description",
        "input_data_path",
        "output_data_path",
        "output_logs_path",
        "dev_mode",
        "dev_max_transcripts",
        "llm_config",
        "processing_config",
        "financial_categories"
    ]

    for param in required_stage_06_params:
        if param not in stage_06_config:
            error_msg = f"Missing required stage_06_llm_classification parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"stage_06_llm_classification.{param}"})
            raise ValueError(error_msg)

    # Build category registry from financial_categories
    if "financial_categories" in stage_06_config:
        categories = stage_06_config["financial_categories"]
        for cat in categories:
            cat_id = cat["id"]
            CATEGORY_REGISTRY[cat_id] = {
                "id": cat_id,
                "name": cat["name"],
                "description": cat["description"]
            }
        log_execution(f"Built category registry with {len(CATEGORY_REGISTRY)} categories")
    else:
        raise ValueError("No financial categories found in configuration")

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "total_categories": len(CATEGORY_REGISTRY),
        "llm_model": stage_06_config["llm_config"]["model"]
    })


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


def setup_proxy_configuration() -> str:
    """Set up proxy configuration with proper credential escaping."""

    try:
        proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")
        proxy_user = os.getenv("PROXY_USER")
        proxy_password = os.getenv("PROXY_PASSWORD")
        proxy_url = os.getenv("PROXY_URL")

        # Escape domain and user for URL
        escaped_domain = quote(proxy_domain + "\\" + proxy_user)
        quoted_password = quote(proxy_password)

        # Construct proxy URL
        proxy_with_auth = f"http://{escaped_domain}:{quoted_password}@{proxy_url}"

        log_execution("Proxy configuration setup complete", {
            "proxy_domain": proxy_domain,
            "proxy_url": proxy_url
        })

        return proxy_with_auth

    except Exception as e:
        error_msg = f"Error setting up proxy configuration: {e}"
        log_error(error_msg, "proxy_setup", {"exception_type": type(e).__name__})
        raise


def get_oauth_token(retry_count: int = 3, retry_delay: float = 1.0) -> Optional[str]:
    """Get OAuth token using client credentials flow with retry logic."""
    global ssl_cert_path
    
    for attempt in range(1, retry_count + 1):
        try:
            token_endpoint = config["stage_06_llm_classification"]["llm_config"]["token_endpoint"]
            
            # Prepare OAuth request
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': os.getenv("LLM_CLIENT_ID"),
                'client_secret': os.getenv("LLM_CLIENT_SECRET")
            }
            
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            
            # Use SSL certificate if available
            verify_ssl = ssl_cert_path if ssl_cert_path else True
            
            response = requests.post(
                token_endpoint,
                data=auth_data,
                headers=headers,
                verify=verify_ssl,
                timeout=30
            )
            
            if response.status_code == 200:
                token_data = response.json()
                log_execution(f"OAuth token acquired successfully (attempt {attempt})")
                return token_data.get('access_token')
            else:
                error_msg = f"OAuth token request failed: {response.status_code}"
                log_error(error_msg, "oauth_token", {"attempt": attempt, "status_code": response.status_code})
                
        except requests.exceptions.RequestException as e:
            error_msg = f"OAuth token request error (attempt {attempt}): {e}"
            log_error(error_msg, "oauth_token", {"attempt": attempt, "error": str(e)})
            
        if attempt < retry_count:
            time.sleep(retry_delay * attempt)
    
    return None


def setup_llm_client(oauth_token: str) -> Optional[OpenAI]:
    """Initialize OpenAI client with OAuth token."""
    
    try:
        client = OpenAI(
            api_key=oauth_token,
            base_url=config["stage_06_llm_classification"]["llm_config"]["base_url"],
            timeout=config["stage_06_llm_classification"]["llm_config"]["timeout"]
        )
        
        log_execution("LLM client initialized successfully")
        return client
        
    except Exception as e:
        error_msg = f"LLM client initialization failed: {e}"
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
    
    prompt_cost_per_1k = config["stage_06_llm_classification"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_06_llm_classification"]["llm_config"]["cost_per_1k_completion_tokens"]
    
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
    """Save results incrementally after each transcript."""
    global logger
    
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        if is_first_batch:
            # Initialize with opening bracket
            output_bytes = io.BytesIO(b"[\n")
            if not nas_upload_file(nas_conn, output_bytes, output_path):
                raise RuntimeError("Failed to initialize output file")
            log_console(f"Initialized output file for incremental saving")
        
        # Append records
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
    """Append records to a JSON array file."""
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
    """Close a JSON array file by appending the closing bracket."""
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


def nas_file_exists(nas_conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on NAS."""
    try:
        # Parse path
        path_parts = file_path.replace('\\', '/').split('/')
        file_name = path_parts[-1]
        dir_path = '/'.join(path_parts[:-1]) if len(path_parts) > 1 else '/'
        
        # List directory contents
        files = nas_conn.listPath(os.environ.get('NAS_SHARE_NAME'), dir_path)
        
        # Check if file exists
        for file_info in files:
            if file_info.filename == file_name:
                return True
        return False
        
    except Exception as e:
        log_error(f"Error checking file existence: {e}", "nas_operation", {"path": file_path, "error": str(e)})
        return False


def load_existing_output(nas_conn: SMBConnection, output_path: str) -> set:
    """
    Load existing output file and extract completed transcript IDs.
    Returns a set of transcript IDs that have already been processed.
    Adapted from Stage 5 for Stage 6 resume capability.
    """
    try:
        output_filename = "stage_06_classified_content.json"
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
            
            # Extract unique transcript IDs
            completed_ids = set()
            for record in records:
                # Build transcript ID from record fields
                # Use filename if available, otherwise construct from components
                if 'filename' in record:
                    transcript_id = record['filename']
                else:
                    ticker = record.get('ticker', 'unknown')
                    event_id = record.get('event_id', 'unknown')
                    transcript_id = f"{ticker}_{event_id}"
                completed_ids.add(transcript_id)
            
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


# Security validation functions
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


# NAS utility functions
def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "nas_security", {"path": nas_file_path})
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS: {e}", "nas_download", {"path": nas_file_path})
        return None


def nas_upload_file(conn: SMBConnection, file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS file path: {nas_file_path}", "nas_security", {"path": nas_file_path})
        return False

    try:
        # Ensure parent directory exists
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory_recursive(conn, parent_dir)

        file_obj.seek(0)
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {e}", "nas_upload", {"path": nas_file_path})
        return False


def list_nas_directory(conn: SMBConnection, dir_path: str) -> List[str]:
    """List all files in NAS directory."""
    try:
        shared_files = conn.listPath(os.getenv("NAS_SHARE_NAME"), dir_path)
        return [f.filename for f in shared_files if not f.filename.startswith('.')]
    except Exception as e:
        log_error(f"Failed to list directory {dir_path}", "nas_list", {"error": str(e)})
        return []


def nas_rename_file(conn: SMBConnection, old_path: str, new_path: str) -> bool:
    """Rename file on NAS (copy → delete pattern for SMB)."""
    try:
        content = nas_download_file(conn, old_path)
        if not content:
            return False
        file_obj = io.BytesIO(content)
        if not nas_upload_file(conn, file_obj, new_path):
            return False
        nas_delete_file(conn, old_path)
        return True
    except Exception as e:
        log_error(f"Failed to rename file {old_path} → {new_path}", "nas_rename", {"error": str(e)})
        return False


def nas_delete_file(conn: SMBConnection, file_path: str) -> bool:
    """Delete file from NAS."""
    try:
        conn.deleteFiles(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception as e:
        log_error(f"Failed to delete file {file_path}", "nas_delete", {"error": str(e)})
        return False


def initialize_empty_manifest() -> Dict:
    """Initialize a new empty manifest."""
    return {
        "version": "6.0.0",
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_transcripts": 0,
        "completed_transcripts": 0,
        "failed_transcripts": 0,
        "consolidation_status": "not_started",
        "transcripts": {}
    }


def load_manifest_with_recovery() -> Dict:
    """Load manifest with crash recovery support."""
    nas_conn = get_nas_connection()
    if not nas_conn:
        return initialize_empty_manifest()

    try:
        manifest_path = nas_path_join(
            config["stage_06_llm_classification"]["output_data_path"],
            "manifest.json"
        )
        temp_path = nas_path_join(
            config["stage_06_llm_classification"]["output_data_path"],
            "manifest.json.tmp"
        )

        manifest_content = nas_download_file(nas_conn, manifest_path)
        if manifest_content:
            try:
                manifest = json.loads(manifest_content.decode('utf-8'))
                nas_conn.close()
                return manifest
            except json.JSONDecodeError:
                log_console("Main manifest corrupted, trying temp...", "WARNING")

        temp_content = nas_download_file(nas_conn, temp_path)
        if temp_content:
            try:
                manifest = json.loads(temp_content.decode('utf-8'))
                log_console("Recovered from temp manifest", "WARNING")
                nas_conn.close()
                return manifest
            except json.JSONDecodeError:
                log_console("Temp manifest also corrupted", "WARNING")

        log_console("No valid manifest found, initializing new one")
        nas_conn.close()
        return initialize_empty_manifest()

    except Exception as e:
        log_error(f"Failed to load manifest: {e}", "manifest_load")
        if nas_conn:
            nas_conn.close()
        return initialize_empty_manifest()


def update_manifest_atomic(transcript_id: str, updates: Dict) -> bool:
    """Atomically update manifest using temp file pattern."""
    nas_conn = get_nas_connection()
    if not nas_conn:
        log_error("Failed to connect to NAS for manifest update", "manifest_update")
        return False

    try:
        manifest_path = nas_path_join(
            config["stage_06_llm_classification"]["output_data_path"],
            "manifest.json"
        )
        temp_path = nas_path_join(
            config["stage_06_llm_classification"]["output_data_path"],
            "manifest.json.tmp"
        )

        manifest_content = nas_download_file(nas_conn, manifest_path)
        if manifest_content:
            try:
                manifest = json.loads(manifest_content.decode('utf-8'))
            except json.JSONDecodeError:
                manifest = initialize_empty_manifest()
        else:
            manifest = initialize_empty_manifest()

        if transcript_id not in manifest["transcripts"]:
            manifest["transcripts"][transcript_id] = {}

        manifest["transcripts"][transcript_id].update(updates)
        manifest["last_updated"] = datetime.now().isoformat()

        manifest["total_transcripts"] = len(manifest["transcripts"])
        manifest["completed_transcripts"] = sum(
            1 for t in manifest["transcripts"].values() if t.get("status") == "completed"
        )
        manifest["failed_transcripts"] = sum(
            1 for t in manifest["transcripts"].values() if t.get("status") == "failed"
        )

        temp_obj = io.BytesIO(json.dumps(manifest, indent=2).encode('utf-8'))
        if not nas_upload_file(nas_conn, temp_obj, temp_path):
            nas_conn.close()
            return False

        if nas_file_exists(nas_conn, manifest_path):
            nas_delete_file(nas_conn, manifest_path)
        nas_rename_file(nas_conn, temp_path, manifest_path)

        nas_conn.close()
        return True

    except Exception as e:
        log_error(f"Failed to update manifest: {e}", "manifest_update", {"error": str(e)})
        if nas_conn:
            nas_conn.close()
        return False


def create_json_from_records(records: List[Dict]) -> str:
    """Create JSON content from records."""
    return json.dumps(records, indent=2, default=str)


def create_metadata_file(nas_conn: SMBConnection, transcript_id: str, json_path: str, record_count: int) -> Optional[Dict]:
    """Create validation metadata after successful write."""
    import hashlib

    try:
        json_content = nas_download_file(nas_conn, json_path)
        if not json_content:
            raise ValueError("Failed to download JSON for metadata creation")

        metadata = {
            "transcript_id": transcript_id,
            "created_at": datetime.now().isoformat(),
            "file_size_bytes": len(json_content),
            "sha256_checksum": hashlib.sha256(json_content).hexdigest(),
            "expected_records": record_count,
            "stage_version": "6.0.0",
            "processing_host": os.getenv("CLIENT_MACHINE_NAME")
        }

        meta_path = f"{json_path}.meta"
        meta_content = json.dumps(metadata, indent=2)
        meta_obj = io.BytesIO(meta_content.encode('utf-8'))

        if not nas_upload_file(nas_conn, meta_obj, meta_path):
            raise ValueError("Failed to upload metadata file")

        return metadata

    except Exception as e:
        log_error(f"Failed to create metadata for {transcript_id}: {e}", "metadata_create")
        return None


def validate_json_integrity(nas_conn: SMBConnection, json_path: str, meta_path: str) -> bool:
    """Validate JSON file against metadata."""
    import hashlib

    try:
        meta_content = nas_download_file(nas_conn, meta_path)
        if not meta_content:
            return False

        metadata = json.loads(meta_content.decode('utf-8'))
        json_content = nas_download_file(nas_conn, json_path)
        if not json_content:
            return False

        checks = {
            "file_size": len(json_content) == metadata["file_size_bytes"],
            "checksum": hashlib.sha256(json_content).hexdigest() == metadata["sha256_checksum"],
        }

        try:
            records = json.loads(json_content.decode('utf-8'))
            checks["valid_json"] = True
            checks["record_count"] = len(records) == metadata["expected_records"]
        except:
            checks["valid_json"] = False
            checks["record_count"] = False

        return all(checks.values())

    except Exception as e:
        log_console(f"Validation error for {json_path}: {e}", "ERROR")
        return False


def cleanup_tmp_file(nas_conn: SMBConnection, transcript_id: str):
    """Remove temporary file after crash or error."""
    output_path = config["stage_06_llm_classification"]["output_data_path"]
    tmp_path = nas_path_join(output_path, "individual", f"{transcript_id}.json.tmp")
    if nas_file_exists(nas_conn, tmp_path):
        nas_delete_file(nas_conn, tmp_path)


def quarantine_corrupt_file(nas_conn: SMBConnection, transcript_id: str):
    """Move corrupted file to failed/ directory."""
    output_path = config["stage_06_llm_classification"]["output_data_path"]
    json_path = nas_path_join(output_path, "individual", f"{transcript_id}.json")
    meta_path = f"{json_path}.meta"
    tmp_path = f"{json_path}.tmp"

    failed_dir = nas_path_join(output_path, "failed")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if nas_file_exists(nas_conn, json_path):
        nas_rename_file(nas_conn, json_path, nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.json.corrupt"))
    if nas_file_exists(nas_conn, meta_path):
        nas_rename_file(nas_conn, meta_path, nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.meta.corrupt"))
    if nas_file_exists(nas_conn, tmp_path):
        nas_rename_file(nas_conn, tmp_path, nas_path_join(failed_dir, f"{transcript_id}_{timestamp}.tmp.corrupt"))


def cleanup_individual_files(nas_conn: SMBConnection, file_list: List[str]):
    """Delete individual files after successful consolidation."""
    output_path = config["stage_06_llm_classification"]["output_data_path"]
    for json_filename in file_list:
        json_path = nas_path_join(output_path, "individual", json_filename)
        meta_path = f"{json_path}.meta"
        nas_delete_file(nas_conn, json_path)
        nas_delete_file(nas_conn, meta_path)
    log_console(f"Cleaned up {len(file_list)} individual files")


def process_and_save_transcript_resilient(
    transcript_id: str,
    transcript_records: List[Dict],
    enhanced_error_logger: EnhancedErrorLogger,
    max_retries: int = 3
) -> bool:
    """Crash-resilient transcript processing with error handling."""
    retry_delay = 5

    for attempt in range(max_retries):
        nas_conn = None
        try:
            update_manifest_atomic(transcript_id, {
                "status": "in_progress",
                "started_at": datetime.now().isoformat(),
                "attempt": attempt + 1
            })

            log_console(f"  Processing classifications for {transcript_id} (attempt {attempt + 1}/{max_retries})")
            enhanced_records = process_transcript(transcript_records, transcript_id, enhanced_error_logger)

            if not enhanced_records:
                raise ValueError(f"No records generated for {transcript_id}")

            json_content = create_json_from_records(enhanced_records)

            nas_conn = get_nas_connection()
            if not nas_conn:
                raise RuntimeError("Failed to connect to NAS")

            output_path = config["stage_06_llm_classification"]["output_data_path"]
            individual_path = nas_path_join(output_path, "individual", f"{transcript_id}.json")
            temp_path = f"{individual_path}.tmp"

            log_console(f"  Writing to temp file...")
            temp_obj = io.BytesIO(json_content.encode('utf-8'))
            if not nas_upload_file(nas_conn, temp_obj, temp_path):
                raise IOError(f"Failed to upload temp file for {transcript_id}")

            log_console(f"  Finalizing file...")
            if not nas_rename_file(nas_conn, temp_path, individual_path):
                raise IOError(f"Failed to rename temp file for {transcript_id}")

            log_console(f"  Creating metadata...")
            metadata = create_metadata_file(nas_conn, transcript_id, individual_path, len(enhanced_records))
            if not metadata:
                raise ValueError(f"Failed to create metadata for {transcript_id}")

            log_console(f"  Validating file integrity...")
            if not validate_json_integrity(nas_conn, individual_path, f"{individual_path}.meta"):
                raise ValueError(f"Final validation failed for {transcript_id}")

            update_manifest_atomic(transcript_id, {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "record_count": len(enhanced_records),
                "file_size_bytes": len(json_content),
                "individual_file": individual_path,
                "metadata_file": f"{individual_path}.meta"
            })

            log_console(f"✓ Successfully saved {transcript_id} ({len(enhanced_records)} records)")

            if nas_conn:
                nas_conn.close()

            return True

        except Exception as e:
            log_console(f"Error processing {transcript_id} (attempt {attempt + 1}/{max_retries}): {e}", "WARNING")

            if nas_conn:
                try:
                    cleanup_tmp_file(nas_conn, transcript_id)
                except:
                    pass
                nas_conn.close()

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            else:
                log_console(f"✗ Failed to process {transcript_id} after {max_retries} attempts", "ERROR")

                nas_conn_cleanup = get_nas_connection()
                if nas_conn_cleanup:
                    try:
                        quarantine_corrupt_file(nas_conn_cleanup, transcript_id)
                    except:
                        pass
                    nas_conn_cleanup.close()

                update_manifest_atomic(transcript_id, {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat(),
                    "retry_count": max_retries
                })

                return False

    return False


def analyze_resume_state(all_transcripts: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str]]:
    """Analyze which transcripts need processing based on manifest."""
    manifest = load_manifest_with_recovery()

    to_process = []
    completed = []
    failed_list = []

    for transcript_id in all_transcripts.keys():
        transcript_info = manifest.get("transcripts", {}).get(transcript_id, {})
        status = transcript_info.get("status", "not_started")

        if status == "completed":
            nas_conn = get_nas_connection()
            if nas_conn:
                output_path = config["stage_06_llm_classification"]["output_data_path"]
                json_path = nas_path_join(output_path, "individual", f"{transcript_id}.json")
                meta_path = f"{json_path}.meta"

                if nas_file_exists(nas_conn, json_path) and validate_json_integrity(nas_conn, json_path, meta_path):
                    completed.append(transcript_id)
                    nas_conn.close()
                else:
                    log_console(f"  {transcript_id}: marked complete but file invalid - will reprocess", "WARNING")
                    to_process.append(transcript_id)
                    nas_conn.close()
            else:
                completed.append(transcript_id)

        elif status == "failed":
            failed_list.append(transcript_id)
            retry_failed = config["stage_06_llm_classification"].get("retry_failed_on_resume", False)
            if retry_failed:
                log_console(f"  {transcript_id}: previously failed - will retry", "WARNING")
                to_process.append(transcript_id)

        elif status == "in_progress":
            log_console(f"  {transcript_id}: was in progress - will reprocess", "WARNING")
            to_process.append(transcript_id)

        else:
            to_process.append(transcript_id)

    return to_process, completed, failed_list


def consolidate_individual_files() -> Optional[str]:
    """Consolidate all individual transcript JSONs into master file."""
    nas_conn = get_nas_connection()
    if not nas_conn:
        log_console("Failed to connect to NAS for consolidation", "ERROR")
        return None

    try:
        output_path = config["stage_06_llm_classification"]["output_data_path"]
        individual_dir = nas_path_join(output_path, "individual")
        consolidated_dir = nas_path_join(output_path, "consolidated")

        log_console("Scanning for individual transcript files...")
        all_files = list_nas_directory(nas_conn, individual_dir)
        json_files = [f for f in all_files if f.endswith('.json') and not f.endswith('.tmp')]

        log_console(f"Found {len(json_files)} individual JSON files to consolidate")

        if len(json_files) == 0:
            log_console("No files to consolidate", "WARNING")
            nas_conn.close()
            return None

        valid_files = []
        invalid_files = []

        log_console("Validating individual files...")
        for json_file in json_files:
            json_path = nas_path_join(individual_dir, json_file)
            meta_path = f"{json_path}.meta"

            if validate_json_integrity(nas_conn, json_path, meta_path):
                valid_files.append(json_file)
            else:
                invalid_files.append(json_file)
                transcript_id = json_file.replace('.json', '')
                quarantine_corrupt_file(nas_conn, transcript_id)

        if invalid_files:
            log_console(f"WARNING: {len(invalid_files)} files failed validation - quarantined", "WARNING")

        log_console(f"Consolidating {len(valid_files)} validated files...")

        temp_consolidated_path = nas_path_join(consolidated_dir, "stage_06_classifications.json.tmp")
        final_consolidated_path = nas_path_join(consolidated_dir, "stage_06_classifications.json")

        all_records = []

        for i, json_file in enumerate(valid_files):
            json_path = nas_path_join(individual_dir, json_file)
            json_content = nas_download_file(nas_conn, json_path)

            if not json_content:
                log_console(f"ERROR: Could not download {json_file}", "ERROR")
                continue

            records = json.loads(json_content.decode('utf-8'))
            all_records.extend(records)

            if (i + 1) % 10 == 0 or (i + 1) == len(valid_files):
                log_console(f"  Progress: {i + 1}/{len(valid_files)} files, {len(all_records)} records")

        log_console("Uploading consolidated file...")
        consolidated_content = json.dumps(all_records, indent=2, default=str)
        temp_file_obj = io.BytesIO(consolidated_content.encode('utf-8'))

        if not nas_upload_file(nas_conn, temp_file_obj, temp_consolidated_path):
            raise RuntimeError("Failed to write consolidated temp file")

        log_console("Finalizing consolidated file...")
        if nas_file_exists(nas_conn, final_consolidated_path):
            nas_delete_file(nas_conn, final_consolidated_path)
        if not nas_rename_file(nas_conn, temp_consolidated_path, final_consolidated_path):
            raise RuntimeError("Failed to finalize consolidated file")

        log_console(f"✓ Consolidation complete: {len(all_records)} total records in {len(valid_files)} transcripts")

        cleanup_enabled = config["stage_06_llm_classification"].get("cleanup_after_consolidation", True)
        if cleanup_enabled:
            log_console("Cleaning up individual files...")
            cleanup_individual_files(nas_conn, valid_files)
            log_console("✓ Cleanup complete")
        else:
            log_console("Individual files preserved (cleanup disabled in config)")

        nas_conn.close()
        return final_consolidated_path

    except Exception as e:
        log_console(f"Consolidation failed: {e}", "ERROR")
        log_error(f"Consolidation error: {e}", "consolidation_error")
        if nas_conn:
            nas_conn.close()
        return None


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory recursively on NAS."""
    if not validate_nas_path(dir_path):
        return False

    path_parts = [part for part in dir_path.strip("/").split("/") if part]
    current_path = ""
    
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part
        try:
            conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
        except Exception:
            # Directory might already exist
            pass
    
    return True


# ============================================================================
# CLASSIFICATION HELPER FUNCTIONS
# ============================================================================

def build_numbered_category_list() -> str:
    """Build formatted category list with IDs and descriptions."""
    global CATEGORY_REGISTRY
    
    lines = []
    for cat_id in sorted(CATEGORY_REGISTRY.keys()):
        cat = CATEGORY_REGISTRY[cat_id]
        lines.append(f"{cat_id}: {cat['name']} - {cat['description']}")
    
    return "\n".join(lines)


def format_previous_speaker_blocks_with_classifications(
    previous_blocks: List[Dict], 
    max_blocks: int = 2
) -> str:
    """Format previous speaker blocks with their classifications for context."""
    
    if not previous_blocks:
        return "No previous speaker blocks available for context."
    
    # Take only the most recent blocks
    recent_blocks = previous_blocks[-max_blocks:] if len(previous_blocks) > max_blocks else previous_blocks
    
    context_parts = []
    context_parts.append("\n=== PREVIOUS SPEAKER BLOCKS ===")
    
    for block_idx, block in enumerate(recent_blocks, start=1):
        speaker_name = block["speaker_name"]
        paragraphs = block["paragraphs"]
        
        # Add speaker name
        context_parts.append(f"\n[Previous Block {block_idx}: {speaker_name}]:")
        
        # Show paragraphs (limited for context)
        for para in paragraphs[:3]:  # Show first 3 paragraphs as preview
            content = para.get('paragraph_content', para.get('content', ''))
            if len(content) > 200:
                content = content[:200] + "..."
            context_parts.append(content)
        
        if len(paragraphs) > 3:
            context_parts.append(f"... and {len(paragraphs) - 3} more paragraphs")
        
        # Show classifications at the end (all paragraphs have same in new system)
        if paragraphs:
            classification_ids = paragraphs[0].get('classification_ids', [])
            if classification_ids:
                classification_names = [f"{cid}: {CATEGORY_REGISTRY[cid]['name']}" for cid in classification_ids]
                context_parts.append(f"\nClassifications: {', '.join(classification_names)}")
    
    return "\n".join(context_parts)


def format_current_speaker_block(speaker_block_records: List[Dict]) -> str:
    """Format current speaker block for classification."""
    
    if not speaker_block_records:
        return ""
    
    # Get speaker name from first record
    first_record = speaker_block_records[0]
    speaker_name = first_record.get("speaker", first_record.get("speaker_name", "Unknown"))
    
    context_parts = []
    context_parts.append("\n=== CURRENT SPEAKER BLOCK ===")
    context_parts.append(f"\n[{speaker_name}]:")
    
    # Add all paragraphs without labeling
    for record in speaker_block_records:
        content = record.get("paragraph_content", record.get("content", ""))
        context_parts.append(content)
    
    return "\n".join(context_parts)




def process_qa_conversation_single_pass(
    qa_conversation_records: List[Dict],
    transcript_info: Dict[str, Any],
    qa_group_id: int,
    total_qa_groups: int,
    enhanced_error_logger: EnhancedErrorLogger
) -> List[Dict]:
    """Process Q&A conversation with single-pass classification at conversation level."""
    global llm_client, config
    
    log_console(f"Processing Q&A group {qa_group_id} ({len(qa_conversation_records)} paragraphs)")
    
    # Build conversation text for classification
    sorted_records = sorted(qa_conversation_records, key=lambda x: x["paragraph_id"])
    
    # Group by speaker blocks for readability in prompt
    qa_speaker_blocks = defaultdict(list)
    for record in sorted_records:
        qa_speaker_blocks[record["speaker_block_id"]].append(record)
    
    # Format conversation for prompt - show as current Q&A group
    conversation_parts = []
    conversation_parts.append("\n=== CURRENT Q&A CONVERSATION ===")
    conversation_parts.append(f"Q&A Group ID: {qa_group_id}")
    
    speaker_block_count = 0
    for speaker_block_id, block_records in sorted(qa_speaker_blocks.items()):
        speaker_block_count += 1
        speaker_name = block_records[0].get("speaker", block_records[0].get("speaker_name", "Unknown"))
        
        # Label each speaker block within the Q&A
        conversation_parts.append(f"\n[Speaker Block {speaker_block_count}: {speaker_name}]:")
        
        # Add all paragraphs for this speaker
        for record in block_records:
            content = record.get("paragraph_content", record.get("content", ""))
            conversation_parts.append(content)
    
    full_conversation = "\n".join(conversation_parts)
    
    # ========== SINGLE-PASS CLASSIFICATION (One set for entire conversation) ==========
    try:
        categories_list = build_numbered_category_list()
        
        system_prompt = f"""You are a financial analyst specializing in earnings call transcript classification for a retrieval system.

<document_context>
Institution: {transcript_info.get("company_name", transcript_info.get("ticker", "Unknown"))}
Fiscal Period: {transcript_info.get("fiscal_quarter", "Q1")} {transcript_info.get("fiscal_year", 2024)}
Section: Q&A Conversation
Q&A Group: {qa_group_id} of {total_qa_groups}
</document_context>

<task_description>
You are performing comprehensive classification for the ENTIRE Q&A conversation.
Assign ALL applicable category IDs to this complete Q&A exchange.
All paragraphs in this conversation will receive the same classification list.
Include all topics discussed in both questions and answers.
</task_description>

<retrieval_context>
This classification enables comprehensive retrieval of Q&A conversations:
- Users searching for ANY topic mentioned will find this entire Q&A exchange
- The complete question-answer context is preserved
- Multiple categories ensure no relevant content is missed
- The goal is complete retrieval of all financial insights discussed
</retrieval_context>

<classification_guidelines>
1. Classify the ENTIRE conversation as a single unit
2. Include ALL topics mentioned in questions AND answers
3. Be comprehensive - any topic discussed gets included
4. Do NOT use previous conversation context (each Q&A is independent)
5. Think broadly about what searches should surface this conversation
</classification_guidelines>

<financial_categories>
{categories_list}
</financial_categories>

<conversation>
{full_conversation}
</conversation>

<response_format>
Use the classify_qa_conversation function to provide ALL applicable category IDs for this entire Q&A conversation.
</response_format>"""

        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please classify this Q&A conversation with ALL applicable categories."}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "classify_qa_conversation",
                    "description": "Assign ALL applicable category IDs to the Q&A conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category_ids": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 22
                                },
                                "description": "ALL applicable category IDs for this Q&A conversation"
                            }
                        },
                        "required": ["category_ids"]
                    }
                }
            }],
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse classifications
        classification_ids = []
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Validate category IDs
            for cat_id in result.get("category_ids", []):
                if cat_id in CATEGORY_REGISTRY:
                    classification_ids.append(cat_id)
                else:
                    log_console(f"Invalid category ID {cat_id}, skipping", "WARNING")
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            })
        
        # If no categories assigned, use fallback
        if not classification_ids:
            classification_ids = [0]
            log_console(f"No categories assigned to Q&A group {qa_group_id}, using fallback", "WARNING")
        
    except Exception as e:
        error_msg = f"Q&A classification failed for group {qa_group_id}: {e}"
        log_error(error_msg, "qa_classification", {"qa_group": qa_group_id, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            transcript_info.get("ticker", "unknown"),
            "qa_single_pass",
            error_msg
        )
        
        # Fallback to category 0
        classification_ids = [0]
        log_console(f"Applying fallback category 0 to Q&A group {qa_group_id}", "WARNING")
    
    # Apply same classifications to all records in the conversation
    classification_names = [CATEGORY_REGISTRY[cid]["name"] for cid in classification_ids]
    
    for record in sorted_records:
        record["classification_ids"] = classification_ids
        record["classification_names"] = classification_names
    
    return sorted_records


def process_speaker_block_single_pass(
    speaker_block_records: List[Dict],
    transcript_info: Dict[str, Any],
    speaker_block_number: int,
    total_speaker_blocks: int,
    previous_blocks: List[Dict],
    enhanced_error_logger: EnhancedErrorLogger
) -> List[Dict]:
    """Process a speaker block with single-pass classification."""
    global llm_client, config
    
    # Sort records by paragraph_id to maintain order
    sorted_records = sorted(speaker_block_records, key=lambda x: x["paragraph_id"])
    
    # Extract speaker and section info
    first_record = sorted_records[0]
    # Fix field name mismatch: Stage 3 uses 'speaker', not 'speaker_name'
    speaker_name = first_record.get("speaker", first_record.get("speaker_name", "Unknown"))
    speaker_role = ""  # Could be extracted from speaker name if formatted
    section_name = first_record.get("section_name", "Unknown")
    
    log_console(f"Processing speaker block #{speaker_block_number} - {speaker_name} ({len(sorted_records)} paragraphs)")
    
    # ========== SINGLE-PASS CLASSIFICATION ==========
    try:
        # Build prompt for single-pass classification
        categories_list = build_numbered_category_list()
        # Reduce context to last 2 blocks
        previous_context = format_previous_speaker_blocks_with_classifications(previous_blocks[-2:] if len(previous_blocks) > 2 else previous_blocks)
        current_block = format_current_speaker_block(sorted_records)
        
        system_prompt = f"""You are a financial analyst specializing in earnings call transcript classification for major financial institutions.

<document_context>
You are analyzing an earnings call transcript with the following context:
- Institution: {transcript_info.get("company_name", transcript_info.get("ticker", "Unknown"))}
- Document Type: Quarterly Earnings Call Transcript
- Fiscal Period: {transcript_info.get("fiscal_quarter", "Q1")} {transcript_info.get("fiscal_year", 2024)}
- Section: {section_name}
- Current Speaker: {speaker_name}, {speaker_role if speaker_role else 'Role Unknown'}
- Speaker Block: #{speaker_block_number} of {total_speaker_blocks}
</document_context>

<task_description>
You are performing comprehensive classification for a retrieval system. 
Assign ALL applicable category IDs to this entire speaker block.
All paragraphs in this speaker block will receive the same classification list.
Include both the main topic and any secondary topics mentioned or referenced.
</task_description>

<retrieval_context>
This classification enables a search system where:
- Users query for specific financial topics (e.g., "show me discussions about margins")
- Content matching ANY of the assigned categories will be found
- Multiple categories ensure comprehensive coverage
- The goal is complete retrieval of all relevant financial insights
</retrieval_context>

<classification_guidelines>
1. Assign ALL applicable categories for this speaker block
2. Include the main topic AND all secondary topics mentioned or referenced
3. Be comprehensive - include any category that has relevance to the content
4. Consider the context from the last 2 speaker blocks when classifying
5. For {transcript_info.get("company_name", transcript_info.get("ticker", "Unknown"))}, be aware of their specific terminology and reporting structure
6. Only use category 0 (Non-Relevant) alone for pure procedural content with NO financial information
7. If content has ANY financial or business relevance, include all applicable categories
8. Think broadly about what searches should surface this content
</classification_guidelines>

<financial_categories>
{categories_list}
</financial_categories>

<previous_context>
{previous_context}
</previous_context>

<current_task>
Classify the entire speaker block with ALL applicable category IDs (0-22).
All paragraphs in this block will receive the same classification list.
{current_block}
</current_task>

<response_format>
Use the classify_speaker_block function to provide ALL applicable category IDs for this speaker block.
</response_format>"""
        
        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please classify this speaker block with ALL applicable categories."}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "classify_speaker_block",
                    "description": "Assign ALL applicable category IDs to the speaker block",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category_ids": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 22
                                },
                                "description": "ALL applicable category IDs for this speaker block (0-22)"
                            }
                        },
                        "required": ["category_ids"]
                    }
                }
            }],
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse classifications
        classification_ids = []
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Validate category IDs
            for cat_id in result.get("category_ids", []):
                if cat_id in CATEGORY_REGISTRY:
                    classification_ids.append(cat_id)
                else:
                    log_console(f"Invalid category ID {cat_id}, skipping", "WARNING")
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            })
        
        # If no categories assigned or only 0, use fallback
        if not classification_ids:
            classification_ids = [0]
            log_console(f"No categories assigned, using fallback category 0", "WARNING")
        
    except Exception as e:
        error_msg = f"Classification failed for speaker block: {e}"
        log_error(error_msg, "classification", {"speaker": speaker_name, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            transcript_info.get("ticker", "unknown"),
            "single_pass",
            error_msg
        )
        
        # Fallback to category 0
        classification_ids = [0]
        log_console(f"Applying fallback category 0 due to classification error", "WARNING")
    
    # Apply same classifications to all paragraphs in the speaker block
    classification_names = [CATEGORY_REGISTRY[cid]["name"] for cid in classification_ids]
    
    for record in sorted_records:
        record["classification_ids"] = classification_ids
        record["classification_names"] = classification_names
    
    return sorted_records


def process_transcript_v2(
    transcript_key: str,
    transcript_data: Dict,
    enhanced_error_logger: EnhancedErrorLogger
) -> Tuple[List[Dict], bool]:
    """Process a single transcript with two-pass classification and context."""
    global logger
    
    try:
        # Extract transcript info
        if ".xml" in transcript_key:
            # Filename format - extract info from first record
            first_record = None
            if transcript_data["management_discussion"]:
                first_record = transcript_data["management_discussion"][0]
            elif transcript_data["qa_groups"]:
                first_record = list(transcript_data["qa_groups"].values())[0][0]
            
            if first_record:
                transcript_info = {
                    "ticker": first_record.get("ticker", "unknown"),
                    "company_name": first_record.get("company_name", "Unknown"),
                    "fiscal_year": first_record.get("fiscal_year", 2024),
                    "fiscal_quarter": first_record.get("fiscal_quarter", "Q1")
                }
            else:
                transcript_info = {
                    "ticker": "unknown",
                    "company_name": "Unknown",
                    "fiscal_year": 2024,
                    "fiscal_quarter": "Q1"
                }
        else:
            # ticker_event_id format
            parts = transcript_key.split("_")
            transcript_info = {
                "ticker": parts[0] if parts else "unknown",
                "company_name": "Unknown",
                "fiscal_year": 2024,
                "fiscal_quarter": "Q1"
            }
        
        # Refresh OAuth token per transcript
        refresh_oauth_token_for_transcript(transcript_info)
        
        classified_records = []
        previous_md_blocks = []
        previous_qa_blocks = []
        
        # Process Management Discussion speaker blocks
        if transcript_data["management_discussion"]:
            # Group by speaker blocks
            md_speaker_blocks = defaultdict(list)
            for record in sorted(transcript_data["management_discussion"], key=lambda x: x["paragraph_id"]):
                md_speaker_blocks[record["speaker_block_id"]].append(record)
            
            total_md_blocks = len(md_speaker_blocks)
            
            for block_num, (block_id, block_records) in enumerate(md_speaker_blocks.items(), start=1):
                # Process with single-pass classification
                classified_block = process_speaker_block_single_pass(
                    block_records,
                    transcript_info,
                    block_num,
                    total_md_blocks,
                    previous_md_blocks[-2:],  # Last 2 blocks for context
                    enhanced_error_logger
                )
                
                classified_records.extend(classified_block)
                
                # Add to previous blocks for next iteration
                # Fix field name: use 'speaker' field from Stage 3
                previous_md_blocks.append({
                    "speaker_name": block_records[0].get("speaker", "Unknown"),
                    "section_name": "MANAGEMENT DISCUSSION SECTION",
                    "paragraphs": classified_block
                })
        
        # Process Q&A groups - each group is a complete conversation
        for qa_group_id, qa_records in transcript_data["qa_groups"].items():
            # Process entire Q&A conversation as one unit (not split by speaker blocks)
            # This ensures question context is preserved for answer classification
            log_console(f"Processing Q&A group {qa_group_id} ({len(qa_records)} paragraphs)")
            
            # Sort records by paragraph_id to maintain order
            sorted_qa_records = sorted(qa_records, key=lambda x: x["paragraph_id"])
            
            # Process the entire Q&A conversation together
            classified_conversation = process_qa_conversation_single_pass(
                sorted_qa_records,
                transcript_info,
                qa_group_id,
                len(transcript_data["qa_groups"]),
                enhanced_error_logger
            )
            
            classified_records.extend(classified_conversation)
            
            # Add to previous blocks for context in next Q&A
            previous_qa_blocks.append({
                "qa_group_id": qa_group_id,
                "section_name": "Q&A",
                "paragraphs": classified_conversation
            })
        
        # Process any unpaired Q&A records (shouldn't happen if Stage 5 worked correctly)
        if transcript_data.get("unpaired_qa"):
            log_console(f"WARNING: Processing {len(transcript_data['unpaired_qa'])} unpaired Q&A records as speaker blocks", "WARNING")
            
            # Group by speaker blocks
            unpaired_speaker_blocks = defaultdict(list)
            for record in sorted(transcript_data["unpaired_qa"], key=lambda x: x["paragraph_id"]):
                unpaired_speaker_blocks[record["speaker_block_id"]].append(record)
            
            for block_records in unpaired_speaker_blocks.values():
                # Process as speaker blocks (fallback)
                classified_block = process_speaker_block_single_pass(
                    block_records,
                    transcript_info,
                    1,
                    1,
                    [],
                    enhanced_error_logger
                )
                classified_records.extend(classified_block)
        
        return classified_records, True
        
    except Exception as e:
        error_msg = f"Failed to process transcript {transcript_key}: {e}"
        log_error(error_msg, "transcript_processing", {"transcript": transcript_key, "error": str(e)})
        enhanced_error_logger.log_processing_error(transcript_key, error_msg)
        return [], False


def main():
    """Main execution function."""
    global config, logger, ssl_cert_path, llm_client, oauth_token
    
    # Initialize
    logger = setup_logging()
    log_console("=" * 60)
    log_console("Stage 6: LLM-Based Financial Content Classification")
    log_console("Single-pass classification with all applicable categories")
    log_console("=" * 60)
    
    # Stage tracking
    start_time = datetime.now()
    enhanced_error_logger = EnhancedErrorLogger()
    failed_transcripts = []
    
    try:
        # Validate environment
        validate_environment_variables()
        
        # Connect to NAS
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS")
        
        # Load configuration
        config = load_config_from_nas(nas_conn)
        stage_config = config["stage_06_llm_classification"]
        
        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Setup proxy
        proxy_url = setup_proxy_configuration()
        
        # Load Stage 5 output
        input_path = nas_path_join(stage_config["input_data_path"])
        log_console(f"Loading Stage 5 output from: {input_path}")
        
        input_data = nas_download_file(nas_conn, input_path)
        if not input_data:
            raise FileNotFoundError(f"Stage 5 output not found at {input_path}")
        
        stage5_data = json.loads(input_data.decode('utf-8'))
        # Stage 5 outputs a JSON array directly
        if isinstance(stage5_data, list):
            all_records = stage5_data
        else:
            all_records = stage5_data.get("records", [])
        log_console(f"Loaded {len(all_records)} records from Stage 5")
        
        # Normalize field names from Stage 3/5 to what Stage 6 expects
        log_console("Normalizing field names for compatibility...")
        for record in all_records:
            # Map 'speaker' to 'speaker_name' for consistency
            if 'speaker' in record and 'speaker_name' not in record:
                record['speaker_name'] = record['speaker']
            
            # Add company_name if missing (use ticker as fallback)
            if 'company_name' not in record:
                record['company_name'] = record.get('ticker', 'Unknown')
            
            # Ensure content field exists (map from paragraph_content if needed)
            if 'paragraph_content' in record and 'content' not in record:
                record['content'] = record['paragraph_content']
        
        # Apply development mode limits
        if stage_config.get("dev_mode", False):
            max_transcripts = stage_config.get("dev_max_transcripts", 2)
            log_console(f"Development mode: limiting to {max_transcripts} transcripts")
            
            # Group by transcript
            transcript_groups = defaultdict(list)
            for record in all_records:
                transcript_key = record.get("filename", f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
                transcript_groups[transcript_key].append(record)
            
            limited_records = []
            for i, (transcript_key, transcript_records) in enumerate(transcript_groups.items()):
                if i >= max_transcripts:
                    break
                limited_records.extend(transcript_records)
            
            all_records = limited_records
            log_console(f"Limited to {len(all_records)} records from {min(max_transcripts, len(transcript_groups))} transcripts")
        
        # Group records by transcript and section type
        transcripts = defaultdict(lambda: {"management_discussion": [], "qa_groups": defaultdict(list), "unpaired_qa": []})
        
        # Track statistics
        total_md_records = 0
        total_qa_with_group = 0
        total_qa_without_group = 0
        
        for record in all_records:
            transcript_key = record.get("filename", f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
            section_name = record.get("section_name")
            
            if section_name == "MANAGEMENT DISCUSSION SECTION":
                transcripts[transcript_key]["management_discussion"].append(record)
                total_md_records += 1
            elif section_name == "Q&A":
                if record.get("qa_group_id"):
                    qa_group_id = record["qa_group_id"]
                    transcripts[transcript_key]["qa_groups"][qa_group_id].append(record)
                    total_qa_with_group += 1
                else:
                    # Q&A records without group ID (shouldn't happen if Stage 5 worked correctly)
                    transcripts[transcript_key]["unpaired_qa"].append(record)
                    total_qa_without_group += 1
                    log_console(f"WARNING: Q&A record without qa_group_id: {record.get('paragraph_id')}", "WARNING")
        
        log_console(f"Found {len(transcripts)} transcripts to process with two-pass classification")
        log_console(f"  - MD records: {total_md_records}")
        log_console(f"  - Q&A records with group ID: {total_qa_with_group}")
        if total_qa_without_group > 0:
            log_console(f"  - Q&A records WITHOUT group ID: {total_qa_without_group} (will process as speaker blocks)", "WARNING")
        
        # Analyze resume state
        log_console("=" * 50)
        log_console("Analyzing resume state...")
        to_process_ids, completed_ids, failed_ids = analyze_resume_state(transcripts)

        log_console(f"Resume analysis:")
        log_console(f"  To process: {len(to_process_ids)}")
        log_console(f"  Already completed: {len(completed_ids)}")
        log_console(f"  Previously failed: {len(failed_ids)}")
        log_console("=" * 50)

        # Filter transcripts to process
        transcripts_to_process = {
            tid: transcripts[tid] for tid in to_process_ids if tid in transcripts
        }

        if len(transcripts_to_process) == 0:
            log_console("All transcripts already processed. Moving to consolidation...")
        else:
            log_console(f"Processing {len(transcripts_to_process)} transcripts...")

        # Process each transcript with crash-resilient save
        successful_count = 0

        for i, (transcript_key, transcript_data) in enumerate(transcripts_to_process.items(), 1):
            transcript_start = datetime.now()

            try:
                log_console(f"\n[{i}/{len(transcripts_to_process)}] {transcript_key}")

                # Process transcript - this returns (records, success)
                classified_records, success_flag = process_transcript_v2(transcript_key, transcript_data, enhanced_error_logger)

                if not success_flag or not classified_records:
                    raise ValueError(f"Processing failed for {transcript_key}")

                # Add filename field
                for record in classified_records:
                    if 'filename' not in record:
                        record['filename'] = transcript_key

                # Save with resilient error handling (wrapped)
                nas_conn_save = get_nas_connection()
                if not nas_conn_save:
                    raise RuntimeError("Failed to connect to NAS for saving")

                try:
                    output_path = config["stage_06_llm_classification"]["output_data_path"]
                    individual_path = nas_path_join(output_path, "individual", f"{transcript_key}.json")
                    temp_path = f"{individual_path}.tmp"

                    update_manifest_atomic(transcript_key, {
                        "status": "in_progress",
                        "started_at": transcript_start.isoformat()
                    })

                    json_content = create_json_from_records(classified_records)
                    temp_obj = io.BytesIO(json_content.encode('utf-8'))

                    if not nas_upload_file(nas_conn_save, temp_obj, temp_path):
                        raise IOError(f"Failed to upload temp file for {transcript_key}")

                    if not nas_rename_file(nas_conn_save, temp_path, individual_path):
                        raise IOError(f"Failed to rename temp file for {transcript_key}")

                    metadata = create_metadata_file(nas_conn_save, transcript_key, individual_path, len(classified_records))
                    if not metadata:
                        raise ValueError(f"Failed to create metadata for {transcript_key}")

                    if not validate_json_integrity(nas_conn_save, individual_path, f"{individual_path}.meta"):
                        raise ValueError(f"Validation failed for {transcript_key}")

                    update_manifest_atomic(transcript_key, {
                        "status": "completed",
                        "completed_at": datetime.now().isoformat(),
                        "record_count": len(classified_records),
                        "file_size_bytes": len(json_content),
                        "individual_file": individual_path
                    })

                    log_console(f"✓ Successfully saved {transcript_key} ({len(classified_records)} records)")
                    successful_count += 1

                finally:
                    nas_conn_save.close()

                transcript_time = datetime.now() - transcript_start
                log_console(f"  Time: {transcript_time}")
                time.sleep(1)

            except Exception as e:
                failed_transcripts.append({
                    "transcript": transcript_key,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
                log_console(f"✗ Failed to process {transcript_key}: {e}", "ERROR")

                update_manifest_atomic(transcript_key, {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                })

        # Save failed transcripts
        if failed_transcripts:
            failed_path = nas_path_join(stage_config["output_data_path"], "stage_06_failed_transcripts.json")
            failed_data = {
                "timestamp": datetime.now().isoformat(),
                "total_failed": len(failed_transcripts),
                "failed_transcripts": failed_transcripts
            }
            failed_json = json.dumps(failed_data, indent=2)
            failed_bytes = io.BytesIO(failed_json.encode('utf-8'))
            if nas_upload_file(nas_conn, failed_bytes, failed_path):
                log_console(f"Saved {len(failed_transcripts)} failed transcripts", "WARNING")

        # Consolidate individual files
        log_console("=" * 50)
        log_console("Starting consolidation of individual files...")
        log_console("=" * 50)

        consolidated_path = consolidate_individual_files()

        if consolidated_path:
            log_console(f"✓ Consolidated file: {consolidated_path}")
        else:
            log_console("✗ Consolidation failed - individual files preserved", "WARNING")
        
        # Calculate final summary
        end_time = datetime.now()
        processing_time = end_time - start_time

        stage_summary = {
            "stage": "06_llm_classification",
            "mode": "crash_resilient_individual_files",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_time": str(processing_time),
            "total_transcripts": len(transcripts),
            "transcripts_to_process": len(transcripts_to_process),
            "transcripts_completed": len(completed_ids) + successful_count,
            "transcripts_skipped": len(completed_ids),
            "transcripts_newly_processed": successful_count,
            "transcripts_failed": len(failed_transcripts),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens,
            "classification_errors": len(enhanced_error_logger.classification_errors),
            "processing_errors": len(enhanced_error_logger.processing_errors),
            "output_format": "Individual JSON files with consolidation",
            "consolidated_file": consolidated_path if consolidated_path else "consolidation_failed"
        }

        # Display summary
        log_console("=" * 50)
        log_console("STAGE 6 SUMMARY")
        log_console("=" * 50)
        log_console(f"Total time: {processing_time}")
        log_console(f"Total transcripts: {len(transcripts)}")
        log_console(f"  Already completed (skipped): {len(completed_ids)}")
        log_console(f"  Newly processed: {successful_count}")
        log_console(f"  Failed: {len(failed_transcripts)}")
        log_console(f"Total cost: ${enhanced_error_logger.total_cost:.4f}")
        log_console(f"Total tokens: {enhanced_error_logger.total_tokens:,}")
        if consolidated_path:
            log_console(f"Consolidated file: {consolidated_path}")
        log_console("=" * 50)
        
        # Save logs
        save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)
        
        # Close NAS connection
        nas_conn.close()
        
    except Exception as e:
        error_msg = f"Stage 6 execution failed: {e}"
        log_error(error_msg, "stage_execution", {"error": str(e)})
        log_console(error_msg, "ERROR")
        
        # Save whatever logs we have
        try:
            if 'nas_conn' in locals() and nas_conn:
                stage_summary = {
                    "status": "failed",
                    "version": "1.0",
                    "error": str(e),
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds()
                }
                save_logs_to_nas(nas_conn, stage_summary, enhanced_error_logger)
                nas_conn.close()
        except:
            pass
        
        raise
    
    finally:
        # Cleanup
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            try:
                os.unlink(ssl_cert_path)
                log_console("SSL certificate cleaned up")
            except:
                pass


if __name__ == "__main__":
    main()