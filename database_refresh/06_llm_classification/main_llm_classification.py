"""
Stage 6: LLM-Based Financial Content Classification
ID-based classification with two-pass system and sliding context windows.
Processes entire speaker blocks with historical context for improved accuracy.
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
        self.primary_pass_tokens = 0
        self.secondary_pass_tokens = 0

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

    def accumulate_costs(self, token_usage: dict = None, pass_type: str = "primary"):
        """Accumulate total cost and token usage for final summary."""
        if token_usage:
            if "cost" in token_usage:
                self.total_cost += token_usage["cost"]["total_cost"]
            if "total_tokens" in token_usage:
                self.total_tokens += token_usage["total_tokens"]
                if pass_type == "primary":
                    self.primary_pass_tokens += token_usage["total_tokens"]
                else:
                    self.secondary_pass_tokens += token_usage["total_tokens"]

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
# NEW V2 CLASSIFICATION FUNCTIONS
# ============================================================================

def index_speaker_block_paragraphs(speaker_block_records: List[Dict]) -> List[Dict]:
    """Add paragraph index (starting at 1) within each speaker block."""
    indexed_records = []
    for idx, record in enumerate(sorted(speaker_block_records, key=lambda x: x["paragraph_id"]), start=1):
        record_copy = record.copy()
        record_copy["paragraph_index"] = idx
        indexed_records.append(record_copy)
    return indexed_records


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
    max_blocks: int = 5
) -> str:
    """Format previous speaker blocks with their classifications for context."""
    
    if not previous_blocks:
        return "No previous speaker blocks available for context."
    
    # Take only the most recent blocks
    recent_blocks = previous_blocks[-max_blocks:] if len(previous_blocks) > max_blocks else previous_blocks
    
    context_parts = []
    for block_idx, block in enumerate(recent_blocks, start=1):
        speaker_name = block["speaker_name"]
        section_name = block["section_name"]
        paragraphs = block["paragraphs"]
        
        context_parts.append(f"\n[Previous Block #{block_idx} - {speaker_name} - {section_name} - {len(paragraphs)} paragraphs]")
        
        for para in paragraphs[:3]:  # Show first 3 paragraphs as preview
            # Fix field name: Stage 3 uses 'paragraph_content', not 'content'
            content = para.get('paragraph_content', para.get('content', ''))
            content_preview = content[:150] + "..." if len(content) > 150 else content
            primary_id = para.get('primary_classification', 0)
            primary_name = CATEGORY_REGISTRY[primary_id]['name']
            
            context_parts.append(f"P{para['paragraph_index']}: \"{content_preview}\"")
            context_parts.append(f"  → Primary: {primary_id} ({primary_name})")
            
            if para.get('secondary_classifications'):
                secondary_names = [f"{sid} ({CATEGORY_REGISTRY[sid]['name']})" for sid in para['secondary_classifications']]
                context_parts.append(f"  → Secondary: {', '.join(secondary_names)}")
        
        if len(paragraphs) > 3:
            context_parts.append(f"  ... and {len(paragraphs) - 3} more paragraphs")
    
    return "\n".join(context_parts)


def format_current_speaker_block(speaker_block_records: List[Dict]) -> str:
    """Format current speaker block for classification."""
    
    context_parts = []
    context_parts.append("\n=== CURRENT SPEAKER BLOCK ===")
    
    for record in speaker_block_records:
        para_idx = record["paragraph_index"]
        content = record["paragraph_content"]
        context_parts.append(f"\nP{para_idx}: {content}")
    
    return "\n".join(context_parts)


def create_primary_classification_prompt(
    bank_name: str,
    fiscal_year: int,
    fiscal_quarter: str,
    section_type: str,
    speaker_name: str,
    speaker_role: str,
    speaker_block_number: int,
    total_speaker_blocks: int,
    previous_blocks: List[Dict],
    current_block_paragraphs: List[Dict]
) -> str:
    """Create comprehensive system prompt for primary classification."""
    
    categories_list = build_numbered_category_list()
    previous_context = format_previous_speaker_blocks_with_classifications(previous_blocks)
    current_block = format_current_speaker_block(current_block_paragraphs)
    
    return f"""You are a financial analyst specializing in earnings call transcript classification for major financial institutions.

<document_context>
You are analyzing an earnings call transcript with the following context:
- Institution: {bank_name}
- Document Type: Quarterly Earnings Call Transcript
- Fiscal Period: {fiscal_quarter} {fiscal_year}
- Section: {section_type}
- Current Speaker: {speaker_name}, {speaker_role if speaker_role else 'Role Unknown'}
- Speaker Block: #{speaker_block_number} of {total_speaker_blocks}
</document_context>

<task_description>
You are performing PRIMARY classification. Assign exactly ONE category ID (0-22) to each paragraph that best captures its main financial topic.
Every paragraph MUST receive a classification. Category 0 (Non-Relevant) should ONLY be used for content with absolutely NO financial or business substance.
</task_description>

<classification_guidelines>
1. Choose the SINGLE MOST relevant category for each paragraph
2. Focus on the financial substance, not speaker pleasantries
3. Consider the context from previous speaker blocks when classifying
4. For {bank_name}, be aware of their specific terminology and reporting structure
5. IMPORTANT: Only use category 0 (Non-Relevant) for pure procedural content like "Thank you", "Next question", "Good morning" with NO financial information
6. If content has ANY financial or business relevance (including ESG, community initiatives, donations, etc.), assign an appropriate non-zero category
7. When in doubt between category 0 and another category, choose the other category
8. Assign based on the primary topic, even if other topics are mentioned
</classification_guidelines>

<financial_categories>
{categories_list}
</financial_categories>

<previous_context>
{previous_context}
</previous_context>

<current_task>
Classify each paragraph in the current speaker block with a PRIMARY category ID (0-22).
{current_block}
</current_task>

<response_format>
Use the classify_speaker_block_primary function to provide the primary category ID for each paragraph by its index number (P1, P2, P3, etc.).
</response_format>"""


def create_secondary_classification_prompt(
    bank_name: str,
    fiscal_year: int,
    fiscal_quarter: str,
    section_type: str,
    speaker_name: str,
    speaker_role: str,
    speaker_block_number: int,
    total_speaker_blocks: int,
    previous_blocks: List[Dict],
    current_block_paragraphs: List[Dict]
) -> str:
    """Create comprehensive system prompt for secondary classification."""
    
    categories_list = build_numbered_category_list()
    
    # Format current block with primary classifications
    context_parts = []
    context_parts.append("\n=== CURRENT SPEAKER BLOCK WITH PRIMARY CLASSIFICATIONS ===")
    
    for record in current_block_paragraphs:
        para_idx = record["paragraph_index"]
        content = record["paragraph_content"]
        primary_id = record.get("primary_classification", 0)
        primary_name = CATEGORY_REGISTRY[primary_id]["name"]
        
        context_parts.append(f"\nP{para_idx}: {content}")
        context_parts.append(f"  Primary Classification: {primary_id} ({primary_name})")
    
    current_block_with_primary = "\n".join(context_parts)
    previous_context = format_previous_speaker_blocks_with_classifications(previous_blocks)
    
    return f"""You are a financial analyst specializing in earnings call transcript classification for major financial institutions.

<document_context>
You are analyzing an earnings call transcript with the following context:
- Institution: {bank_name}
- Document Type: Quarterly Earnings Call Transcript
- Fiscal Period: {fiscal_quarter} {fiscal_year}
- Section: {section_type}
- Current Speaker: {speaker_name}, {speaker_role if speaker_role else 'Role Unknown'}
- Speaker Block: #{speaker_block_number} of {total_speaker_blocks}
</document_context>

<task_description>
You are performing SECONDARY classification. For each paragraph that has already been assigned a primary category, identify ANY ADDITIONAL relevant category IDs beyond the primary.
Return an empty list if no secondary categories apply. Do not repeat the primary category as a secondary.
</task_description>

<classification_guidelines>
1. Include ALL relevant secondary categories that apply to the content
2. Do NOT include the primary category in secondary classifications
3. Return empty list [] if only the primary category applies
4. Consider multiple aspects of the content that might relate to different categories
5. Look for cross-cutting themes and related topics
6. Secondary categories should add meaningful classification depth
</classification_guidelines>

<financial_categories>
{categories_list}
</financial_categories>

<previous_context>
{previous_context}
</previous_context>

<current_task>
Identify SECONDARY category IDs for each paragraph (if any) beyond their primary classification.
{current_block_with_primary}
</current_task>

<response_format>
Use the classify_speaker_block_secondary function to provide secondary category IDs for each paragraph.
Return empty list [] for paragraphs with no relevant secondary categories.
</response_format>"""


def create_primary_classification_tools() -> List[Dict]:
    """Function calling schema for primary classification."""
    
    return [{
        "type": "function",
        "function": {
            "name": "classify_speaker_block_primary",
            "description": "Assign primary category ID to each paragraph in the speaker block",
            "parameters": {
                "type": "object",
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paragraph_index": {
                                    "type": "integer",
                                    "description": "Paragraph index within speaker block (1, 2, 3, etc.)"
                                },
                                "primary_category_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 22,
                                    "description": "Primary category ID (0-22)"
                                }
                            },
                            "required": ["paragraph_index", "primary_category_id"]
                        },
                        "description": "Classification for each paragraph"
                    }
                },
                "required": ["classifications"]
            }
        }
    }]


def create_secondary_classification_tools() -> List[Dict]:
    """Function calling schema for secondary classification."""
    
    return [{
        "type": "function",
        "function": {
            "name": "classify_speaker_block_secondary",
            "description": "Assign secondary category IDs to each paragraph in the speaker block",
            "parameters": {
                "type": "object",
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paragraph_index": {
                                    "type": "integer",
                                    "description": "Paragraph index within speaker block (1, 2, 3, etc.)"
                                },
                                "secondary_category_ids": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": 22
                                    },
                                    "description": "List of secondary category IDs (can be empty)"
                                }
                            },
                            "required": ["paragraph_index", "secondary_category_ids"]
                        },
                        "description": "Secondary classifications for each paragraph"
                    }
                },
                "required": ["classifications"]
            }
        }
    }]


def process_qa_conversation_two_pass(
    qa_conversation_records: List[Dict],
    transcript_info: Dict[str, Any],
    qa_group_id: int,
    total_qa_groups: int,
    previous_conversations: List[Dict],
    enhanced_error_logger: EnhancedErrorLogger
) -> List[Dict]:
    """Process an entire Q&A conversation (question + answer) with two-pass classification."""
    global llm_client, config
    
    # Index all paragraphs in the conversation
    indexed_records = []
    for idx, record in enumerate(qa_conversation_records, start=1):
        record_copy = record.copy()
        record_copy["paragraph_index"] = idx
        indexed_records.append(record_copy)
    
    log_console(f"Processing Q&A conversation {qa_group_id} ({len(indexed_records)} paragraphs total)")
    
    # Create a single prompt for the entire conversation
    conversation_text = []
    speakers_in_conversation = set()
    
    for record in indexed_records:
        speaker = record.get("speaker", record.get("speaker_name", "Unknown"))
        speakers_in_conversation.add(speaker)
        conversation_text.append(f"P{record['paragraph_index']} [{speaker}]: {record.get('paragraph_content', record.get('content', ''))}")
    
    full_conversation = "\n".join(conversation_text)
    
    # ========== PASS 1: PRIMARY CLASSIFICATION (Same for all paragraphs) ==========
    try:
        system_prompt = f"""You are a financial analyst specializing in earnings call transcript classification.

<document_context>
Institution: {transcript_info.get("company_name", transcript_info.get("ticker", "Unknown"))}
Fiscal Period: {transcript_info.get("fiscal_quarter", "Q1")} {transcript_info.get("fiscal_year", 2024)}
Section: Q&A Conversation
Q&A Group: {qa_group_id} of {total_qa_groups}
Speakers in conversation: {', '.join(speakers_in_conversation)}
</document_context>

<task_description>
This is a complete Q&A conversation including analyst question(s) and management response(s).
Classify the PRIMARY financial topic of this entire conversation.
All paragraphs in this conversation should receive the SAME classification since they are part of the same Q&A exchange.
IMPORTANT: Q&A conversations almost always have financial substance - avoid category 0 (Non-Relevant) unless the entire conversation is purely procedural.
</task_description>

<financial_categories>
{build_numbered_category_list()}
</financial_categories>

<conversation>
{full_conversation}
</conversation>

Use the classify_qa_conversation_primary function to assign ONE primary category ID to ALL paragraphs in this conversation."""

        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please classify this Q&A conversation with its primary category."}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "classify_qa_conversation_primary",
                    "description": "Assign primary category to Q&A conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "primary_category_id": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 22,
                                "description": "Primary category ID for the entire conversation"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Confidence score for classification"
                            }
                        },
                        "required": ["primary_category_id", "confidence"]
                    }
                }
            }],
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse primary classification
        primary_id = 0  # Default
        confidence = 0.0
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            primary_id = result.get("primary_category_id", 0)
            confidence = result.get("confidence", 0.0)
            
            # Validate category ID
            if primary_id not in CATEGORY_REGISTRY:
                log_console(f"Invalid category ID {primary_id}, defaulting to 0", "WARNING")
                primary_id = 0
        
        # Apply same primary classification to all paragraphs in conversation
        for record in indexed_records:
            record["primary_classification"] = primary_id
            record["classification_confidence"] = confidence
            record["classification_method"] = "qa_conversation"
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            }, pass_type="qa_primary")
        
    except Exception as e:
        error_msg = f"Q&A primary classification failed for group {qa_group_id}: {e}"
        log_error(error_msg, "qa_classification", {"qa_group": qa_group_id, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            transcript_info.get("ticker", "unknown"),
            "qa_primary",
            error_msg
        )
        
        # Fallback to category 0
        log_console(f"Applying fallback category 0 to Q&A group {qa_group_id}", "WARNING")
        for record in indexed_records:
            record["primary_classification"] = 0
            record["classification_confidence"] = 0.0
            record["classification_method"] = "qa_fallback"
    
    # ========== PASS 2: SECONDARY CLASSIFICATION (Optional additional categories) ==========
    try:
        # Secondary classification for Q&A conversations
        system_prompt = f"""You are a financial analyst specializing in earnings call transcript classification.

<document_context>
Institution: {transcript_info.get("company_name", transcript_info.get("ticker", "Unknown"))}
Fiscal Period: {transcript_info.get("fiscal_quarter", "Q1")} {transcript_info.get("fiscal_year", 2024)}
Section: Q&A Conversation
Primary Classification: {CATEGORY_REGISTRY[primary_id]['name']}
</document_context>

<task_description>
This Q&A conversation has been classified with primary category: {primary_id} ({CATEGORY_REGISTRY[primary_id]['name']})
Identify ANY ADDITIONAL relevant financial categories that apply to this conversation.
</task_description>

<financial_categories>
{build_numbered_category_list()}
</financial_categories>

<conversation>
{full_conversation}
</conversation>

Use the classify_qa_conversation_secondary function to identify additional relevant categories."""

        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please identify any secondary categories for this Q&A conversation."}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "classify_qa_conversation_secondary",
                    "description": "Assign secondary categories to Q&A conversation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "secondary_category_ids": {
                                "type": "array",
                                "items": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 22
                                },
                                "description": "List of secondary category IDs (empty if none)"
                            }
                        },
                        "required": ["secondary_category_ids"]
                    }
                }
            }],
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse secondary classifications
        secondary_ids = []
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            secondary_ids = result.get("secondary_category_ids", [])
            
            # Validate and filter secondary IDs
            valid_secondary = []
            for sid in secondary_ids:
                if sid in CATEGORY_REGISTRY and sid != primary_id:
                    valid_secondary.append(sid)
                elif sid not in CATEGORY_REGISTRY:
                    log_console(f"Invalid secondary category ID {sid}, skipping", "WARNING")
            
            secondary_ids = valid_secondary
        
        # Apply same secondary classifications to all paragraphs
        for record in indexed_records:
            record["secondary_classifications"] = secondary_ids
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            }, pass_type="qa_secondary")
        
    except Exception as e:
        error_msg = f"Q&A secondary classification failed for group {qa_group_id}: {e}"
        log_error(error_msg, "qa_secondary_classification", {"qa_group": qa_group_id, "error": str(e)})
        # Secondary classification failure is not critical - keep primary
        for record in indexed_records:
            record["secondary_classifications"] = []
    
    # ========== VALIDATION: Fix Non-Relevant primary with secondary categories ==========
    # Check if all records have same primary (they should for Q&A conversations)
    primary_id = indexed_records[0].get("primary_classification", 0) if indexed_records else 0
    secondary_ids = indexed_records[0].get("secondary_classifications", []) if indexed_records else []
    
    if primary_id == 0 and secondary_ids:
        # Log the issue
        log_console(f"WARNING: Q&A group {qa_group_id} has Non-Relevant primary but {len(secondary_ids)} secondary categories", "WARNING")
        log_console(f"  Promoting first secondary category to primary: {CATEGORY_REGISTRY[secondary_ids[0]]['name']}", "WARNING")
        
        # Promote first secondary to primary for ALL records in conversation
        for record in indexed_records:
            record["primary_classification"] = secondary_ids[0]
            record["secondary_classifications"] = secondary_ids[1:]
        
        # Update primary_id for the conversion below
        primary_id = secondary_ids[0]
    
    # Map IDs to names for output - SAME FORMAT AS MD SPEAKER BLOCKS
    for record in indexed_records:
        primary_id = record.get("primary_classification", 0)
        record["primary_category_name"] = CATEGORY_REGISTRY[primary_id]["name"]
        
        secondary_ids = record.get("secondary_classifications", [])
        record["secondary_category_names"] = [CATEGORY_REGISTRY[sid]["name"] for sid in secondary_ids]
        
        # Add classification method (different from MD to indicate conversation-level classification)
        record["classification_method"] = "qa_conversation_two_pass"
        
        # Clean up temporary fields to match MD output
        fields_to_remove = ["paragraph_index", "primary_classification", "secondary_classifications", 
                          "classification_confidence", "classification_method_temp"]
        for field in fields_to_remove:
            record.pop(field, None)
    
    return indexed_records


def process_speaker_block_two_pass(
    speaker_block_records: List[Dict],
    transcript_info: Dict[str, Any],
    speaker_block_number: int,
    total_speaker_blocks: int,
    previous_blocks: List[Dict],
    enhanced_error_logger: EnhancedErrorLogger
) -> List[Dict]:
    """Process a speaker block with two-pass classification."""
    global llm_client, config
    
    # Index paragraphs
    indexed_records = index_speaker_block_paragraphs(speaker_block_records)
    
    # Extract speaker and section info
    first_record = indexed_records[0]
    # Fix field name mismatch: Stage 3 uses 'speaker', not 'speaker_name'
    speaker_name = first_record.get("speaker", first_record.get("speaker_name", "Unknown"))
    speaker_role = ""  # Could be extracted from speaker name if formatted
    section_name = first_record.get("section_name", "Unknown")
    
    log_console(f"Processing speaker block #{speaker_block_number} - {speaker_name} ({len(indexed_records)} paragraphs)")
    
    # ========== PASS 1: PRIMARY CLASSIFICATION ==========
    try:
        system_prompt = create_primary_classification_prompt(
            # Use ticker as fallback for company_name since Stage 3 doesn't provide it
            bank_name=transcript_info.get("company_name", transcript_info.get("ticker", "Unknown")),
            fiscal_year=transcript_info.get("fiscal_year", 2024),
            fiscal_quarter=transcript_info.get("fiscal_quarter", "Q1"),
            section_type=section_name,
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            speaker_block_number=speaker_block_number,
            total_speaker_blocks=total_speaker_blocks,
            previous_blocks=previous_blocks,
            current_block_paragraphs=indexed_records
        )
        
        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please classify each paragraph with its primary category."}
            ],
            tools=create_primary_classification_tools(),
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse primary classifications
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Apply primary classifications
            for classification in result["classifications"]:
                para_idx = classification["paragraph_index"]
                primary_id = classification["primary_category_id"]
                
                # Validate category ID
                if primary_id not in CATEGORY_REGISTRY:
                    log_console(f"Invalid category ID {primary_id}, defaulting to 0", "WARNING")
                    primary_id = 0
                
                # Find and update the record
                for record in indexed_records:
                    if record["paragraph_index"] == para_idx:
                        record["primary_classification"] = primary_id
                        break
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            }, pass_type="primary")
        
    except Exception as e:
        error_msg = f"Primary classification failed for speaker block: {e}"
        log_error(error_msg, "primary_classification", {"speaker": speaker_name, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            transcript_info.get("ticker", "unknown"),
            "primary",
            error_msg
        )
        
        # Fallback to category 0 for all
        log_console(f"Applying fallback category 0 to {len(indexed_records)} records due to classification error", "WARNING")
        for record in indexed_records:
            record["primary_classification"] = 0
    
    # ========== PASS 2: SECONDARY CLASSIFICATION ==========
    try:
        system_prompt = create_secondary_classification_prompt(
            # Use ticker as fallback for company_name since Stage 3 doesn't provide it
            bank_name=transcript_info.get("company_name", transcript_info.get("ticker", "Unknown")),
            fiscal_year=transcript_info.get("fiscal_year", 2024),
            fiscal_quarter=transcript_info.get("fiscal_quarter", "Q1"),
            section_type=section_name,
            speaker_name=speaker_name,
            speaker_role=speaker_role,
            speaker_block_number=speaker_block_number,
            total_speaker_blocks=total_speaker_blocks,
            previous_blocks=previous_blocks,
            current_block_paragraphs=indexed_records
        )
        
        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please identify any secondary categories for each paragraph."}
            ],
            tools=create_secondary_classification_tools(),
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse secondary classifications
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Apply secondary classifications
            for classification in result["classifications"]:
                para_idx = classification["paragraph_index"]
                secondary_ids = classification["secondary_category_ids"]
                
                # Validate and filter secondary IDs
                valid_secondary_ids = []
                for sec_id in secondary_ids:
                    if sec_id in CATEGORY_REGISTRY:
                        valid_secondary_ids.append(sec_id)
                    else:
                        log_console(f"Invalid secondary category ID {sec_id}, skipping", "WARNING")
                
                # Find and update the record
                for record in indexed_records:
                    if record["paragraph_index"] == para_idx:
                        # Don't include primary in secondary
                        primary = record.get("primary_classification", 0)
                        filtered_secondary = [sid for sid in valid_secondary_ids if sid != primary]
                        record["secondary_classifications"] = filtered_secondary
                        break
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            }, pass_type="secondary")
        
    except Exception as e:
        error_msg = f"Secondary classification failed for speaker block: {e}"
        log_error(error_msg, "secondary_classification", {"speaker": speaker_name, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            transcript_info.get("ticker", "unknown"),
            "secondary",
            error_msg
        )
        
        # No secondary classifications on error
        for record in indexed_records:
            record["secondary_classifications"] = []
    
    # ========== VALIDATION: Fix Non-Relevant primary with secondary categories ==========
    # If primary is 0 (Non-Relevant) but secondary categories exist, promote first secondary to primary
    for record in indexed_records:
        primary_id = record.get("primary_classification", 0)
        secondary_ids = record.get("secondary_classifications", [])
        
        if primary_id == 0 and secondary_ids:
            # Log the issue
            log_console(f"WARNING: P{record['paragraph_index']} has Non-Relevant primary but {len(secondary_ids)} secondary categories", "WARNING")
            log_console(f"  Promoting first secondary category to primary: {CATEGORY_REGISTRY[secondary_ids[0]]['name']}", "WARNING")
            
            # Promote first secondary to primary
            record["primary_classification"] = secondary_ids[0]
            # Remove the promoted category from secondary list
            record["secondary_classifications"] = secondary_ids[1:]
    
    # Map IDs to names for output
    for record in indexed_records:
        primary_id = record.get("primary_classification", 0)
        record["primary_category_name"] = CATEGORY_REGISTRY[primary_id]["name"]
        
        secondary_ids = record.get("secondary_classifications", [])
        record["secondary_category_names"] = [CATEGORY_REGISTRY[sid]["name"] for sid in secondary_ids]
        
        # Add classification method
        record["classification_method"] = "two_pass_with_context"
        
        # Clean up temporary fields
        fields_to_remove = ["paragraph_index", "primary_classification", "secondary_classifications"]
        for field in fields_to_remove:
            record.pop(field, None)
    
    return indexed_records


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
                # Process with two-pass classification
                classified_block = process_speaker_block_two_pass(
                    block_records,
                    transcript_info,
                    block_num,
                    total_md_blocks,
                    previous_md_blocks[-5:],  # Last 5 blocks for context
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
            classified_conversation = process_qa_conversation_two_pass(
                sorted_qa_records,
                transcript_info,
                qa_group_id,
                len(transcript_data["qa_groups"]),
                previous_qa_blocks[-1:] if previous_qa_blocks else [],
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
                classified_block = process_speaker_block_two_pass(
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
    log_console("Two-pass classification with sliding context windows")
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
        
        log_console(f"Processing {len(transcripts)} transcripts with two-pass classification")
        log_console(f"  - MD records: {total_md_records}")
        log_console(f"  - Q&A records with group ID: {total_qa_with_group}")
        if total_qa_without_group > 0:
            log_console(f"  - Q&A records WITHOUT group ID: {total_qa_without_group} (will process as speaker blocks)", "WARNING")
        
        # Process transcripts with incremental saving
        output_path = nas_path_join(stage_config["output_data_path"], "stage_06_classified_content.json")
        is_first_batch = True
        
        for i, (transcript_key, transcript_data) in enumerate(transcripts.items(), 1):
            log_console(f"\nProcessing transcript {i}/{len(transcripts)}: {transcript_key}")
            
            # Process transcript
            classified_records, success = process_transcript_v2(transcript_key, transcript_data, enhanced_error_logger)
            
            if success and classified_records:
                # Save incrementally
                save_results_incrementally(classified_records, output_path, is_first_batch)
                is_first_batch = False
                log_console(f"✓ Saved {len(classified_records)} classified records")
            else:
                failed_transcripts.append({
                    "transcript": transcript_key,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "Processing failed - see error logs"
                })
                log_console(f"✗ Failed to process transcript", "ERROR")
            
            # Brief pause between transcripts
            if i < len(transcripts):
                time.sleep(1)
        
        # Close the JSON array
        successful_transcripts = len(transcripts) - len(failed_transcripts)
        if successful_transcripts > 0:
            if not close_json_array(nas_conn, output_path):
                log_console("Warning: Failed to properly close main output file", "WARNING")
        
        # Save failed transcripts if any
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
                log_console(f"Saved {len(failed_transcripts)} failed transcripts to separate file", "WARNING")
        
        # Calculate final summary
        end_time = datetime.now()
        processing_time = end_time - start_time
        
        stage_summary = {
            "status": "completed" if not failed_transcripts else "completed_with_errors",
            "version": "2.0",
            "classification_method": "two_pass_with_context",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_time_seconds": processing_time.total_seconds(),
            "transcripts_processed": len(transcripts) - len(failed_transcripts),
            "transcripts_failed": len(failed_transcripts),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens,
            "primary_pass_tokens": enhanced_error_logger.primary_pass_tokens,
            "secondary_pass_tokens": enhanced_error_logger.secondary_pass_tokens,
            "errors": {
                "classification_errors": len(enhanced_error_logger.classification_errors),
                "authentication_errors": len(enhanced_error_logger.authentication_errors),
                "validation_errors": len(enhanced_error_logger.validation_errors),
                "processing_errors": len(enhanced_error_logger.processing_errors)
            }
        }
        
        # Display summary
        log_console("\n" + "=" * 60)
        log_console("STAGE 6 EXECUTION SUMMARY")
        log_console("=" * 60)
        log_console(f"Status: {stage_summary['status']}")
        log_console(f"Processing time: {processing_time}")
        log_console(f"Transcripts processed: {stage_summary['transcripts_processed']}")
        log_console(f"Transcripts failed: {stage_summary['transcripts_failed']}")
        log_console(f"Total LLM cost: ${stage_summary['total_cost']:.4f}")
        log_console(f"Total tokens used: {stage_summary['total_tokens']:,}")
        log_console(f"Primary pass tokens: {stage_summary['primary_pass_tokens']:,}")
        log_console(f"Secondary pass tokens: {stage_summary['secondary_pass_tokens']:,}")
        
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