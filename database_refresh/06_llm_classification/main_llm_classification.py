"""
Stage 6: LLM-Based Financial Content Classification
Processes Stage 5 output to add detailed financial category classification.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 5 patterns with Stage 6 classification logic.
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

# Valid financial categories (loaded from config)
VALID_CATEGORIES = set()


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

    def log_classification_error(self, transcript_id: str, section_type: str, error: str):
        """Log classification-specific errors."""
        self.classification_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "section_type": section_type,
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
    """Validate configuration structure and required parameters."""
    global VALID_CATEGORIES

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

    # Validate LLM config structure
    llm_config = stage_06_config["llm_config"]
    required_llm_params = [
        "base_url", "model", "temperature", "max_tokens",
        "timeout", "max_retries", "token_endpoint",
        "cost_per_1k_prompt_tokens", "cost_per_1k_completion_tokens"
    ]

    for param in required_llm_params:
        if param not in llm_config:
            error_msg = f"Missing required LLM config parameter: {param}"
            log_error(error_msg, "config_validation", {"missing_parameter": f"llm_config.{param}"})
            raise ValueError(error_msg)

    # Load valid categories from config
    if "financial_categories" in stage_06_config:
        VALID_CATEGORIES = {cat["name"] for cat in stage_06_config["financial_categories"]}
        log_execution(f"Loaded {len(VALID_CATEGORIES)} financial categories from config")
    else:
        raise ValueError("No financial categories found in configuration")

    log_execution("Configuration validation successful", {
        "total_institutions": len(config["monitored_institutions"]),
        "llm_model": llm_config["model"],
        "financial_categories": len(VALID_CATEGORIES)
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
    """Save results incrementally after each transcript (following Stage 5 pattern)."""
    global logger
    
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to connect to NAS for incremental save")
        
        # For first batch, create new file with proper structure
        if is_first_batch:
            output_data = {
                "schema_version": "1.0",
                "processing_timestamp": datetime.now().isoformat(),
                "stage": "stage_06_llm_classification",
                "description": "Financial content classification using LLM",
                "records": results
            }
            mode = "w"
            log_console(f"Creating new output file with {len(results)} records")
        else:
            # For subsequent batches, load existing and append
            existing_data = nas_download_file(nas_conn, output_path)
            if existing_data:
                output_data = json.loads(existing_data.decode('utf-8'))
                output_data["records"].extend(results)
                output_data["processing_timestamp"] = datetime.now().isoformat()
                mode = "a"
                log_console(f"Appending {len(results)} records (total: {len(output_data['records'])})")
            else:
                # Fallback if file doesn't exist
                output_data = {
                    "schema_version": "1.0",
                    "processing_timestamp": datetime.now().isoformat(),
                    "stage": "stage_06_llm_classification",
                    "description": "Financial content classification using LLM",
                    "records": results
                }
                mode = "w"
        
        # Upload to NAS
        output_json = json.dumps(output_data, indent=2)
        output_bytes = io.BytesIO(output_json.encode('utf-8'))
        
        if nas_upload_file(nas_conn, output_bytes, output_path):
            log_execution(f"Incrementally saved {len(results)} records", {
                "output_path": output_path,
                "total_records": len(output_data["records"]),
                "mode": mode
            })
        else:
            raise RuntimeError("Failed to upload incremental results to NAS")
            
        nas_conn.close()
        
    except Exception as e:
        error_msg = f"Failed to save results incrementally: {e}"
        log_error(error_msg, "incremental_save", {"error": str(e)})
        raise


# Security validation functions (from Stage 5)
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


# NAS utility functions (from Stage 5)
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


# Stage 6 specific functions
def validate_categories(categories: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that all categories are in the expected set.
    
    Returns:
        (is_valid, valid_categories, invalid_categories)
    """
    if not categories:
        return True, [], []
    
    valid_cats = []
    invalid_cats = []
    
    for category in categories:
        if category in VALID_CATEGORIES:
            valid_cats.append(category)
        else:
            invalid_cats.append(category)
    
    is_valid = len(invalid_cats) == 0
    return is_valid, valid_cats, invalid_cats


def build_categories_description() -> str:
    """Build category descriptions from config for prompts."""
    global config
    
    if "financial_categories" not in config["stage_06_llm_classification"]:
        return "Categories not available in configuration"
    
    categories = config["stage_06_llm_classification"]["financial_categories"]
    category_lines = []
    
    for i, cat in enumerate(categories, 1):
        name = cat["name"]
        description = cat["description"]
        
        # Start with basic info
        line = f"{i}. **{name}**: {description}"
        
        # Add additional details if available
        if "key_indicators" in cat:
            line += f"\n   - **Key Indicators**: {cat['key_indicators']}"
        
        if "use_when" in cat:
            line += f"\n   - **Use When**: {cat['use_when']}"
        
        if "do_not_use_when" in cat:
            line += f"\n   - **Do NOT Use When**: {cat['do_not_use_when']}"
        
        if "example_phrases" in cat:
            line += f"\n   - **Example Phrases**: {cat['example_phrases']}"
        
        category_lines.append(line)
    
    return "\n\n".join(category_lines)


def create_management_discussion_tools() -> List[Dict]:
    """Function calling schema for Management Discussion classification."""
    global VALID_CATEGORIES
    
    return [{
        "type": "function",
        "function": {
            "name": "classify_management_discussion_paragraphs",
            "description": "Classify Management Discussion paragraphs with applicable financial categories",
            "parameters": {
                "type": "object",
                "properties": {
                    "paragraph_classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paragraph_number": {"type": "integer", "description": "Paragraph number in window"},
                                "categories": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": sorted(list(VALID_CATEGORIES))},
                                    "description": "All applicable categories for this paragraph. Use 'Other' for non-contributory content."
                                },
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["paragraph_number", "categories", "confidence"]
                        }
                    }
                },
                "required": ["paragraph_classifications"]
            }
        }
    }]


def create_qa_conversation_tools() -> List[Dict]:
    """Function calling schema for Q&A conversation classification."""
    global VALID_CATEGORIES
    
    return [{
        "type": "function",
        "function": {
            "name": "classify_qa_conversation",
            "description": "Classify complete Q&A conversation with applicable financial categories",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_classification": {
                        "type": "object",
                        "properties": {
                            "categories": {
                                "type": "array",
                                "items": {"type": "string", "enum": sorted(list(VALID_CATEGORIES))},
                                "description": "All applicable categories for this conversation. Use 'Other' for non-contributory content."
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["categories", "confidence"]
                    }
                },
                "required": ["conversation_classification"]
            }
        }
    }]


def create_management_discussion_costar_prompt(company_name: str, fiscal_info: str,
                                             speaker: str, window_position: str,
                                             total_paragraphs: int) -> str:
    """CO-STAR prompt with full category descriptions for Management Discussion."""
    categories_description = build_categories_description()
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <speaker>{speaker}</speaker>
  <classification_window>{window_position} of {total_paragraphs} total</classification_window>
  <section_type>Management Discussion</section_type>
</context>

<objective>
Classify each paragraph in the current window with ALL applicable financial content categories. 
Each paragraph should receive whatever categories apply based on its content and context.

IMPORTANT GUIDELINES:
- Apply ALL relevant categories - there is no minimum or maximum number required
- Use "Other" for non-contributory content like introductions, pleasantries, or transitions
- Never leave categories empty - if no financial categories apply, use ["Other"]
- Focus on actual financial substance, not speaker pleasantries
</objective>

<style>
Analyze content against category descriptions. Apply all relevant categories - there is no 
minimum or maximum number required. Base decisions on actual content themes and business context.
</style>

<tone>
Professional financial analysis focused on content categorization.
</tone>

<audience>
Financial analysts requiring detailed content categorization for earnings analysis.
</audience>

<categories>
{categories_description}
</categories>

<response_format>
Use the classify_management_discussion_paragraphs function to classify each paragraph 
with applicable categories and confidence scores.
</response_format>
"""


def create_qa_conversation_costar_prompt(company_name: str, fiscal_info: str,
                                       conversation_length: int) -> str:
    """CO-STAR prompt with category descriptions for Q&A conversations."""
    categories_description = build_categories_description()
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <conversation_paragraphs>{conversation_length}</conversation_paragraphs>
  <section_type>Q&A Conversation</section_type>
</context>

<objective>
Analyze this complete Q&A conversation to identify all applicable financial content categories
based on the topics discussed between analysts and management.

IMPORTANT GUIDELINES:
- Apply ALL relevant categories based on conversation content
- Use "Other" for conversations that are purely procedural or non-financial
- Never leave categories empty - if no financial categories apply, use ["Other"]
- Focus on the financial topics discussed, not pleasantries
</objective>

<style>
Focus on financial topics and business themes discussed in both analyst questions and
management responses. Apply all relevant categories based on conversation content.
</style>

<tone>
Analytical and comprehensive. Consider both question topics and response content.
</tone>

<audience>
Financial analysts studying earnings call topic coverage and conversation themes.
</audience>

<categories>
{categories_description}
</categories>

<response_format>
Use the classify_qa_conversation function to provide comprehensive conversation analysis
with all applicable categories and confidence score.
</response_format>
"""


def format_management_discussion_context(speaker_block_records: List[Dict], 
                                       paragraph_window: List[Dict]) -> str:
    """Format MD speaker block context for classification."""
    context_parts = []
    
    context_parts.append(f"=== SPEAKER BLOCK ===")
    context_parts.append(f"Speaker: {speaker_block_records[0]['speaker']}")
    context_parts.append(f"Total Paragraphs: {len(speaker_block_records)}")
    
    # Show all paragraphs in speaker block
    for i, record in enumerate(speaker_block_records):
        para_id = record["paragraph_id"]
        content = record["paragraph_content"]
        
        if record in paragraph_window:
            # Current window - to be classified
            context_parts.append(f"\nP{i+1} (ID:{para_id}) [TO_CLASSIFY]:")
            context_parts.append(content)
        else:
            # Context paragraphs
            context_parts.append(f"\nP{i+1} (ID:{para_id}) [CONTEXT]:")
            context_parts.append(content)
    
    # Highlight current classification window
    window_start = speaker_block_records.index(paragraph_window[0]) + 1
    window_end = speaker_block_records.index(paragraph_window[-1]) + 1
    context_parts.append(f"\n=== CLASSIFY PARAGRAPHS P{window_start}-P{window_end} ===")
    
    return "\n".join(context_parts)


def format_qa_group_context(qa_group_records: List[Dict]) -> str:
    """Format complete Q&A conversation for single classification call."""
    context_parts = []
    context_parts.append("=== COMPLETE Q&A CONVERSATION ===")
    context_parts.append(f"Q&A Group ID: {qa_group_records[0]['qa_group_id']}")
    context_parts.append(f"Total Paragraphs: {len(qa_group_records)}")
    
    # Show all paragraphs in conversation order
    for i, record in enumerate(sorted(qa_group_records, key=lambda x: x["paragraph_id"])):
        speaker = record["speaker"]
        content = record["paragraph_content"]
        
        # Add speaker role context
        if "analyst" in speaker.lower():
            role_indicator = "[ANALYST QUESTION]"
        elif any(title in speaker.lower() for title in ["ceo", "cfo", "president", "chief"]):
            role_indicator = "[MANAGEMENT RESPONSE]"
        else:
            role_indicator = "[OTHER]"
        
        context_parts.append(f"\nP{i+1}: {speaker} {role_indicator}")
        context_parts.append(content)
    
    context_parts.append("\n=== CLASSIFY THIS COMPLETE CONVERSATION ===")
    
    return "\n".join(context_parts)


def process_management_discussion_section(md_records: List[Dict], enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process Management Discussion using speaker block windowing approach."""
    global logger, llm_client, config
    
    # Group by speaker blocks
    speaker_blocks = defaultdict(list)
    for record in sorted(md_records, key=lambda x: x["paragraph_id"]):
        speaker_blocks[record["speaker_block_id"]].append(record)
    
    classified_records = []
    
    for block_id, block_records in speaker_blocks.items():
        log_console(f"Processing MD speaker block {block_id} ({len(block_records)} paragraphs)")
        
        # Process in configurable windows
        window_size = config["stage_06_llm_classification"]["processing_config"]["md_paragraph_window_size"]
        
        for window_start in range(0, len(block_records), window_size):
            window_end = min(window_start + window_size, len(block_records))
            paragraph_window = block_records[window_start:window_end]
            
            # Format context
            context = format_management_discussion_context(block_records, paragraph_window)
            
            # Create CO-STAR prompt
            system_prompt = create_management_discussion_costar_prompt(
                company_name=block_records[0].get("company_name", "Unknown"),
                fiscal_info=f"{block_records[0].get('fiscal_year')} {block_records[0].get('fiscal_quarter')}",
                speaker=block_records[0]["speaker"],
                window_position=f"paragraphs {window_start+1}-{window_end}",
                total_paragraphs=len(block_records)
            )
            
            # Call LLM
            try:
                response = llm_client.chat.completions.create(
                    model=config["stage_06_llm_classification"]["llm_config"]["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    tools=create_management_discussion_tools(),
                    tool_choice="required",
                    temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
                    max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
                )
                
                # Parse and apply classifications
                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = json.loads(tool_call.function.arguments)
                    
                    # Apply classifications to each paragraph
                    for i, para_class in enumerate(result["paragraph_classifications"]):
                        if i < len(paragraph_window):
                            record = paragraph_window[i]
                            categories = para_class["categories"]
                            
                            # Validate and fix categories
                            is_valid, valid_cats, invalid_cats = validate_categories(categories)
                            if not is_valid:
                                log_console(f"Fixed invalid categories: {invalid_cats} → {valid_cats or ['Other']}", "WARNING")
                                categories = valid_cats or ["Other"]
                            
                            # Handle empty categories
                            if not categories:
                                categories = ["Other"]
                            
                            record["category_type"] = categories
                            record["category_type_confidence"] = para_class["confidence"]
                            record["category_type_method"] = "speaker_block_windowing"
                            
                            classified_records.append(record)
                
                # Track costs
                if hasattr(response, 'usage') and response.usage:
                    cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    enhanced_error_logger.accumulate_costs({
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_info
                    })
                
            except Exception as e:
                error_msg = f"MD classification failed for block {block_id}: {e}"
                log_error(error_msg, "md_classification", {"block_id": block_id, "error": str(e)})
                enhanced_error_logger.log_classification_error(
                    block_records[0].get("ticker", "unknown"), 
                    "Management Discussion", 
                    error_msg
                )
                
                # Fallback to "Other" for failed paragraphs
                for record in paragraph_window:
                    record["category_type"] = ["Other"]
                    record["category_type_confidence"] = 0.0
                    record["category_type_method"] = "error_fallback"
                    classified_records.append(record)
    
    return classified_records


def process_qa_group(qa_group_records: List[Dict], enhanced_error_logger: EnhancedErrorLogger) -> List[Dict]:
    """Process complete Q&A group as single conversation."""
    global logger, llm_client, config
    
    qa_group_id = qa_group_records[0]["qa_group_id"]
    log_console(f"Processing Q&A group {qa_group_id} ({len(qa_group_records)} paragraphs)")
    
    # Format complete conversation context
    conversation_context = format_qa_group_context(qa_group_records)
    
    # Create CO-STAR prompt
    system_prompt = create_qa_conversation_costar_prompt(
        company_name=qa_group_records[0].get("company_name", "Unknown"),
        fiscal_info=f"{qa_group_records[0].get('fiscal_year')} {qa_group_records[0].get('fiscal_quarter')}",
        conversation_length=len(qa_group_records)
    )
    
    try:
        # Single LLM call for entire conversation
        response = llm_client.chat.completions.create(
            model=config["stage_06_llm_classification"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_context}
            ],
            tools=create_qa_conversation_tools(),
            tool_choice="required",
            temperature=config["stage_06_llm_classification"]["llm_config"]["temperature"],
            max_tokens=config["stage_06_llm_classification"]["llm_config"]["max_tokens"]
        )
        
        # Parse and apply to ALL records in group
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Apply same classification to all paragraphs in conversation
            categories = result["conversation_classification"]["categories"]
            confidence = result["conversation_classification"]["confidence"]
            
            # Validate and fix categories
            is_valid, valid_cats, invalid_cats = validate_categories(categories)
            if not is_valid:
                log_console(f"Fixed invalid categories: {invalid_cats} → {valid_cats or ['Other']}", "WARNING")
                categories = valid_cats or ["Other"]
            
            # Handle empty categories
            if not categories:
                categories = ["Other"]
            
            for record in qa_group_records:
                record["category_type"] = categories
                record["category_type_confidence"] = confidence
                record["category_type_method"] = "complete_conversation"
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            enhanced_error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            })
        
        return qa_group_records
        
    except Exception as e:
        error_msg = f"Q&A classification failed for group {qa_group_id}: {e}"
        log_error(error_msg, "qa_classification", {"qa_group_id": qa_group_id, "error": str(e)})
        enhanced_error_logger.log_classification_error(
            qa_group_records[0].get("ticker", "unknown"),
            "Investor Q&A",
            error_msg
        )
        
        # Fallback to "Other" for failed Q&A group
        for record in qa_group_records:
            record["category_type"] = ["Other"]
            record["category_type_confidence"] = 0.0
            record["category_type_method"] = "error_fallback"
        
        return qa_group_records


def process_transcript(transcript_key: str, transcript_data: Dict, enhanced_error_logger: EnhancedErrorLogger) -> Tuple[List[Dict], bool]:
    """Process a single transcript and return classified records."""
    global logger
    
    try:
        # Extract transcript info for OAuth refresh
        ticker, fiscal_year, fiscal_quarter = transcript_key.split("_")
        transcript_info = {
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter
        }
        
        # Refresh OAuth token per transcript
        refresh_oauth_token_for_transcript(transcript_info)
        
        classified_records = []
        
        # Process Management Discussion sections
        if transcript_data["management_discussion"]:
            md_classified = process_management_discussion_section(
                transcript_data["management_discussion"],
                enhanced_error_logger
            )
            classified_records.extend(md_classified)
        
        # Process Q&A groups
        for qa_group_id, qa_records in transcript_data["qa_groups"].items():
            qa_classified = process_qa_group(qa_records, enhanced_error_logger)
            classified_records.extend(qa_classified)
        
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
        all_records = stage5_data.get("records", [])
        log_console(f"Loaded {len(all_records)} records from Stage 5")
        
        # Apply development mode limits
        if stage_config.get("dev_mode", False):
            max_transcripts = stage_config.get("dev_max_transcripts", 2)
            log_console(f"Development mode: limiting to {max_transcripts} transcripts")
            
            # Group by transcript and take first N
            transcript_groups = defaultdict(list)
            for record in all_records:
                transcript_key = f"{record.get('ticker')}_{record.get('fiscal_year')}_{record.get('fiscal_quarter')}"
                transcript_groups[transcript_key].append(record)
            
            limited_records = []
            for i, (transcript_key, transcript_records) in enumerate(transcript_groups.items()):
                if i >= max_transcripts:
                    break
                limited_records.extend(transcript_records)
            
            all_records = limited_records
            log_console(f"Limited to {len(all_records)} records from {min(max_transcripts, len(transcript_groups))} transcripts")
        
        # Group records by transcript and section type
        transcripts = defaultdict(lambda: {"management_discussion": [], "qa_groups": defaultdict(list)})
        
        for record in all_records:
            transcript_key = f"{record.get('ticker')}_{record.get('fiscal_year')}_{record.get('fiscal_quarter')}"
            
            if record.get("section_type") == "Management Discussion":
                transcripts[transcript_key]["management_discussion"].append(record)
            elif record.get("section_type") == "Investor Q&A" and record.get("qa_group_id"):
                qa_group_id = record["qa_group_id"]
                transcripts[transcript_key]["qa_groups"][qa_group_id].append(record)
        
        log_console(f"Processing {len(transcripts)} transcripts")
        
        # Process transcripts with incremental saving
        output_path = nas_path_join(stage_config["output_data_path"], "stage_06_classified_content.json")
        is_first_batch = True
        
        for i, (transcript_key, transcript_data) in enumerate(transcripts.items(), 1):
            log_console(f"\nProcessing transcript {i}/{len(transcripts)}: {transcript_key}")
            
            # Process transcript
            classified_records, success = process_transcript(transcript_key, transcript_data, enhanced_error_logger)
            
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
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_time_seconds": processing_time.total_seconds(),
            "transcripts_processed": len(transcripts) - len(failed_transcripts),
            "transcripts_failed": len(failed_transcripts),
            "total_cost": enhanced_error_logger.total_cost,
            "total_tokens": enhanced_error_logger.total_tokens,
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