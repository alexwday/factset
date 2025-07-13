"""
Stage 4: LLM-Based Transcript Classification
Processes Stage 3 output to add section type classification using 3-level progressive analysis.
Self-contained standalone script that loads config from NAS at runtime.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import json
import tempfile
import io
import requests
import time
from smb.SMBConnection import SMBConnection
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from dotenv import load_dotenv
import re
import logging
from openai import OpenAI
from collections import defaultdict

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
PROXY_URL = os.getenv("PROXY_URL")
NAS_USERNAME = os.getenv("NAS_USERNAME")
NAS_PASSWORD = os.getenv("NAS_PASSWORD")
NAS_SERVER_IP = os.getenv("NAS_SERVER_IP")
NAS_SERVER_NAME = os.getenv("NAS_SERVER_NAME")
NAS_SHARE_NAME = os.getenv("NAS_SHARE_NAME")
NAS_BASE_PATH = os.getenv("NAS_BASE_PATH")
NAS_PORT = int(os.getenv("NAS_PORT", 445))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CLIENT_MACHINE_NAME = os.getenv("CLIENT_MACHINE_NAME")
PROXY_DOMAIN = os.getenv("PROXY_DOMAIN", "MAPLE")

# LLM-specific environment variables
LLM_CLIENT_ID = os.getenv("LLM_CLIENT_ID")
LLM_CLIENT_SECRET = os.getenv("LLM_CLIENT_SECRET")

# Validate required environment variables
required_env_vars = [
    "API_USERNAME", "API_PASSWORD", "PROXY_USER", "PROXY_PASSWORD", "PROXY_URL",
    "NAS_USERNAME", "NAS_PASSWORD", "NAS_SERVER_IP", "NAS_SERVER_NAME", 
    "NAS_SHARE_NAME", "NAS_BASE_PATH", "CONFIG_PATH", "CLIENT_MACHINE_NAME",
    "LLM_CLIENT_ID", "LLM_CLIENT_SECRET"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Global variables for configuration
config = {}
logger = None
error_logger = None
llm_client = None
oauth_token = None
ssl_cert_path = None


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".log",
        prefix=f'stage_4_llm_classification_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
        delete=False,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(temp_log_file.name), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.temp_log_file = temp_log_file.name
    return logger


class EnhancedErrorLogger:
    """Handles separate error logging for different failure types and LLM audit trails."""

    def __init__(self):
        self.llm_errors = []
        self.authentication_errors = []
        self.classification_errors = []
        self.processing_errors = []
        self.llm_audit_trail = []

    def log_llm_error(self, transcript_id: str, section_id: int, error: str):
        """Log LLM API errors."""
        self.llm_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "section_id": section_id,
            "error": error,
            "action_required": "Check LLM API connectivity and authentication"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_classification_error(self, transcript_id: str, error: str):
        """Log classification logic errors."""
        self.classification_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review classification logic and data structure"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review processing logic and input data"
        })

    def log_llm_interaction(self, level: str, section_name: str, classification: str, confidence: float, reasoning: str, token_usage: dict = None):
        """Log LLM classification decisions for audit trail."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "section_name": section_name,
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning
        }
        if token_usage:
            interaction["token_usage"] = token_usage
        self.llm_audit_trail.append(interaction)

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Logs", "Errors")
        nas_create_directory(nas_conn, error_base_path)

        error_types = [
            ("llm_errors", self.llm_errors),
            ("authentication_errors", self.authentication_errors),
            ("classification_errors", self.classification_errors),
            ("processing_errors", self.processing_errors),
        ]
        
        # Save LLM audit trail separately
        if self.llm_audit_trail:
            audit_filename = f"stage_4_llm_audit_trail_{timestamp}.json"
            audit_file_path = nas_path_join(error_base_path, audit_filename)
            audit_content = json.dumps({
                "run_timestamp": timestamp,
                "total_interactions": len(self.llm_audit_trail),
                "interactions": self.llm_audit_trail
            }, indent=2)
            audit_file_obj = io.BytesIO(audit_content.encode("utf-8"))
            nas_upload_file(nas_conn, audit_file_obj, audit_file_path)
            logger.info(f"Saved {len(self.llm_audit_trail)} LLM interactions to {audit_filename}")

        summary = {
            "run_timestamp": timestamp,
            "total_errors": sum(len(errors) for _, errors in error_types),
            "errors_by_type": {error_type: len(errors) for error_type, errors in error_types},
        }

        for error_type, errors in error_types:
            if errors:
                filename = f"stage_4_{error_type}_{timestamp}.json"
                file_path = nas_path_join(error_base_path, filename)
                content = json.dumps({"summary": summary, "errors": errors}, indent=2)
                file_obj = io.BytesIO(content.encode("utf-8"))
                nas_upload_file(nas_conn, file_obj, file_path)
                logger.warning(f"Saved {len(errors)} {error_type} to {filename}")


def get_oauth_token() -> Optional[str]:
    """Get OAuth token using client credentials flow."""
    global logger, error_logger
    
    try:
        token_endpoint = config["stage_4"]["llm_config"]["token_endpoint"]
        
        # Prepare OAuth request
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': LLM_CLIENT_ID,
            'client_secret': LLM_CLIENT_SECRET
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
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
            access_token = token_data.get('access_token')
            if access_token:
                logger.info("Successfully obtained OAuth token")
                return access_token
            else:
                error_msg = "No access token in OAuth response"
                logger.error(error_msg)
                error_logger.log_authentication_error(error_msg)
                return None
        else:
            error_msg = f"OAuth token request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            error_logger.log_authentication_error(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download and setup SSL certificate for LLM API calls."""
    global logger, error_logger
    
    try:
        # Download certificate from NAS using config path
        cert_data = nas_download_file(nas_conn, config['ssl_cert_nas_path'])
        
        if not cert_data:
            error_msg = "Failed to download SSL certificate from NAS"
            logger.error(error_msg)
            error_logger.log_authentication_error(error_msg)
            return None
        
        # Save to temporary file
        cert_temp_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".cer", prefix="llm_cert_", delete=False
        )
        cert_temp_file.write(cert_data)
        cert_temp_file.close()
        
        # Set SSL environment variables
        os.environ["SSL_CERT_FILE"] = cert_temp_file.name
        os.environ["REQUESTS_CA_BUNDLE"] = cert_temp_file.name
        
        logger.info(f"SSL certificate setup complete: {cert_temp_file.name}")
        return cert_temp_file.name
        
    except Exception as e:
        error_msg = f"SSL certificate setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def setup_llm_client() -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    global logger, error_logger, oauth_token
    
    try:
        # Get OAuth token
        oauth_token = get_oauth_token()
        if not oauth_token:
            return None
        
        # Setup OpenAI client with custom configuration
        llm_config = config["stage_4"]["llm_config"]
        
        client = OpenAI(
            api_key=oauth_token,  # Use OAuth token as API key
            base_url=llm_config["base_url"],
            timeout=llm_config.get("timeout", 60),
            max_retries=llm_config.get("max_retries", 3)
        )
        
        logger.info("LLM client setup complete")
        return client
        
    except Exception as e:
        error_msg = f"LLM client setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def refresh_oauth_token_if_needed() -> bool:
    """Refresh OAuth token if needed and update client."""
    global oauth_token, llm_client
    
    # For simplicity, refresh every time (could be optimized with expiry tracking)
    new_token = get_oauth_token()
    if new_token and new_token != oauth_token:
        oauth_token = new_token
        # Update client with new token
        llm_client = setup_llm_client()
        return llm_client is not None
    return oauth_token is not None


# Copy all NAS utility functions from Stage 3 (to maintain standalone nature)
def sanitize_url_for_logging(url: str) -> str:
    """Sanitize URL for logging by removing query parameters and auth tokens."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.warning(f"URL sanitization failed: {e}")
        return "[URL_SANITIZED]"


def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    global logger
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )

        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            logger.info("Connected to NAS successfully")
            return conn
        else:
            logger.error("Failed to connect to NAS")
            return None

    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    global logger
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except Exception as e:
        logger.debug(f"File existence check failed for {file_path}: {e}")
        return False


def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""
    global logger

    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        logger.error("Cannot create directory with empty path")
        return False

    path_parts = [part for part in normalized_path.split("/") if part]
    if not path_parts:
        logger.error("Cannot create directory with invalid path")
        return False

    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part

        if nas_file_exists(conn, current_path):
            continue

        try:
            conn.createDirectory(NAS_SHARE_NAME, current_path)
            logger.debug(f"Created directory: {current_path}")
        except Exception as e:
            if not nas_file_exists(conn, current_path):
                logger.error(f"Failed to create directory {current_path}: {e}")
                return False

    return True


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    global logger

    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return False

    try:
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory(conn, parent_dir)

        conn.storeFile(NAS_SHARE_NAME, nas_file_path, local_file_obj)
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to NAS {nas_file_path}: {e}")
        return False


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    global logger

    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return None

    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(NAS_SHARE_NAME, nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        logger.error(f"Failed to download file from NAS {nas_file_path}: {e}")
        return None


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False

    if ".." in path or path.startswith("/"):
        return False

    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
    if any(char in path for char in invalid_chars):
        return False

    if len(path) > 260:
        return False

    return True


def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure."""
    global logger
    
    if not path or not isinstance(path, str):
        logger.debug(f"NAS path validation failed: empty or not string: '{path}'")
        return False

    normalized = path.strip("/")
    if not normalized:
        logger.debug(f"NAS path validation failed: empty after normalization: '{path}'")
        return False
        
    parts = normalized.split("/")

    for part in parts:
        if not part or part in [".", ".."]:
            logger.debug(f"NAS path validation failed: invalid part '{part}' in path: '{path}'")
            return False
        if not validate_file_path(part):
            logger.debug(f"NAS path validation failed: file path validation failed for part '{part}' in path: '{path}'")
            return False

    return True


def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate configuration schema and parameters."""
    global logger

    required_structure = {
        "ssl_cert_nas_path": str,
        "stage_4": {
            "dev_mode": bool,
            "dev_max_transcripts": int,
            "input_source": str,
            "output_file": str,
            "output_path": str,
            "llm_config": dict,
            "classification_thresholds": dict,
            "content_limits": dict,
        },
        "monitored_institutions": dict,
    }

    for top_key, top_value in required_structure.items():
        if top_key not in config:
            raise ValueError(f"Missing required configuration section: {top_key}")

        if isinstance(top_value, dict):
            for nested_key, expected_type in top_value.items():
                if nested_key not in config[top_key]:
                    raise ValueError(f"Missing required configuration parameter: {top_key}.{nested_key}")

                actual_value = config[top_key][nested_key]
                if not isinstance(actual_value, expected_type):
                    raise ValueError(f"Invalid type for {top_key}.{nested_key}: expected {expected_type}, got {type(actual_value)}")

    if not config["monitored_institutions"]:
        raise ValueError("monitored_institutions cannot be empty")

    logger.info("Configuration validation successful")


def load_stage_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate shared configuration from NAS."""
    global logger
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)

        if config_data:
            stage_config = json.loads(config_data.decode("utf-8"))
            logger.info("Successfully loaded shared configuration from NAS")

            validate_config_schema(stage_config)
            return stage_config
        else:
            logger.error("Config file not found on NAS - script cannot proceed")
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config from NAS: {e}")
        raise


# Classification functions with CO-STAR prompts and function calling
def create_section_classification_tools():
    """Create function calling tools for section classification."""
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_section",
                "description": "Classify entire section as uniform or mixed content type",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["Management Discussion", "Investor Q&A", "Mixed", "Administrative"],
                            "description": "Primary classification of the section content"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the classification"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of classification decision"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key phrases or patterns that supported the classification"
                        }
                    },
                    "required": ["classification", "confidence", "reasoning"]
                }
            }
        }
    ]


def create_breakpoint_detection_tools():
    """Create function calling tools for breakpoint detection."""
    return [
        {
            "type": "function",
            "function": {
                "name": "identify_breakpoint",
                "description": "Identify speaker block where section transitions from management discussion to Q&A",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "breakpoint_speaker_block": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Speaker block number where Q&A begins (1-based indexing)"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the breakpoint identification"
                        },
                        "transition_indicator": {
                            "type": "string",
                            "description": "Key phrase or pattern indicating the transition"
                        }
                    },
                    "required": ["breakpoint_speaker_block", "confidence"]
                }
            }
        }
    ]


def format_paragraph_content(content: str) -> str:
    """Format paragraph content with configurable truncation."""
    max_chars = config["stage_4"]["content_limits"]["max_paragraph_chars"]
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "..."


def create_individual_classification_tools():
    """Create function calling tools for individual speaker block classification."""
    return [
        {
            "type": "function",
            "function": {
                "name": "classify_speaker_block",
                "description": "Classify individual speaker block with contextual information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["Management Discussion", "Investor Q&A"],
                            "description": "Classification of the speaker block"
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Confidence level in the classification"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation considering context and content"
                        }
                    },
                    "required": ["classification", "confidence", "reasoning"]
                }
            }
        }
    ]


def classify_section_level_1(section_records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Level 1: Direct section classification using full section context."""
    global logger, llm_client, error_logger
    
    if not section_records:
        return None
    
    try:
        # Get section metadata
        first_record = section_records[0]
        section_name = first_record["section_name"]
        company_name = first_record.get("company_name", "Unknown Company")
        fiscal_year = first_record.get("fiscal_year", "Unknown")
        fiscal_quarter = first_record.get("fiscal_quarter", "Unknown")
        title = first_record.get("title", "Earnings Call")
        
        # Get unique speakers
        speakers = list(set(record["speaker"] for record in section_records))
        
        # Build content preview
        content_preview = []
        for record in section_records:  # ALL paragraphs
            formatted_content = format_paragraph_content(record['paragraph_content'])
            content_preview.append(f"{record['speaker']}: {formatted_content}")
        
        # CO-STAR prompt
        system_prompt = f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_year} {fiscal_quarter}</fiscal_period>
  <call_title>{title}</call_title>
  <section_name>{section_name}</section_name>
  <paragraph_count>{len(section_records)}</paragraph_count>
  <unique_speakers>{len(speakers)}</unique_speakers>
</context>

<objective>
Classify this earnings call section into one of these categories:
1. "Management Discussion" - Prepared remarks, presentations, financial guidance
2. "Investor Q&A" - Interactive questions and answers between analysts and management
3. "Mixed" - Contains both management discussion and Q&A elements
4. "Administrative" - Operator instructions, disclaimers, introductions
</objective>

<style>
Provide decisive classification based on content structure, speaker patterns, and interaction dynamics.
</style>

<tone>
Professional and analytical. Focus on structural patterns rather than subjective interpretation.
</tone>

<audience>
Financial analysts and automated systems requiring accurate content segmentation.
</audience>

<response_format>
Use the classify_section function to return structured classification.
</response_format>

Section Content Preview:
{chr(10).join(content_preview)}

Unique Speakers in Section:
{chr(10).join(f"- {speaker}" for speaker in speakers[:10])}
"""

        # Call LLM with function calling
        response = llm_client.chat.completions.create(
            model=config["stage_4"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please classify the section '{section_name}' with {len(section_records)} paragraphs."}
            ],
            tools=create_section_classification_tools(),
            tool_choice="required",
            temperature=config["stage_4"]["llm_config"]["temperature"],
            max_tokens=config["stage_4"]["llm_config"]["max_tokens"]
        )
        
        # Parse response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Log token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                logger.info(f"Level 1 tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
            
            logger.info(f"Level 1 classification: {section_name} -> {result['classification']} (confidence: {result['confidence']}) | Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            
            # Log to audit trail
            error_logger.log_llm_interaction(
                level="Level 1",
                section_name=section_name,
                classification=result['classification'],
                confidence=result['confidence'],
                reasoning=result.get('reasoning', 'No reasoning provided'),
                token_usage=token_usage
            )
            
            return {
                "method": "section_uniform",
                "classification": result["classification"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "indicators": result.get("indicators", [])
            }
        else:
            logger.warning(f"No tool call in LLM response for section {section_name}")
            return None
            
    except Exception as e:
        error_msg = f"Level 1 classification failed for section {section_name}: {e}"
        logger.error(error_msg)
        error_logger.log_llm_error(
            first_record.get("filename", "unknown"),
            first_record.get("section_id", 0),
            error_msg
        )
        return None


def classify_section_level_2(section_records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Level 2: Breakpoint detection for mixed sections."""
    global logger, llm_client, error_logger
    
    try:
        # Group by speaker blocks
        speaker_blocks = defaultdict(list)
        for record in sorted(section_records, key=lambda x: x["paragraph_id"]):
            speaker_blocks[record["speaker_block_id"]].append(record)
        
        # Create indexed speaker block list
        block_index = []
        for i, (block_id, block_records) in enumerate(sorted(speaker_blocks.items()), 1):
            speaker = block_records[0]["speaker"]
            # Format ALL paragraphs in this speaker block with truncation
            formatted_paragraphs = []
            for record in block_records:
                formatted_content = format_paragraph_content(record["paragraph_content"])
                formatted_paragraphs.append(formatted_content)
            content_preview = " ".join(formatted_paragraphs)
            block_index.append(f"{i}. [{speaker}] {content_preview}")
        
        # Get section metadata
        first_record = section_records[0]
        section_name = first_record["section_name"]
        company_name = first_record.get("company_name", "Unknown Company")
        fiscal_year = first_record.get("fiscal_year", "Unknown")
        fiscal_quarter = first_record.get("fiscal_quarter", "Unknown")
        
        # CO-STAR prompt for breakpoint detection
        system_prompt = f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_year} {fiscal_quarter}</fiscal_period>
  <section_name>{section_name}</section_name>
  <speaker_block_count>{len(speaker_blocks)}</speaker_block_count>
</context>

<objective>
Identify the exact speaker block where this section transitions from Management Discussion 
(prepared remarks) to Investor Q&A (interactive discussion).
</objective>

<style>
Look for clear transition markers like operator announcements, question-taking phrases, 
or shifts from monologue to dialogue patterns.
</style>

<tone>
Precise and analytical. Focus on identifying the specific transition point.
</tone>

<audience>
Automated classification system requiring exact breakpoint identification.
</audience>

<response_format>
Use the identify_breakpoint function to return the speaker block number where Q&A begins.
</response_format>

Speaker Block Index:
{chr(10).join(block_index)}

Common transition indicators:
- "now open for questions"
- "turn to Q&A"
- "take your questions"
- Operator: "question-and-answer session"
- Switch from single speaker to multiple speakers
"""

        # Call LLM
        response = llm_client.chat.completions.create(
            model=config["stage_4"]["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Find the breakpoint where '{section_name}' transitions from management discussion to Q&A."}
            ],
            tools=create_breakpoint_detection_tools(),
            tool_choice="required",
            temperature=config["stage_4"]["llm_config"]["temperature"],
            max_tokens=config["stage_4"]["llm_config"]["max_tokens"]
        )
        
        # Parse response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            breakpoint_block = result["breakpoint_speaker_block"]
            confidence = result["confidence"]
            
            # Log token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                logger.info(f"Level 2 tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
            
            logger.info(f"Level 2 breakpoint: {section_name} -> block {breakpoint_block} (confidence: {confidence}) | Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            
            # Log to audit trail
            error_logger.log_llm_interaction(
                level="Level 2",
                section_name=section_name,
                classification=f"Breakpoint at block {breakpoint_block}",
                confidence=confidence,
                reasoning=result.get('reasoning', 'No reasoning provided'),
                token_usage=token_usage
            )
            
            return {
                "method": "breakpoint_detection",
                "breakpoint_block": breakpoint_block,
                "confidence": confidence,
                "transition_indicator": result.get("transition_indicator", ""),
                "total_blocks": len(speaker_blocks)
            }
        else:
            logger.warning(f"No tool call in breakpoint detection for section {section_name}")
            return None
            
    except Exception as e:
        error_msg = f"Level 2 breakpoint detection failed for section {section_name}: {e}"
        logger.error(error_msg)
        error_logger.log_llm_error(
            first_record.get("filename", "unknown"),
            first_record.get("section_id", 0),
            error_msg
        )
        return None


def classify_section_level_3(section_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Level 3: Individual speaker block classification with context."""
    global logger, llm_client, error_logger
    
    # Group by speaker blocks
    speaker_blocks = defaultdict(list)
    for record in sorted(section_records, key=lambda x: x["paragraph_id"]):
        speaker_blocks[record["speaker_block_id"]].append(record)
    
    block_classifications = {}
    
    # Sort blocks by paragraph order
    sorted_block_ids = sorted(speaker_blocks.keys(), 
                             key=lambda bid: min(r["paragraph_id"] for r in speaker_blocks[bid]))
    
    for i, block_id in enumerate(sorted_block_ids):
        block_records = speaker_blocks[block_id]
        
        try:
            # Get context - previous blocks (up to 3) and next block (1)
            prev_context = []
            for j in range(max(0, i-3), i):
                prev_block_id = sorted_block_ids[j]
                prev_records = speaker_blocks[prev_block_id]
                prev_speaker = prev_records[0]["speaker"]
                # Format ALL paragraphs in previous block with truncation
                formatted_paragraphs = []
                for record in prev_records:
                    formatted_content = format_paragraph_content(record["paragraph_content"])
                    formatted_paragraphs.append(formatted_content)
                prev_content = " ".join(formatted_paragraphs)
                prev_classification = block_classifications.get(prev_block_id, {}).get("classification", "Unknown")
                prev_context.append(f"Block {j+1} [{prev_speaker}] ({prev_classification}): {prev_content}")
            
            next_context = []
            if i + 1 < len(sorted_block_ids):
                next_block_id = sorted_block_ids[i + 1]
                next_records = speaker_blocks[next_block_id]
                next_speaker = next_records[0]["speaker"]
                # Format ALL paragraphs in next block with truncation
                formatted_paragraphs = []
                for record in next_records:
                    formatted_content = format_paragraph_content(record["paragraph_content"])
                    formatted_paragraphs.append(formatted_content)
                next_content = " ".join(formatted_paragraphs)
                next_context.append(f"Next Block [{next_speaker}]: {next_content}")
            
            # Current block
            current_speaker = block_records[0]["speaker"]
            # Format ALL paragraphs in current block with truncation
            formatted_paragraphs = []
            for record in block_records:
                formatted_content = format_paragraph_content(record["paragraph_content"])
                formatted_paragraphs.append(formatted_content)
            current_content = " ".join(formatted_paragraphs)
            
            # Get section metadata
            first_record = section_records[0]
            company_name = first_record.get("company_name", "Unknown Company")
            fiscal_year = first_record.get("fiscal_year", "Unknown")
            fiscal_quarter = first_record.get("fiscal_quarter", "Unknown")
            section_name = first_record["section_name"]
            
            # CO-STAR prompt for individual classification
            system_prompt = f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_year} {fiscal_quarter}</fiscal_period>
  <section_name>{section_name}</section_name>
  <current_block>{i+1} of {len(sorted_block_ids)}</current_block>
</context>

<objective>
Classify this individual speaker block as either "Management Discussion" or "Investor Q&A" 
considering the surrounding context and maintaining consistency with previous classifications.
</objective>

<style>
Consider speaker roles, content structure, and flow patterns. Maintain logical consistency 
with context while focusing on the current block's primary function.
</style>

<tone>
Contextual and consistent. Balance current content with surrounding classifications.
</tone>

<audience>
Classification system requiring precise paragraph-level accuracy.
</audience>

<response_format>
Use the classify_speaker_block function to return the classification.
</response_format>

Previous Context:
{chr(10).join(prev_context) if prev_context else "No previous context"}

Current Speaker Block:
Speaker: {current_speaker}
Content: {current_content}

Future Context:
{chr(10).join(next_context) if next_context else "No future context"}
"""

            # Call LLM
            response = llm_client.chat.completions.create(
                model=config["stage_4"]["llm_config"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Classify speaker block {i+1} by {current_speaker} considering the context."}
                ],
                tools=create_individual_classification_tools(),
                tool_choice="required",
                temperature=config["stage_4"]["llm_config"]["temperature"],
                max_tokens=config["stage_4"]["llm_config"]["max_tokens"]
            )
            
            # Parse response
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                result = json.loads(tool_call.function.arguments)
                
                block_classifications[block_id] = {
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "reasoning": result["reasoning"]
                }
                
                # Log token usage
                token_usage = None
                if hasattr(response, 'usage') and response.usage:
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                    logger.debug(f"Level 3 block {i+1} tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
                
                logger.debug(f"Level 3 block {i+1}: {result['classification']} (confidence: {result['confidence']}) | Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                
                # Log to audit trail
                error_logger.log_llm_interaction(
                    level="Level 3",
                    section_name=f"{section_name} - Block {i+1}",
                    classification=result['classification'],
                    confidence=result['confidence'],
                    reasoning=result.get('reasoning', 'No reasoning provided'),
                    token_usage=token_usage
                )
            else:
                # Fallback classification
                block_classifications[block_id] = {
                    "classification": "Management Discussion",
                    "confidence": 0.5,
                    "reasoning": "Fallback classification due to LLM response error"
                }
                logger.warning(f"No tool call for block {block_id}, using fallback")
                
        except Exception as e:
            error_msg = f"Level 3 classification failed for block {block_id}: {e}"
            logger.error(error_msg)
            error_logger.log_llm_error(
                first_record.get("filename", "unknown"),
                first_record.get("section_id", 0),
                error_msg
            )
            
            # Fallback classification
            block_classifications[block_id] = {
                "classification": "Management Discussion",
                "confidence": 0.3,
                "reasoning": f"Fallback due to processing error: {str(e)[:100]}"
            }
    
    return {
        "method": "contextual_individual",
        "block_classifications": block_classifications
    }


def classify_transcript_sections(transcript_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Main classification function implementing 3-level progressive analysis."""
    global logger, config
    
    logger.info(f"Starting classification for transcript with {len(transcript_records)} paragraphs")
    
    # Group by sections
    sections = defaultdict(list)
    for record in transcript_records:
        sections[record["section_id"]].append(record)
    
    classified_records = []
    
    for section_id, section_records in sections.items():
        section_name = section_records[0]["section_name"]
        logger.info(f"Processing section {section_id}: {section_name} ({len(section_records)} paragraphs)")
        
        # Level 1: Direct section classification
        level1_result = classify_section_level_1(section_records)
        
        if level1_result and level1_result["classification"] != "Mixed" and \
           level1_result["confidence"] >= config["stage_4"]["classification_thresholds"]["min_confidence"]:
            
            # Apply uniform classification
            classification = level1_result["classification"]
            for record in section_records:
                record["section_type"] = classification
                record["section_type_confidence"] = level1_result["confidence"]
                record["section_type_method"] = level1_result["method"]
                classified_records.append(record)
            
            logger.info(f"Section {section_id} classified uniformly as {classification}")
            continue
        
        # Level 2: Breakpoint detection
        level2_result = classify_section_level_2(section_records)
        
        if level2_result and level2_result["confidence"] >= config["stage_4"]["classification_thresholds"]["min_confidence"]:
            
            # Apply breakpoint-based classification
            breakpoint_block = level2_result["breakpoint_block"]
            
            # Group by speaker blocks for breakpoint application
            speaker_blocks = defaultdict(list)
            for record in section_records:
                speaker_blocks[record["speaker_block_id"]].append(record)
            
            # Sort speaker blocks by min paragraph_id
            sorted_block_ids = sorted(speaker_blocks.keys(), 
                                     key=lambda bid: min(r["paragraph_id"] for r in speaker_blocks[bid]))
            
            for i, block_id in enumerate(sorted_block_ids, 1):
                block_records = speaker_blocks[block_id]
                
                if i < breakpoint_block:
                    classification = "Management Discussion"
                else:
                    classification = "Investor Q&A"
                
                for record in block_records:
                    record["section_type"] = classification
                    record["section_type_confidence"] = level2_result["confidence"]
                    record["section_type_method"] = level2_result["method"]
                    classified_records.append(record)
            
            logger.info(f"Section {section_id} classified with breakpoint at block {breakpoint_block}")
            continue
        
        # Level 3: Individual speaker block classification
        level3_result = classify_section_level_3(section_records)
        
        for record in section_records:
            block_id = record["speaker_block_id"]
            block_classification = level3_result["block_classifications"].get(block_id, {
                "classification": "Management Discussion",
                "confidence": 0.3,
                "reasoning": "Fallback classification"
            })
            
            record["section_type"] = block_classification["classification"]
            record["section_type_confidence"] = block_classification["confidence"]
            record["section_type_method"] = level3_result["method"]
            classified_records.append(record)
        
        logger.info(f"Section {section_id} classified individually by speaker blocks")
    
    return classified_records


def load_stage3_output(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load Stage 3 output for processing."""
    global logger, config
    
    input_path = nas_path_join(NAS_BASE_PATH, config["stage_4"]["input_source"])
    
    if not nas_file_exists(nas_conn, input_path):
        logger.error(f"Stage 3 output not found at {input_path}")
        raise FileNotFoundError(f"Stage 3 output not found at {input_path}")
    
    try:
        stage3_data = nas_download_file(nas_conn, input_path)
        if stage3_data:
            data = json.loads(stage3_data.decode("utf-8"))
            logger.info(f"Loaded Stage 3 output: {len(data.get('records', []))} records")
            return data
        else:
            logger.error(f"Failed to download Stage 3 output from {input_path}")
            raise RuntimeError(f"Failed to download Stage 3 output")
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in Stage 3 output: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading Stage 3 output: {e}")
        raise


def save_classified_output(nas_conn: SMBConnection, classified_records: List[Dict[str, Any]]) -> bool:
    """Save classified records to NAS."""
    global logger, config
    
    output_data = {
        "schema_version": "1.0",
        "processing_timestamp": datetime.now().isoformat(),
        "total_records": len(classified_records),
        "total_transcripts_processed": len(set(record.get('filename', '') for record in classified_records)),
        "classification_summary": {
            "management_discussion": len([r for r in classified_records if r.get("section_type") == "Management Discussion"]),
            "investor_qa": len([r for r in classified_records if r.get("section_type") == "Investor Q&A"]),
            "other": len([r for r in classified_records if r.get("section_type") not in ["Management Discussion", "Investor Q&A"]])
        },
        "records": classified_records
    }
    
    output_path = nas_path_join(
        NAS_BASE_PATH,
        config["stage_4"]["output_path"],
        config["stage_4"]["output_file"]
    )
    
    try:
        output_content = json.dumps(output_data, indent=2)
        output_file_obj = io.BytesIO(output_content.encode("utf-8"))
        
        if nas_upload_file(nas_conn, output_file_obj, output_path):
            logger.info(f"Saved {len(classified_records)} classified records to {output_path}")
            logger.info(f"Classification summary: {output_data['classification_summary']}")
            return True
        else:
            logger.error(f"Failed to upload classified output to {output_path}")
            return False
    
    except Exception as e:
        logger.error(f"Error saving classified output: {e}")
        return False


def upload_logs_to_nas(nas_conn: SMBConnection, logger: logging.Logger, error_logger) -> None:
    """Upload logs to NAS."""
    try:
        error_logger.save_error_logs(nas_conn)
        
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        logging.shutdown()

        log_file_path = nas_path_join(
            NAS_BASE_PATH,
            "Outputs",
            "Logs",
            f"stage_4_llm_classification_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        )

        with open(logger.temp_log_file, "rb") as log_file:
            log_content = log_file.read()

        log_file_obj = io.BytesIO(log_content)
        nas_upload_file(nas_conn, log_file_obj, log_file_path)
        print(f"Log upload completed to: {log_file_path}")
        
    except Exception as e:
        print(f"Log upload failed: {e}")


def main() -> None:
    """Main function to orchestrate Stage 4 LLM classification."""
    global config, logger, error_logger, llm_client, ssl_cert_path

    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    print(f"Local log file: {logger.temp_log_file}")
    logger.info("STAGE 4: LLM-BASED TRANSCRIPT CLASSIFICATION")

    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting classification")
        return

    try:
        # Load configuration
        config = load_stage_config(nas_conn)
        logger.info("Loaded configuration for Stage 4")
        logger.info(f"Development mode: {config['stage_4']['dev_mode']}")
        
        if config['stage_4']['dev_mode']:
            logger.info(f"Max transcripts in dev mode: {config['stage_4']['dev_max_transcripts']}")

        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        if not ssl_cert_path:
            logger.error("Failed to setup SSL certificate - aborting")
            return

        # Setup LLM client
        llm_client = setup_llm_client()
        if not llm_client:
            logger.error("Failed to setup LLM client - aborting")
            return

        start_time = datetime.now()

        # Step 1: Load Stage 3 output
        logger.info("Step 1: Loading Stage 3 output...")
        stage3_data = load_stage3_output(nas_conn)
        
        if not stage3_data.get("records"):
            logger.warning("No records to process - exiting")
            return
        
        records = stage3_data["records"]
        
        # Apply development mode limit
        if config['stage_4']['dev_mode']:
            # Group by transcript first, then limit
            transcripts = defaultdict(list)
            for record in records:
                transcript_key = f"{record.get('ticker', 'unknown')}_{record.get('fiscal_year', 'unknown')}_{record.get('fiscal_quarter', 'unknown')}"
                transcripts[transcript_key].append(record)
            
            # Limit number of transcripts
            limited_transcripts = dict(list(transcripts.items())[:config['stage_4']['dev_max_transcripts']])
            records = []
            for transcript_records in limited_transcripts.values():
                records.extend(transcript_records)
            
            logger.info(f"Development mode: Processing {len(limited_transcripts)} transcripts with {len(records)} total records")

        # Step 2: Group by transcript and classify
        logger.info("Step 2: Classifying transcript sections...")
        
        # Group records by transcript
        transcripts = defaultdict(list)
        for record in records:
            transcript_key = f"{record.get('ticker', 'unknown')}_{record.get('fiscal_year', 'unknown')}_{record.get('fiscal_quarter', 'unknown')}"
            transcripts[transcript_key].append(record)
        
        all_classified_records = []
        
        for i, (transcript_key, transcript_records) in enumerate(transcripts.items(), 1):
            logger.info(f"Processing transcript {i}/{len(transcripts)}: {transcript_key}")
            
            # Refresh OAuth token periodically
            if i % 5 == 0:  # Every 5 transcripts
                if not refresh_oauth_token_if_needed():
                    logger.error("Failed to refresh OAuth token")
                    break
            
            classified_records = classify_transcript_sections(transcript_records)
            all_classified_records.extend(classified_records)
            
            # Rate limiting
            time.sleep(1)  # 1 second between transcripts
        
        # Step 3: Save classified output
        logger.info("Step 3: Saving classified output...")
        if not save_classified_output(nas_conn, all_classified_records):
            logger.error("Failed to save classified output")
            upload_logs_to_nas(nas_conn, logger, error_logger)
            return

        end_time = datetime.now()
        execution_time = end_time - start_time

        # Final summary
        logger.info("STAGE 4 LLM CLASSIFICATION COMPLETE")
        logger.info(f"Transcripts processed: {len(transcripts)}")
        logger.info(f"Total records classified: {len(all_classified_records)}")
        logger.info(f"Average records per transcript: {len(all_classified_records) / len(transcripts) if transcripts else 0:.1f}")
        logger.info(f"Execution time: {execution_time}")

        # Upload logs
        upload_logs_to_nas(nas_conn, logger, error_logger)

    except Exception as e:
        logger.error(f"Stage 4 classification failed: {e}")
        try:
            error_logger.save_error_logs(nas_conn)
        except:
            pass
    finally:
        if nas_conn:
            nas_conn.close()

        # Cleanup
        try:
            if ssl_cert_path and os.path.exists(ssl_cert_path):
                os.unlink(ssl_cert_path)
            os.unlink(logger.temp_log_file)
        except (OSError, FileNotFoundError):
            pass


if __name__ == "__main__":
    main()