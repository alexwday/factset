"""
Stage 7: Content Enhancement System
Processes Stage 6 output to add paragraph-level content enhancements.
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
    try:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    except Exception as e:
        print(f"Error in environment variable check join: missing_vars={missing_vars}, type={type(missing_vars)}, error={e}")
        raise ValueError(f"Missing required environment variables: {str(missing_vars)}")

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
        prefix=f'stage_7_content_enhancement_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
    """Handles separate error logging for different failure types."""

    def __init__(self):
        self.llm_errors = []
        self.authentication_errors = []
        self.enhancement_errors = []
        self.processing_errors = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.api_calls = 0

    def log_llm_error(self, transcript_id: str, error: str):
        """Log LLM-specific errors."""
        self.llm_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "error_type": "llm_processing"
        })

    def log_authentication_error(self, error: str):
        """Log authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "error_type": "authentication"
        })

    def log_enhancement_error(self, transcript_id: str, paragraph_id: int, error: str):
        """Log enhancement-specific errors."""
        self.enhancement_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "paragraph_id": paragraph_id,
            "error": error,
            "error_type": "enhancement"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "error_type": "processing"
        })

    def accumulate_costs(self, cost_data: Dict):
        """Accumulate token usage and costs."""
        self.total_cost += cost_data.get("cost", {}).get("total_cost", 0.0)
        self.total_tokens += cost_data.get("total_tokens", 0)
        self.api_calls += 1

    def get_summary(self) -> Dict:
        """Get comprehensive error and cost summary."""
        total_errors = (len(self.llm_errors) + len(self.authentication_errors) + 
                       len(self.enhancement_errors) + len(self.processing_errors))
        
        return {
            "total_errors": total_errors,
            "llm_errors": len(self.llm_errors),
            "authentication_errors": len(self.authentication_errors),
            "enhancement_errors": len(self.enhancement_errors),
            "processing_errors": len(self.processing_errors),
            "total_cost": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "total_api_calls": self.api_calls,
            "average_cost_per_call": round(self.total_cost / max(self.api_calls, 1), 4),
            "average_tokens_per_call": round(self.total_tokens / max(self.api_calls, 1), 0)
        }

    def save_error_logs(self, nas_conn: SMBConnection) -> bool:
        """Save detailed error logs to NAS only if there are errors."""
        total_errors = (len(self.llm_errors) + len(self.authentication_errors) + 
                       len(self.enhancement_errors) + len(self.processing_errors))
        
        # Only save error logs if there are actual errors
        if total_errors == 0:
            logger.info("No errors to log - skipping error log creation")
            return True
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            error_data = {
                "processing_timestamp": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "detailed_errors": {
                    "llm_errors": self.llm_errors,
                    "authentication_errors": self.authentication_errors,
                    "enhancement_errors": self.enhancement_errors,
                    "processing_errors": self.processing_errors
                }
            }
            
            error_log_content = json.dumps(error_data, indent=2)
            error_log_file = f"stage_7_content_enhancement_{timestamp}_errors.json"
            
            nas_upload_file(
                nas_conn,
                io.BytesIO(error_log_content.encode('utf-8')),
                f"{NAS_BASE_PATH}/Outputs/Logs/Errors/{error_log_file}"
            )
            
            logger.info(f"Error logs saved to NAS: {error_log_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save error logs to NAS: {e}")
            return False


def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    try:
        normalized = os.path.normpath(path)
        return not ('..' in normalized or normalized.startswith('/'))
    except Exception:
        return False


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
    try:
        return "/".join(clean_parts)
    except Exception as e:
        logger.error(f"Error in nas_path_join: parts={parts}, clean_parts={clean_parts}, error={e}")
        return "ERROR_PATH"


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    global logger
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except Exception as e:
        logger.debug(f"File existence check failed for {file_path}: {e}")
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
        
        try:
            conn.createDirectory(NAS_SHARE_NAME, current_path)
        except Exception:
            # Directory might already exist, continue
            pass
    
    return True


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    global logger

    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return False

    try:
        path_parts = nas_file_path.split("/")[:-1]
        parent_dir = "/".join(path_parts)
        if parent_dir:
            nas_create_directory(conn, parent_dir)

        conn.storeFile(NAS_SHARE_NAME, nas_file_path, local_file_obj)
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to NAS {nas_file_path}: {e}")
        return False


def load_stage_config(nas_conn: SMBConnection) -> Dict:
    """Load and validate Stage 7 configuration from NAS."""
    global logger, error_logger
    
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)

        if config_data:
            full_config = json.loads(config_data.decode("utf-8"))
            logger.info("Successfully loaded shared configuration from NAS")

            if "stage_7" not in full_config:
                raise ValueError("Stage 7 configuration not found in config file")
                
            stage_config = full_config["stage_7"]
            
            # Validate required configuration sections
            required_sections = ["llm_config", "processing_config"]
            for section in required_sections:
                if section not in stage_config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return full_config  # Return full config like other stages
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


def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download and setup SSL certificate for LLM API calls."""
    global logger, error_logger, config
    
    try:
        # Download certificate from NAS using config path
        cert_data = nas_download_file(nas_conn, config.get('ssl_cert_nas_path', 'Inputs/certificate/certificate.cer'))
        
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


def get_oauth_token() -> Optional[str]:
    """Obtain OAuth token for LLM API access."""
    global logger, error_logger, config
    
    try:
        token_endpoint = config["stage_7"]["llm_config"]["token_endpoint"]
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': LLM_CLIENT_ID,
            'client_secret': LLM_CLIENT_SECRET
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
            logger.error(error_msg)
            error_logger.log_authentication_error(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"OAuth token acquisition failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def setup_llm_client() -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    global logger, error_logger, config, oauth_token
    
    try:
        if not oauth_token:
            logger.error("No OAuth token available for LLM client")
            return None
            
        client = OpenAI(
            api_key=oauth_token,
            base_url=config["stage_7"]["llm_config"]["base_url"],
            timeout=config["stage_7"]["llm_config"]["timeout"]
        )
        
        logger.info("LLM client setup completed")
        return client
        
    except Exception as e:
        error_msg = f"LLM client setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def refresh_oauth_token_for_transcript() -> bool:
    """Refresh OAuth token per transcript to eliminate expiration issues."""
    global oauth_token, llm_client, logger
    
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client()
        return llm_client is not None
    return False


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> Dict:
    """Calculate token costs based on configuration."""
    global config
    
    prompt_cost_per_1k = config["stage_7"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_7"]["llm_config"]["cost_per_1k_completion_tokens"]
    
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


def load_stage6_output(nas_conn: SMBConnection) -> Dict:
    """Load Stage 6 output data from NAS."""
    global logger, error_logger, config
    
    try:
        input_path = f"{NAS_BASE_PATH}/{config['stage_7']['input_source']}"
        
        if not validate_nas_path(input_path):
            raise ValueError("Invalid input file path")
            
        input_buffer = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, input_path, input_buffer)
        input_buffer.seek(0)
        
        stage6_data = json.loads(input_buffer.read().decode('utf-8'))
        
        logger.info(f"Loaded {len(stage6_data.get('records', []))} records from Stage 6")
        return stage6_data
        
    except Exception as e:
        error_msg = f"Failed to load Stage 6 data: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("stage6_load", error_msg)
        raise


def create_indexed_summary_tools(batch_size: int) -> List[Dict]:
    """Function calling schema for indexed paragraph enhancement."""
    return [{
        "type": "function",
        "function": {
            "name": "summarize_indexed_paragraphs",
            "description": f"Provide summaries and importance scores for exactly {batch_size} indexed paragraphs",
            "parameters": {
                "type": "object",
                "properties": {
                    "paragraph_summaries": {
                        "type": "array",
                        "items": {
                            "type": "object", 
                            "properties": {
                                "index": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": batch_size,
                                    "description": f"Index number from current batch (1-{batch_size})"
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Condensed summary focusing on financial substance, written for retrieval systems"
                                },
                                "importance": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Relative importance within speaker block context"
                                }
                            },
                            "required": ["index", "summary", "importance"]
                        },
                        "minItems": batch_size,
                        "maxItems": batch_size,
                        "description": f"Exactly {batch_size} summaries matching the indexed paragraphs"
                    }
                },
                "required": ["paragraph_summaries"]
            }
        }
    }]


def create_content_enhancement_prompt(company_name: str, fiscal_info: str, 
                                    speaker: str, batch_size: int, 
                                    financial_categories: List[str],
                                    speaker_block_total_paragraphs: int) -> str:
    """CO-STAR prompt for content enhancement with sliding window context."""
    try:
        if financial_categories and isinstance(financial_categories, list):
            categories_str = ', '.join(financial_categories)
        else:
            categories_str = 'General Business'
    except Exception as e:
        logger.error(f"Error in create_content_enhancement_prompt join: financial_categories={financial_categories}, type={type(financial_categories)}, error={e}")
        categories_str = 'General Business'
    
    return f"""
<context>
  <institution>{company_name}</institution>
  <fiscal_period>{fiscal_info}</fiscal_period>
  <speaker_role>{speaker}</speaker_role>
  <financial_focus>{categories_str}</financial_focus>
  <speaker_block_size>{speaker_block_total_paragraphs} total paragraphs</speaker_block_size>
  <current_batch_size>{batch_size} paragraphs to process</current_batch_size>
</context>

<objective>
You are creating summaries for a financial transcript retrieval system. These summaries will be used by a smaller model to judge relevance to user queries about earnings calls, financial performance, guidance, and business developments.

Process ONLY the numbered paragraphs [1] through [{batch_size}] in the current batch.
For each numbered paragraph, provide:

1. SUMMARY: A reranking-optimized summary that scales with content importance
2. IMPORTANCE: A relative importance score within this speaker block (0.0-1.0)

CONTENT-ADAPTIVE SUMMARIZATION RULES:
- NEVER skip content - summarize everything, even single sentences
- Scale detail proportionally: short content = minimal summary, long content = detailed summary
- Preserve key financial terms, numbers, and speaker attributions
- Focus on searchable concepts that users might query

SUMMARY SCALING BY CONTENT LENGTH:
- For content under 50 characters: Create 5-15 word summaries capturing speaker type, topic, and key sentiment
  Example: "Good morning everyone" → "[EXECUTIVE] Call opening greeting"
  
- For content 50-200 characters: Create 15-30 word summaries including who spoke, specific topic, and quantitative information
  Example: "Thank you for joining us today. Let's review our Q3 results." → "[EXECUTIVE] Opens call with Q3 financial results review"
  
- For content over 200 characters: Create proportional summaries (20-40% of original length) preserving all key financial metrics, guidance, strategic initiatives, and analyst concerns
  Example: Long detailed response → Comprehensive summary with all financial data and strategic insights

FINANCIAL RELEVANCE MARKERS (Always include when present):
- Financial metrics (revenue, earnings, margins, ROE, etc.)
- Forward-looking statements and guidance
- Market conditions and competitive positioning
- Strategic initiatives and business developments
- Analyst questions and management responses

SPEAKER ATTRIBUTION FOR RERANKING:
Include speaker type markers:
- [ANALYST]: For analyst questions and comments
- [EXECUTIVE]: For management responses and prepared remarks
- [OPERATOR]: For call logistics and transitions

CRITICAL REQUIREMENTS FOR IMPORTANCE SCORES:
- Score RELATIVE to other paragraphs in this {speaker_block_total_paragraphs}-paragraph speaker block
- 0.8-1.0: Contains specific metrics, financial guidance, or material business decisions
- 0.6-0.8: Contains meaningful business commentary or strategic insights  
- 0.4-0.6: Contains general business context or supporting information
- 0.2-0.4: Contains procedural information or basic acknowledgments
- 0.0-0.2: Contains only pleasantries, transitions, or non-business content

RERANKING OPTIMIZATION:
- Include relevant financial terminology and context for semantic search
- Preserve quantitative data and comparisons
- Maintain forward-looking statements and guidance
- Keep analyst concerns and management responses connected
</objective>

<style>
Write summaries in third person, present tense. Focus on business facts and metrics.
Be concise and analytical. Scale detail with content importance.
</style>

<tone>
Professional financial analysis. Summaries should read like executive briefing points optimized for relevance filtering.
</tone>

<audience>
A smaller model that will judge relevance to user queries about earnings calls and filter results accordingly.
</audience>

<response_format>
Use the summarize_indexed_paragraphs function. 
- Provide exactly one entry per numbered paragraph [1] through [{batch_size}]
- Match your response indexes exactly to the input paragraph numbers
- Scale summary detail with content length and importance
- Include speaker attribution markers for reranking context
- Preserve financial terminology and quantitative data

VALIDATION: Before responding, verify:
1. You have exactly {batch_size} summaries
2. Each summary is appropriately scaled to content length and importance
3. Each importance score considers the full speaker block context
4. Summaries include speaker attribution and financial context for reranking
</response_format>
"""


def format_sliding_window_context(
    previous_speaker_blocks: List[Dict],
    current_speaker_block_records: List[Dict],
    current_batch_start: int,
    batch_size: int,
    processed_summaries: List[Dict] = None
) -> str:
    """Format sliding window context for content enhancement."""
    
    context_parts = []
    
    # Previous speaker blocks context
    if previous_speaker_blocks:
        context_parts.append("=== PREVIOUS SPEAKER BLOCKS (CONTEXT ONLY - DO NOT PROCESS) ===")
        for prev_block in previous_speaker_blocks[-2:]:
            context_parts.append(f"\nSpeaker Block {prev_block['speaker_block_id']}: {prev_block['speaker']}")
            context_parts.append("Key topics covered:")
            for summary in prev_block['paragraph_summaries']:
                context_parts.append(f"  • {summary}")
    
    # Current speaker block header
    total_paragraphs = len(current_speaker_block_records)
    context_parts.append(f"\n=== CURRENT SPEAKER BLOCK {current_speaker_block_records[0]['speaker_block_id']} ===")
    context_parts.append(f"Speaker: {current_speaker_block_records[0]['speaker']}")
    context_parts.append(f"Total Paragraphs in Block: {total_paragraphs}")
    categories = current_speaker_block_records[0].get('category_type', ['General'])
    try:
        if categories and isinstance(categories, list):
            categories_str = ', '.join(categories)
        else:
            categories_str = 'General'
    except Exception as e:
        logger.error(f"Error in format_sliding_window_context join: categories={categories}, type={type(categories)}, error={e}")
        categories_str = 'General'
    context_parts.append(f"Financial Categories: {categories_str}")
    
    # Already processed paragraphs
    if processed_summaries:
        context_parts.append(f"\n--- Already Processed in This Block (CONTEXT ONLY) ---")
        context_parts.append("Previous summaries for context:")
        for i, summary_data in enumerate(processed_summaries):
            importance = summary_data['importance']
            context_parts.append(f"  • (Importance: {importance:.1f}) {summary_data['summary']}")
    
    # Current batch - very clear marking
    context_parts.append(f"\n--- CURRENT BATCH TO PROCESS ---")
    context_parts.append(f"TASK: Summarize and score the following {batch_size} paragraphs:")
    context_parts.append("IMPORTANT: These are the ONLY paragraphs you should process in your response.")
    
    batch_end = min(current_batch_start + batch_size, total_paragraphs)
    
    for i in range(current_batch_start, batch_end):
        local_index = i - current_batch_start + 1
        record = current_speaker_block_records[i]
        context_parts.append(f"\n[{local_index}] PARAGRAPH {record['paragraph_id']} (to be processed):")
        context_parts.append(f"Content: {record['paragraph_content']}")
    
    # Remaining paragraphs context
    if batch_end < total_paragraphs:
        context_parts.append(f"\n--- Remaining in Speaker Block (CONTEXT ONLY - DO NOT PROCESS) ---")
        context_parts.append("Upcoming content for importance calibration:")
        for i in range(batch_end, min(batch_end + 3, total_paragraphs)):  # Show next 3 for context
            record = current_speaker_block_records[i]
            preview = record['paragraph_content'][:150]
            context_parts.append(f"  • Paragraph {record['paragraph_id']}: {preview}...")
    
    # Final instructions
    context_parts.append(f"\n=== PROCESSING INSTRUCTIONS ===")
    context_parts.append(f"1. Process ONLY the {batch_size} numbered paragraphs [1] through [{batch_size}]")
    context_parts.append(f"2. Consider importance relative to all {total_paragraphs} paragraphs in this speaker block")
    context_parts.append(f"3. Write summaries for retrieval systems (what would analysts search for?)")
    context_parts.append(f"4. Use the context above and below to calibrate importance scores")
    
    try:
        return "\n".join(context_parts)
    except Exception as e:
        logger.error(f"Error in format_sliding_window_context final join: context_parts={len(context_parts) if context_parts else 'None'}, error={e}")
        return "ERROR: Context formatting failed"


def validate_summary_response(response_data: Dict, expected_batch_size: int, 
                            current_batch_records: List[Dict]) -> Tuple[bool, str, Dict]:
    """Minimal validation of summary response."""
    
    try:
        summaries = response_data.get("paragraph_summaries", [])
        
        # Check count
        if len(summaries) != expected_batch_size:
            return False, f"Expected {expected_batch_size} summaries, got {len(summaries)}", response_data
        
        # Check indexes
        expected_indexes = set(range(1, expected_batch_size + 1))
        received_indexes = set(s.get("index") for s in summaries)
        
        if expected_indexes != received_indexes:
            missing = expected_indexes - received_indexes
            extra = received_indexes - expected_indexes
            return False, f"Index mismatch - Missing: {missing}, Extra: {extra}", response_data
        
        # Validate importance scores
        for summary_data in summaries:
            importance = summary_data.get("importance")
            if not isinstance(importance, (int, float)) or not (0.0 <= importance <= 1.0):
                return False, f"Invalid importance score: {importance}", response_data
        
        return True, "", response_data
        
    except Exception as e:
        return False, f"Validation error: {e}", response_data


def process_transcript_with_sliding_window(transcript_records: List[Dict], transcript_id: str) -> List[Dict]:
    """Process transcript using sliding window approach for content enhancement."""
    global logger, llm_client, error_logger, config
    
    try:
        # Group by speaker blocks in chronological order
        speaker_blocks = defaultdict(list)
        for record in sorted(transcript_records, key=lambda x: x["paragraph_id"]):
            speaker_blocks[record["speaker_block_id"]].append(record)
        
        # Sort speaker blocks chronologically
        sorted_block_ids = sorted(speaker_blocks.keys())
        
        enhanced_records = []
        previous_speaker_blocks = []  # Store processed blocks with summaries
        batch_size = config["stage_7"]["processing_config"]["batch_size"]
        
        for block_id in sorted_block_ids:
            current_block_records = speaker_blocks[block_id]
            logger.info(f"Processing speaker block {block_id} with {len(current_block_records)} paragraphs")
            
            # Process current speaker block in batches
            processed_summaries = []
            
            for batch_start in range(0, len(current_block_records), batch_size):
                batch_end = min(batch_start + batch_size, len(current_block_records))
                current_batch = current_block_records[batch_start:batch_end]
                
                # Process ALL content - no skipping based on length
                valid_batch = current_batch
                
                # Format sliding window context
                context = format_sliding_window_context(
                    previous_speaker_blocks=previous_speaker_blocks,
                    current_speaker_block_records=current_block_records,
                    current_batch_start=batch_start,
                    batch_size=len(valid_batch),
                    processed_summaries=processed_summaries
                )
                
                # Create CO-STAR prompt
                # Extract and validate financial categories
                raw_categories = valid_batch[0].get("category_type")
                if isinstance(raw_categories, list):
                    financial_categories = raw_categories
                else:
                    logger.warning(f"Invalid category_type for batch: {raw_categories}, type: {type(raw_categories)}")
                    financial_categories = []
                
                system_prompt = create_content_enhancement_prompt(
                    company_name=valid_batch[0].get("company_name", "Unknown"),
                    fiscal_info=f"{valid_batch[0].get('fiscal_year')} {valid_batch[0].get('fiscal_quarter')}",
                    speaker=valid_batch[0]["speaker"],
                    batch_size=len(valid_batch),
                    financial_categories=financial_categories,
                    speaker_block_total_paragraphs=len(current_block_records)
                )
                
                try:
                    # LLM call for current batch
                    response = llm_client.chat.completions.create(
                        model=config["stage_7"]["llm_config"]["model"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": context}
                        ],
                        tools=create_indexed_summary_tools(len(valid_batch)),
                        tool_choice="required",
                        temperature=config["stage_7"]["llm_config"]["temperature"],
                        max_tokens=config["stage_7"]["llm_config"]["max_tokens"]
                    )
                    
                    # Parse and validate response
                    if response.choices[0].message.tool_calls:
                        tool_call = response.choices[0].message.tool_calls[0]
                        result = json.loads(tool_call.function.arguments)
                        
                        # Validate response
                        is_valid, error_msg, validated_result = validate_summary_response(
                            result, len(valid_batch), valid_batch
                        )
                        
                        if not is_valid:
                            logger.warning(f"Validation failed for batch: {error_msg}")
                            error_logger.log_enhancement_error(transcript_id, current_batch[0]["paragraph_id"], error_msg)
                            
                            # Set null fields for failed batch
                            for record in valid_batch:
                                record["paragraph_summary"] = None
                                record["paragraph_importance"] = None
                        else:
                            summaries = validated_result["paragraph_summaries"]
                            
                            # Apply summaries to records
                            for summary_data in summaries:
                                index = summary_data["index"] - 1  # Convert to 0-based
                                if 0 <= index < len(valid_batch):
                                    record = valid_batch[index]
                                    record["paragraph_summary"] = summary_data["summary"]
                                    record["paragraph_importance"] = summary_data["importance"]
                                    
                                    # Track for next batch context
                                    processed_summaries.append({
                                        "paragraph_id": record["paragraph_id"],
                                        "summary": summary_data["summary"],
                                        "importance": summary_data["importance"]
                                    })
                        
                        enhanced_records.extend(valid_batch)
                        
                        # Track costs
                        if hasattr(response, 'usage') and response.usage:
                            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                            error_logger.accumulate_costs({
                                "total_tokens": response.usage.total_tokens,
                                "cost": cost_info
                            })
                            logger.info(f"Batch processed - tokens: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed for speaker block {block_id}, batch {batch_start}: {e}")
                    error_logger.log_enhancement_error(transcript_id, current_batch[0]["paragraph_id"], str(e))
                    
                    # Set null fields for failed batch
                    for record in valid_batch:
                        record["paragraph_summary"] = None
                        record["paragraph_importance"] = None
                    enhanced_records.extend(valid_batch)
            
            # Add current block to previous blocks context for next iteration
            previous_speaker_blocks.append({
                "speaker_block_id": block_id,
                "speaker": current_block_records[0]["speaker"],
                "paragraph_summaries": [s["summary"] for s in processed_summaries if s["summary"]]
            })
            
            # Keep only last 2 speaker blocks for context
            if len(previous_speaker_blocks) > 2:
                previous_speaker_blocks = previous_speaker_blocks[-2:]
        
        return enhanced_records
        
    except Exception as e:
        error_msg = f"Transcript processing failed: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error(transcript_id, error_msg)
        
        # Return original records with null enhancement fields
        for record in transcript_records:
            record["paragraph_summary"] = None
            record["paragraph_importance"] = None
        return transcript_records


def save_enhanced_output(nas_conn: SMBConnection, enhanced_records: List[Dict]) -> bool:
    """Save enhanced records to NAS following Stage 4/5/6 output pattern."""
    global logger, error_logger, config
    
    try:
        # Create output data structure matching Stage 4/5/6 pattern
        output_data = {
            "schema_version": "1.0",
            "processing_timestamp": datetime.now().isoformat(),
            "total_records": len(enhanced_records),
            "enhancement_summary": {
                "paragraphs_enhanced": len([r for r in enhanced_records if r.get("paragraph_summary") is not None]),
                "paragraphs_skipped": len([r for r in enhanced_records if r.get("paragraph_summary") is None]),
                "average_importance_score": round(sum(r.get("paragraph_importance", 0) for r in enhanced_records if r.get("paragraph_importance") is not None) / 
                                                max(len([r for r in enhanced_records if r.get("paragraph_importance") is not None]), 1), 3)
            },
            "cost_summary": error_logger.get_summary(),
            "records": enhanced_records
        }
        
        # Save to NAS
        output_path = f"{NAS_BASE_PATH}/{config['stage_7']['output_path']}/{config['stage_7']['output_file']}"
        
        if not validate_nas_path(output_path):
            raise ValueError("Invalid output file path")
            
        output_content = json.dumps(output_data, indent=2)
        
        nas_upload_file(
            nas_conn,
            io.BytesIO(output_content.encode('utf-8')),
            output_path
        )
        
        logger.info(f"Enhanced output saved to NAS: {config['stage_7']['output_file']}")
        logger.info(f"Total records: {len(enhanced_records)}")
        logger.info(f"Records enhanced: {output_data['enhancement_summary']['paragraphs_enhanced']}")
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to save enhanced output: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("output_save", error_msg)
        return False


def upload_logs_to_nas(nas_conn: SMBConnection, main_logger: logging.Logger, 
                      enhanced_error_logger: EnhancedErrorLogger) -> bool:
    """Upload execution logs to NAS following Stage 4/5/6 pattern."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Upload main execution log
        if hasattr(main_logger, 'temp_log_file'):
            with open(main_logger.temp_log_file, 'r') as f:
                log_content = f.read()
            
            log_filename = f"stage_7_content_enhancement_{timestamp}.log"
            nas_upload_file(
                nas_conn,
                io.BytesIO(log_content.encode('utf-8')),
                f"{NAS_BASE_PATH}/Outputs/Logs/{log_filename}"
            )
            
            logger.info(f"Execution log uploaded to NAS: {log_filename}")
        
        # Upload error logs
        enhanced_error_logger.save_error_logs(nas_conn)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload logs to NAS: {e}")
        return False


def cleanup_resources():
    """Clean up temporary files and resources."""
    global ssl_cert_path, logger
    
    try:
        # Clean up SSL certificate
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            os.unlink(ssl_cert_path)
            logger.info("SSL certificate cleaned up")
        
        # Clean up temp log file
        if hasattr(logger, 'temp_log_file') and os.path.exists(logger.temp_log_file):
            os.unlink(logger.temp_log_file)
            logger.info("Temporary log file cleaned up")
            
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def main() -> None:
    """Main function orchestrating Stage 7 content enhancement."""
    global config, logger, error_logger, llm_client, ssl_cert_path

    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    logger.info("STAGE 7: CONTENT ENHANCEMENT SYSTEM")

    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting")
        return

    try:
        start_time = datetime.now()
        
        # Load configuration
        config = load_stage_config(nas_conn)
        logger.info("Loaded configuration for Stage 7")
        logger.info(f"Development mode: {config['stage_7']['dev_mode']}")
        
        if config['stage_7']['dev_mode']:
            logger.info(f"Max transcripts in dev mode: {config['stage_7']['dev_max_transcripts']}")

        # Setup SSL certificate
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        if not ssl_cert_path:
            logger.error("Failed to setup SSL certificate - aborting")
            return
        
        # Load Stage 6 output
        stage6_data = load_stage6_output(nas_conn)
        records = stage6_data["records"]
        
        # Apply dev mode limits
        if config['stage_7']['dev_mode']:
            max_transcripts = config['stage_7']['dev_max_transcripts']
            logger.info(f"Development mode: limiting to {max_transcripts} transcripts")
            
            # Group by transcript and take first N
            transcript_groups = defaultdict(list)
            for record in records:
                transcript_key = f"{record.get('ticker')}_{record.get('fiscal_year')}_{record.get('fiscal_quarter')}"
                transcript_groups[transcript_key].append(record)
            
            limited_records = []
            for i, (transcript_key, transcript_records) in enumerate(transcript_groups.items()):
                if i >= max_transcripts:
                    break
                limited_records.extend(transcript_records)
            
            records = limited_records
            logger.info(f"Limited to {len(records)} records from {max_transcripts} transcripts")
        
        # Group by transcript for processing
        transcripts = defaultdict(list)
        for record in records:
            transcript_key = f"{record.get('ticker')}_{record.get('fiscal_year')}_{record.get('fiscal_quarter')}"
            transcripts[transcript_key].append(record)
        
        all_enhanced_records = []
        per_file_metrics = []  # Track per-transcript metrics
        
        # Process each transcript
        for i, (transcript_key, transcript_records) in enumerate(transcripts.items(), 1):
            transcript_start_time = datetime.now()
            logger.info(f"Processing transcript {i}/{len(transcripts)}: {transcript_key}")
            
            # Capture initial cost state
            initial_cost = error_logger.total_cost
            initial_tokens = error_logger.total_tokens
            
            # Refresh OAuth token per transcript
            if not refresh_oauth_token_for_transcript():
                logger.error(f"Failed to refresh OAuth token for {transcript_key}")
                continue
            
            # Process transcript with sliding window
            enhanced_records = process_transcript_with_sliding_window(transcript_records, transcript_key)
            all_enhanced_records.extend(enhanced_records)
            
            # Calculate per-transcript metrics
            transcript_end_time = datetime.now()
            transcript_time = transcript_end_time - transcript_start_time
            transcript_cost = error_logger.total_cost - initial_cost
            transcript_tokens = error_logger.total_tokens - initial_tokens
            
            per_file_metrics.append({
                "transcript": transcript_key,
                "processing_time": transcript_time,
                "cost": transcript_cost,
                "tokens": transcript_tokens,
                "records": len(enhanced_records)
            })
            
            logger.info(f"Transcript {transcript_key} completed - Time: {transcript_time}, Cost: ${transcript_cost:.4f}, Tokens: {transcript_tokens:,}")
            
            time.sleep(1)  # Rate limiting
        
        # Add records that don't need enhancement (maintain complete dataset)
        for record in records:
            if record not in all_enhanced_records:
                record["paragraph_summary"] = None
                record["paragraph_importance"] = None
                all_enhanced_records.append(record)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Final summary
        total_cost = error_logger.total_cost
        total_tokens = error_logger.total_tokens
        total_calls = error_logger.api_calls
        
        logger.info("="*60)
        logger.info("STAGE 7 CONTENT ENHANCEMENT COMPLETE")
        logger.info("="*60)
        logger.info(f"Transcripts processed: {len(transcripts)}")
        logger.info(f"Total records processed: {len(all_enhanced_records)}")
        logger.info(f"Records enhanced: {len([r for r in all_enhanced_records if r.get('paragraph_summary') is not None])}")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Total LLM API calls: {total_calls}")
        logger.info(f"Total tokens used: {total_tokens:,}")
        logger.info(f"Total LLM cost: ${total_cost:.4f}")
        if total_calls > 0:
            logger.info(f"Average cost per call: ${total_cost/total_calls:.4f}")
            logger.info(f"Average tokens per call: {total_tokens/total_calls:.0f}")
        
        # Add per-transcript averages
        if per_file_metrics:
            avg_time_per_transcript = sum(m["processing_time"].total_seconds() for m in per_file_metrics) / len(per_file_metrics)
            avg_cost_per_transcript = sum(m["cost"] for m in per_file_metrics) / len(per_file_metrics)
            logger.info(f"Average time per transcript: {avg_time_per_transcript:.1f} seconds")
            logger.info(f"Average cost per transcript: ${avg_cost_per_transcript:.4f}")
        
        logger.info("="*60)
        
        # Save output and upload logs
        save_enhanced_output(nas_conn, all_enhanced_records)
        upload_logs_to_nas(nas_conn, logger, error_logger)
        
    except Exception as e:
        logger.error(f"Stage 7 failed: {e}")
        error_logger.log_processing_error("main_execution", str(e))
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()