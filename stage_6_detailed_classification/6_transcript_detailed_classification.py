"""
Stage 6: Detailed Classification System
Processes Stage 5 output to add detailed financial category classification.
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
        prefix=f'stage_6_detailed_classification_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
        self.classification_errors = []
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

    def log_classification_error(self, transcript_id: str, section_type: str, error: str):
        """Log classification-specific errors."""
        self.classification_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "section_type": section_type,
            "error": error,
            "error_type": "classification"
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
                       len(self.classification_errors) + len(self.processing_errors))
        
        return {
            "total_errors": total_errors,
            "llm_errors": len(self.llm_errors),
            "authentication_errors": len(self.authentication_errors),
            "classification_errors": len(self.classification_errors),
            "processing_errors": len(self.processing_errors),
            "total_cost": round(self.total_cost, 4),
            "total_tokens": self.total_tokens,
            "total_api_calls": self.api_calls,
            "average_cost_per_call": round(self.total_cost / max(self.api_calls, 1), 4),
            "average_tokens_per_call": round(self.total_tokens / max(self.api_calls, 1), 0)
        }

    def save_error_logs(self, nas_conn: SMBConnection) -> bool:
        """Save detailed error logs to NAS."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            error_data = {
                "processing_timestamp": datetime.now().isoformat(),
                "summary": self.get_summary(),
                "detailed_errors": {
                    "llm_errors": self.llm_errors,
                    "authentication_errors": self.authentication_errors,
                    "classification_errors": self.classification_errors,
                    "processing_errors": self.processing_errors
                }
            }
            
            error_log_content = json.dumps(error_data, indent=2)
            error_log_file = f"stage_6_detailed_classification_{timestamp}_errors.json"
            
            nas_conn.storeFile(
                NAS_SHARE_NAME,
                f"{NAS_BASE_PATH}/Outputs/Logs/Errors/{error_log_file}",
                io.BytesIO(error_log_content.encode('utf-8'))
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
    """Ensure safe NAS paths only."""
    try:
        if not path or not isinstance(path, str):
            return False
        safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./\\')
        return all(c in safe_chars for c in path) and validate_file_path(path)
    except Exception:
        return False


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
    """Establish secure NAS connection with validation."""
    global logger
    
    try:
        if not all([NAS_USERNAME, NAS_PASSWORD, NAS_SERVER_NAME, CLIENT_MACHINE_NAME]):
            logger.error("Missing NAS connection parameters")
            return None
            
        conn = SMBConnection(
            NAS_USERNAME, NAS_PASSWORD, CLIENT_MACHINE_NAME, NAS_SERVER_NAME,
            domain=PROXY_DOMAIN, use_ntlm_v2=True, is_direct_tcp=True
        )
        
        if not conn.connect(NAS_SERVER_IP, NAS_PORT, timeout=30):
            logger.error("Failed to establish NAS connection")
            return None
            
        logger.info("Successfully connected to NAS")
        return conn
        
    except Exception as e:
        logger.error(f"NAS connection failed: {e}")
        return None


def load_stage_config(nas_conn: SMBConnection) -> Dict:
    """Load and validate Stage 6 configuration from NAS."""
    global logger, error_logger
    
    try:
        if not validate_nas_path(CONFIG_PATH):
            raise ValueError("Invalid CONFIG_PATH")
            
        config_buffer = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, CONFIG_PATH, config_buffer)
        config_buffer.seek(0)
        
        full_config = json.loads(config_buffer.read().decode('utf-8'))
        
        if "stage_6" not in full_config:
            raise ValueError("Stage 6 configuration not found in config file")
            
        stage_config = full_config["stage_6"]
        
        # Validate required configuration sections
        required_sections = ["llm_config", "processing_config"]
        for section in required_sections:
            if section not in stage_config:
                raise ValueError(f"Missing required config section: {section}")
        
        logger.info("Successfully loaded Stage 6 configuration from NAS")
        return stage_config
        
    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("config_load", error_msg)
        raise


def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download and setup SSL certificate for LLM API connections."""
    global logger, error_logger
    
    try:
        cert_path = f"{NAS_BASE_PATH}/Inputs/certificate/certificate.cer"
        
        if not validate_nas_path(cert_path):
            raise ValueError("Invalid certificate path")
            
        cert_buffer = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, cert_path, cert_buffer)
        cert_buffer.seek(0)
        
        # Save to temporary file
        temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
        temp_cert.write(cert_buffer.read())
        temp_cert.close()
        
        # Set environment variable for SSL verification
        os.environ['SSL_CERT_FILE'] = temp_cert.name
        
        logger.info(f"SSL certificate setup completed: {temp_cert.name}")
        return temp_cert.name
        
    except Exception as e:
        error_msg = f"SSL certificate setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        raise


def get_oauth_token() -> Optional[str]:
    """Obtain OAuth token for LLM API access."""
    global logger, error_logger, config
    
    try:
        token_endpoint = config["llm_config"]["token_endpoint"]
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': LLM_CLIENT_ID,
            'client_secret': LLM_CLIENT_SECRET
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(
            token_endpoint,
            data=auth_data,
            headers=headers,
            timeout=30,
            verify=ssl_cert_path
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
            base_url=config["llm_config"]["base_url"],
            timeout=config["llm_config"]["timeout"]
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
    
    prompt_cost_per_1k = config["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["llm_config"]["cost_per_1k_completion_tokens"]
    
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


def load_stage5_output(nas_conn: SMBConnection) -> Dict:
    """Load Stage 5 output data from NAS."""
    global logger, error_logger, config
    
    try:
        input_path = f"{NAS_BASE_PATH}/{config['input_source']}"
        
        if not validate_nas_path(input_path):
            raise ValueError("Invalid input file path")
            
        input_buffer = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, input_path, input_buffer)
        input_buffer.seek(0)
        
        stage5_data = json.loads(input_buffer.read().decode('utf-8'))
        
        logger.info(f"Loaded {len(stage5_data.get('records', []))} records from Stage 5")
        return stage5_data
        
    except Exception as e:
        error_msg = f"Failed to load Stage 5 data: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("stage5_load", error_msg)
        raise


def create_management_discussion_tools() -> List[Dict]:
    """Function calling schema for Management Discussion classification."""
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
                                    "items": {"type": "string", "enum": [
                                        "Financial Performance & Results",
                                        "Credit Quality & Risk Management", 
                                        "Capital & Regulatory Management",
                                        "Strategic Initiatives & Transformation",
                                        "Market Environment & Outlook", 
                                        "Operating Efficiency & Expenses",
                                        "Asset & Liability Management",
                                        "Non-Interest Revenue & Segments",
                                        "ESG & Sustainability",
                                        "Insurance Operations"
                                    ]},
                                    "description": "All applicable categories for this paragraph"
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
                                "items": {"type": "string", "enum": [
                                    "Financial Performance & Results",
                                    "Credit Quality & Risk Management",
                                    "Capital & Regulatory Management", 
                                    "Strategic Initiatives & Transformation",
                                    "Market Environment & Outlook",
                                    "Operating Efficiency & Expenses",
                                    "Asset & Liability Management",
                                    "Non-Interest Revenue & Segments", 
                                    "ESG & Sustainability",
                                    "Insurance Operations"
                                ]},
                                "description": "All applicable categories for this conversation"
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
1. **Financial Performance & Results**: Overall financial results including revenue, earnings per share, profitability metrics (ROE/ROA), net income, and general financial health indicators. High-level performance commentary without diving into specific drivers.

2. **Credit Quality & Risk Management**: Asset quality metrics including loan loss provisions, non-performing loans, charge-offs, delinquency trends, reserve coverage ratios, and portfolio risk assessment. Includes both current credit performance and forward-looking credit expectations.

3. **Capital & Regulatory Management**: Capital adequacy ratios (CET1, Tier 1, leverage), stress test results, capital allocation decisions (dividends, buybacks), regulatory requirements, and compliance matters.

4. **Strategic Initiatives & Transformation**: Forward-looking strategic moves including digital transformation, technology investments, M&A activity, market expansion, new products/services, efficiency programs, and organizational changes.

5. **Market Environment & Outlook**: Macroeconomic commentary, interest rate environment, competitive landscape, industry trends, AND forward guidance/outlook. Includes both external factors and company's expectations.

6. **Operating Efficiency & Expenses**: Operating costs, expense management, efficiency ratios, technology spend, personnel costs, and productivity initiatives. Includes both absolute expense levels and efficiency metrics.

7. **Asset & Liability Management**: Net interest income/margin dynamics, deposit trends and costs, loan yields, funding strategies, and balance sheet optimization. Encompasses the core spread business of banking.

8. **Non-Interest Revenue & Segments**: Fee-based income streams, trading revenues, wealth management, investment banking, and performance by business division. Captures all non-spread revenue and divisional breakdowns.

9. **ESG & Sustainability**: Environmental commitments, social responsibility initiatives, governance improvements, climate risk management, and sustainability reporting. Includes regulatory ESG requirements and voluntary programs.

10. **Insurance Operations**: Insurance-specific metrics including premiums, underwriting results, combined ratios, catastrophe losses, reserve development, and reinsurance. Only for institutions with insurance operations.
</categories>

<response_format>
Use the classify_management_discussion_paragraphs function to classify each paragraph 
with applicable categories and confidence scores.
</response_format>
"""


def create_qa_conversation_costar_prompt(company_name: str, fiscal_info: str,
                                       conversation_length: int) -> str:
    """CO-STAR prompt with category descriptions for Q&A conversations."""
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
1. **Financial Performance & Results**: Overall financial results including revenue, earnings per share, profitability metrics (ROE/ROA), net income, and general financial health indicators.

2. **Credit Quality & Risk Management**: Asset quality metrics including loan loss provisions, non-performing loans, charge-offs, delinquency trends, reserve coverage ratios, and portfolio risk assessment.

3. **Capital & Regulatory Management**: Capital adequacy ratios (CET1, Tier 1, leverage), stress test results, capital allocation decisions (dividends, buybacks), regulatory requirements, and compliance matters.

4. **Strategic Initiatives & Transformation**: Forward-looking strategic moves including digital transformation, technology investments, M&A activity, market expansion, new products/services, efficiency programs, and organizational changes.

5. **Market Environment & Outlook**: Macroeconomic commentary, interest rate environment, competitive landscape, industry trends, AND forward guidance/outlook.

6. **Operating Efficiency & Expenses**: Operating costs, expense management, efficiency ratios, technology spend, personnel costs, and productivity initiatives.

7. **Asset & Liability Management**: Net interest income/margin dynamics, deposit trends and costs, loan yields, funding strategies, and balance sheet optimization.

8. **Non-Interest Revenue & Segments**: Fee-based income streams, trading revenues, wealth management, investment banking, and performance by business division.

9. **ESG & Sustainability**: Environmental commitments, social responsibility initiatives, governance improvements, climate risk management, and sustainability reporting.

10. **Insurance Operations**: Insurance-specific metrics including premiums, underwriting results, combined ratios, catastrophe losses, reserve development, and reinsurance.
</categories>

<response_format>
Use the classify_qa_conversation function to provide comprehensive conversation analysis
with all applicable categories and confidence score.
</response_format>
"""


def get_prior_blocks_context(speaker_blocks: Dict, current_block_id: int, 
                           max_blocks: int = 2, preview_chars: int = 750) -> str:
    """Get prior speaker blocks with 750-char previews and classifications."""
    
    # Get previous speaker blocks
    prior_blocks = []
    for block_id in sorted(speaker_blocks.keys()):
        if block_id < current_block_id:
            prior_blocks.append((block_id, speaker_blocks[block_id]))
    
    if not prior_blocks:
        return ""
    
    # Take last 2 blocks
    recent_blocks = prior_blocks[-max_blocks:]
    context_parts = []
    
    for block_id, block_records in recent_blocks:
        speaker = block_records[0]["speaker"]
        context_parts.append(f"\n--- PRIOR SPEAKER BLOCK {block_id}: {speaker} ---")
        
        for i, record in enumerate(block_records):
            # Show 750-char preview with classification if available
            content_preview = record["paragraph_content"][:preview_chars]
            if len(record["paragraph_content"]) > preview_chars:
                content_preview += "..."
            
            # Check if this paragraph has detailed classification
            if "detailed_classification" in record and record["detailed_classification"]:
                categories = record["detailed_classification"]["categories"]
                confidence = record["detailed_classification"]["confidence"]
                context_parts.append(f"P{i+1}: [CLASSIFIED: {', '.join(categories)} ({confidence:.2f})]")
                context_parts.append(content_preview)
            else:
                context_parts.append(f"P{i+1}: [NO_CLASSIFICATION]")
                context_parts.append(content_preview)
    
    return "\n".join(context_parts)


def format_management_discussion_context(current_block_records: List[Dict], 
                                       current_paragraph_window: List[Dict],
                                       prior_blocks_context: str,
                                       previous_classifications: List[Dict]) -> str:
    """Format context with 750-char prior block previews."""
    
    context_parts = []
    
    # Prior speaker blocks (750-char previews with classifications)
    if prior_blocks_context:
        context_parts.append("=== PRIOR SPEAKER BLOCKS (CONTEXT) ===")
        context_parts.append(prior_blocks_context)
    
    # Current speaker block (full content always)
    context_parts.append(f"\n=== CURRENT SPEAKER BLOCK ===")
    context_parts.append(f"Speaker: {current_block_records[0]['speaker']}")
    context_parts.append(f"Total Paragraphs: {len(current_block_records)}")
    
    # Show ALL paragraphs in current speaker block
    for i, record in enumerate(current_block_records):
        para_id = record["paragraph_id"]
        content = record["paragraph_content"]
        
        # Check if this paragraph was previously classified in this session
        prev_class = next((c for c in previous_classifications if c["paragraph_id"] == para_id), None)
        
        if prev_class:
            # Previously classified - show classification
            categories = ', '.join(prev_class['categories'])
            context_parts.append(f"\nP{i+1} (ID:{para_id}) [CLASSIFIED: {categories} ({prev_class['confidence']:.2f})]:")
            context_parts.append(content)
        elif record in current_paragraph_window:
            # Current window - to be classified
            context_parts.append(f"\nP{i+1} (ID:{para_id}) [TO_CLASSIFY]:")
            context_parts.append(content)
        else:
            # Future paragraphs in this speaker block
            context_parts.append(f"\nP{i+1} (ID:{para_id}) [FUTURE]:")
            context_parts.append(content)
    
    # Highlight current classification window
    window_start = current_block_records.index(current_paragraph_window[0]) + 1
    window_end = current_block_records.index(current_paragraph_window[-1]) + 1
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


def process_management_discussion_section(md_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process Management Discussion using speaker block windowing approach."""
    global logger, llm_client, error_logger, config
    
    # Group by speaker blocks
    speaker_blocks = defaultdict(list)
    for record in sorted(md_records, key=lambda x: x["paragraph_id"]):
        speaker_blocks[record["speaker_block_id"]].append(record)
    
    classified_records = []
    
    for block_id, block_records in speaker_blocks.items():
        logger.info(f"Processing speaker block {block_id} with {len(block_records)} paragraphs")
        
        # Process in 5-paragraph windows
        previous_classifications = []
        window_size = config["processing_config"]["md_paragraph_window_size"]
        
        for window_start in range(0, len(block_records), window_size):
            window_end = min(window_start + window_size, len(block_records))
            paragraph_window = block_records[window_start:window_end]
            
            # Get prior blocks context (750-char previews)
            prior_blocks_context = get_prior_blocks_context(
                speaker_blocks, block_id, 
                max_blocks=config["processing_config"]["max_speaker_blocks_context"], 
                preview_chars=config["processing_config"]["prior_block_preview_chars"]
            )
            
            # Format context
            context = format_management_discussion_context(
                current_block_records=block_records,
                current_paragraph_window=paragraph_window,
                prior_blocks_context=prior_blocks_context,
                previous_classifications=previous_classifications
            )
            
            # Create CO-STAR prompt with full category descriptions
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
                    model=config["llm_config"]["model"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": context}
                    ],
                    tools=create_management_discussion_tools(),
                    tool_choice="required",
                    temperature=config["llm_config"]["temperature"],
                    max_tokens=config["llm_config"]["max_tokens"]
                )
                
                # Parse and apply classifications
                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    result = json.loads(tool_call.function.arguments)
                    
                    # Apply single classification field to each paragraph
                    for i, para_class in enumerate(result["paragraph_classifications"]):
                        if i < len(paragraph_window):
                            record = paragraph_window[i]
                            record["detailed_classification"] = {
                                "categories": para_class["categories"],
                                "confidence": para_class["confidence"],
                                "method": "speaker_block_windowing"
                            }
                            
                            classified_records.append(record)
                            previous_classifications.append({
                                "paragraph_id": record["paragraph_id"],
                                "categories": para_class["categories"],
                                "confidence": para_class["confidence"]
                            })
                
                # Track costs
                if hasattr(response, 'usage') and response.usage:
                    cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    error_logger.accumulate_costs({
                        "total_tokens": response.usage.total_tokens,
                        "cost": cost_info
                    })
                    logger.debug(f"MD window classification - tokens: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                
            except Exception as e:
                error_msg = f"Management Discussion classification failed for block {block_id}: {e}"
                logger.error(error_msg)
                error_logger.log_classification_error(
                    block_records[0].get("filename", "unknown"), "Management Discussion", error_msg
                )
                
                # No fallback - set classification to None for failed paragraphs
                for record in paragraph_window:
                    record["detailed_classification"] = None
                    classified_records.append(record)
    
    return classified_records


def process_qa_group(qa_group_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process complete Q&A group as single conversation."""
    global logger, llm_client, error_logger, config
    
    qa_group_id = qa_group_records[0]["qa_group_id"]
    logger.info(f"Processing Q&A group {qa_group_id} with {len(qa_group_records)} paragraphs")
    
    # Format complete conversation context
    conversation_context = format_qa_group_context(qa_group_records)
    
    # Create CO-STAR prompt with category descriptions
    system_prompt = create_qa_conversation_costar_prompt(
        company_name=qa_group_records[0].get("company_name", "Unknown"),
        fiscal_info=f"{qa_group_records[0].get('fiscal_year')} {qa_group_records[0].get('fiscal_quarter')}",
        conversation_length=len(qa_group_records)
    )
    
    try:
        # Single LLM call for entire conversation
        response = llm_client.chat.completions.create(
            model=config["llm_config"]["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation_context}
            ],
            tools=create_qa_conversation_tools(),
            tool_choice="required",
            temperature=config["llm_config"]["temperature"],
            max_tokens=config["llm_config"]["max_tokens"]
        )
        
        # Parse and apply to ALL records in group
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            # Apply same classification to all paragraphs in conversation
            conversation_classification = {
                "categories": result["conversation_classification"]["categories"],
                "confidence": result["conversation_classification"]["confidence"],
                "method": "complete_conversation"
            }
            
            for record in qa_group_records:
                record["detailed_classification"] = conversation_classification
        
        # Track costs
        if hasattr(response, 'usage') and response.usage:
            cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            error_logger.accumulate_costs({
                "total_tokens": response.usage.total_tokens,
                "cost": cost_info
            })
            logger.info(f"Q&A group {qa_group_id} - tokens: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
        
        return qa_group_records
        
    except Exception as e:
        error_msg = f"Q&A conversation classification failed for group {qa_group_id}: {e}"
        logger.error(error_msg)
        error_logger.log_classification_error(
            qa_group_records[0].get("filename", "unknown"), "Investor Q&A", error_msg
        )
        
        # No fallback - set classification to None
        for record in qa_group_records:
            record["detailed_classification"] = None
        
        return qa_group_records


def save_classified_output(nas_conn: SMBConnection, classified_records: List[Dict]) -> bool:
    """Save classified records to NAS following Stage 4/5 output pattern."""
    global logger, error_logger, config
    
    try:
        # Create output data structure matching Stage 4/5 pattern
        output_data = {
            "schema_version": "1.0",
            "processing_timestamp": datetime.now().isoformat(),
            "total_records": len(classified_records),
            "classification_summary": {
                "management_discussion_classified": len([r for r in classified_records 
                                                       if r.get("section_type") == "Management Discussion" 
                                                       and r.get("detailed_classification")]),
                "qa_groups_classified": len(set(r.get("qa_group_id") for r in classified_records 
                                              if r.get("section_type") == "Investor Q&A" 
                                              and r.get("detailed_classification") 
                                              and r.get("qa_group_id"))),
                "total_with_classifications": len([r for r in classified_records if r.get("detailed_classification")])
            },
            "cost_summary": error_logger.get_summary(),
            "records": classified_records
        }
        
        # Save to NAS
        output_path = f"{NAS_BASE_PATH}/{config['output_path']}/{config['output_file']}"
        
        if not validate_nas_path(output_path):
            raise ValueError("Invalid output file path")
            
        output_content = json.dumps(output_data, indent=2)
        
        nas_conn.storeFile(
            NAS_SHARE_NAME,
            output_path,
            io.BytesIO(output_content.encode('utf-8'))
        )
        
        logger.info(f"Classified output saved to NAS: {config['output_file']}")
        logger.info(f"Total records: {len(classified_records)}")
        logger.info(f"Records with classifications: {output_data['classification_summary']['total_with_classifications']}")
        
        return True
        
    except Exception as e:
        error_msg = f"Failed to save classified output: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("output_save", error_msg)
        return False


def upload_logs_to_nas(nas_conn: SMBConnection, main_logger: logging.Logger, 
                      enhanced_error_logger: EnhancedErrorLogger) -> bool:
    """Upload execution logs to NAS following Stage 4/5 pattern."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Upload main execution log
        if hasattr(main_logger, 'temp_log_file'):
            with open(main_logger.temp_log_file, 'r') as f:
                log_content = f.read()
            
            log_filename = f"stage_6_detailed_classification_{timestamp}.log"
            nas_conn.storeFile(
                NAS_SHARE_NAME,
                f"{NAS_BASE_PATH}/Outputs/Logs/{log_filename}",
                io.BytesIO(log_content.encode('utf-8'))
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
    """Main function orchestrating Stage 6 detailed classification."""
    global config, logger, error_logger, llm_client, ssl_cert_path

    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    logger.info("STAGE 6: DETAILED CLASSIFICATION SYSTEM")

    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting")
        return

    try:
        # Setup (identical to Stage 4/5)
        config = load_stage_config(nas_conn)
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        
        # Load Stage 5 output
        stage5_data = load_stage5_output(nas_conn)
        records = stage5_data["records"]
        
        # Apply dev mode limits
        if config.get("dev_mode", False):
            max_transcripts = config.get("dev_max_transcripts", 2)
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
            logger.info(f"Limited to {len(records)} records from {len(transcript_groups)} transcripts")
        
        # Group by transcript and section type
        transcripts = defaultdict(lambda: {"management_discussion": [], "qa_groups": defaultdict(list)})
        
        for record in records:
            transcript_key = f"{record.get('ticker')}_{record.get('fiscal_year')}_{record.get('fiscal_quarter')}"
            
            if record.get("section_type") == "Management Discussion":
                transcripts[transcript_key]["management_discussion"].append(record)
            elif record.get("section_type") == "Investor Q&A" and record.get("qa_group_id"):
                qa_group_id = record["qa_group_id"]
                transcripts[transcript_key]["qa_groups"][qa_group_id].append(record)
        
        all_classified_records = []
        
        # Process each transcript
        for transcript_key, transcript_data in transcripts.items():
            logger.info(f"Processing transcript: {transcript_key}")
            
            # Refresh OAuth token per transcript
            if not refresh_oauth_token_for_transcript():
                logger.error(f"Failed to refresh OAuth token for {transcript_key}")
                continue
            
            # Process Management Discussion sections
            if transcript_data["management_discussion"]:
                md_classified = process_management_discussion_section(
                    transcript_data["management_discussion"]
                )
                all_classified_records.extend(md_classified)
            
            # Process Q&A groups
            for qa_group_id, qa_records in transcript_data["qa_groups"].items():
                qa_classified = process_qa_group(qa_records)
                all_classified_records.extend(qa_classified)
            
            time.sleep(1)  # Rate limiting
        
        # Add records that don't need classification
        for record in records:
            if (record.get("section_type") not in ["Management Discussion", "Investor Q&A"] or
                (record.get("section_type") == "Investor Q&A" and not record.get("qa_group_id"))):
                record["detailed_classification"] = None
                all_classified_records.append(record)
        
        # Final summary
        total_cost = error_logger.total_cost
        total_tokens = error_logger.total_tokens
        total_calls = error_logger.api_calls
        
        logger.info("=== STAGE 6 EXECUTION SUMMARY ===")
        logger.info(f"Total records processed: {len(all_classified_records)}")
        logger.info(f"Records with detailed classifications: {len([r for r in all_classified_records if r.get('detailed_classification')])}")
        logger.info(f"Total LLM API calls: {total_calls}")
        logger.info(f"Total tokens used: {total_tokens:,}")
        logger.info(f"Total LLM cost: ${total_cost:.4f}")
        if total_calls > 0:
            logger.info(f"Average cost per call: ${total_cost/total_calls:.4f}")
            logger.info(f"Average tokens per call: {total_tokens/total_calls:.0f}")
        
        # Save output and upload logs
        save_classified_output(nas_conn, all_classified_records)
        upload_logs_to_nas(nas_conn, logger, error_logger)
        
    except Exception as e:
        logger.error(f"Stage 6 failed: {e}")
        error_logger.log_processing_error("main_execution", str(e))
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()