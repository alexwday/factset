"""
Stage 5: Q&A Pairing System
Processes Stage 4 output to identify and group question-answer conversation boundaries.
Uses speaker block-based sliding window analysis with LLM boundary detection.
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
        prefix=f'stage_05_qa_pairing_qa_pairing_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
        self.boundary_detection_errors = []
        self.authentication_errors = []
        self.validation_errors = []
        self.processing_errors = []
        self.total_cost = 0.0
        self.total_tokens = 0

    def log_boundary_error(self, transcript_id: str, speaker_block_id: int, error: str):
        """Log Q&A boundary detection errors."""
        self.boundary_detection_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "speaker_block_id": speaker_block_id,
            "error": error,
            "action_required": "Review speaker block context and retry"
        })

    def log_authentication_error(self, error: str):
        """Log OAuth/SSL authentication errors."""
        self.authentication_errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "action_required": "Check LLM credentials and SSL certificate"
        })

    def log_validation_error(self, transcript_id: str, validation_issue: str):
        """Log Q&A validation errors."""
        self.validation_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "validation_issue": validation_issue,
            "action_required": "Review Q&A group assignments"
        })

    def log_processing_error(self, transcript_id: str, error: str):
        """Log general processing errors."""
        self.processing_errors.append({
            "timestamp": datetime.now().isoformat(),
            "transcript_id": transcript_id,
            "error": error,
            "action_required": "Review transcript structure and processing logic"
        })

    def accumulate_costs(self, token_usage: dict = None):
        """Accumulate total cost and token usage for final summary."""
        if token_usage:
            # Accumulate total cost and tokens
            if "cost" in token_usage:
                self.total_cost += token_usage["cost"]["total_cost"]
            if "total_tokens" in token_usage:
                self.total_tokens += token_usage["total_tokens"]

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        error_types = {
            "boundary_detection": self.boundary_detection_errors,
            "authentication": self.authentication_errors, 
            "validation": self.validation_errors,
            "processing": self.processing_errors
        }
        
        for error_type, errors in error_types.items():
            if errors:
                filename = f"stage_05_qa_pairing_qa_pairing_{error_type}_errors_{timestamp}.json"
                nas_path = f"{NAS_BASE_PATH}/Outputs/Logs/Errors/{filename}"
                
                try:
                    error_data = {
                        "error_type": error_type,
                        "timestamp": timestamp,
                        "total_errors": len(errors),
                        "errors": errors
                    }
                    
                    error_json = json.dumps(error_data, indent=2)
                    error_bytes = io.BytesIO(error_json.encode('utf-8'))
                    
                    nas_conn.storeFile(NAS_SHARE_NAME, nas_path, error_bytes)
                    logger.info(f"Uploaded {error_type} error log: {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to upload {error_type} error log: {e}")


def get_oauth_token() -> Optional[str]:
    """Get OAuth token using client credentials flow."""
    global logger, error_logger
    
    try:
        token_endpoint = config["stage_05_qa_pairing"]["llm_config"]["token_endpoint"]
        
        # Prepare OAuth request
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': LLM_CLIENT_ID,
            'client_secret': LLM_CLIENT_SECRET
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


def setup_llm_client() -> Optional[OpenAI]:
    """Setup OpenAI client with custom base URL and OAuth token."""
    global logger, error_logger, oauth_token
    
    try:
        # Get OAuth token
        oauth_token = get_oauth_token()
        if not oauth_token:
            return None
        
        # Setup OpenAI client with custom configuration
        llm_config = config["stage_05_qa_pairing"]["llm_config"]
        
        client = OpenAI(
            api_key=oauth_token,  # Use OAuth token as API key
            base_url=llm_config["base_url"],
            timeout=llm_config.get("timeout", 60),
            max_retries=llm_config.get("max_retries", 3)
        )
        
        logger.info("Successfully configured LLM client")
        return client
        
    except Exception as e:
        error_msg = f"LLM client setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return None


def refresh_oauth_token_for_transcript() -> bool:
    """Refresh OAuth token for each new transcript."""
    global oauth_token, llm_client
    
    # Get fresh token for each transcript
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        # Update client with new token
        llm_client = setup_llm_client()
        return llm_client is not None
    return False


# Copy all NAS utility functions from Stage 4 (to maintain standalone nature)
def sanitize_url_for_logging(url: str) -> str:
    """Sanitize URL for logging by removing query parameters and auth tokens."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        return str(url)


def validate_file_path(path: str) -> bool:
    """Validate file path to prevent directory traversal attacks."""
    try:
        # Normalize the path and check for directory traversal attempts
        normalized_path = os.path.normpath(path)
        if ".." in normalized_path or normalized_path.startswith("/"):
            return False
        return True
    except (ValueError, TypeError) as e:
        return False


def validate_nas_path(path: str) -> bool:
    """Ensure NAS paths are safe and within expected directory structure."""
    try:
        # Check that path starts with expected base path structure
        expected_prefixes = [
            NAS_BASE_PATH + "/Inputs",
            NAS_BASE_PATH + "/Outputs"
        ]
        return any(path.startswith(prefix) for prefix in expected_prefixes)
    except (ValueError, TypeError) as e:
        return False


def connect_to_nas() -> Optional[SMBConnection]:
    """Connect to NAS using NTLM authentication."""
    global logger, error_logger
    
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            logger.info("Successfully connected to NAS")
            return conn
        else:
            error_msg = "Failed to connect to NAS"
            logger.error(error_msg)
            error_logger.log_processing_error("system", error_msg)
            return None
            
    except Exception as e:
        error_msg = f"NAS connection failed: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("system", error_msg)
        return None


def setup_ssl_certificate(nas_conn: SMBConnection) -> bool:
    """Download and setup SSL certificate from NAS."""
    global logger, error_logger, ssl_cert_path
    
    try:
        # Use config path like Stage 4
        cert_nas_path = config.get('ssl_cert_nas_path', 'Inputs/certificate/certificate.cer')
        
        # Create temporary certificate file
        temp_cert_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".cer",
            prefix="stage_05_qa_pairing_cert_",
            delete=False
        )
        
        # Download certificate from NAS
        nas_conn.retrieveFile(NAS_SHARE_NAME, cert_nas_path, temp_cert_file)
        temp_cert_file.close()
        
        ssl_cert_path = temp_cert_file.name
        os.environ['SSL_CERT_FILE'] = ssl_cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
        
        logger.info(f"SSL certificate setup complete: {ssl_cert_path}")
        return True
        
    except Exception as e:
        error_msg = f"SSL certificate setup failed: {e}"
        logger.error(error_msg)
        error_logger.log_authentication_error(error_msg)
        return False


def load_config_from_nas(nas_conn: SMBConnection) -> bool:
    """Load configuration from NAS."""
    global logger, error_logger, config
    
    try:
        config_nas_path = f"{NAS_BASE_PATH}/Inputs/config/config.json"
        
        if not validate_nas_path(config_nas_path):
            error_msg = "Invalid config path"
            logger.error(error_msg)
            error_logger.log_processing_error("system", error_msg)
            return False
        
        # Download config from NAS
        config_file = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, config_nas_path, config_file)
        config_file.seek(0)
        
        config = json.loads(config_file.read().decode('utf-8'))
        logger.info("Successfully loaded configuration from NAS")
        
        # Validate stage_05_qa_pairing configuration exists
        if "stage_05_qa_pairing" not in config:
            error_msg = "stage_05_qa_pairing configuration section missing"
            logger.error(error_msg)
            error_logger.log_processing_error("system", error_msg)
            return False
            
        return True
        
    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error("system", error_msg)
        return False


def group_records_by_transcript(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by transcript for processing."""
    transcripts = defaultdict(list)
    
    for record in records:
        # Create transcript key using filename or event identifiers
        transcript_key = record.get("filename", f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
        transcripts[transcript_key].append(record)
    
    return dict(transcripts)


def group_records_by_speaker_block(records: List[Dict]) -> List[Dict]:
    """Group records by speaker blocks within a transcript."""
    speaker_blocks = defaultdict(list)
    
    # Group paragraphs by speaker block
    for record in records:
        speaker_block_id = record["speaker_block_id"]
        speaker_blocks[speaker_block_id].append(record)
    
    # Create speaker block objects
    speaker_block_list = []
    for block_id in sorted(speaker_blocks.keys()):
        paragraphs = speaker_blocks[block_id]
        
        # Use first paragraph for speaker block metadata
        first_paragraph = paragraphs[0]
        
        speaker_block = {
            "speaker_block_id": block_id,
            "speaker": first_paragraph["speaker"],
            "question_answer_flag": first_paragraph.get("question_answer_flag"),
            "section_type": first_paragraph.get("section_type"),
            "paragraphs": paragraphs
        }
        
        speaker_block_list.append(speaker_block)
    
    return speaker_block_list


def create_speaker_block_window(current_block_index: int, 
                               speaker_blocks: List[Dict],
                               qa_state: Dict = None) -> List[Dict]:
    """
    Create context window around current speaker block with dynamic extension.
    
    For question end decisions: Include all blocks back to question start
    For other decisions: Use standard window size
    """
    window_config = config["stage_05_qa_pairing"]["window_config"]
    context_before = window_config["context_blocks_before"]
    context_after = window_config["context_blocks_after"]
    
    # Dynamic extension for question continuation decisions
    if qa_state and qa_state.get("extends_question_start"):
        question_start_index = qa_state.get("question_start_index", 0)
        start_index = max(0, question_start_index)
    else:
        # Standard window
        start_index = max(0, current_block_index - context_before)
    
    end_index = min(len(speaker_blocks), current_block_index + context_after + 1)
    
    return speaker_blocks[start_index:end_index]


def format_speaker_block_context(window_blocks: List[Dict], current_block_index: int) -> str:
    """
    Format speaker blocks with clear context boundaries for LLM analysis.
    """
    context_sections = []
    current_block_id = window_blocks[current_block_index]["speaker_block_id"]
    
    # 1. PRIOR BLOCK CONTEXT
    prior_blocks = [b for b in window_blocks if b["speaker_block_id"] < current_block_id]
    if prior_blocks:
        context_sections.append("=== PRIOR SPEAKER BLOCK CONTEXT ===")
        for block in prior_blocks:
            context_sections.append(format_single_speaker_block(block))
    
    # 2. CURRENT BLOCK CONTEXT (DECISION POINT)
    current_block = next(b for b in window_blocks if b["speaker_block_id"] == current_block_id)
    context_sections.append("=== CURRENT SPEAKER BLOCK (DECISION POINT) ===")
    context_sections.append(f"**ANALYZE THIS BLOCK FOR Q&A BOUNDARY DECISION:**")
    context_sections.append(format_single_speaker_block(current_block))
    
    # 3. FUTURE BLOCK CONTEXT
    future_blocks = [b for b in window_blocks if b["speaker_block_id"] > current_block_id]
    if future_blocks:
        context_sections.append("=== FUTURE SPEAKER BLOCK CONTEXT ===")
        for block in future_blocks:
            context_sections.append(format_single_speaker_block(block))
    
    return "\n\n".join(context_sections)


def format_single_speaker_block(block: Dict) -> str:
    """Format a single speaker block with full paragraph content and clear structure."""
    paragraphs_text = []
    
    for para in block["paragraphs"]:
        # Clean formatting with clear paragraph separation
        content = para['paragraph_content'].strip()
        paragraphs_text.append(f"    • {content}")
    
    # Extract speaker role information for context
    speaker = block['speaker']
    xml_role = block.get('question_answer_flag', 'general')
    section_type = block.get('section_type', 'unknown')
    
    # Determine speaker type context
    speaker_context = ""
    if "analyst" in speaker.lower():
        speaker_context = " [ANALYST - typically asks questions]"
    elif any(title in speaker.lower() for title in ["ceo", "cfo", "president", "chief", "executive"]):
        speaker_context = " [EXECUTIVE - typically provides answers]"
    elif "operator" in speaker.lower():
        speaker_context = " [OPERATOR - manages call flow]"
    
    return f"""
SPEAKER BLOCK {block['speaker_block_id']}:
  Speaker: {speaker}{speaker_context}
  XML Role: {xml_role}
  Section Type: {section_type}
  Content:
{chr(10).join(paragraphs_text)}"""


def create_qa_boundary_prompt(formatted_context: str, current_block_id: int, qa_state: Dict = None) -> str:
    """
    Create state-driven prompt for Q&A boundary detection with clear decision options.
    """
    # Extract state information
    current_group_active = qa_state.get("group_active", False) if qa_state else False
    current_group_id = qa_state.get("current_qa_group_id") if qa_state else None
    last_status = qa_state.get("last_decision_status") if qa_state else None
    
    # Determine valid options based on current state
    if current_group_active:
        valid_options = "CONTINUE current group OR END current group"
        primary_objective = f"Decide if current block continues or ends Q&A group {current_group_id}"
    else:
        valid_options = "START new group OR mark as STANDALONE (operator/non-Q&A)"
        primary_objective = "Decide if current block starts a new Q&A exchange"
    
    # Create state context
    state_context = f"""**CURRENT STATE**:
- Q&A Group Active: {"YES" if current_group_active else "NO"}
- Active Group ID: {current_group_id if current_group_id else "None"}
- Last Decision: {last_status if last_status else "None"}
- Your Options: {valid_options}"""
    
    return f"""**CONTEXT**: You are analyzing earnings call speaker blocks to capture complete analyst-responder exchanges. Focus on speaker patterns to determine conversation boundaries.

{state_context}

**OBJECTIVE**: {primary_objective}

{formatted_context}

**DECISION LOGIC**:

**Speaker Pattern Analysis**:
- **ANALYST speakers**: Usually start new Q&A exchanges with questions
- **EXECUTIVE speakers**: Usually respond to analyst questions  
- **OPERATOR speakers**: Manage call flow - always mark as STANDALONE

**Exchange Completion Rules**:
1. Capture ENTIRE analyst question → executive response → any follow-up → closing
2. An exchange ends when: 
   - Executive finishes responding AND next speaker is different analyst
   - Executive finishes responding AND operator speaks
   - Analyst says brief thanks/closing after getting response
3. Continue exchange when:
   - Same executive continues multi-part answer
   - Same analyst asks follow-up to same executive
   - Brief acknowledgments within ongoing conversation

**Current Block Analysis**:
- Look at CURRENT speaker role and content
- Consider PRIOR speaker context for conversation flow
- Consider NEXT speaker to avoid cutting off mid-exchange

**Response Requirements**:
- Brief reasoning (max 100 chars): focus on speaker pattern
- Examples: "Analyst starts Q", "Exec continues answer", "Exchange complete", "Operator manages"
- High confidence (0.8+) for clear patterns, lower for ambiguous cases"""


# Q&A Boundary Detection Schema
def create_qa_boundary_detection_tools(qa_state: Dict = None):
    """Create dynamic function calling tools based on current Q&A state."""
    
    # Extract state information
    current_group_active = qa_state.get("group_active", False) if qa_state else False
    current_group_id = qa_state.get("current_qa_group_id") if qa_state else None
    
    # Create dynamic enum based on current state
    if current_group_active:
        # If group is active, can only continue or end
        status_enum = ["group_continue", "group_end", "standalone"]
        description = f"Continue or end active Q&A group {current_group_id}, or mark operator as standalone"
    else:
        # If no active group, can only start new or mark standalone
        status_enum = ["group_start", "standalone"]
        description = "Start new Q&A group or mark operator/non-Q&A as standalone"
    
    # Remove qa_group_id from LLM schema - we'll assign it automatically
    # LLM only decides the action, system assigns the ID
    
    return [
        {
            "type": "function",
            "function": {
                "name": "analyze_speaker_block_boundaries",
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "qa_group_decision": {
                            "type": "object",
                            "properties": {
                                "current_block_id": {"type": "integer"},
                                "group_status": {
                                    "type": "string",
                                    "enum": status_enum
                                },
                                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "reasoning": {
                                    "type": "string",
                                    "maxLength": 100,
                                    "description": "Brief reasoning: speaker role and conversation flow"
                                }
                            },
                            "required": ["current_block_id", "group_status", "confidence_score", "reasoning"]
                        }
                    },
                    "required": ["qa_group_decision"]
                }
            }
        }
    ]


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost based on token usage and configured rates."""
    prompt_cost_per_1k = config["stage_05_qa_pairing"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_05_qa_pairing"]["llm_config"]["cost_per_1k_completion_tokens"]
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": round(total_cost, 6)
    }


def is_operator_block(speaker_block: Dict) -> bool:
    """
    Detect if a speaker block is from an operator and should be excluded from Q&A assignments.
    Uses both speaker name and content pattern detection.
    """
    speaker = speaker_block.get("speaker", "").lower()
    
    # Check for operator indicators in speaker name
    speaker_indicators = ["operator", "conference", "moderator", "host", "call operator"]
    
    for indicator in speaker_indicators:
        if indicator in speaker:
            return True
    
    # Also check content for operator language patterns
    all_content = ""
    for para in speaker_block.get("paragraphs", []):
        content = para.get("paragraph_content", "")
        all_content += " " + content.lower()
    
    # Operator content patterns
    operator_content_patterns = [
        "next question", "our next question", "next caller", "next participant",
        "question comes from", "question is from", "caller is from",
        "please hold", "please stand by", "one moment please",
        "we have no further questions", "no more questions", "end of q&a",
        "concludes our question", "concludes the q&a", "end of our q&a"
    ]
    
    # If content contains operator patterns, it's likely an operator
    for pattern in operator_content_patterns:
        if pattern in all_content:
            return True
    
    # Special case: "thank you" + "next question" is operator language
    if "thank you" in all_content and any(pattern in all_content for pattern in ["next question", "next caller", "question comes from", "question is from"]):
        return True
    
    return False


def assign_group_id_to_decision(decision: Dict, qa_state: Dict, current_block_id: int) -> Dict:
    """
    Automatically assign appropriate group ID based on decision status and current state.
    """
    status = decision.get("group_status")
    current_group_active = qa_state.get("group_active", False)
    current_group_id = qa_state.get("current_qa_group_id")
    next_group_id = qa_state.get("next_group_id", 1)
    
    # Create enhanced decision with proper group ID
    enhanced_decision = decision.copy()
    
    if status == "group_start":
        # Assign next available group ID
        enhanced_decision["qa_group_id"] = next_group_id
        qa_state["next_group_id"] = next_group_id + 1
        qa_state["assigned_group_ids"].add(next_group_id)
        
    elif status in ["group_continue", "group_end"]:
        # Use current active group ID
        if current_group_active and current_group_id:
            enhanced_decision["qa_group_id"] = current_group_id
        else:
            # This shouldn't happen with proper validation, but handle gracefully
            enhanced_decision["qa_group_id"] = None
            
    elif status == "standalone":
        # Operator/non-Q&A blocks get null ID
        enhanced_decision["qa_group_id"] = None
    
    return enhanced_decision


def validate_llm_decision(decision: Dict, qa_state: Dict, current_block_id: int) -> str:
    """
    Validate LLM decision against current state. Returns error message if invalid, None if valid.
    Note: We no longer validate group IDs since they're auto-assigned.
    """
    current_group_active = qa_state.get("group_active", False)
    status = decision.get("group_status")
    
    # Validate status based on current state
    if current_group_active:
        # Active group: can only continue, end, or standalone
        if status not in ["group_continue", "group_end", "standalone"]:
            return f"Invalid status '{status}' for active group state"
    else:
        # No active group: can only start or standalone
        if status not in ["group_start", "standalone"]:
            return f"Invalid status '{status}' for no active group state"
    
    return None  # Valid decision


def retry_with_validation_feedback(current_block_index: int, speaker_blocks: List[Dict], 
                                 transcript_id: str, qa_state: Dict, validation_error: str) -> Optional[Dict]:
    """
    Retry LLM call with validation feedback. Single retry only.
    """
    global logger, error_logger, llm_client
    
    try:
        current_block = speaker_blocks[current_block_index]
        current_block_id = current_block["speaker_block_id"]
        
        logger.info(f"Block {current_block_id}: Retrying with validation feedback: {validation_error}")
        
        # Create context window
        window_blocks = create_speaker_block_window(current_block_index, speaker_blocks, qa_state)
        
        # Format context for LLM
        formatted_context = format_speaker_block_context(window_blocks, 
                                                       next(i for i, b in enumerate(window_blocks) 
                                                           if b["speaker_block_id"] == current_block_id))
        
        # Create prompt with validation feedback
        base_prompt = create_qa_boundary_prompt(formatted_context, current_block_id, qa_state)
        retry_prompt = f"""{base_prompt}
        
**VALIDATION ERROR FROM PREVIOUS ATTEMPT**: {validation_error}
**RETRY INSTRUCTION**: Please provide a decision that matches the current state requirements."""
        
        # Make retry LLM API call
        response = llm_client.chat.completions.create(
            model=config["stage_05_qa_pairing"]["llm_config"]["model"],
            messages=[{"role": "user", "content": retry_prompt}],
            tools=create_qa_boundary_detection_tools(qa_state),
            tool_choice="required",
            temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
            max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
        )
        
        # Parse retry response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            decision = result["qa_group_decision"]
            
            # Validate retry decision
            retry_validation_error = validate_llm_decision(decision, qa_state, current_block_id)
            if retry_validation_error:
                logger.error(f"Block {current_block_id}: Retry also failed validation - {retry_validation_error}")
                error_logger.log_boundary_error(transcript_id, current_block_id, f"Retry validation failed: {retry_validation_error}")
                return None
            
            # Auto-assign group ID for retry decision
            decision = assign_group_id_to_decision(decision, qa_state, current_block_id)
            
            logger.info(f"Block {current_block_id}: Retry successful - {decision['group_status']} (group: {decision.get('qa_group_id')})")
            return decision
        else:
            logger.error(f"Block {current_block_id}: No tool call in retry response")
            return None
            
    except Exception as e:
        error_msg = f"Retry failed for block {current_block_id}: {e}"
        logger.error(error_msg)
        error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
        return None


# Note: is_closing_pleasantries_block function removed - conflicts with operator detection
# Enhanced operator detection handles "thank you, next question" patterns


def analyze_speaker_block_boundary(current_block_index: int, 
                                 speaker_blocks: List[Dict],
                                 transcript_id: str,
                                 qa_state: Dict = None) -> Optional[Dict]:
    """
    Analyze a single speaker block for Q&A boundary decisions using LLM.
    Handles operator blocks with special exclusion logic.
    """
    global logger, error_logger, llm_client
    
    try:
        current_block = speaker_blocks[current_block_index]
        current_block_id = current_block["speaker_block_id"]
        
        # Check if this is an operator block - exclude from Q&A assignments
        if is_operator_block(current_block):
            logger.info(f"Block {current_block_id} identified as operator block - excluding from Q&A groups")
            return {
                "current_block_id": current_block_id,
                "qa_group_id": None,  # No Q&A group assignment
                "qa_group_start_block": None,
                "qa_group_end_block": None,
                "group_status": "standalone",  # Operator blocks are standalone
                "confidence_score": 1.0,  # High confidence in operator detection
                "reasoning": f"Operator block detected (speaker: {current_block.get('speaker', 'unknown')}) - excluded from Q&A assignments"
            }
        
        # Note: Removed closing pleasantries detection as it conflicts with operator detection
        # Enhanced operator detection above should catch operator "thank you" statements
        
        # Create context window for non-operator blocks
        window_blocks = create_speaker_block_window(current_block_index, speaker_blocks, qa_state)
        
        # Format context for LLM
        formatted_context = format_speaker_block_context(window_blocks, 
                                                       next(i for i, b in enumerate(window_blocks) 
                                                           if b["speaker_block_id"] == current_block_id))
        
        # Create prompt with state awareness
        prompt = create_qa_boundary_prompt(formatted_context, current_block_id, qa_state)
        
        # Make LLM API call with dynamic tools
        response = llm_client.chat.completions.create(
            model=config["stage_05_qa_pairing"]["llm_config"]["model"],
            messages=[{"role": "user", "content": prompt}],
            tools=create_qa_boundary_detection_tools(qa_state),
            tool_choice="required",
            temperature=config["stage_05_qa_pairing"]["llm_config"]["temperature"],
            max_tokens=config["stage_05_qa_pairing"]["llm_config"]["max_tokens"]
        )
        
        # Parse and validate response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            decision = result["qa_group_decision"]
            
            # Validate decision against current state
            validation_error = validate_llm_decision(decision, qa_state, current_block_id)
            if validation_error:
                logger.warning(f"Block {current_block_id}: Invalid LLM decision - {validation_error}")
                # Retry with corrected prompt (implement single retry)
                return retry_with_validation_feedback(current_block_index, speaker_blocks, transcript_id, qa_state, validation_error)
            
            # Automatically assign appropriate group ID
            decision = assign_group_id_to_decision(decision, qa_state, current_block_id)
            
            # Log token usage and cost
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                cost_info = calculate_token_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "cost": cost_info
                }
                logger.info(f"Block {current_block_id} tokens - input: {response.usage.prompt_tokens}, output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}, cost: ${cost_info['total_cost']:.4f}")
                
                # Accumulate costs for final summary
                error_logger.accumulate_costs(token_usage)
            
            logger.info(f"Block {current_block_id} boundary decision: {decision['group_status']} (group: {decision.get('qa_group_id')}, confidence: {decision['confidence_score']:.2f}) | Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
            
            return decision
        else:
            error_msg = f"No tool call response for block {current_block_id}"
            logger.error(error_msg)
            error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
            return None
            
    except Exception as e:
        error_msg = f"LLM boundary analysis failed for block {current_block_id}: {e}"
        logger.error(error_msg)
        error_logger.log_boundary_error(transcript_id, current_block_id, error_msg)
        return None


def calculate_qa_group_confidence(block_decisions: List[Dict]) -> float:
    """
    Calculate aggregated confidence for Q&A group spanning multiple blocks.
    
    Strategy: Use weighted average based on decision criticality
    - Group start/end decisions: Higher weight (0.4 each)
    - Group continuation decisions: Lower weight (0.2 / num_continuations)
    """
    if len(block_decisions) == 1:
        return block_decisions[0]["confidence_score"]
    
    weighted_confidence = 0.0
    total_weight = 0.0
    
    continuation_count = len([d for d in block_decisions if d["group_status"] == "group_continue"])
    
    for decision in block_decisions:
        if decision["group_status"] in ["group_start", "group_end"]:
            weight = 0.4  # Critical boundary decisions
        else:
            weight = 0.2 / max(1, continuation_count)  # Distribute continuation weight
        
        weighted_confidence += decision["confidence_score"] * weight
        total_weight += weight
    
    return round(weighted_confidence / total_weight, 3)


def determine_qa_group_method(block_decisions: List[Dict]) -> str:
    """Determine method based on decision quality and confidence."""
    avg_confidence = calculate_qa_group_confidence(block_decisions)
    
    if avg_confidence >= 0.8:
        return "llm_detection"
    elif avg_confidence >= 0.6:
        return "llm_detection_medium_confidence"
    else:
        return "xml_fallback"


def detect_decision_inconsistencies(decisions: List[Dict]) -> List[str]:
    """Detect logical inconsistencies in Q&A group assignments with awareness of real-time corrections."""
    issues = []
    
    # Track state machine progression
    current_qa_group = None
    group_state = "none"  # none, active, ended
    auto_corrections_detected = 0
    
    for i, decision in enumerate(decisions):
        block_id = decision["current_block_id"]
        status = decision["group_status"]
        qa_group_id = decision.get("qa_group_id")
        
        # Skip operator blocks (they should be standalone)
        if status == "standalone" and qa_group_id is None:
            continue
            
        # State machine validation with awareness of real-time corrections
        if status == "group_start":
            if group_state == "active" and current_qa_group is not None:
                # This indicates a real-time correction was made - note it but don't flag as error
                auto_corrections_detected += 1
                # Force-end the previous group to match real-time behavior
                group_state = "ended"
            current_qa_group = qa_group_id
            group_state = "active"
            
        elif status == "group_continue":
            if group_state == "ended":
                # This might be a continue after we force-ended a group
                # Convert the state back to active to match the real correction
                group_state = "active"
                current_qa_group = qa_group_id
            elif group_state != "active":
                # Only flag as error if it's truly inconsistent
                issues.append(f"Block {block_id}: Continuing group {qa_group_id} but no active group (state: {group_state})")
            elif qa_group_id != current_qa_group:
                # This might be a corrected continue->start conversion
                auto_corrections_detected += 1
                current_qa_group = qa_group_id
                
        elif status == "group_end":
            if group_state != "active":
                # Only flag if it's a genuine inconsistency (not a corrected case)
                if group_state == "none":
                    issues.append(f"Block {block_id}: Ending group {qa_group_id} but no group was ever started")
            elif qa_group_id != current_qa_group:
                # This might be ending a different group due to corrections
                auto_corrections_detected += 1
            
            group_state = "ended"
            current_qa_group = qa_group_id  # Track the group that was ended
                
        # Check for multiple consecutive group ends (this is still a real issue)
        if i > 0:
            prev_decision = decisions[i-1]
            if (prev_decision["group_status"] == "group_end" and 
                status == "group_end" and 
                prev_decision.get("qa_group_id") == qa_group_id):
                issues.append(f"Block {block_id}: Multiple consecutive group_end for same group {qa_group_id}")
    
    # Log auto-corrections as info, not errors
    if auto_corrections_detected > 0:
        issues.append(f"INFO: {auto_corrections_detected} real-time auto-corrections were applied during processing")
    
    return issues


def validate_qa_group_completeness(qa_groups: List[Dict], speaker_blocks: List[Dict]) -> List[str]:
    """Ensure each Q&A group has logical question-answer patterns."""
    issues = []
    
    for group in qa_groups:
        group_blocks = [b for b in speaker_blocks 
                       if group["start_block_id"] <= b["speaker_block_id"] <= group["end_block_id"]]
        
        # Check for presence of both questions and answers based on speaker patterns
        has_analyst = any("analyst" in block.get("speaker", "").lower() for block in group_blocks)
        has_executive = any("executive" in block.get("speaker", "").lower() or 
                           any(title in block.get("speaker", "").lower() 
                               for title in ["ceo", "cfo", "president", "chief"]) 
                           for block in group_blocks)
        
        if has_analyst and not has_executive:
            issues.append(f"Group {group['qa_group_id']} has analyst but no executive response")
        elif has_executive and not has_analyst:
            issues.append(f"Group {group['qa_group_id']} has executive but no analyst question")
    
    return issues


def apply_xml_fallback_grouping(speaker_blocks: List[Dict]) -> List[Dict]:
    """
    Fallback Q&A grouping using XML type attributes when LLM analysis fails.
    """
    global logger
    
    logger.info("Applying XML fallback grouping for Q&A boundary detection")
    
    qa_groups = []
    current_group = []
    group_id = 1
    
    for block in speaker_blocks:
        xml_type = block.get("question_answer_flag")
        
        if xml_type == "question":
            # Start new group if we have an existing group
            if current_group:
                qa_groups.append({
                    "qa_group_id": group_id,
                    "start_block_id": current_group[0]["speaker_block_id"],
                    "end_block_id": current_group[-1]["speaker_block_id"],
                    "confidence": 0.5,  # Lower confidence for fallback
                    "method": "xml_fallback",
                    "blocks": current_group
                })
                group_id += 1
            
            current_group = [block]
        
        elif xml_type == "answer" and current_group:
            # Add to current group
            current_group.append(block)
        
        else:
            # Handle blocks without clear XML type
            if current_group:
                current_group.append(block)
    
    # Add final group if exists
    if current_group:
        qa_groups.append({
            "qa_group_id": group_id,
            "start_block_id": current_group[0]["speaker_block_id"],
            "end_block_id": current_group[-1]["speaker_block_id"],
            "confidence": 0.5,
            "method": "xml_fallback",
            "blocks": current_group
        })
    
    return qa_groups


def process_qa_boundaries_with_fallbacks(speaker_blocks: List[Dict], transcript_id: str) -> List[Dict]:
    """
    Main Q&A boundary processing with comprehensive fallback handling.
    """
    global logger, error_logger
    
    try:
        # Filter to only Q&A sections
        qa_speaker_blocks = [block for block in speaker_blocks 
                           if block.get("section_type") == "Investor Q&A"]
        
        if not qa_speaker_blocks:
            logger.info(f"No Q&A sections found in transcript {transcript_id}")
            return []
        
        logger.info(f"Processing {len(qa_speaker_blocks)} Q&A speaker blocks")
        
        # Process each speaker block for boundary decisions with enhanced state tracking
        all_decisions = []
        qa_state = {
            "current_qa_group_id": None,
            "group_active": False,
            "last_decision_status": None,
            "last_block_id": None,
            "question_start_index": None,
            "extends_question_start": False,
            "next_group_id": 1,  # Auto-increment group ID counter
            "assigned_group_ids": set()  # Track used IDs for validation
        }
        
        for i, block in enumerate(qa_speaker_blocks):
            # Add look-ahead context for better end decisions
            next_block = qa_speaker_blocks[i + 1] if i + 1 < len(qa_speaker_blocks) else None
            qa_state["next_block"] = next_block
            qa_state["current_block_index"] = i
            
            decision = analyze_speaker_block_boundary(i, qa_speaker_blocks, transcript_id, qa_state)
            
            if decision:
                all_decisions.append(decision)
                
                # Update comprehensive QA state tracking
                block_id = decision["current_block_id"]
                status = decision["group_status"]
                group_id = decision.get("qa_group_id")
                
                # Real-time state validation and updates
                if status == "group_start":
                    # Validate: can't start if group already active
                    if qa_state["group_active"]:
                        logger.error(f"Block {block_id}: Cannot start group {group_id} - group {qa_state['current_qa_group_id']} already active")
                        # Force end previous group first
                        qa_state["group_active"] = False
                        logger.warning(f"Block {block_id}: Force-ended previous group {qa_state['current_qa_group_id']}")
                    
                    qa_state["current_qa_group_id"] = group_id
                    qa_state["group_active"] = True
                    qa_state["question_start_index"] = i
                    qa_state["extends_question_start"] = True
                    logger.info(f"Block {block_id}: Started Q&A group {group_id}")
                    
                elif status == "group_continue":
                    # Validate: must have active group
                    if not qa_state["group_active"]:
                        logger.error(f"Block {block_id}: Cannot continue group {group_id} - no active group")
                        # Convert to group start
                        qa_state["current_qa_group_id"] = group_id
                        qa_state["group_active"] = True
                        logger.warning(f"Block {block_id}: Converted continue to start for group {group_id}")
                    elif qa_state["current_qa_group_id"] != group_id:
                        logger.error(f"Block {block_id}: Group ID mismatch - continuing {group_id} but active is {qa_state['current_qa_group_id']}")
                    
                elif status == "group_end":
                    # Validate: must have active group to end
                    if not qa_state["group_active"]:
                        logger.warning(f"Block {block_id}: Cannot end group {group_id} - no active group")
                    elif qa_state["current_qa_group_id"] != group_id:
                        logger.warning(f"Block {block_id}: Ending group {group_id} but active is {qa_state['current_qa_group_id']}")
                    
                    qa_state["group_active"] = False
                    qa_state["extends_question_start"] = False
                    logger.info(f"Block {block_id}: Ended Q&A group {group_id}")
                    
                elif status == "standalone":
                    # Operator blocks don't change group state
                    logger.debug(f"Block {block_id}: Standalone block (operator/non-Q&A)")
                
                # Update last decision tracking
                qa_state["last_decision_status"] = status
                qa_state["last_block_id"] = block_id
                
                logger.debug(f"Block {block_id} state update: active={qa_state['group_active']}, current_group={qa_state['current_qa_group_id']}, status={status}")
                
            else:
                # LLM analysis failed, use XML fallback
                logger.warning(f"LLM analysis failed for block {block['speaker_block_id']}, falling back to XML grouping")
                return apply_xml_fallback_grouping(qa_speaker_blocks)
        
        # Validate decision consistency with awareness of real-time corrections
        consistency_issues = detect_decision_inconsistencies(all_decisions)
        if consistency_issues:
            # Separate INFO messages from actual errors
            actual_errors = [issue for issue in consistency_issues if not issue.startswith("INFO:")]
            info_messages = [issue for issue in consistency_issues if issue.startswith("INFO:")]
            
            # Log INFO messages as info, not warnings
            for info_msg in info_messages:
                logger.info(info_msg)
            
            # Only treat actual errors as warnings and trigger fallback
            if actual_errors:
                logger.warning(f"Decision inconsistencies detected: {actual_errors}")
                error_logger.log_validation_error(transcript_id, f"Inconsistent decisions: {actual_errors}")
                return apply_xml_fallback_grouping(qa_speaker_blocks)
            else:
                logger.info("All detected inconsistencies were auto-corrected during real-time processing")
        
        # Group decisions into Q&A groups
        qa_groups = []
        current_group_decisions = []
        current_group_id = None
        
        for decision in all_decisions:
            if decision["group_status"] == "group_start":
                # Finalize previous group
                if current_group_decisions:
                    qa_groups.append(finalize_qa_group(current_group_decisions, qa_speaker_blocks))
                
                # Start new group
                current_group_decisions = [decision]
                current_group_id = decision["qa_group_id"]
                
            elif decision["group_status"] == "standalone":
                # Skip standalone blocks (operators) - they don't belong to any group
                continue
                
            elif decision["qa_group_id"] == current_group_id:
                current_group_decisions.append(decision)
            
            else:
                # Handle group ID mismatch (real issues only)
                decision_group_id = decision.get("qa_group_id")
                decision_status = decision.get("group_status")
                logger.warning(f"Group ID mismatch in decision sequence: decision has group {decision_group_id} (status: {decision_status}) but current group is {current_group_id}")
                current_group_decisions.append(decision)
        
        # Finalize last group
        if current_group_decisions:
            qa_groups.append(finalize_qa_group(current_group_decisions, qa_speaker_blocks))
        
        # Validate group completeness
        completeness_issues = validate_qa_group_completeness(qa_groups, qa_speaker_blocks)
        if completeness_issues:
            logger.warning(f"Q&A completeness issues: {completeness_issues}")
            error_logger.log_validation_error(transcript_id, f"Completeness issues: {completeness_issues}")
        
        logger.info(f"Successfully identified {len(qa_groups)} Q&A groups")
        return qa_groups
        
    except Exception as e:
        error_msg = f"Q&A boundary processing failed: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error(transcript_id, error_msg)
        return apply_xml_fallback_grouping(speaker_blocks)


def finalize_qa_group(group_decisions: List[Dict], speaker_blocks: List[Dict]) -> Dict:
    """Create final Q&A group from boundary decisions."""
    
    # Get block range
    block_ids = [d["current_block_id"] for d in group_decisions]
    start_block_id = min(block_ids)
    end_block_id = max(block_ids)
    
    # Calculate aggregated confidence and method
    confidence = calculate_qa_group_confidence(group_decisions)
    method = determine_qa_group_method(group_decisions)
    
    # Get group ID (should be consistent across decisions)
    qa_group_id = group_decisions[0]["qa_group_id"]
    
    return {
        "qa_group_id": qa_group_id,
        "start_block_id": start_block_id,
        "end_block_id": end_block_id,
        "confidence": confidence,
        "method": method,
        "block_decisions": group_decisions
    }


def apply_qa_assignments_to_records(records: List[Dict], qa_groups: List[Dict]) -> List[Dict]:
    """
    Apply Q&A group assignments to paragraph records.
    Only adds: qa_group_id, qa_group_confidence, qa_group_method
    """
    
    # Create mapping from speaker block to Q&A group
    block_to_qa_map = {}
    
    for group in qa_groups:
        for block_id in range(group["start_block_id"], group["end_block_id"] + 1):
            block_to_qa_map[block_id] = {
                "qa_group_id": group["qa_group_id"],
                "qa_group_confidence": group["confidence"],
                "qa_group_method": group["method"]
            }
    
    # Apply to records
    enhanced_records = []
    
    for record in records:
        enhanced_record = record.copy()
        
        # Only apply Q&A assignments to Q&A section records
        if record.get("section_type") == "Investor Q&A":
            speaker_block_id = record["speaker_block_id"]
            
            if speaker_block_id in block_to_qa_map:
                qa_info = block_to_qa_map[speaker_block_id]
                enhanced_record.update(qa_info)
            else:
                # No Q&A group assignment
                enhanced_record.update({
                    "qa_group_id": None,
                    "qa_group_confidence": None,
                    "qa_group_method": None
                })
        else:
            # Non-Q&A sections don't get Q&A assignments
            enhanced_record.update({
                "qa_group_id": None,
                "qa_group_confidence": None,
                "qa_group_method": None
            })
        
        enhanced_records.append(enhanced_record)
    
    return enhanced_records


def process_transcript_qa_pairing(transcript_records: List[Dict], transcript_id: str) -> List[Dict]:
    """
    Process a single transcript for Q&A pairing with per-transcript OAuth refresh.
    """
    global logger, error_logger
    
    try:
        # Refresh OAuth token for each transcript
        if not refresh_oauth_token_for_transcript():
            logger.error(f"Failed to refresh OAuth token for transcript {transcript_id}")
            error_logger.log_authentication_error(f"Token refresh failed for {transcript_id}")
            return transcript_records  # Return original records without Q&A assignments
        
        logger.info(f"Processing Q&A pairing for transcript: {transcript_id}")
        
        # Group records by speaker blocks
        speaker_blocks = group_records_by_speaker_block(transcript_records)
        
        # Process Q&A boundaries
        qa_groups = process_qa_boundaries_with_fallbacks(speaker_blocks, transcript_id)
        
        # Apply Q&A assignments to records
        enhanced_records = apply_qa_assignments_to_records(transcript_records, qa_groups)
        
        logger.info(f"Completed Q&A pairing for transcript {transcript_id}: {len(qa_groups)} groups identified")
        
        return enhanced_records, len(qa_groups)
        
    except Exception as e:
        error_msg = f"Transcript Q&A pairing failed: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error(transcript_id, error_msg)
        return transcript_records, 0  # Return original records on failure


def upload_logs_to_nas(nas_conn: SMBConnection, logger: logging.Logger, error_logger: EnhancedErrorLogger):
    """Upload execution logs to NAS."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Upload main log file
        log_filename = f"stage_05_qa_pairing_qa_pairing_execution_log_{timestamp}.log"
        log_nas_path = f"{NAS_BASE_PATH}/Outputs/Logs/{log_filename}"
        
        with open(logger.temp_log_file, 'rb') as log_file:
            nas_conn.storeFile(NAS_SHARE_NAME, log_nas_path, log_file)
        
        logger.info(f"Uploaded execution log: {log_filename}")
        
        # Upload error logs
        error_logger.save_error_logs(nas_conn)
        
        # Clean up temp log file
        os.unlink(logger.temp_log_file)
        
    except Exception as e:
        logger.error(f"Failed to upload logs: {e}")


def main():
    """Main execution function for Stage 5 Q&A Pairing."""
    global logger, error_logger, config, llm_client, ssl_cert_path
    
    try:
        # Setup logging
        logger = setup_logging()
        error_logger = EnhancedErrorLogger()
        
        # Start timing
        start_time = datetime.now()
        
        logger.info("="*60)
        logger.info("STAGE 5: Q&A PAIRING SYSTEM")
        logger.info("="*60)
        logger.info(f"Execution started at: {start_time.isoformat()}")
        
        # Connect to NAS
        nas_conn = connect_to_nas()
        if not nas_conn:
            raise Exception("Failed to connect to NAS")
        
        # Load configuration
        if not load_config_from_nas(nas_conn):
            raise Exception("Configuration loading failed")
        
        # Setup SSL certificate
        if not setup_ssl_certificate(nas_conn):
            raise Exception("SSL certificate setup failed")
        
        # Setup LLM client
        llm_client = setup_llm_client()
        if not llm_client:
            raise Exception("LLM client setup failed")
        
        # Load input data
        stage_05_qa_pairing_config = config["stage_05_qa_pairing"]
        input_file_path = f"{NAS_BASE_PATH}/{stage_05_qa_pairing_config['input_source']}"
        
        logger.info(f"Loading input data from: {input_file_path}")
        
        input_file = io.BytesIO()
        nas_conn.retrieveFile(NAS_SHARE_NAME, input_file_path, input_file)
        input_file.seek(0)
        
        input_data = json.loads(input_file.read().decode('utf-8'))
        records = input_data["records"]
        
        logger.info(f"Loaded {len(records)} records from Stage 4 output")
        
        # Group records by transcript
        transcripts = group_records_by_transcript(records)
        logger.info(f"Processing {len(transcripts)} transcripts")
        
        # Apply development mode limits if enabled
        if stage_05_qa_pairing_config.get("dev_mode", False):
            max_transcripts = stage_05_qa_pairing_config.get("dev_max_transcripts", 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            logger.info(f"Development mode: Limited to {len(transcripts)} transcripts")
        
        # Process each transcript
        all_enhanced_records = []
        processed_count = 0
        total_qa_groups_count = 0  # Track total Q&A groups across all transcripts
        per_file_metrics = []  # Track per-file metrics
        
        for transcript_id, transcript_records in transcripts.items():
            transcript_start_time = datetime.now()
            logger.info(f"Processing transcript {processed_count + 1}/{len(transcripts)}: {transcript_id}")
            
            # Capture initial cost state
            initial_cost = error_logger.total_cost
            initial_tokens = error_logger.total_tokens
            
            enhanced_records, qa_groups_count = process_transcript_qa_pairing(transcript_records, transcript_id)
            all_enhanced_records.extend(enhanced_records)
            total_qa_groups_count += qa_groups_count
            
            # Calculate per-file metrics
            transcript_end_time = datetime.now()
            transcript_time = transcript_end_time - transcript_start_time
            transcript_cost = error_logger.total_cost - initial_cost
            transcript_tokens = error_logger.total_tokens - initial_tokens
            
            per_file_metrics.append({
                "transcript": transcript_id,
                "processing_time": transcript_time,
                "cost": transcript_cost,
                "tokens": transcript_tokens,
                "qa_groups": qa_groups_count,
                "records": len(enhanced_records)
            })
            
            logger.info(f"Transcript {transcript_id} completed - Time: {transcript_time}, Cost: ${transcript_cost:.4f}, Tokens: {transcript_tokens:,}, Q&A Groups: {qa_groups_count}")
            
            processed_count += 1
            
            # Add delay between transcripts
            time.sleep(1)
        
        # Create output data
        output_data = {
            "schema_version": "1.0",
            "processing_timestamp": datetime.now().isoformat(),
            "total_records": len(all_enhanced_records),
            "total_transcripts_processed": processed_count,
            "qa_pairing_summary": {
                "transcripts_with_qa_groups": len([t for t in transcripts.keys() 
                                                 if any(r.get("qa_group_id") is not None 
                                                       for r in all_enhanced_records
                                                       if r.get("filename", "").startswith(t.split("_")[0]))]),
                "total_qa_groups": total_qa_groups_count
            },
            "records": all_enhanced_records
        }
        
        # Save output to NAS
        output_filename = stage_05_qa_pairing_config["output_file"]
        output_path = f"{NAS_BASE_PATH}/{stage_05_qa_pairing_config['output_path']}/{output_filename}"
        
        logger.info(f"Saving output to: {output_path}")
        
        output_json = json.dumps(output_data, indent=2)
        output_bytes = io.BytesIO(output_json.encode('utf-8'))
        
        nas_conn.storeFile(NAS_SHARE_NAME, output_path, output_bytes)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Final summary
        logger.info("="*60)
        logger.info("STAGE 5 Q&A PAIRING COMPLETE")
        logger.info("="*60)
        logger.info(f"Transcripts processed: {processed_count}")
        logger.info(f"Total records processed: {len(all_enhanced_records)}")
        logger.info(f"Transcripts with Q&A groups: {output_data['qa_pairing_summary']['transcripts_with_qa_groups']}")
        logger.info(f"Total Q&A groups identified: {output_data['qa_pairing_summary']['total_qa_groups']}")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Total tokens used: {error_logger.total_tokens:,}")
        logger.info(f"Total LLM cost: ${error_logger.total_cost:.4f}")
        if error_logger.total_tokens > 0:
            avg_cost_per_1k_tokens = (error_logger.total_cost / error_logger.total_tokens) * 1000
            logger.info(f"Average cost per 1K tokens: ${avg_cost_per_1k_tokens:.4f}")
        
        # Add per-file averages
        if per_file_metrics:
            avg_time_per_file = sum(m["processing_time"].total_seconds() for m in per_file_metrics) / len(per_file_metrics)
            avg_cost_per_file = sum(m["cost"] for m in per_file_metrics) / len(per_file_metrics)
            logger.info(f"Average time per transcript: {avg_time_per_file:.1f} seconds")
            logger.info(f"Average cost per transcript: ${avg_cost_per_file:.4f}")
        
        logger.info(f"Output file: {output_filename}")
        logger.info("="*60)
        
        # Upload logs
        upload_logs_to_nas(nas_conn, logger, error_logger)
        
        # Clean up
        if ssl_cert_path and os.path.exists(ssl_cert_path):
            os.unlink(ssl_cert_path)
        
        nas_conn.close()
        
        logger.info("Stage 5 Q&A Pairing completed successfully")
        
    except Exception as e:
        error_msg = f"Stage 5 execution failed: {e}"
        if logger:
            logger.error(error_msg)
        if error_logger:
            error_logger.log_processing_error("system", error_msg)
        raise


if __name__ == "__main__":
    main()