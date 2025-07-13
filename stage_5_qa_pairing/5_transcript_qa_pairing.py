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
        prefix=f'stage_5_qa_pairing_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
                filename = f"stage_5_qa_pairing_{error_type}_errors_{timestamp}.json"
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
        token_endpoint = config["stage_5"]["llm_config"]["token_endpoint"]
        
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
        llm_config = config["stage_5"]["llm_config"]
        
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
            domain=PROXY_DOMAIN,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if conn.connect(NAS_SERVER_IP, NAS_PORT, timeout=30):
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
        cert_nas_path = f"{NAS_BASE_PATH}/Inputs/certificate/certificate.cer"
        
        if not validate_nas_path(cert_nas_path):
            error_msg = "Invalid SSL certificate path"
            logger.error(error_msg)
            error_logger.log_authentication_error(error_msg)
            return False
        
        # Create temporary certificate file
        temp_cert_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".cer",
            prefix="stage_5_cert_",
            delete=False
        )
        
        # Download certificate from NAS
        nas_conn.retrieveFile(NAS_SHARE_NAME, cert_nas_path, temp_cert_file)
        temp_cert_file.close()
        
        ssl_cert_path = temp_cert_file.name
        os.environ['SSL_CERT_FILE'] = ssl_cert_path
        
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
        
        # Validate stage_5 configuration exists
        if "stage_5" not in config:
            error_msg = "stage_5 configuration section missing"
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
    window_config = config["stage_5"]["window_config"]
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


def create_qa_boundary_prompt(formatted_context: str, current_block_id: int) -> str:
    """
    Create sophisticated CO-STAR prompt for Q&A boundary detection.
    """
    return f"""**CONTEXT**: You are analyzing earnings call speaker blocks to determine Q&A conversation boundaries. Each speaker block contains one person's complete statement with their role and content clearly marked.

**OBJECTIVE**: For the current speaker block (marked as DECISION POINT), determine:
1. Does this block start a new Q&A group?
2. Does this block continue an existing Q&A group?  
3. Does this block end the current Q&A group?
4. What is the complete Q&A group span (start_block_id to end_block_id)?

**STYLE**: Analyze complete speaker blocks as units. Pay attention to:
- Speaker roles: [ANALYST] typically ask questions, [EXECUTIVE] typically answer
- XML Role indicators: "question" vs "answer" vs "general"
- Content flow: Look for question/answer patterns and natural conversation breaks
- Speaker transitions: Changes between analyst→executive or executive→analyst

**TONE**: Conservative and methodical. When uncertain about boundaries, prefer to keep related exchanges together rather than split them. Focus on speaker transitions and content flow.

**AUDIENCE**: Financial analysts who need accurate question-answer relationship mapping for research and analysis.

**RESPONSE**: Use the analyze_speaker_block_boundaries function with structured output.

{formatted_context}

**ANALYSIS INSTRUCTIONS**:
- Pay close attention to speaker role contexts ([ANALYST], [EXECUTIVE], [OPERATOR])
- Consider XML Role indicators as additional guidance ("question", "answer", "general")
- Look for natural conversation breaks vs. continuation patterns
- Evaluate if current block completes a logical Q&A exchange
- Use future context to determine if answer continues beyond current block
- Assign confidence based on clarity of speaker patterns and content flow
- For multi-part questions or answers, keep all parts in the same Q&A group
- Consider follow-up questions as part of extended Q&A groups
- Operator interventions usually signal transitions between Q&A groups"""


# Q&A Boundary Detection Schema
qa_boundary_detection_schema = {
    "name": "analyze_speaker_block_boundaries",
    "description": "Analyze speaker block boundaries for Q&A group assignment",
    "parameters": {
        "type": "object",
        "properties": {
            "qa_group_decision": {
                "type": "object",
                "properties": {
                    "current_block_id": {"type": "integer"},
                    "qa_group_id": {"type": "integer"},
                    "qa_group_start_block": {"type": "integer"},
                    "qa_group_end_block": {"type": "integer"},
                    "group_status": {
                        "type": "string",
                        "enum": ["group_start", "group_continue", "group_end", "standalone"]
                    },
                    "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                    "continue_to_next_block": {"type": "boolean"}
                },
                "required": ["current_block_id", "qa_group_id", "group_status", "confidence_score", "continue_to_next_block"]
            }
        },
        "required": ["qa_group_decision"]
    }
}


def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> dict:
    """Calculate cost based on token usage and configured rates."""
    prompt_cost_per_1k = config["stage_5"]["llm_config"]["cost_per_1k_prompt_tokens"]
    completion_cost_per_1k = config["stage_5"]["llm_config"]["cost_per_1k_completion_tokens"]
    
    prompt_cost = (prompt_tokens / 1000) * prompt_cost_per_1k
    completion_cost = (completion_tokens / 1000) * completion_cost_per_1k
    total_cost = prompt_cost + completion_cost
    
    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": round(total_cost, 6)
    }


def analyze_speaker_block_boundary(current_block_index: int, 
                                 speaker_blocks: List[Dict],
                                 transcript_id: str,
                                 qa_state: Dict = None) -> Optional[Dict]:
    """
    Analyze a single speaker block for Q&A boundary decisions using LLM.
    """
    global logger, error_logger, llm_client
    
    try:
        current_block = speaker_blocks[current_block_index]
        current_block_id = current_block["speaker_block_id"]
        
        # Create context window
        window_blocks = create_speaker_block_window(current_block_index, speaker_blocks, qa_state)
        
        # Format context for LLM
        formatted_context = format_speaker_block_context(window_blocks, 
                                                       next(i for i, b in enumerate(window_blocks) 
                                                           if b["speaker_block_id"] == current_block_id))
        
        # Create prompt
        prompt = create_qa_boundary_prompt(formatted_context, current_block_id)
        
        # Make LLM API call
        response = llm_client.chat.completions.create(
            model=config["stage_5"]["llm_config"]["model"],
            messages=[{"role": "user", "content": prompt}],
            tools=[qa_boundary_detection_schema],
            tool_choice={"type": "function", "function": {"name": "analyze_speaker_block_boundaries"}},
            temperature=config["stage_5"]["llm_config"]["temperature"],
            max_tokens=config["stage_5"]["llm_config"]["max_tokens"]
        )
        
        # Parse response
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            decision = result["qa_group_decision"]
            
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
            
            logger.info(f"Block {current_block_id} boundary decision: {decision['group_status']} (confidence: {decision['confidence_score']:.2f}) | Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
            
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
    """Detect logical inconsistencies in Q&A group assignments."""
    issues = []
    
    # Check for impossible sequences
    for i in range(len(decisions) - 1):
        current = decisions[i]
        next_decision = decisions[i + 1]
        
        # Group end followed by same group continuation
        if (current["group_status"] == "group_end" and 
            next_decision.get("qa_group_id") == current["qa_group_id"] and
            next_decision["group_status"] in ["group_continue"]):
            issues.append(f"Block {current['current_block_id']} ends group but block {next_decision['current_block_id']} continues same group")
        
        # Group start without proper sequence
        if (current["group_status"] == "group_start" and 
            next_decision.get("qa_group_id") == current["qa_group_id"] and
            next_decision["group_status"] not in ["group_continue", "group_end"]):
            issues.append(f"Block {current['current_block_id']} starts group but next block has invalid status")
    
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
        
        # Process each speaker block for boundary decisions
        all_decisions = []
        qa_state = {}
        
        for i, block in enumerate(qa_speaker_blocks):
            decision = analyze_speaker_block_boundary(i, qa_speaker_blocks, transcript_id, qa_state)
            
            if decision:
                all_decisions.append(decision)
                
                # Update QA state for dynamic window extension
                if decision["group_status"] == "group_start":
                    qa_state["question_start_index"] = i
                    qa_state["extends_question_start"] = True
                elif decision["group_status"] == "group_end":
                    qa_state["extends_question_start"] = False
            else:
                # LLM analysis failed, use XML fallback
                logger.warning(f"LLM analysis failed for block {block['speaker_block_id']}, falling back to XML grouping")
                return apply_xml_fallback_grouping(qa_speaker_blocks)
        
        # Validate decision consistency
        consistency_issues = detect_decision_inconsistencies(all_decisions)
        if consistency_issues:
            logger.warning(f"Decision inconsistencies detected: {consistency_issues}")
            error_logger.log_validation_error(transcript_id, f"Inconsistent decisions: {consistency_issues}")
            return apply_xml_fallback_grouping(qa_speaker_blocks)
        
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
                
            elif decision["qa_group_id"] == current_group_id:
                current_group_decisions.append(decision)
            
            else:
                # Handle group ID mismatch
                logger.warning(f"Group ID mismatch in decision sequence")
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
        
        return enhanced_records
        
    except Exception as e:
        error_msg = f"Transcript Q&A pairing failed: {e}"
        logger.error(error_msg)
        error_logger.log_processing_error(transcript_id, error_msg)
        return transcript_records  # Return original records on failure


def upload_logs_to_nas(nas_conn: SMBConnection, logger: logging.Logger, error_logger: EnhancedErrorLogger):
    """Upload execution logs to NAS."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Upload main log file
        log_filename = f"stage_5_qa_pairing_execution_log_{timestamp}.log"
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
        
        logger.info("="*60)
        logger.info("STAGE 5: Q&A PAIRING SYSTEM")
        logger.info("="*60)
        logger.info(f"Execution started at: {datetime.now().isoformat()}")
        
        # Connect to NAS
        nas_conn = connect_to_nas()
        if not nas_conn:
            raise Exception("Failed to connect to NAS")
        
        # Setup SSL certificate
        if not setup_ssl_certificate(nas_conn):
            raise Exception("SSL certificate setup failed")
        
        # Load configuration
        if not load_config_from_nas(nas_conn):
            raise Exception("Configuration loading failed")
        
        # Setup LLM client
        llm_client = setup_llm_client()
        if not llm_client:
            raise Exception("LLM client setup failed")
        
        # Load input data
        stage_5_config = config["stage_5"]
        input_file_path = f"{NAS_BASE_PATH}/{stage_5_config['input_source']}"
        
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
        if stage_5_config.get("dev_mode", False):
            max_transcripts = stage_5_config.get("dev_max_transcripts", 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            logger.info(f"Development mode: Limited to {len(transcripts)} transcripts")
        
        # Process each transcript
        all_enhanced_records = []
        processed_count = 0
        
        for transcript_id, transcript_records in transcripts.items():
            logger.info(f"Processing transcript {processed_count + 1}/{len(transcripts)}: {transcript_id}")
            
            enhanced_records = process_transcript_qa_pairing(transcript_records, transcript_id)
            all_enhanced_records.extend(enhanced_records)
            
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
                "total_qa_groups": len(set(r.get("qa_group_id") for r in all_enhanced_records 
                                         if r.get("qa_group_id") is not None))
            },
            "records": all_enhanced_records
        }
        
        # Save output to NAS
        output_filename = stage_5_config["output_file"]
        output_path = f"{NAS_BASE_PATH}/{stage_5_config['output_path']}/{output_filename}"
        
        logger.info(f"Saving output to: {output_path}")
        
        output_json = json.dumps(output_data, indent=2)
        output_bytes = io.BytesIO(output_json.encode('utf-8'))
        
        nas_conn.storeFile(NAS_SHARE_NAME, output_path, output_bytes)
        
        # Final summary
        logger.info("="*60)
        logger.info("STAGE 5 EXECUTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total transcripts processed: {processed_count}")
        logger.info(f"Total records processed: {len(all_enhanced_records)}")
        logger.info(f"Transcripts with Q&A groups: {output_data['qa_pairing_summary']['transcripts_with_qa_groups']}")
        logger.info(f"Total Q&A groups identified: {output_data['qa_pairing_summary']['total_qa_groups']}")
        logger.info(f"Output file: {output_filename}")
        logger.info("")
        logger.info("TOKEN USAGE & COST SUMMARY:")
        logger.info(f"Total tokens used: {error_logger.total_tokens:,}")
        logger.info(f"Total LLM cost: ${error_logger.total_cost:.4f}")
        if error_logger.total_tokens > 0:
            avg_cost_per_1k_tokens = (error_logger.total_cost / error_logger.total_tokens) * 1000
            logger.info(f"Average cost per 1K tokens: ${avg_cost_per_1k_tokens:.4f}")
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