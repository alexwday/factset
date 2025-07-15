"""
Stage 1: Daily Transcript Sync
Downloads new/updated earnings transcripts from recent date range for monitored financial institutions.
Self-contained standalone script that loads config from NAS at runtime.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
import os
from urllib.parse import quote
from datetime import datetime, timedelta
import time
import requests
import logging
import re
import json
import tempfile
import io
import xml.etree.ElementTree as ET
from smb.SMBConnection import SMBConnection
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')
PROXY_USER = os.getenv('PROXY_USER')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
PROXY_URL = os.getenv('PROXY_URL')
NAS_USERNAME = os.getenv('NAS_USERNAME')
NAS_PASSWORD = os.getenv('NAS_PASSWORD')
NAS_SERVER_IP = os.getenv('NAS_SERVER_IP')
NAS_SERVER_NAME = os.getenv('NAS_SERVER_NAME')
NAS_SHARE_NAME = os.getenv('NAS_SHARE_NAME')
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH')
NAS_PORT = int(os.getenv('NAS_PORT', 445))
CONFIG_PATH = os.getenv('CONFIG_PATH')
CLIENT_MACHINE_NAME = os.getenv('CLIENT_MACHINE_NAME')
PROXY_DOMAIN = os.getenv('PROXY_DOMAIN', 'MAPLE')

# Validate required environment variables
required_env_vars = [
    'API_USERNAME', 'API_PASSWORD', 'PROXY_USER', 'PROXY_PASSWORD', 'PROXY_URL',
    'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
    'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Constants will be loaded from config

# No default config - script must load from NAS

# Global variables for configuration
config = {}
logger = None
error_logger = None

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        suffix='.log', 
        prefix=f'stage_1_daily_sync_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
        delete=False
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_file.name),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.temp_log_file = temp_log_file.name
    return logger

class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""
    def __init__(self):
        self.parsing_errors = []
        self.download_errors = []
        self.filesystem_errors = []
        self.validation_errors = []
        
    def log_parsing_error(self, ticker: str, filename: str, title: str, error: str):
        """Log parsing errors with actionable information."""
        self.parsing_errors.append({
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'filename': filename,
            'title': title,
            'error': error,
            'location': 'Data/Unknown/Unknown/',
            'action_required': 'Check title format and manually move to correct quarter/year'
        })
        
    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Logs", "Errors")
        nas_create_directory(nas_conn, error_base_path)
        
        # Save each error type if errors exist
        error_types = [
            ('parsing_errors', self.parsing_errors),
            ('download_errors', self.download_errors),
            ('filesystem_errors', self.filesystem_errors),
            ('validation_errors', self.validation_errors)
        ]
        
        summary = {
            'run_timestamp': timestamp,
            'total_errors': sum(len(errors) for _, errors in error_types),
            'errors_by_type': {error_type: len(errors) for error_type, errors in error_types}
        }
        
        for error_type, errors in error_types:
            if errors:
                filename = f"{error_type}_{timestamp}.json"
                file_path = nas_path_join(error_base_path, filename)
                content = json.dumps({
                    'summary': summary,
                    'errors': errors
                }, indent=2)
                file_obj = io.BytesIO(content.encode('utf-8'))
                nas_upload_file(nas_conn, file_obj, file_path)
                logger.warning(f"Saved {len(errors)} {error_type} to {filename}")

def sanitize_url_for_logging(url: str) -> str:
    """Sanitize URL for logging by removing query parameters and auth tokens."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        # Return only scheme, netloc, and path - remove query params and fragments
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
            is_direct_tcp=True
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

def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate configuration schema and parameters."""
    global logger
    
    # Define required configuration structure
    required_structure = {
        'stage_1': {
            'sync_date_range': int,
            'description': str
        },
        'api_settings': {
            'request_delay': (int, float),
            'max_retries': int,
            'retry_delay': (int, float),
            'use_exponential_backoff': bool,
            'max_backoff_delay': (int, float),
            'transcript_types': list
        },
        'monitored_institutions': dict,
        'ssl_cert_nas_path': str
    }
    
    # Validate top-level structure
    for top_key, top_value in required_structure.items():
        if top_key not in config:
            raise ValueError(f"Missing required configuration section: {top_key}")
        
        if isinstance(top_value, dict):
            # Validate nested structure
            for nested_key, expected_type in top_value.items():
                if nested_key not in config[top_key]:
                    raise ValueError(f"Missing required configuration parameter: {top_key}.{nested_key}")
                
                actual_value = config[top_key][nested_key]
                if isinstance(expected_type, tuple):
                    if not isinstance(actual_value, expected_type):
                        raise ValueError(f"Invalid type for {top_key}.{nested_key}: expected {expected_type}, got {type(actual_value)}")
                else:
                    if not isinstance(actual_value, expected_type):
                        raise ValueError(f"Invalid type for {top_key}.{nested_key}: expected {expected_type}, got {type(actual_value)}")
        else:
            # Validate direct parameter
            if not isinstance(config[top_key], top_value):
                raise ValueError(f"Invalid type for {top_key}: expected {top_value}, got {type(config[top_key])}")
    
    # Validate specific parameter values
    try:
        # Validate sync_date_range is non-negative integer
        if not isinstance(config['stage_1']['sync_date_range'], int) or config['stage_1']['sync_date_range'] < 0:
            raise ValueError("sync_date_range must be a non-negative integer")
    except KeyError as e:
        raise ValueError(f"Missing required stage_1 configuration: {e}")
    
    # Validate API settings ranges
    if config['api_settings']['request_delay'] < 0:
        raise ValueError("request_delay must be non-negative")
    
    if config['api_settings']['max_retries'] < 1:
        raise ValueError("max_retries must be at least 1")
    
    if config['api_settings']['retry_delay'] < 0:
        raise ValueError("retry_delay must be non-negative")
    
    # Validate exponential backoff settings
    if 'use_exponential_backoff' in config['api_settings']:
        if config['api_settings']['use_exponential_backoff'] and config['api_settings'].get('max_backoff_delay', 0) <= 0:
            raise ValueError("max_backoff_delay must be positive when exponential backoff is enabled")
    
    # Validate transcript types
    if not config['api_settings']['transcript_types']:
        raise ValueError("transcript_types cannot be empty")
    
    valid_transcript_types = ['Raw', 'Corrected', 'NearRealTime']
    for transcript_type in config['api_settings']['transcript_types']:
        if transcript_type not in valid_transcript_types:
            raise ValueError(f"Invalid transcript type: {transcript_type}. Must be one of: {valid_transcript_types}")
    
    # Validate monitored institutions structure
    if not config['monitored_institutions']:
        raise ValueError("monitored_institutions cannot be empty")
    
    for ticker, institution_info in config['monitored_institutions'].items():
        if not isinstance(institution_info, dict):
            raise ValueError(f"Invalid institution info for {ticker}: must be a dictionary")
        
        required_fields = ['name', 'type', 'path_safe_name']
        for field in required_fields:
            if field not in institution_info:
                raise ValueError(f"Missing required field '{field}' for institution {ticker}")
            if not isinstance(institution_info[field], str):
                raise ValueError(f"Field '{field}' for institution {ticker} must be a string")
        
        # Validate institution type
        valid_types = [
            'Canadian', 'US', 'European', 'Insurance',
            'US_Regional', 'Nordic', 'Australian', 'US_Asset_Manager',
            'US_Boutique', 'Canadian_Asset_Manager', 'UK_Asset_Manager',
            'Canadian_Monoline', 'US_Trust'
        ]
        if institution_info['type'] not in valid_types:
            raise ValueError(f"Invalid institution type for {ticker}: {institution_info['type']}. Must be one of: {valid_types}")
    
    # Validate SSL certificate path
    if not config['ssl_cert_nas_path'].strip():
        raise ValueError("ssl_cert_nas_path cannot be empty")
    
    if not validate_nas_path(config['ssl_cert_nas_path']):
        raise ValueError(f"Invalid SSL certificate path: {config['ssl_cert_nas_path']}")
    
    # Validate ticker formats for security
    for ticker in config['monitored_institutions'].keys():
        if not re.match(r'^[A-Z0-9.-]+$', ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")
    
    logger.info("Configuration validation successful")

def load_stage_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate shared configuration from NAS."""
    global logger
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)
        
        if config_data:
            stage_config = json.loads(config_data.decode('utf-8'))
            logger.info("Successfully loaded shared configuration from NAS")
            
            # Validate configuration schema and parameters
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

def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    return '/'.join(str(part) for part in parts if part)

def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    global logger
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except Exception as e:
        # Log debug info but don't treat as error - file might just not exist
        logger.debug(f"File existence check failed for {file_path}: {e}")
        return False

def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""
    global logger
    
    # Normalize and validate path
    normalized_path = dir_path.strip('/').rstrip('/')
    if not normalized_path:
        logger.error("Cannot create directory with empty path")
        return False
    
    # Split path into components
    path_parts = [part for part in normalized_path.split('/') if part]
    if not path_parts:
        logger.error("Cannot create directory with invalid path")
        return False
    
    # Build path incrementally from root
    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part
        
        # Check if directory exists
        if nas_file_exists(conn, current_path):
            continue
        
        # Try to create directory
        try:
            conn.createDirectory(NAS_SHARE_NAME, current_path)
            logger.debug(f"Created directory: {current_path}")
        except Exception as e:
            # If it fails and doesn't exist, it's a real error
            if not nas_file_exists(conn, current_path):
                logger.error(f"Failed to create directory {current_path}: {e}")
                return False
    
    return True

def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    global logger
    
    # Validate path for security
    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return False
    
    try:
        parent_dir = '/'.join(nas_file_path.split('/')[:-1])
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
    
    # Validate path for security
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

def nas_list_files(conn: SMBConnection, directory_path: str) -> List[str]:
    """List XML files in a NAS directory."""
    global logger
    try:
        files = conn.listPath(NAS_SHARE_NAME, directory_path)
        return [file_info.filename for file_info in files 
                if not file_info.isDirectory and file_info.filename.endswith('.xml')]
    except Exception as e:
        logger.debug(f"Failed to list files in {directory_path}: {e}")
        return []

def nas_list_directories(conn: SMBConnection, directory_path: str) -> List[str]:
    """List subdirectories in a NAS directory."""
    global logger
    try:
        files = conn.listPath(NAS_SHARE_NAME, directory_path)
        return [file_info.filename for file_info in files 
                if file_info.isDirectory and file_info.filename not in ['.', '..']]
    except Exception as e:
        logger.debug(f"Failed to list directories in {directory_path}: {e}")
        return []

def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False
    
    # Check for directory traversal attacks
    if '..' in path or path.startswith('/'):
        return False
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    if any(char in path for char in invalid_chars):
        return False
    
    # Check path length
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False
    
    return True

def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure."""
    if not path or not isinstance(path, str):
        return False
    
    # Ensure path is relative and safe
    normalized = path.strip('/')
    parts = normalized.split('/')
    
    for part in parts:
        if not part or part in ['.', '..']:
            return False
        if not validate_file_path(part):
            return False
    
    return True

def validate_api_response_structure(response) -> bool:
    """Validate API response structure and content."""
    global logger
    
    if not response or not hasattr(response, 'data'):
        logger.warning("API response missing data attribute")
        return False
    
    if not isinstance(response.data, list):
        logger.warning("API response data is not a list")
        return False
    
    for i, transcript in enumerate(response.data):
        transcript_dict = transcript.to_dict()
        
        # Validate required fields
        required_fields = ['primary_ids', 'event_type', 'transcripts_link']
        for field in required_fields:
            if field not in transcript_dict:
                logger.warning(f"Transcript {i} missing required field: {field}")
                return False
        
        # Validate data types
        if not isinstance(transcript_dict.get('primary_ids'), list):
            logger.warning(f"Transcript {i} has invalid primary_ids type")
            return False
        
        # Validate URL format
        url = transcript_dict.get('transcripts_link')
        if url and not isinstance(url, str):
            logger.warning(f"Transcript {i} has invalid transcripts_link type")
            return False
        
        if url and not url.startswith(('http://', 'https://')):
            logger.warning(f"Transcript {i} has invalid URL format: {url}")
            return False
    
    return True

def sanitize_for_filename(text: Any) -> str:
    """Sanitize text to be safe for filename."""
    if pd.isna(text) or text is None:
        return "unknown"
    
    clean_text = str(text)
    clean_text = re.sub(r'[<>:"/\\|?*]', '_', clean_text)
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    clean_text = re.sub(r'[-\s]+', '_', clean_text)
    return clean_text.strip('_')

def parse_quarter_and_year_from_xml(xml_content: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Parse quarter and fiscal year from transcript XML title."""
    try:
        # Parse only until we find the title
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        
        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return None, None
            
        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return None, None
            
        title = title_elem.text.strip()
        
        # Try multiple patterns in order of likelihood
        patterns = [
            r"Q([1-4])\s+(20\d{2})\s+Earnings\s+Call",  # "Q1 2024 Earnings Call"
            r".*Q([1-4])\s+(20\d{2})\s+Earnings\s+Call.*",  # Anywhere in title
            r"(First|Second|Third|Fourth)\s+Quarter\s+(20\d{2})",  # "First Quarter 2024"
            r"(20\d{2})\s+Q([1-4])",  # "2024 Q1"
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                if i == 2:  # "(First|Second|Third|Fourth)\s+Quarter\s+(20\d{2})"
                    quarter_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
                    quarter = quarter_map.get(match.group(1).lower())
                    year = match.group(2)
                    return quarter, year
                elif i == 3:  # "(20\d{2})\s+Q([1-4])"
                    year = match.group(1)
                    quarter = f"Q{match.group(2)}"
                    return quarter, year
                else:  # Standard Q([1-4])\s+(20\d{2}) patterns
                    quarter = f"Q{match.group(1)}"
                    year = match.group(2)
                    return quarter, year
        
        # Final fallback: Find Q and year separately
        quarter_match = re.search(r"Q([1-4])", title)
        year_match = re.search(r"(20\d{2})", title)
        if quarter_match and year_match:
            return f"Q{quarter_match.group(1)}", year_match.group(2)
            
        return None, None
        
    except Exception as e:
        logger.error(f"Error parsing XML for quarter/year: {e}")
        return None, None

def validate_path_length(path: str) -> bool:
    """Validate path length for Windows compatibility."""
    # Windows has 260 character path limit
    if len(path) > 250:  # Leave some buffer
        logger.warning(f"Path length ({len(path)}) approaching Windows limit: {path}")
        return False
    return True

def get_fallback_quarter_year() -> Tuple[str, str]:
    """Get fallback quarter/year when parsing fails."""
    # Don't use current date as it's meaningless for fiscal quarters
    # Instead, use a special "Unknown" folder that operators can manually sort
    return "Unknown", "Unknown"

def create_enhanced_directory_structure(nas_conn: SMBConnection, ticker: str, 
                                      institution_type: str, transcript_type: str,
                                      quarter: str, year: str) -> Optional[str]:
    """Create enhanced directory structure with fiscal year and quarter."""
    global config
    # Build path: Data/YYYY/QX/Type/Company/TranscriptType/
    institution_info = config['monitored_institutions'].get(ticker, {})
    path_safe_name = institution_info.get('path_safe_name', ticker)
    
    # Shorten company folder name if needed for path length
    ticker_folder_name = f"{ticker}_{path_safe_name}"
    if len(ticker_folder_name) > 50:  # Arbitrary limit to keep paths short
        ticker_folder_name = ticker  # Just use ticker if name too long
    
    path_components = [
        NAS_BASE_PATH, "Outputs", "Data",
        year, quarter, institution_type,
        ticker_folder_name, transcript_type
    ]
    
    full_path = nas_path_join(*path_components)
    
    # Validate path length
    if not validate_path_length(full_path):
        # Try with just ticker
        path_components[-2] = ticker
        full_path = nas_path_join(*path_components)
        if not validate_path_length(full_path):
            logger.error(f"Path too long even with shortened name: {full_path}")
            return None
    
    nas_create_directory(nas_conn, full_path)
    return full_path

def create_base_directory_structure(nas_conn: SMBConnection) -> None:
    """Create the base directory structure for the transcript repository on NAS."""
    global logger
    logger.info("Creating base directory structure on NAS...")
    
    inputs_path = nas_path_join(NAS_BASE_PATH, "Inputs")
    outputs_path = nas_path_join(NAS_BASE_PATH, "Outputs")
    
    nas_create_directory(nas_conn, inputs_path)
    nas_create_directory(nas_conn, outputs_path)
    
    certificate_path = nas_path_join(inputs_path, "certificate")
    config_path = nas_path_join(inputs_path, "config")
    data_path = nas_path_join(outputs_path, "Data")
    logs_path = nas_path_join(outputs_path, "Logs")
    
    nas_create_directory(nas_conn, certificate_path)
    nas_create_directory(nas_conn, config_path)
    nas_create_directory(nas_conn, data_path)
    nas_create_directory(nas_conn, logs_path)
    
    # Enhanced structure creates type folders dynamically within fiscal year/quarter folders
    # No need to pre-create legacy type-based folders

# Removed create_ticker_directory_structure - replaced by enhanced directory structure

def create_filename(transcript_data: Dict[str, Any], target_ticker: Optional[str] = None) -> str:
    """Create standardized filename from transcript data."""
    global logger
    try:
        if target_ticker:
            primary_id = sanitize_for_filename(target_ticker)
        else:
            primary_id = sanitize_for_filename(
                transcript_data.get('primary_ids', ['unknown'])[0] 
                if transcript_data.get('primary_ids') else 'unknown'
            )
        
        event_date = transcript_data.get('event_date')
        if pd.notna(event_date):
            try:
                event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                logger.warning(f"Date parsing failed for {event_date}: {e}")
                event_date_str = str(event_date)
        else:
            event_date_str = 'unknown_date'
        
        event_type = sanitize_for_filename(transcript_data.get('event_type', 'unknown'))
        transcript_type = sanitize_for_filename(transcript_data.get('transcript_type', 'unknown'))
        event_id = sanitize_for_filename(transcript_data.get('event_id', 'unknown'))
        report_id = sanitize_for_filename(transcript_data.get('report_id', 'unknown'))
        version_id = sanitize_for_filename(transcript_data.get('version_id', 'unknown'))
        
        return f"{primary_id}_{event_date_str}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml"
        
    except Exception as e:
        logger.error(f"Error creating filename: {e}")
        return f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"

def create_version_agnostic_key(transcript_data: Dict[str, Any], target_ticker: Optional[str] = None) -> str:
    """Create version-agnostic key for duplicate detection (excludes version_id)."""
    global logger
    try:
        if target_ticker:
            primary_id = sanitize_for_filename(target_ticker)
        else:
            primary_id = sanitize_for_filename(
                transcript_data.get('primary_ids', ['unknown'])[0] 
                if transcript_data.get('primary_ids') else 'unknown'
            )
        
        event_date = transcript_data.get('event_date')
        if pd.notna(event_date):
            try:
                event_date_str = pd.to_datetime(event_date).strftime('%Y-%m-%d')
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime) as e:
                logger.warning(f"Date parsing failed for {event_date}: {e}")
                event_date_str = str(event_date)
        else:
            event_date_str = 'unknown_date'
        
        event_type = sanitize_for_filename(transcript_data.get('event_type', 'unknown'))
        transcript_type = sanitize_for_filename(transcript_data.get('transcript_type', 'unknown'))
        event_id = sanitize_for_filename(transcript_data.get('event_id', 'unknown'))
        report_id = sanitize_for_filename(transcript_data.get('report_id', 'unknown'))
        
        # Version-agnostic key excludes version_id
        return f"{primary_id}_{event_date_str}_{event_type}_{transcript_type}_{event_id}_{report_id}"
        
    except Exception as e:
        logger.error(f"Error creating version-agnostic key: {e}")
        return f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def parse_version_from_filename(filename: str) -> Optional[int]:
    """Extract version number from filename. Returns None if not found or invalid."""
    global logger
    try:
        # Remove .xml extension
        basename = filename.replace('.xml', '')
        # Split by underscores and get the last part (version_id)
        parts = basename.split('_')
        if len(parts) >= 6:  # Should have at least 6 parts based on our naming convention
            version_part = parts[-1]
            # Try to convert to integer
            return int(version_part)
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(f"Could not parse version from filename {filename}: {e}")
    
    return None

def get_version_agnostic_key_from_filename(filename: str) -> str:
    """Extract version-agnostic key from existing filename."""
    try:
        # Remove .xml extension
        basename = filename.replace('.xml', '')
        # Split by underscores and remove the last part (version_id)
        parts = basename.split('_')
        if len(parts) >= 6:
            return '_'.join(parts[:-1])  # All parts except the last one (version_id)
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(f"Could not extract version-agnostic key from filename {filename}: {e}")
    
    return basename  # Return as-is if parsing fails

def nas_remove_file(conn: SMBConnection, nas_file_path: str) -> bool:
    """Remove a file from NAS."""
    global logger
    
    # Validate path for security
    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path for removal: {nas_file_path}")
        return False
    
    try:
        conn.deleteFiles(NAS_SHARE_NAME, nas_file_path)
        logger.info(f"Removed old version from NAS: {nas_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove file from NAS {nas_file_path}: {e}")
        return False

def get_existing_files(nas_conn: SMBConnection, ticker: str, transcript_type: str, institution_type: str) -> Set[str]:
    """Get list of existing transcript files for a specific ticker and type from NAS."""
    global config
    # Get path-safe name for the ticker folder
    institution_info = config['monitored_institutions'].get(ticker, {})
    path_safe_name = institution_info.get('path_safe_name', ticker)
    ticker_folder_name = f"{ticker}_{path_safe_name}"
    
    institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Data", institution_type, ticker_folder_name, transcript_type)
    return set(nas_list_files(nas_conn, institution_path))

def get_existing_files_with_version_management(nas_conn: SMBConnection, ticker: str, transcript_type: str, institution_type: str) -> Dict[str, Dict[str, Any]]:
    """Get existing files with version management - keeps only latest version of each transcript."""
    global config, logger
    
    # Get path-safe name for the ticker folder
    institution_info = config['monitored_institutions'].get(ticker, {})
    path_safe_name = institution_info.get('path_safe_name', ticker)
    ticker_folder_name = f"{ticker}_{path_safe_name}"
    
    # Search across ALL quarterly folders in enhanced structure
    data_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Data")
    all_files = []
    
    # Get all years and quarters from enhanced structure
    file_paths = {}  # Track filename -> full_path mapping
    try:
        years = nas_list_directories(nas_conn, data_base_path)
        for year in years:
            if year == "Unknown":
                continue
            year_path = nas_path_join(data_base_path, year)
            quarters = nas_list_directories(nas_conn, year_path)
            for quarter in quarters:
                if quarter == "Unknown":
                    continue
                # Check this quarterly path for our institution
                institution_path = nas_path_join(year_path, quarter, institution_type, ticker_folder_name, transcript_type)
                quarterly_files = nas_list_files(nas_conn, institution_path)
                # Track full paths for each file
                for filename in quarterly_files:
                    file_paths[filename] = nas_path_join(institution_path, filename)
                all_files.extend(quarterly_files)
                logger.debug(f"Found {len(quarterly_files)} files in {year}/{quarter} for {ticker}")
    except Exception as e:
        logger.debug(f"Enhanced structure search failed for {ticker}, trying legacy structure: {e}")
        # Fallback to legacy structure if enhanced structure doesn't exist
        legacy_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Data", institution_type, ticker_folder_name, transcript_type)
        legacy_files = nas_list_files(nas_conn, legacy_path)
        for filename in legacy_files:
            file_paths[filename] = nas_path_join(legacy_path, filename)
        all_files = legacy_files
    
    # Group files by version-agnostic key
    version_groups = {}
    for filename in all_files:
        version_agnostic_key = get_version_agnostic_key_from_filename(filename)
        version_id = parse_version_from_filename(filename)
        
        if version_agnostic_key not in version_groups:
            version_groups[version_agnostic_key] = []
        
        version_groups[version_agnostic_key].append({
            'filename': filename,
            'version_id': version_id,
            'full_path': file_paths[filename]
        })
    
    # Keep only the latest version of each transcript
    latest_versions = {}
    files_to_remove = []
    
    for version_agnostic_key, files in version_groups.items():
        if len(files) == 1:
            # Only one version, keep it
            latest_versions[version_agnostic_key] = files[0]
        else:
            # Multiple versions, find the latest
            files_with_valid_versions = [f for f in files if f['version_id'] is not None]
            files_with_invalid_versions = [f for f in files if f['version_id'] is None]
            
            if files_with_valid_versions:
                # Sort by version_id descending and keep the highest
                latest_file = max(files_with_valid_versions, key=lambda x: x['version_id'])
                latest_versions[version_agnostic_key] = latest_file
                
                # Mark older versions for removal
                for file_info in files_with_valid_versions:
                    if file_info['version_id'] < latest_file['version_id']:
                        files_to_remove.append(file_info)
                        logger.info(f"Marking older version for removal: {file_info['filename']} (version {file_info['version_id']})")
                
                # Also remove files with invalid versions when we have valid ones
                files_to_remove.extend(files_with_invalid_versions)
            else:
                # No valid versions, keep the first one (arbitrary choice)
                latest_versions[version_agnostic_key] = files_with_invalid_versions[0]
                files_to_remove.extend(files_with_invalid_versions[1:])
    
    # Remove old versions from NAS
    for file_info in files_to_remove:
        nas_remove_file(nas_conn, file_info['full_path'])
    
    # Return a dictionary mapping version-agnostic keys to file info
    result = {}
    for version_agnostic_key, file_info in latest_versions.items():
        result[version_agnostic_key] = {
            'filename': file_info['filename'],
            'version_id': file_info['version_id']
        }
    
    return result

def download_transcript_with_enhanced_structure(nas_conn: SMBConnection, transcript_link: str,
                                              ticker: str, institution_type: str, 
                                              transcript_type: str, filename: str,
                                              configuration, proxy_user: str, 
                                              proxy_password: str) -> bool:
    """Download transcript with quarter/year parsing and enhanced folder structure."""
    global logger, config, error_logger
    
    for attempt in range(config['api_settings']['max_retries']):
        try:
            logger.info(f"Downloading from URL: {sanitize_url_for_logging(transcript_link)}")
            headers = {
                'Accept': 'application/xml,*/*',
                'Authorization': configuration.get_basic_auth_token(),
            }
            escaped_domain = quote(PROXY_DOMAIN + '\\' + proxy_user)
            proxy_url = f"http://{escaped_domain}:{quote(proxy_password)}@{PROXY_URL}"
            proxies = {
                'https': proxy_url,
                'http': proxy_url
            }
            
            response = requests.get(
                transcript_link,
                headers=headers,
                proxies=proxies,
                verify=configuration.ssl_ca_cert,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Parse XML to extract quarter and year
            quarter, year = parse_quarter_and_year_from_xml(response.content)
            
            if not quarter or not year:
                # Extract title for error logging
                try:
                    root = ET.parse(io.BytesIO(response.content)).getroot()
                    namespace = ""
                    if root.tag.startswith('{'):
                        namespace = root.tag.split('}')[0] + '}'
                    meta = root.find(f"{namespace}meta" if namespace else "meta")
                    title_elem = meta.find(f"{namespace}title" if namespace else "title") if meta else None
                    title = title_elem.text if title_elem and title_elem.text else "No title found"
                except:
                    title = "Failed to extract title"
                
                # Log parsing error
                error_logger.log_parsing_error(ticker, filename, title, 
                                             "Could not extract quarter/year from title")
                
                # Use fallback
                quarter, year = get_fallback_quarter_year()
                logger.warning(f"Using fallback location Unknown/Unknown for {filename}")
            
            # Create enhanced directory structure
            target_dir = create_enhanced_directory_structure(nas_conn, ticker, 
                                                           institution_type, 
                                                           transcript_type,
                                                           quarter, year)
            
            if not target_dir:
                error_logger.filesystem_errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename,
                    'error': 'Path too long for Windows compatibility'
                })
                return False
            
            # Upload file to new location
            file_path = nas_path_join(target_dir, filename)
            file_obj = io.BytesIO(response.content)
            file_size = len(response.content)
            
            if nas_upload_file(nas_conn, file_obj, file_path):
                logger.info(f"Successfully saved {filename} to {year}/{quarter} ({file_size:,} bytes)")
                return True
            else:
                error_logger.filesystem_errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename,
                    'path': file_path,
                    'error': 'Failed to upload to NAS'
                })
                return False
                
        except requests.exceptions.RequestException as e:
            if attempt < config['api_settings']['max_retries'] - 1:
                # Calculate delay with exponential backoff if enabled
                if config['api_settings'].get('use_exponential_backoff', False):
                    base_delay = config['api_settings']['retry_delay']
                    max_delay = config['api_settings'].get('max_backoff_delay', 120.0)
                    exponential_delay = base_delay * (2 ** attempt)
                    actual_delay = min(exponential_delay, max_delay)
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {actual_delay:.1f}s (exponential backoff): {e}")
                else:
                    actual_delay = config['api_settings']['retry_delay']
                    logger.warning(f"Download attempt {attempt + 1} failed, retrying in {actual_delay:.1f}s: {e}")
                
                time.sleep(actual_delay)
            else:
                error_logger.download_errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'filename': filename,
                    'url': sanitize_url_for_logging(transcript_link),
                    'error': str(e)
                })
                logger.error(f"Failed to download {filename} after {attempt + 1} attempts: {e}")
                return False
        except Exception as e:
            error_logger.download_errors.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'filename': filename,
                'error': str(e)
            })
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False
                
    return False

def get_daily_transcripts_by_date(api_instance: transcripts_api.TranscriptsApi, 
                                 target_date: datetime.date, 
                                 monitored_tickers: List[str]) -> List[Tuple[Dict[str, Any], str]]:
    """Get all transcripts for target date, filter to monitored institutions."""
    global logger, config
    logger.info(f"Querying transcripts for date: {target_date}")
    
    try:
        # Use date-based API endpoint instead of company-based
        response = api_instance.get_transcripts_dates(
            start_date=target_date,
            end_date=target_date,
            sort=["-storyDateTime"],
            pagination_limit=1000  # Maximum to get all results
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            logger.info(f"No transcripts found for date: {target_date}")
            return []
        
        # Validate API response structure
        if not validate_api_response_structure(response):
            logger.error(f"Invalid API response structure for date: {target_date}")
            return []
        
        all_transcripts = [transcript.to_dict() for transcript in response.data]
        logger.info(f"Found {len(all_transcripts)} total transcripts for date: {target_date}")
        
        # Filter to only transcripts where one of our monitored tickers is the ONLY primary ID
        filtered_transcripts = []
        for transcript in all_transcripts:
            primary_ids = transcript.get('primary_ids', [])
            
            # Validate that primary_ids is actually a list
            if not isinstance(primary_ids, list):
                logger.warning(f"Invalid primary_ids type for transcript: {type(primary_ids)}")
                continue
            
            # Check if exactly one primary ID and it's one of our monitored tickers
            if len(primary_ids) == 1 and primary_ids[0] in monitored_tickers:
                # Also filter for earnings transcripts
                if transcript.get('event_type', '') and 'Earnings' in str(transcript.get('event_type', '')):
                    filtered_transcripts.append((transcript, primary_ids[0]))
        
        logger.info(f"Filtered to {len(filtered_transcripts)} earnings transcripts for monitored institutions")
        
        return filtered_transcripts
        
    except Exception as e:
        logger.error(f"Error querying transcripts for date {target_date}: {e}")
        return []


def process_bank(ticker: str, institution_info: Dict[str, str], 
                transcripts_for_ticker: List[Dict[str, Any]],
                nas_conn: SMBConnection, configuration, 
                proxy_user: str, proxy_password: str) -> Dict[str, Any]:
    """Process transcripts for a single bank."""
    global logger, config
    logger.info(f"Processing: {institution_info['name']} ({ticker}) - {len(transcripts_for_ticker)} transcripts")
    
    institution_type = institution_info['type']
    bank_downloads = {
        'total': 0,
        'by_type': {transcript_type: 0 for transcript_type in config['api_settings']['transcript_types']},
        'institution_type': institution_type,
        'found_transcripts': False
    }
    
    try:
        if not transcripts_for_ticker:
            logger.info(f"No transcripts found for {ticker}")
            return bank_downloads
        
        # Directory structure will be created during download with enhanced structure
        bank_downloads['found_transcripts'] = True
        
        total_downloaded = 0
        
        for transcript_type in config['api_settings']['transcript_types']:
            type_transcripts = [t for t in transcripts_for_ticker if t.get('transcript_type') == transcript_type]
            
            if not type_transcripts:
                continue
            
            logger.info(f"Processing {len(type_transcripts)} {transcript_type} transcripts")
            
            # Get existing files with version management (automatically cleans up old versions)
            existing_files_with_versions = get_existing_files_with_version_management(nas_conn, ticker, transcript_type, institution_type)
            
            new_transcripts = []
            updated_transcripts = []
            
            for transcript in type_transcripts:
                version_agnostic_key = create_version_agnostic_key(transcript, target_ticker=ticker)
                new_version_id = transcript.get('version_id')
                
                if version_agnostic_key not in existing_files_with_versions:
                    # Completely new transcript
                    filename = create_filename(transcript, target_ticker=ticker)
                    new_transcripts.append((transcript, filename))
                else:
                    # Check if this is a newer version
                    existing_info = existing_files_with_versions[version_agnostic_key]
                    existing_version_id = existing_info['version_id']
                    
                    # Convert version IDs to integers for comparison
                    try:
                        new_version_int = int(new_version_id) if new_version_id else 0
                        existing_version_int = int(existing_version_id) if existing_version_id else 0
                        
                        if new_version_int > existing_version_int:
                            # This is a newer version, download it (old version already removed)
                            filename = create_filename(transcript, target_ticker=ticker)
                            updated_transcripts.append((transcript, filename))
                            logger.info(f"Found newer version: {filename} (version {new_version_int} > {existing_version_int})")
                    except (ValueError, TypeError) as e:
                        # If version comparison fails, treat as new transcript
                        logger.warning(f"Version comparison failed for {version_agnostic_key}: {e}")
                        filename = create_filename(transcript, target_ticker=ticker)
                        new_transcripts.append((transcript, filename))
            
            total_to_download = len(new_transcripts) + len(updated_transcripts)
            logger.info(f"Found {len(new_transcripts)} new and {len(updated_transcripts)} updated {transcript_type} transcripts to download")
            
            if total_to_download > 0:
                downloaded_count = 0
                
                # Download new transcripts
                for transcript, filename in new_transcripts:
                    if pd.isna(transcript.get('transcripts_link')) or not transcript.get('transcripts_link'):
                        logger.warning(f"No download link for {filename}")
                        continue
                    
                    if download_transcript_with_enhanced_structure(nas_conn, transcript['transcripts_link'],
                                                                    ticker, institution_type, transcript_type, 
                                                                    filename, configuration, 
                                                                    proxy_user, proxy_password):
                        downloaded_count += 1
                        
                    time.sleep(config['api_settings']['request_delay'])
                
                # Download updated transcripts (newer versions)
                for transcript, filename in updated_transcripts:
                    if pd.isna(transcript.get('transcripts_link')) or not transcript.get('transcripts_link'):
                        logger.warning(f"No download link for {filename}")
                        continue
                    
                    if download_transcript_with_enhanced_structure(nas_conn, transcript['transcripts_link'],
                                                                    ticker, institution_type, transcript_type, 
                                                                    filename, configuration, 
                                                                    proxy_user, proxy_password):
                        downloaded_count += 1
                        logger.info(f"Downloaded updated version: {filename}")
                        
                    time.sleep(config['api_settings']['request_delay'])
                
                logger.info(f"Successfully downloaded {downloaded_count}/{total_to_download} {transcript_type} transcripts")
                total_downloaded += downloaded_count
                bank_downloads['by_type'][transcript_type] = downloaded_count
        
        bank_downloads['total'] = total_downloaded
        logger.info(f"Completed {ticker}: Downloaded {total_downloaded} new transcripts total")
        
        return bank_downloads
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return bank_downloads


def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
    global logger, config
    try:
        logger.info("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, config['ssl_cert_nas_path'])
        if cert_data:
            temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()
            
            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name
            
            logger.info("SSL certificate downloaded from NAS")
            return temp_cert.name
        else:
            logger.error("Failed to download SSL certificate from NAS")
            return None
    except Exception as e:
        logger.error(f"Error downloading SSL certificate from NAS: {e}")
        return None

# No automatic config creation - config must be manually placed on NAS

def main() -> None:
    """Main function to orchestrate the Stage 1 daily transcript sync."""
    global config, logger, error_logger
    
    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    logger.info("STAGE 1: DAILY TRANSCRIPT SYNC WITH ENHANCED FOLDER STRUCTURE")
    
    temp_cert_path = None
    
    # Connect to NAS and load configuration
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting sync")
        return
    
    # Load shared configuration from NAS
    config = load_stage_config(nas_conn)
    logger.info(f"Loaded configuration for {len(config['monitored_institutions'])} institutions")
    
    # Setup SSL certificate
    temp_cert_path = setup_ssl_certificate(nas_conn)
    if not temp_cert_path:
        nas_conn.close()
        return
    
    # Configure FactSet API
    user = PROXY_USER
    password = quote(PROXY_PASSWORD)
    
    escaped_domain = quote(PROXY_DOMAIN + '\\' + user)
    proxy_url = f"http://{escaped_domain}:{password}@{PROXY_URL}"
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=API_USERNAME,
        password=API_PASSWORD,
        proxy=proxy_url,
        ssl_ca_cert=temp_cert_path
    )
    configuration.get_basic_auth_token()
    logger.info("FactSet API client configured")
    
    try:
        create_base_directory_structure(nas_conn)
        
        start_time = datetime.now()
        total_institutions = len(config['monitored_institutions'])
        total_downloaded = 0
        
        # Calculate date range for daily sync
        current_date = datetime.now().date()
        sync_date_range = config['stage_1']['sync_date_range']
        
        # Generate list of dates to check
        target_dates = []
        for i in range(sync_date_range + 1):  # Include current date
            target_date = current_date - timedelta(days=i)
            target_dates.append(target_date)
        
        logger.info(f"Starting daily sync for {total_institutions} institutions")
        logger.info(f"Date range: {target_dates[-1]} to {target_dates[0]} ({len(target_dates)} days)")
        
        institutions_with_transcripts = []
        monitored_tickers = list(config['monitored_institutions'].keys())
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Get all transcripts for all target dates - efficient discovery
            all_transcripts_by_ticker = {}
            for ticker in monitored_tickers:
                all_transcripts_by_ticker[ticker] = []
            
            for target_date in target_dates:
                logger.info(f"Querying transcripts for date: {target_date}")
                
                daily_transcripts = get_daily_transcripts_by_date(api_instance, target_date, monitored_tickers)
                
                # Group transcripts by ticker
                for transcript, ticker in daily_transcripts:
                    all_transcripts_by_ticker[ticker].append(transcript)
                
                # Rate limiting between date queries
                if target_date != target_dates[-1]:  # Don't sleep after last date
                    time.sleep(config['api_settings']['request_delay'])
            
            # Log summary of discovery results
            banks_with_transcripts = [ticker for ticker, transcripts in all_transcripts_by_ticker.items() if transcripts]
            total_transcripts_found = sum(len(transcripts) for transcripts in all_transcripts_by_ticker.values())
            
            if banks_with_transcripts:
                logger.info(f"Discovery complete: Found {total_transcripts_found} transcripts for {len(banks_with_transcripts)} banks: {', '.join(banks_with_transcripts)}")
            else:
                logger.info("Discovery complete: No new transcripts found for any monitored banks")
            
            # Process each institution with their accumulated transcripts (proven logic)
            for i, (ticker, institution_info) in enumerate(config['monitored_institutions'].items(), 1):
                transcripts_for_ticker = all_transcripts_by_ticker[ticker]
                
                # Only process if we found transcripts for this bank
                if transcripts_for_ticker:
                    logger.info(f"Processing institution {i}/{total_institutions}: {institution_info['name']} ({ticker})")
                    
                    bank_downloads = process_bank(ticker, institution_info, transcripts_for_ticker, nas_conn, configuration, user, password)
                    total_downloaded += bank_downloads['total']
                    
                    if bank_downloads['found_transcripts']:
                        institutions_with_transcripts.append({
                            'ticker': ticker,
                            'name': institution_info['name'],
                            'type': institution_info['type'],
                            'downloaded': bank_downloads['total']
                        })
                # No else clause needed - we only care about banks with transcripts
                
                # No artificial delays between banks - we already have all the data
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Save error logs to NAS
        error_logger.save_error_logs(nas_conn)
        
        logger.info("STAGE 1 DAILY SYNC WITH ENHANCED STRUCTURE COMPLETE")
        logger.info(f"Total new transcripts downloaded: {total_downloaded}")
        logger.info(f"Execution time: {execution_time}")
        
        # Log summary of institutions with activity
        if institutions_with_transcripts:
            logger.info(f"Institutions with transcripts found: {len(institutions_with_transcripts)}")
            for inst in institutions_with_transcripts:
                logger.info(f"  {inst['ticker']} ({inst['type']}): {inst['downloaded']} transcripts downloaded")
        else:
            logger.info("No new transcripts found for any monitored institutions")
        
        # Upload log file to NAS - properly close logging first
        try:
            # Close all logging handlers to ensure file is not in use
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Force any remaining buffered log data to be written
            logging.shutdown()
            
            # Now safely upload the log file
            log_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Logs", 
                                        f"stage_1_daily_sync_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
            
            # Read the entire log file and convert to BytesIO for upload
            with open(logger.temp_log_file, 'rb') as log_file:
                log_content = log_file.read()
            
            log_file_obj = io.BytesIO(log_content)
            nas_upload_file(nas_conn, log_file_obj, log_file_path)
            
        except Exception as e:
            print(f"Error uploading log file: {e}")  # Can't use logger since it's shut down
        
    except Exception as e:
        logger.error(f"Daily sync failed: {e}")
        # Try to save error logs even if main execution failed
        try:
            error_logger.save_error_logs(nas_conn)
        except:
            pass
    finally:
        if nas_conn:
            nas_conn.close()
        
        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
            except (OSError, FileNotFoundError) as e:
                print(f"Cleanup failed for certificate file: {e}")
        
        try:
            os.unlink(logger.temp_log_file)
        except (OSError, FileNotFoundError) as e:
            print(f"Cleanup failed for log file: {e}")
        except:
            pass

if __name__ == "__main__":
    main()