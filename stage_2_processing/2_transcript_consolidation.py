"""
Stage 2: Transcript Consolidation & Database Management
Identifies the best single transcript per company per fiscal quarter/year and manages master database.
Self-contained standalone script that loads config from NAS at runtime.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import json
import tempfile
import io
import xml.etree.ElementTree as ET
from smb.SMBConnection import SMBConnection
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from dotenv import load_dotenv
import re
import logging

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

# Validate required environment variables
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
    "CONFIG_PATH",
    "CLIENT_MACHINE_NAME",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )

# Global variables for configuration
config = {}
logger = None
error_logger = None


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".log",
        prefix=f'stage_2_consolidation_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
        self.parsing_errors = []
        self.database_errors = []
        self.filesystem_errors = []
        self.validation_errors = []
        self.selection_errors = []
        self.comparison_errors = []

    def log_selection_error(self, ticker: str, quarter: str, year: str, error: str):
        """Log file selection priority resolution errors."""
        self.selection_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "error": error,
                "action_required": "Manual review of transcript selection logic needed",
            }
        )

    def log_database_error(self, operation: str, error: str):
        """Log database schema or corruption errors."""
        self.database_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "error": error,
                "action_required": "Database validation or recovery needed",
            }
        )

    def log_comparison_error(self, ticker: str, quarter: str, year: str, error: str):
        """Log delta detection comparison errors."""
        self.comparison_errors.append(
            {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "quarter": quarter,
                "year": year,
                "error": error,
                "action_required": "Delta detection logic review needed",
            }
        )

    def save_error_logs(self, nas_conn: SMBConnection):
        """Save error logs to separate JSON files on NAS."""
        global logger
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        error_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Logs", "Errors")
        nas_create_directory(nas_conn, error_base_path)

        # Save each error type if errors exist
        error_types = [
            ("parsing_errors", self.parsing_errors),
            ("database_errors", self.database_errors),
            ("filesystem_errors", self.filesystem_errors),
            ("validation_errors", self.validation_errors),
            ("selection_errors", self.selection_errors),
            ("comparison_errors", self.comparison_errors),
        ]

        summary = {
            "run_timestamp": timestamp,
            "total_errors": sum(len(errors) for _, errors in error_types),
            "errors_by_type": {
                error_type: len(errors) for error_type, errors in error_types
            },
        }

        for error_type, errors in error_types:
            if errors:
                filename = f"{error_type}_{timestamp}.json"
                file_path = nas_path_join(error_base_path, filename)
                content = json.dumps({"summary": summary, "errors": errors}, indent=2)
                file_obj = io.BytesIO(content.encode("utf-8"))
                nas_upload_file(nas_conn, file_obj, file_path)
                logger.warning(f"Saved {len(errors)} {error_type} to {filename}")


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


def validate_config_schema(config: Dict[str, Any]) -> None:
    """Validate configuration schema and parameters."""
    global logger

    # Define required configuration structure
    required_structure = {
        "stage_2": {
            "master_database_path": str,
            "refresh_output_path": str,
            "transcript_type_priority": list,
            "include_xml_content": bool,
        },
        "api_settings": {"transcript_types": list},
        "monitored_institutions": dict,
    }

    # Validate top-level structure
    for top_key, top_value in required_structure.items():
        if top_key not in config:
            raise ValueError(f"Missing required configuration section: {top_key}")

        if isinstance(top_value, dict):
            # Validate nested structure
            for nested_key, expected_type in top_value.items():
                if nested_key not in config[top_key]:
                    raise ValueError(
                        f"Missing required configuration parameter: {top_key}.{nested_key}"
                    )

                actual_value = config[top_key][nested_key]
                if not isinstance(actual_value, expected_type):
                    raise ValueError(
                        f"Invalid type for {top_key}.{nested_key}: expected {expected_type}, got {type(actual_value)}"
                    )
        else:
            # Validate direct parameter
            if not isinstance(config[top_key], top_value):
                raise ValueError(
                    f"Invalid type for {top_key}: expected {top_value}, got {type(config[top_key])}"
                )

    # Validate transcript type priority
    priority_list = config["stage_2"]["transcript_type_priority"]
    if not priority_list:
        raise ValueError("transcript_type_priority cannot be empty")

    valid_transcript_types = ["Raw", "Corrected", "NearRealTime"]
    for transcript_type in priority_list:
        if transcript_type not in valid_transcript_types:
            raise ValueError(f"Invalid transcript type in priority: {transcript_type}")

    # Validate monitored institutions structure
    if not config["monitored_institutions"]:
        raise ValueError("monitored_institutions cannot be empty")

    for ticker, institution_info in config["monitored_institutions"].items():
        if not isinstance(institution_info, dict):
            raise ValueError(
                f"Invalid institution info for {ticker}: must be a dictionary"
            )

        required_fields = ["name", "type", "path_safe_name"]
        for field in required_fields:
            if field not in institution_info:
                raise ValueError(
                    f"Missing required field '{field}' for institution {ticker}"
                )
            if not isinstance(institution_info[field], str):
                raise ValueError(
                    f"Field '{field}' for institution {ticker} must be a string"
                )

        # Validate institution type
        valid_types = ["Canadian", "US", "European", "Insurance"]
        if institution_info["type"] not in valid_types:
            raise ValueError(
                f"Invalid institution type for {ticker}: {institution_info['type']}. Must be one of: {valid_types}"
            )

    # Validate ticker formats for security
    for ticker in config["monitored_institutions"].keys():
        if not re.match(r"^[A-Z0-9.-]+$", ticker):
            raise ValueError(f"Invalid ticker format: {ticker}")

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
    # Filter out empty parts and strip slashes
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

    # Normalize and validate path
    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        logger.error("Cannot create directory with empty path")
        return False

    # Split path into components
    path_parts = [part for part in normalized_path.split("/") if part]
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


def nas_upload_file(
    conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str
) -> bool:
    """Upload a file object to NAS."""
    global logger

    # Validate path for security
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
        return [
            file_info.filename
            for file_info in files
            if not file_info.isDirectory and file_info.filename.endswith(".xml")
        ]
    except Exception as e:
        logger.debug(f"Failed to list files in {directory_path}: {e}")
        return []


def nas_list_directories(conn: SMBConnection, directory_path: str) -> List[str]:
    """List subdirectories in a NAS directory."""
    global logger
    try:
        files = conn.listPath(NAS_SHARE_NAME, directory_path)
        return [
            file_info.filename
            for file_info in files
            if file_info.isDirectory and file_info.filename not in [".", ".."]
        ]
    except Exception as e:
        logger.debug(f"Failed to list directories in {directory_path}: {e}")
        return []


def get_file_modified_time(conn: SMBConnection, file_path: str) -> Optional[datetime]:
    """Get the last modified time of a file on NAS."""
    global logger
    try:
        attrs = conn.getAttributes(NAS_SHARE_NAME, file_path)
        return datetime.fromtimestamp(attrs.last_write_time)
    except Exception as e:
        logger.debug(f"Failed to get file modified time for {file_path}: {e}")
        return None


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False

    # Check for directory traversal attacks
    if ".." in path or path.startswith("/"):
        return False

    # Check for invalid characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
    if any(char in path for char in invalid_chars):
        return False

    # Check path length
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False

    return True


def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure."""
    global logger
    
    if not path or not isinstance(path, str):
        logger.debug(f"NAS path validation failed: empty or not string: '{path}'")
        return False

    # Ensure path is relative and safe
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


def sanitize_for_filename(text: Any) -> str:
    """Sanitize text to be safe for filename."""
    if pd.isna(text) or text is None:
        return "unknown"

    clean_text = str(text)
    clean_text = re.sub(r'[<>:"/\\|?*]', "_", clean_text)
    clean_text = re.sub(r"[^\w\s-]", "", clean_text)
    clean_text = re.sub(r"[-\s]+", "_", clean_text)
    return clean_text.strip("_")


def parse_quarter_and_year_from_xml(
    xml_content: bytes,
) -> Tuple[Optional[str], Optional[str]]:
    """Parse quarter and fiscal year from transcript XML title."""
    try:
        # Parse only until we find the title
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

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
                    quarter_map = {
                        "first": "Q1",
                        "second": "Q2",
                        "third": "Q3",
                        "fourth": "Q4",
                    }
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


def extract_title_from_xml(xml_content: bytes) -> Optional[str]:
    """Extract title from transcript XML."""
    try:
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return None

        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return None

        return title_elem.text.strip()
    except Exception as e:
        logger.error(f"Error extracting title from XML: {e}")
        return None


def classify_earnings_call(title: str) -> Tuple[str, Optional[str]]:
    """Classify earnings call type based on title.
    Returns (call_type, call_suffix)
    """
    if not title:
        return "unknown", None
    
    # Check for exact main earnings call pattern
    if re.match(r"^Q[1-4] 20\d{2} Earnings Call$", title):
        return "primary", None
    
    # Check for earnings call with suffix
    suffix_match = re.search(r"Earnings Call\s*-\s*(.+)$", title)
    if suffix_match:
        return "secondary", suffix_match.group(1).strip()
    
    # Check if it's still an earnings call without suffix
    if "Earnings Call" in title and " - " not in title:
        return "primary", None
    
    return "other", None


def parse_version_from_filename(filename: str) -> Optional[int]:
    """Extract version number from filename. Returns None if not found or invalid."""
    global logger
    try:
        # Remove .xml extension
        basename = filename.replace(".xml", "")
        # Split by underscores and get the last part (version_id)
        parts = basename.split("_")
        if (
            len(parts) >= 6
        ):  # Should have at least 6 parts based on our naming convention
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
        basename = filename.replace(".xml", "")
        # Split by underscores and remove the last part (version_id)
        parts = basename.split("_")
        if len(parts) >= 6:
            return "_".join(parts[:-1])  # All parts except the last one (version_id)
    except (ValueError, IndexError, AttributeError) as e:
        logger.debug(
            f"Could not extract version-agnostic key from filename {filename}: {e}"
        )

    return basename  # Return as-is if parsing fails


def scan_nas_for_transcripts(nas_conn: SMBConnection) -> Dict[str, Dict[str, Any]]:
    """Scan NAS for all transcript files and organize by company+quarter+year."""
    global logger, config, error_logger
    logger.info("Starting comprehensive NAS transcript scan...")

    data_base_path = nas_path_join(NAS_BASE_PATH, "Outputs", "Data")
    transcript_inventory = {}

    # Scan all years
    years = nas_list_directories(nas_conn, data_base_path)
    logger.info(f"Found years: {sorted(years)}")

    for year in years:
        if year == "Unknown":
            continue  # Skip Unknown folder for now

        year_path = nas_path_join(data_base_path, year)
        quarters = nas_list_directories(nas_conn, year_path)

        for quarter in quarters:
            if quarter == "Unknown":
                continue  # Skip Unknown folder for now

            quarter_path = nas_path_join(year_path, quarter)
            institution_types = nas_list_directories(nas_conn, quarter_path)

            for institution_type in institution_types:
                type_path = nas_path_join(quarter_path, institution_type)
                companies = nas_list_directories(nas_conn, type_path)

                for company in companies:
                    # Extract ticker from company folder name (format: TICKER_Company_Name)
                    ticker = company.split("_")[0] if "_" in company else company

                    # Only process monitored institutions
                    if ticker not in config["monitored_institutions"]:
                        continue

                    company_path = nas_path_join(type_path, company)
                    transcript_types = nas_list_directories(nas_conn, company_path)

                    # Collect ALL earnings calls for this company+quarter+year
                    all_quarterly_calls = []
                    for priority_type in config["stage_2"]["transcript_type_priority"]:
                        if priority_type in transcript_types:
                            type_path_full = nas_path_join(company_path, priority_type)
                            files = nas_list_files(nas_conn, type_path_full)

                            if files:
                                # Get all files with their metadata
                                type_files = get_all_files_with_metadata(
                                    nas_conn,
                                    type_path_full,
                                    files,
                                    ticker,
                                    quarter,
                                    year,
                                    priority_type
                                )
                                all_quarterly_calls.extend(type_files)
                    
                    # Select the primary earnings call from all available
                    selected_file = select_primary_earnings_call(
                        nas_conn,
                        all_quarterly_calls,
                        ticker,
                        quarter,
                        year
                    )

                    if selected_file:
                        # Create comparison key
                        comparison_key = f"{ticker}_{year}_{quarter}"

                        # Build full transcript record
                        institution_info = config["monitored_institutions"][ticker]
                        # Create output record with requested field ordering
                        transcript_record = {
                            "title": selected_file.get("title", ""),
                            "filename": selected_file["filename"],
                            "date_last_modified": selected_file["date_last_modified"],
                            "transcript_type": selected_file["transcript_type"],
                            "event_id": selected_file.get("event_id", "unknown"),
                            "report_id": selected_file.get("report_id", "unknown"),
                            "version_id": selected_file.get("version_id", "unknown"),
                            "file_path": selected_file["file_path"],
                            "fiscal_year": year,
                            "fiscal_quarter": quarter,
                            "institution_type": institution_type,
                            "ticker": ticker,
                            "company_name": institution_info["name"]
                            # Note: version_agnostic_key not included in output (internal use only)
                        }

                        transcript_inventory[comparison_key] = transcript_record
                        logger.debug(
                            f"Selected {ticker} {year} {quarter}: {selected_file['filename']}"
                        )

    logger.info(
        f"NAS scan complete: Found {len(transcript_inventory)} optimal transcripts"
    )
    return transcript_inventory


def get_all_files_with_metadata(
    nas_conn: SMBConnection,
    type_path: str,
    files: List[str],
    ticker: str,
    quarter: str,
    year: str,
    transcript_type: str
) -> List[Dict[str, Any]]:
    """Get all files in a transcript type folder with their metadata and titles."""
    global logger, error_logger
    
    result_files = []
    
    for filename in files:
        file_path = nas_path_join(type_path, filename)
        modified_time = get_file_modified_time(nas_conn, file_path)
        version_id = parse_version_from_filename(filename)
        version_agnostic_key = get_version_agnostic_key_from_filename(filename)
        
        # Download file to extract title
        try:
            xml_content = nas_download_file(nas_conn, file_path)
            if xml_content:
                title = extract_title_from_xml(xml_content)
            else:
                title = None
                error_logger.validation_errors.append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'filename': filename,
                    'error': 'Failed to download file for title extraction'
                })
        except Exception as e:
            logger.warning(f"Failed to extract title from {filename}: {e}")
            title = None
        
        # Parse metadata from filename  
        parts = filename.replace(".xml", "").split("_")
        # Actual format: RY_CA_2024-01-25_Earnings_Corrected_12345_67890_1
        # parts[0]: RY, parts[1]: CA, parts[2]: date, parts[3]: event_type, parts[4]: transcript_type, parts[5]: event_id, parts[6]: report_id, parts[7]: version_id
        if len(parts) >= 8:
            # Full format with ticker containing underscore
            event_id = parts[5]
            report_id = parts[6]
        elif len(parts) >= 7:
            # Missing event_type or different format
            event_id = parts[4]
            report_id = parts[5]
        else:
            event_id = "unknown"
            report_id = "unknown"
        
        file_record = {
            "filename": filename,
            "file_path": file_path,
            "transcript_type": transcript_type,
            "version_id": version_id or 0,
            "date_last_modified": modified_time.isoformat() if modified_time else datetime.now().isoformat(),
            "version_agnostic_key": version_agnostic_key,  # Keep for internal processing
            "event_id": event_id,
            "report_id": report_id,
            "title": title
        }
        
        result_files.append(file_record)
        logger.debug(f"Found call: {title} ({filename})")
    
    return result_files


def select_primary_earnings_call(
    nas_conn: SMBConnection,
    all_calls: List[Dict[str, Any]], 
    ticker: str,
    quarter: str,
    year: str
) -> Optional[Dict[str, Any]]:
    """Select the primary earnings call from all available calls using enhanced selection logic."""
    global logger, error_logger
    
    if not all_calls:
        return None
    
    if len(all_calls) == 1:
        logger.debug(f"Single call found for {ticker} {year} {quarter}")
        return all_calls[0]
    
    # Apply version management first - group by version-agnostic key and keep latest versions
    version_groups = {}
    for call in all_calls:
        key = call["version_agnostic_key"]
        if key not in version_groups:
            version_groups[key] = []
        version_groups[key].append(call)
    
    # Keep only latest version of each unique call
    latest_calls = []
    for group in version_groups.values():
        if len(group) == 1:
            latest_calls.append(group[0])
        else:
            # Select highest version
            latest_call = max(group, key=lambda x: x["version_id"])
            latest_calls.append(latest_call)
    
    # Now apply primary call selection logic
    if len(latest_calls) == 1:
        logger.debug(f"Single latest call for {ticker} {year} {quarter}")
        return latest_calls[0]
    
    # Tier 1: Look for exact primary calls first
    # Check for primary calls using title patterns instead of removed call_type field
    primary_calls = []
    for call in latest_calls:
        title = call.get("title", "")
        # Primary call patterns: exact match or no suffix
        if re.match(r"^Q[1-4] 20\d{2} Earnings Call$", title) or \
           ("Earnings Call" in title and " - " not in title):
            primary_calls.append(call)
    
    if len(primary_calls) == 1:
        logger.info(f"Selected primary call for {ticker} {year} {quarter}: {primary_calls[0]['title']}")
        return primary_calls[0]
    elif len(primary_calls) > 1:
        # Multiple primary calls - use existing 3-tier selection
        selected = apply_three_tier_selection(primary_calls, ticker, quarter, year)
        logger.info(f"Selected from multiple primary calls for {ticker} {year} {quarter}: {selected['title']}")
        return selected
    
    # Tier 2: No primary calls, look for any earnings calls
    earnings_calls = [call for call in latest_calls if "Earnings Call" in (call["title"] or "")]
    
    if earnings_calls:
        selected = apply_three_tier_selection(earnings_calls, ticker, quarter, year)
        logger.warning(f"No primary call found for {ticker} {year} {quarter}, selected: {selected['title']}")
        error_logger.log_selection_error(
            ticker, quarter, year,
            f"No primary earnings call found, selected: {selected['title']}"
        )
        return selected
    
    # Tier 3: Fallback to best available
    selected = apply_three_tier_selection(latest_calls, ticker, quarter, year)
    logger.warning(f"No earnings calls found for {ticker} {year} {quarter}, selected best available: {selected['title']}")
    error_logger.log_selection_error(
        ticker, quarter, year,
        f"No earnings calls found, selected: {selected['title']}"
    )
    return selected


def apply_three_tier_selection(
    calls: List[Dict[str, Any]],
    ticker: str,
    quarter: str, 
    year: str
) -> Dict[str, Any]:
    """Apply 3-tier selection: transcript type > version ID > date modified."""
    global logger, error_logger
    
    if len(calls) == 1:
        return calls[0]
    
    # Tier 1: Transcript type priority (already grouped by type during collection)
    # Find the highest priority transcript type available
    best_priority = min(call["selection_priority"] for call in calls)
    best_type_calls = [call for call in calls if call["selection_priority"] == best_priority]
    
    if len(best_type_calls) == 1:
        return best_type_calls[0]
    
    # Tier 2: Highest version ID
    max_version = max(call["version_id"] for call in best_type_calls)
    best_version_calls = [call for call in best_type_calls if call["version_id"] == max_version]
    
    if len(best_version_calls) == 1:
        return best_version_calls[0]
    
    # Tier 3: Latest modified date
    latest_call = max(best_version_calls, key=lambda x: x["date_last_modified"])
    
    if len(best_version_calls) > 1:
        error_logger.log_selection_error(
            ticker, quarter, year,
            f"Multiple files after 3-tier selection, chose by date: {[c['filename'] for c in best_version_calls]}"
        )
    
    return latest_call


def select_best_file_from_type(
    nas_conn: SMBConnection,
    type_path: str,
    files: List[str],
    ticker: str,
    quarter: str,
    year: str,
) -> Optional[Dict[str, Any]]:
    """Select the best single file from a transcript type folder using 3-tier priority."""
    global logger, error_logger

    if not files:
        return None

    if len(files) == 1:
        # Single file - easy case
        file_path = nas_path_join(type_path, files[0])
        modified_time = get_file_modified_time(nas_conn, file_path)
        return create_file_record(type_path, files[0], modified_time)

    # Multiple files - apply 3-tier selection
    file_candidates = []

    for filename in files:
        file_path = nas_path_join(type_path, filename)
        modified_time = get_file_modified_time(nas_conn, file_path)
        version_id = parse_version_from_filename(filename)
        version_agnostic_key = get_version_agnostic_key_from_filename(filename)

        file_candidates.append(
            {
                "filename": filename,
                "file_path": file_path,
                "version_id": version_id or 0,
                "date_last_modified": modified_time.isoformat() if modified_time else datetime.now().isoformat(),
                "version_agnostic_key": version_agnostic_key,
            }
        )

    # Tier 2: Group by version-agnostic key and select highest version
    version_groups = {}
    for candidate in file_candidates:
        key = candidate["version_agnostic_key"]
        if key not in version_groups:
            version_groups[key] = []
        version_groups[key].append(candidate)

    best_candidates = []
    for key, group in version_groups.items():
        if len(group) == 1:
            best_candidates.append(group[0])
        else:
            # Select highest version ID
            best_version = max(group, key=lambda x: x["version_id"])
            best_candidates.append(best_version)

    # Tier 3: If still multiple files, select by latest modified date
    if len(best_candidates) == 1:
        return best_candidates[0]
    elif len(best_candidates) > 1:
        # Log this as unusual but handle it
        error_logger.log_selection_error(
            ticker,
            quarter,
            year,
            f"Multiple unique files in {type_path}: {[c['filename'] for c in best_candidates]}",
        )
        # Select by latest modified date
        latest_file = max(best_candidates, key=lambda x: x["date_last_modified"])
        logger.warning(
            f"Selected {latest_file['filename']} by modified date from {len(best_candidates)} candidates"
        )
        return latest_file

    return None


def create_file_record(
    type_path: str, filename: str, modified_time: Optional[datetime]
) -> Dict[str, Any]:
    """Create a standardized file record."""
    file_path = nas_path_join(type_path, filename)
    version_id = parse_version_from_filename(filename)
    version_agnostic_key = get_version_agnostic_key_from_filename(filename)

    # Parse metadata from filename
    parts = filename.replace(".xml", "").split("_")
    # Actual format: RY_CA_2024-01-25_Earnings_Corrected_12345_67890_1
    # parts[0]: RY, parts[1]: CA, parts[2]: date, parts[3]: event_type, parts[4]: transcript_type, parts[5]: event_id, parts[6]: report_id, parts[7]: version_id
    if len(parts) >= 8:
        # Full format with ticker containing underscore
        event_id = parts[5]
        report_id = parts[6]
    elif len(parts) >= 7:
        # Missing event_type or different format
        event_id = parts[4]
        report_id = parts[5]
    else:
        event_id = "unknown"
        report_id = "unknown"

    return {
        "filename": filename,
        "file_path": file_path,
        "version_id": version_id or 0,
        "date_last_modified": (
            modified_time.isoformat() if modified_time else datetime.now().isoformat()
        ),
        "version_agnostic_key": version_agnostic_key,  # Keep for internal processing
        "event_id": event_id,
        "report_id": report_id,
    }


def get_priority_score(transcript_type: str) -> int:
    """Get priority score for transcript type (lower = higher priority)."""
    global config
    priority_list = config["stage_2"]["transcript_type_priority"]
    try:
        return priority_list.index(transcript_type) + 1
    except ValueError:
        return 999  # Unknown types get lowest priority


def load_master_database(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load existing master database for comparison (read-only). Returns empty structure if none exists."""
    global logger, config, error_logger

    db_path = nas_path_join(NAS_BASE_PATH, config["stage_2"]["master_database_path"])

    if nas_file_exists(nas_conn, db_path):
        logger.info("Loading existing master database...")
        try:
            db_data = nas_download_file(nas_conn, db_path)
            if db_data:
                database = json.loads(db_data.decode("utf-8"))
                logger.info(
                    f"Loaded master database with {len(database.get('transcripts', []))} records"
                )
                return database
        except json.JSONDecodeError as e:
            error_logger.log_database_error("load", f"JSON decode error: {e}")
            logger.error(f"Database corruption detected: {e}")
        except Exception as e:
            error_logger.log_database_error("load", f"Database load error: {e}")
            logger.error(f"Error loading database: {e}")

    # No existing database - return empty structure for comparison
    logger.info("No existing master database found - all files will be marked for processing")
    return {
        "schema_version": "2.0",
        "last_updated": "never",
        "transcripts": [],
    }


def save_master_database(nas_conn: SMBConnection, database: Dict[str, Any]) -> bool:
    """Save master database to NAS."""
    global logger, config

    # Update timestamp
    database["last_updated"] = datetime.now().isoformat()

    db_path = nas_path_join(NAS_BASE_PATH, config["stage_2"]["master_database_path"])
    
    # Ensure Database directory exists
    db_dir = nas_path_join(NAS_BASE_PATH, "Outputs", "Database")
    nas_create_directory(nas_conn, db_dir)

    try:
        db_content = json.dumps(database, indent=2)
        db_file_obj = io.BytesIO(db_content.encode("utf-8"))

        if nas_upload_file(nas_conn, db_file_obj, db_path):
            logger.info(
                f"Saved master database with {len(database['transcripts'])} records"
            )
            return True
        else:
            logger.error("Failed to upload master database to NAS")
            return False
    except Exception as e:
        logger.error(f"Error saving master database: {e}")
        return False


def database_to_comparison_format(
    database: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Convert database to comparison format (key -> record)."""
    comparison_data = {}

    for record in database.get("transcripts", []):
        comparison_key = (
            f"{record['ticker']}_{record['fiscal_year']}_{record['fiscal_quarter']}"
        )
        comparison_data[comparison_key] = record

    return comparison_data


def detect_changes(
    nas_listing: Dict[str, Dict[str, Any]], database_listing: Dict[str, Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Detect changes between NAS and database listings."""
    global logger, error_logger

    files_to_process = []
    files_to_remove = []

    # Check for new and changed files
    for comparison_key, nas_record in nas_listing.items():
        db_record = database_listing.get(comparison_key)

        if not db_record:
            # New file - add to processing
            files_to_process.append(nas_record)
            logger.info(f"New: {comparison_key}")
        else:
            # Check for changes
            try:
                nas_path = nas_record["file_path"]
                db_path = db_record["file_path"]
                nas_modified = nas_record["date_last_modified"]
                db_modified = db_record["date_last_modified"]

                if nas_path != db_path or nas_modified != db_modified:
                    # Changed file - remove old, add new
                    files_to_remove.append(db_record)
                    files_to_process.append(nas_record)
                    logger.info(f"Changed: {comparison_key}")
            except KeyError as e:
                error_logger.log_comparison_error(
                    nas_record.get("ticker", "unknown"),
                    nas_record.get("fiscal_quarter", "unknown"),
                    nas_record.get("fiscal_year", "unknown"),
                    f"Missing key during comparison: {e}",
                )

    # Check for deleted files (in database but not on NAS)
    for comparison_key, db_record in database_listing.items():
        if comparison_key not in nas_listing:
            files_to_remove.append(db_record)
            logger.info(f"Deleted: {comparison_key}")

    logger.info(
        f"Delta detection: {len(files_to_process)} to process, {len(files_to_remove)} to remove"
    )
    return files_to_process, files_to_remove


def update_master_database(
    database: Dict[str, Any],
    files_to_process: List[Dict[str, Any]],
    files_to_remove: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Update master database with changes."""
    global logger

    # Remove outdated records
    if files_to_remove:
        remove_keys = set()
        for record in files_to_remove:
            remove_key = (
                f"{record['ticker']}_{record['fiscal_year']}_{record['fiscal_quarter']}"
            )
            remove_keys.add(remove_key)

        original_count = len(database["transcripts"])
        database["transcripts"] = [
            record
            for record in database["transcripts"]
            if f"{record['ticker']}_{record['fiscal_year']}_{record['fiscal_quarter']}"
            not in remove_keys
        ]
        removed_count = original_count - len(database["transcripts"])
        logger.info(f"Removed {removed_count} outdated records from database")

    # Add new records
    if files_to_process:
        database["transcripts"].extend(files_to_process)
        logger.info(f"Added {len(files_to_process)} new records to database")

    return database


def save_processing_queues(
    nas_conn: SMBConnection,
    files_to_process: List[Dict[str, Any]],
    files_to_remove: List[Dict[str, Any]],
) -> bool:
    """Save processing queues to NAS refresh folder."""
    global logger, config

    refresh_base_path = nas_path_join(
        NAS_BASE_PATH, config["stage_2"]["refresh_output_path"]
    )
    nas_create_directory(nas_conn, refresh_base_path)

    timestamp = datetime.now().isoformat()

    # Save files to process
    process_data = {
        "timestamp": timestamp,
        "total_files": len(files_to_process),
        "files": files_to_process,
    }

    process_path = nas_path_join(refresh_base_path, "files_to_process.json")
    process_content = json.dumps(process_data, indent=2)
    process_file_obj = io.BytesIO(process_content.encode("utf-8"))

    # Save files to remove
    remove_data = {
        "timestamp": timestamp,
        "total_files": len(files_to_remove),
        "files": files_to_remove,
    }

    remove_path = nas_path_join(refresh_base_path, "files_to_remove.json")
    remove_content = json.dumps(remove_data, indent=2)
    remove_file_obj = io.BytesIO(remove_content.encode("utf-8"))

    # Upload both files
    success = True
    if not nas_upload_file(nas_conn, process_file_obj, process_path):
        logger.error(f"Failed to save files_to_process.json to path: {process_path}")
        success = False

    if not nas_upload_file(nas_conn, remove_file_obj, remove_path):
        logger.error(f"Failed to save files_to_remove.json to path: {remove_path}")
        success = False

    if success:
        logger.info(
            f"Saved processing queues: {len(files_to_process)} to process, {len(files_to_remove)} to remove"
        )

    return success


def upload_logs_to_nas(nas_conn: SMBConnection, logger: logging.Logger, error_logger) -> None:
    """Upload logs to NAS immediately (for critical failures)."""
    try:
        # Save error logs first
        error_logger.save_error_logs(nas_conn)
        
        # Close all logging handlers to ensure file is not in use
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Force any remaining buffered log data to be written
        logging.shutdown()

        # Upload the log file
        log_file_path = nas_path_join(
            NAS_BASE_PATH,
            "Outputs",
            "Logs",
            f"stage_2_consolidation_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
        )

        # Read the entire log file and convert to BytesIO for upload
        with open(logger.temp_log_file, "rb") as log_file:
            log_content = log_file.read()

        log_file_obj = io.BytesIO(log_content)
        nas_upload_file(nas_conn, log_file_obj, log_file_path)
        print(f"Emergency log upload completed to: {log_file_path}")
        
    except Exception as e:
        print(f"Emergency log upload failed: {e}")


def main() -> None:
    """Main function to orchestrate Stage 2 transcript consolidation."""
    global config, logger, error_logger

    logger = setup_logging()
    error_logger = EnhancedErrorLogger()
    print(f"Local log file: {logger.temp_log_file}")  # Always show where local log is
    logger.info("STAGE 2: TRANSCRIPT CONSOLIDATION & DATABASE MANAGEMENT")

    # Connect to NAS and load configuration
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting consolidation")
        return

    try:
        # Load shared configuration from NAS
        config = load_stage_config(nas_conn)
        logger.info(
            f"Loaded configuration for {len(config['monitored_institutions'])} institutions"
        )

        start_time = datetime.now()

        # Step 1: Scan NAS for transcript files and select best ones
        logger.info("Step 1: Scanning NAS for optimal transcript selection...")
        nas_inventory = scan_nas_for_transcripts(nas_conn)

        if not nas_inventory:
            logger.warning("No transcripts found on NAS - nothing to process")
            return

        # Step 2: Load or create master database
        logger.info("Step 2: Loading master database...")
        master_database = load_master_database(nas_conn)

        # Step 3: Convert database to comparison format
        logger.info("Step 3: Preparing database comparison...")
        database_inventory = database_to_comparison_format(master_database)

        # Step 4: Detect changes between NAS and database
        logger.info("Step 4: Detecting changes...")
        files_to_process, files_to_remove = detect_changes(
            nas_inventory, database_inventory
        )

        # Step 5: Save processing queues
        logger.info("Step 5: Saving processing queues...")
        if not save_processing_queues(nas_conn, files_to_process, files_to_remove):
            logger.error("Failed to save processing queues")
            # Upload logs immediately on critical failure
            try:
                upload_logs_to_nas(nas_conn, logger, error_logger)
            except:
                pass
            return

        end_time = datetime.now()
        execution_time = end_time - start_time

        # Save error logs
        error_logger.save_error_logs(nas_conn)

        # Final summary
        logger.info("STAGE 2 CONSOLIDATION COMPLETE")
        logger.info(f"Total transcripts in inventory: {len(nas_inventory)}")
        logger.info(f"Files to process: {len(files_to_process)}")
        logger.info(f"Files to remove: {len(files_to_remove)}")
        logger.info(f"Existing database records: {len(database_inventory)}")
        logger.info(f"Execution time: {execution_time}")

        # Upload log file to NAS
        try:
            # Close all logging handlers to ensure file is not in use
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Force any remaining buffered log data to be written
            logging.shutdown()

            # Now safely upload the log file
            log_file_path = nas_path_join(
                NAS_BASE_PATH,
                "Outputs",
                "Logs",
                f"stage_2_consolidation_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log",
            )

            # Read the entire log file and convert to BytesIO for upload
            with open(logger.temp_log_file, "rb") as log_file:
                log_content = log_file.read()

            log_file_obj = io.BytesIO(log_content)
            nas_upload_file(nas_conn, log_file_obj, log_file_path)

        except Exception as e:
            print(
                f"Error uploading log file: {e}"
            )  # Can't use logger since it's shut down

    except Exception as e:
        logger.error(f"Stage 2 consolidation failed: {e}")
        # Try to save error logs even if main execution failed
        try:
            error_logger.save_error_logs(nas_conn)
        except:
            pass
    finally:
        if nas_conn:
            nas_conn.close()

        try:
            os.unlink(logger.temp_log_file)
        except (OSError, FileNotFoundError) as e:
            print(f"Cleanup failed for log file: {e}")
        except:
            pass


if __name__ == "__main__":
    main()
