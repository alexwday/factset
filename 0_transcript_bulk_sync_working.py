"""
Earnings Transcript Repository Sync v2
Maintains NAS-based repository of earnings transcripts for monitored financial institutions.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
import os
from urllib.parse import quote
from datetime import datetime
import time
import requests
import logging
import re
import json
import tempfile
import io
from smb.SMBConnection import SMBConnection
from smb.base import SMBTimeout
from typing import Dict, List, Optional, Set, Any, Tuple
import warnings
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from dataclasses import dataclass
from collections import defaultdict

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables - handles both notebook and script execution
load_dotenv(dotenv_path='.env')  # For script execution
if not os.getenv('API_USERNAME'):  # If not found, try parent directory
    load_dotenv(dotenv_path='../.env')  # For notebook execution

# Configuration from environment
SSL_CERT_NAS_PATH = os.getenv('SSL_CERT_NAS_PATH', 'Inputs/certificate/certificate.cer')
PROXY_USER = os.getenv('PROXY_USER')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
PROXY_URL = os.getenv('PROXY_URL', 'oproxy.fg.rbc.com:8080')

API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')

NAS_USERNAME = os.getenv('NAS_USERNAME')
NAS_PASSWORD = os.getenv('NAS_PASSWORD')
NAS_SERVER_IP = os.getenv('NAS_SERVER_IP', '192.168.1.100')
NAS_SERVER_NAME = os.getenv('NAS_SERVER_NAME', 'NAS-SERVER')
NAS_SHARE_NAME = os.getenv('NAS_SHARE_NAME', 'shared_folder')
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH', 'transcript_repository')
NAS_PORT = 445
CLIENT_MACHINE_NAME = "SYNC-CLIENT"

# Performance settings
MAX_CONCURRENT_DOWNLOADS = int(os.getenv('MAX_CONCURRENT_DOWNLOADS', '5'))
RATE_LIMIT_PER_SECOND = int(os.getenv('RATE_LIMIT_PER_SECOND', '10'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Validate required environment variables
required_vars = ['PROXY_USER', 'PROXY_PASSWORD', 'API_USERNAME', 'API_PASSWORD', 
                 'NAS_USERNAME', 'NAS_PASSWORD']
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

SYNC_START_DATE = "2023-01-01"

MONITORED_INSTITUTIONS = {
    "RY-CA": {"name": "Royal Bank of Canada", "region": "CA", "type": "Bank"},
    "TD-CA": {"name": "Toronto-Dominion Bank", "region": "CA", "type": "Bank"},
    "BNS-CA": {"name": "Bank of Nova Scotia", "region": "CA", "type": "Bank"},
    "BMO-CA": {"name": "Bank of Montreal", "region": "CA", "type": "Bank"},
    "CM-CA": {"name": "Canadian Imperial Bank of Commerce", "region": "CA", "type": "Bank"},
    "NA-CA": {"name": "National Bank of Canada", "region": "CA", "type": "Bank"},
    "JPM-US": {"name": "JPMorgan Chase & Co.", "region": "US", "type": "Bank"},
    "BAC-US": {"name": "Bank of America Corporation", "region": "US", "type": "Bank"},
    "WFC-US": {"name": "Wells Fargo & Company", "region": "US", "type": "Bank"},
    "C-US": {"name": "Citigroup Inc.", "region": "US", "type": "Bank"},
    "GS-US": {"name": "Goldman Sachs Group Inc.", "region": "US", "type": "Bank"},
    "MS-US": {"name": "Morgan Stanley", "region": "US", "type": "Bank"},
    "MFC-CA": {"name": "Manulife Financial Corporation", "region": "CA", "type": "Insurance"},
    "SLF-CA": {"name": "Sun Life Financial Inc.", "region": "CA", "type": "Insurance"},
    "UNH-US": {"name": "UnitedHealth Group Incorporated", "region": "US", "type": "Insurance"},
}

INDUSTRY_CATEGORIES = ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"]
TRANSCRIPT_TYPES = ["Corrected", "Raw", "NearRealTime"]
SORT_ORDER = ["-storyDateTime"]
PAGINATION_LIMIT = 1000
PAGINATION_OFFSET = 0
REQUEST_DELAY = 2.0
MAX_RETRIES = 3
RETRY_DELAY = 5.0

user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Rate limiter
rate_limiter = Semaphore(RATE_LIMIT_PER_SECOND)
last_request_time = 0

@dataclass
class DownloadFailure:
    """Track failed downloads for retry"""
    ticker: str
    transcript_type: str
    filename: str
    transcript_link: str
    error: str
    attempts: int = 0

def setup_logging() -> logging.Logger:
    """Set up logging configuration with debug mode support."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        suffix='.log', 
        prefix=f'sync_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
        delete=False
    )
    
    log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        handlers=[
            logging.FileHandler(temp_log_file.name),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.temp_log_file = temp_log_file.name
    
    if DEBUG_MODE:
        logger.debug("Debug mode enabled")
    
    return logger

def rate_limit_request():
    """Ensure we don't exceed rate limit."""
    global last_request_time
    with rate_limiter:
        current_time = time.time()
        time_since_last = current_time - last_request_time
        if time_since_last < (1.0 / RATE_LIMIT_PER_SECOND):
            time.sleep((1.0 / RATE_LIMIT_PER_SECOND) - time_since_last)
        last_request_time = time.time()

def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
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
            logger.info(f"Connected to NAS: {NAS_SERVER_IP}")
            return conn
        else:
            logger.error(f"Failed to connect to NAS: {NAS_SERVER_IP}")
            return None
            
    except SMBTimeout as e:
        logger.error(f"NAS connection timeout: {e}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None

def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    return '/'.join(str(part) for part in parts if part)

def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except Exception as e:
        logger.debug(f"File not found or error checking: {file_path} - {e}")
        return False

def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with parent directory creation."""
    try:
        conn.createDirectory(NAS_SHARE_NAME, dir_path)
        return True
    except Exception as e:
        if nas_file_exists(conn, dir_path):
            return True
        
        parent_dir = '/'.join(dir_path.split('/')[:-1])
        if parent_dir and parent_dir != dir_path:
            nas_create_directory(conn, parent_dir)
            try:
                conn.createDirectory(NAS_SHARE_NAME, dir_path)
                return True
            except Exception as nested_e:
                logger.error(f"Failed to create directory {dir_path}: {nested_e}")
                return False
        return False

def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    try:
        parent_dir = '/'.join(nas_file_path.split('/')[:-1])
        if parent_dir:
            nas_create_directory(conn, parent_dir)
        
        local_file_obj.seek(0)
        conn.storeFile(NAS_SHARE_NAME, nas_file_path, local_file_obj)
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to NAS {nas_file_path}: {e}")
        return False

def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
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
    try:
        files = conn.listPath(NAS_SHARE_NAME, directory_path)
        return [file_info.filename for file_info in files 
                if not file_info.isDirectory and file_info.filename.endswith('.xml')]
    except Exception as e:
        logger.debug(f"Directory doesn't exist or error listing: {directory_path} - {e}")
        return []

def sanitize_for_filename(text: Any) -> str:
    """Sanitize text to be safe for filename."""
    if pd.isna(text) or text is None:
        return "unknown"
    
    clean_text = str(text)
    clean_text = re.sub(r'[<>:"/\\|?*]', '_', clean_text)
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    clean_text = re.sub(r'[-\s]+', '_', clean_text)
    return clean_text.strip('_')

def validate_transcript_data(transcript: Dict[str, Any]) -> bool:
    """Validate that transcript data contains required fields."""
    required_fields = ['event_type', 'transcript_type', 'event_id', 'report_id']
    
    for field in required_fields:
        if field not in transcript or transcript.get(field) is None:
            logger.warning(f"Missing required field '{field}' in transcript data")
            return False
    
    # Validate transcript link
    if not transcript.get('transcripts_link'):
        logger.warning("Missing transcript download link")
        return False
    
    # Validate link format
    link = transcript.get('transcripts_link', '')
    if not link.startswith(('http://', 'https://')):
        logger.warning(f"Invalid transcript link format: {link}")
        return False
    
    return True

def create_directory_structure(nas_conn: SMBConnection) -> None:
    """Create the directory structure for the transcript repository on NAS."""
    logger.info("Creating directory structure on NAS...")
    
    inputs_path = nas_path_join(NAS_BASE_PATH, "Inputs")
    outputs_path = nas_path_join(NAS_BASE_PATH, "Outputs")
    
    nas_create_directory(nas_conn, inputs_path)
    nas_create_directory(nas_conn, outputs_path)
    
    certificate_path = nas_path_join(inputs_path, "certificate")
    data_path = nas_path_join(outputs_path, "data")
    logs_path = nas_path_join(outputs_path, "logs")
    listing_path = nas_path_join(outputs_path, "listing")
    
    nas_create_directory(nas_conn, certificate_path)
    nas_create_directory(nas_conn, data_path)
    nas_create_directory(nas_conn, logs_path)
    nas_create_directory(nas_conn, listing_path)
    
    for transcript_type in TRANSCRIPT_TYPES:
        type_path = nas_path_join(data_path, transcript_type.lower())
        nas_create_directory(nas_conn, type_path)
        
        for ticker in MONITORED_INSTITUTIONS.keys():
            institution_path = nas_path_join(type_path, ticker)
            nas_create_directory(nas_conn, institution_path)

def create_filename(transcript_data: Dict[str, Any], target_ticker: Optional[str] = None) -> str:
    """Create standardized filename from transcript data."""
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
            except Exception:
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

def get_existing_files(nas_conn: SMBConnection, ticker: str, transcript_type: str) -> Set[str]:
    """Get list of existing transcript files for a specific ticker and type from NAS."""
    institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker)
    return set(nas_list_files(nas_conn, institution_path))

def download_transcript_with_retry(nas_conn: SMBConnection, transcript_link: str, 
                                 nas_file_path: str, transcript_id: str) -> Tuple[bool, Optional[str]]:
    """Download transcript with retry logic and upload to NAS. Returns (success, error_message)."""
    for attempt in range(MAX_RETRIES):
        try:
            rate_limit_request()
            
            headers = {
                'Accept': 'application/json',
                'Authorization': configuration.get_basic_auth_token(),
            }
            proxies = {
                'https': "http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
                'http': "http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL)
            }
            
            response = requests.get(
                transcript_link,
                headers=headers,
                proxies=proxies,
                verify=configuration.ssl_ca_cert,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Validate response content
            if len(response.content) == 0:
                raise ValueError("Empty response content")
            
            if len(response.content) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError(f"Response too large: {len(response.content)} bytes")
            
            file_obj = io.BytesIO(response.content)
            file_size = len(response.content)
            
            if nas_upload_file(nas_conn, file_obj, nas_file_path):
                logger.info(f"Downloaded and uploaded {transcript_id} ({file_size:,} bytes)")
                return True, None
            else:
                error_msg = f"Failed to upload {transcript_id} to NAS"
                logger.error(error_msg)
                return False, error_msg
                
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e}"
        except requests.exceptions.Timeout as e:
            error_msg = f"Request timeout: {e}"
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {e}"
        except ValueError as e:
            error_msg = f"Validation error: {e}"
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
        
        logger.warning(f"Download attempt {attempt + 1} failed for {transcript_id}: {error_msg}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
        else:
            logger.error(f"Failed to download {transcript_id} after {MAX_RETRIES} attempts")
            return False, error_msg
    
    return False, "Max retries exceeded"

def download_transcripts_concurrent(nas_conn: SMBConnection, 
                                  new_transcripts: List[Tuple[Dict, str]], 
                                  ticker: str, 
                                  transcript_type: str) -> Tuple[int, List[DownloadFailure]]:
    """Download transcripts concurrently with thread pool."""
    downloaded_count = 0
    failed_downloads = []
    
    def download_single(transcript_data: Tuple[Dict, str]) -> Tuple[bool, Optional[DownloadFailure]]:
        transcript, filename = transcript_data
        
        nas_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", 
                                    transcript_type.lower(), ticker, filename)
        
        success, error_msg = download_transcript_with_retry(
            nas_conn, transcript['transcripts_link'], nas_file_path, filename
        )
        
        if not success:
            failure = DownloadFailure(
                ticker=ticker,
                transcript_type=transcript_type,
                filename=filename,
                transcript_link=transcript['transcripts_link'],
                error=error_msg or "Unknown error",
                attempts=1
            )
            return False, failure
        
        return True, None
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS) as executor:
        future_to_transcript = {
            executor.submit(download_single, t): t 
            for t in new_transcripts
        }
        
        for future in as_completed(future_to_transcript):
            try:
                success, failure = future.result()
                if success:
                    downloaded_count += 1
                elif failure:
                    failed_downloads.append(failure)
            except Exception as e:
                transcript, filename = future_to_transcript[future]
                logger.error(f"Exception in concurrent download for {filename}: {e}")
                failed_downloads.append(DownloadFailure(
                    ticker=ticker,
                    transcript_type=transcript_type,
                    filename=filename,
                    transcript_link=transcript['transcripts_link'],
                    error=str(e),
                    attempts=1
                ))
    
    return downloaded_count, failed_downloads

def save_failed_downloads(nas_conn: SMBConnection, all_failures: List[DownloadFailure]) -> None:
    """Save list of failed downloads for analysis."""
    if not all_failures:
        return
    
    failed_data = {
        'timestamp': datetime.now().isoformat(),
        'total_failures': len(all_failures),
        'failures': [
            {
                'ticker': f.ticker,
                'transcript_type': f.transcript_type,
                'filename': f.filename,
                'error': f.error,
                'attempts': f.attempts
            }
            for f in all_failures
        ]
    }
    
    failed_json = json.dumps(failed_data, indent=2)
    failed_path = nas_path_join(NAS_BASE_PATH, "Outputs", "logs", 
                              f"failed_downloads_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    
    file_obj = io.BytesIO(failed_json.encode('utf-8'))
    if nas_upload_file(nas_conn, file_obj, failed_path):
        logger.info(f"Saved {len(all_failures)} failed downloads to {failed_path}")

def process_bank(ticker: str, institution_info: Dict[str, str], 
                api_instance: transcripts_api.TranscriptsApi, 
                nas_conn: SMBConnection) -> Tuple[Dict[str, Any], List[DownloadFailure]]:
    """Process a single bank: query API, check NAS files, download new transcripts."""
    logger.info(f"Processing: {institution_info['name']} ({ticker})")
    
    bank_downloads = {
        'total': 0,
        'by_type': {transcript_type: 0 for transcript_type in TRANSCRIPT_TYPES}
    }
    all_failures = []
    
    try:
        rate_limit_request()
        
        start_date = datetime.strptime(SYNC_START_DATE, "%Y-%m-%d").date()
        end_date = datetime.now().date()
        
        api_params = {
            'ids': [ticker],
            'start_date': start_date,
            'end_date': end_date,
            'categories': INDUSTRY_CATEGORIES,
            'sort': SORT_ORDER,
            'pagination_limit': PAGINATION_LIMIT,
            'pagination_offset': PAGINATION_OFFSET
        }
        
        response = api_instance.get_transcripts_ids(**api_params)
        
        if not response or not hasattr(response, 'data') or not response.data:
            logger.warning(f"No transcripts found for {ticker}")
            return bank_downloads, all_failures
        
        # Validate response
        if len(response.data) > 10000:  # Sanity check
            logger.warning(f"Unusually large response for {ticker}: {len(response.data)} items")
        
        all_transcripts = [transcript.to_dict() for transcript in response.data]
        logger.info(f"Found {len(all_transcripts)} total transcripts for {ticker}")
        
        # Filter to only transcripts where our ticker is the ONLY primary ID
        filtered_transcripts = []
        for transcript in all_transcripts:
            primary_ids = transcript.get('primary_ids', [])
            if primary_ids == [ticker]:
                filtered_transcripts.append(transcript)
        
        logger.info(f"Filtered to {len(filtered_transcripts)} transcripts where {ticker} is sole primary company")
        
        # Filter for earnings transcripts
        earnings_transcripts = [
            transcript for transcript in filtered_transcripts
            if transcript.get('event_type', '') and 'Earnings' in str(transcript.get('event_type', ''))
            and validate_transcript_data(transcript)
        ]
        
        logger.info(f"Found {len(earnings_transcripts)} valid earnings transcripts for {ticker}")
        
        if not earnings_transcripts:
            return bank_downloads, all_failures
        
        total_downloaded = 0
        
        for transcript_type in TRANSCRIPT_TYPES:
            type_transcripts = [t for t in earnings_transcripts if t.get('transcript_type') == transcript_type]
            
            if not type_transcripts:
                continue
            
            logger.info(f"Processing {len(type_transcripts)} {transcript_type} transcripts")
            
            existing_files = get_existing_files(nas_conn, ticker, transcript_type)
            
            new_transcripts = []
            for transcript in type_transcripts:
                filename = create_filename(transcript, target_ticker=ticker)
                if filename not in existing_files:
                    new_transcripts.append((transcript, filename))
            
            logger.info(f"Found {len(new_transcripts)} new {transcript_type} transcripts to download")
            
            if new_transcripts:
                # Download concurrently
                downloaded_count, failures = download_transcripts_concurrent(
                    nas_conn, new_transcripts, ticker, transcript_type
                )
                
                logger.info(f"Successfully downloaded {downloaded_count}/{len(new_transcripts)} {transcript_type} transcripts")
                total_downloaded += downloaded_count
                bank_downloads['by_type'][transcript_type] = downloaded_count
                all_failures.extend(failures)
        
        bank_downloads['total'] = total_downloaded
        logger.info(f"Completed {ticker}: Downloaded {total_downloaded} new transcripts total, {len(all_failures)} failures")
        
        return bank_downloads, all_failures
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return bank_downloads, all_failures

def generate_final_inventory(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Generate final inventory of all downloaded files from NAS."""
    logger.info("Generating final inventory from NAS")
    
    inventory = {}
    summary_stats = {
        'total_files': 0,
        'by_type': defaultdict(int),
        'by_institution': defaultdict(int)
    }
    
    for transcript_type in TRANSCRIPT_TYPES:
        inventory[transcript_type] = {}
        
        for ticker in MONITORED_INSTITUTIONS.keys():
            institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker)
            files = nas_list_files(nas_conn, institution_path)
            count = len(files)
            inventory[transcript_type][ticker] = count
            
            summary_stats['total_files'] += count
            summary_stats['by_type'][transcript_type] += count
            summary_stats['by_institution'][ticker] += count
    
    # Log summary
    logger.info(f"Total files in repository: {summary_stats['total_files']}")
    for transcript_type in TRANSCRIPT_TYPES:
        logger.info(f"Total {transcript_type}: {summary_stats['by_type'][transcript_type]} files")
    
    # Save inventory files
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    for transcript_type in TRANSCRIPT_TYPES:
        inventory_json = json.dumps(inventory[transcript_type], indent=2)
        inventory_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", 
                                          f"{transcript_type.lower()}_listing.json")
        
        inventory_file_obj = io.BytesIO(inventory_json.encode('utf-8'))
        nas_upload_file(nas_conn, inventory_file_obj, inventory_file_path)
    
    # Save complete inventory with metadata
    complete_inventory = {
        'timestamp': timestamp,
        'summary': dict(summary_stats),
        'inventory': inventory
    }
    
    complete_inventory_json = json.dumps(complete_inventory, indent=2)
    complete_inventory_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", 
                                          f"complete_inventory_{timestamp}.json")
    
    complete_inventory_obj = io.BytesIO(complete_inventory_json.encode('utf-8'))
    nas_upload_file(nas_conn, complete_inventory_obj, complete_inventory_path)
    
    return complete_inventory

def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
    try:
        logger.info("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, SSL_CERT_NAS_PATH)
        if cert_data:
            # Validate certificate data
            if len(cert_data) < 100:  # Basic sanity check
                logger.error("SSL certificate appears to be invalid (too small)")
                return None
            
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

def main() -> None:
    """Main function to orchestrate the transcript repository sync."""
    logger.info("EARNINGS TRANSCRIPT REPOSITORY SYNC v2")
    logger.info(f"Environment: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
    
    temp_cert_path = None
    start_time = datetime.now()
    
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting sync")
        return
    
    temp_cert_path = setup_ssl_certificate(nas_conn)
    if not temp_cert_path:
        nas_conn.close()
        return
    
    global configuration
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=API_USERNAME,
        password=API_PASSWORD,
        proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
        ssl_ca_cert=temp_cert_path
    )
    configuration.get_basic_auth_token()
    logger.info("FactSet API client configured")
    
    nas_conn.close()
    
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to reconnect to NAS - aborting sync")
        return
    
    try:
        create_directory_structure(nas_conn)
        
        total_institutions = len(MONITORED_INSTITUTIONS)
        total_downloaded = 0
        all_failures = []
        download_summary = {}
        
        logger.info(f"Starting sync for {total_institutions} institutions")
        logger.info(f"Date range: {SYNC_START_DATE} to {datetime.now().date()}")
        logger.info(f"Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
        logger.info(f"Rate limit: {RATE_LIMIT_PER_SECOND} requests/second")
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            for i, (ticker, institution_info) in enumerate(MONITORED_INSTITUTIONS.items(), 1):
                logger.info(f"\nProcessing institution {i}/{total_institutions}")
                
                bank_downloads, failures = process_bank(ticker, institution_info, api_instance, nas_conn)
                download_summary[ticker] = bank_downloads
                total_downloaded += bank_downloads['total']
                all_failures.extend(failures)
                
                if i < total_institutions:
                    time.sleep(REQUEST_DELAY)
        
        # Save failed downloads if any
        if all_failures:
            save_failed_downloads(nas_conn, all_failures)
        
        # Generate final inventory
        final_inventory = generate_final_inventory(nas_conn)
        
        # Final summary
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("SYNC COMPLETE")
        logger.info("="*60)
        logger.info(f"Total new transcripts downloaded: {total_downloaded}")
        logger.info(f"Total failures: {len(all_failures)}")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Total files in repository: {final_inventory['summary']['total_files']}")
        
        # Upload log file to NAS
        log_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "logs", 
                                    f"sync_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        with open(logger.temp_log_file, 'rb') as log_file:
            nas_upload_file(nas_conn, log_file, log_file_path)
        
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error during sync: {e}", exc_info=True)
    finally:
        if nas_conn:
            nas_conn.close()
        
        if temp_cert_path and os.path.exists(temp_cert_path):
            try:
                os.unlink(temp_cert_path)
            except Exception:
                pass
        
        if hasattr(logger, 'temp_log_file') and os.path.exists(logger.temp_log_file):
            try:
                os.unlink(logger.temp_log_file)
            except Exception:
                pass

if __name__ == "__main__":
    logger = setup_logging()
    main()