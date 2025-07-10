"""
Stage 0: Bulk Transcript Sync
Downloads ALL historical earnings transcripts from 2023-present for monitored financial institutions.
Self-contained standalone script that loads config from NAS at runtime.
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
user = None
password = None

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        suffix='.log', 
        prefix=f'stage0_sync_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
            
    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None

def load_stage_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load shared configuration from NAS."""
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)
        
        if config_data:
            stage_config = json.loads(config_data.decode('utf-8'))
            logger.info("Successfully loaded shared configuration from NAS")
            return stage_config
        else:
            logger.error("Config file not found on NAS - script cannot proceed")
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
            
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config from NAS: {e}")
        raise

def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    return '/'.join(str(part) for part in parts if part)

def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except:
        return False

def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with parent directory creation."""
    try:
        conn.createDirectory(NAS_SHARE_NAME, dir_path)
        return True
    except Exception:
        if nas_file_exists(conn, dir_path):
            return True
        
        parent_dir = '/'.join(dir_path.split('/')[:-1])
        if parent_dir and parent_dir != dir_path:
            nas_create_directory(conn, parent_dir)
            try:
                conn.createDirectory(NAS_SHARE_NAME, dir_path)
                return True
            except:
                return False
        return False

def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
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
    except Exception:
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

def create_directory_structure(nas_conn: SMBConnection) -> None:
    """Create the directory structure for the transcript repository on NAS."""
    logger.info("Creating directory structure on NAS...")
    
    inputs_path = nas_path_join(NAS_BASE_PATH, "Inputs")
    outputs_path = nas_path_join(NAS_BASE_PATH, "Outputs")
    
    nas_create_directory(nas_conn, inputs_path)
    nas_create_directory(nas_conn, outputs_path)
    
    certificate_path = nas_path_join(inputs_path, "certificate")
    config_path = nas_path_join(inputs_path, "config")
    data_path = nas_path_join(outputs_path, "data")
    logs_path = nas_path_join(outputs_path, "logs")
    listing_path = nas_path_join(outputs_path, "listing")
    
    nas_create_directory(nas_conn, certificate_path)
    nas_create_directory(nas_conn, config_path)
    nas_create_directory(nas_conn, data_path)
    nas_create_directory(nas_conn, logs_path)
    nas_create_directory(nas_conn, listing_path)
    
    for transcript_type in config['api_settings']['transcript_types']:
        type_path = nas_path_join(data_path, transcript_type.lower())
        nas_create_directory(nas_conn, type_path)
        
        for ticker in config['monitored_institutions'].keys():
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
            except:
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
                                 nas_file_path: str, transcript_id: str, configuration) -> bool:
    """Download transcript with retry logic and upload to NAS."""
    
    for attempt in range(config['api_settings']['max_retries']):
        try:
            logger.info(f"Downloading from URL: {transcript_link}")
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
            
            file_obj = io.BytesIO(response.content)
            file_size = len(response.content)
            
            if nas_upload_file(nas_conn, file_obj, nas_file_path):
                logger.info(f"Downloaded and uploaded {transcript_id} ({file_size:,} bytes)")
                return True
            else:
                logger.error(f"Failed to upload {transcript_id} to NAS")
                return False
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed for {transcript_id}: {e}")
            if attempt < config['api_settings']['max_retries'] - 1:
                time.sleep(config['api_settings']['retry_delay'])
            else:
                logger.error(f"Failed to download {transcript_id} after {config['api_settings']['max_retries']} attempts")
                return False
    
    return False

def process_bank(ticker: str, institution_info: Dict[str, str], 
                api_instance: transcripts_api.TranscriptsApi, 
                nas_conn: SMBConnection, configuration) -> Dict[str, Any]:
    """Process a single bank: query API, check NAS files, download new transcripts."""
    logger.info(f"Processing: {institution_info['name']} ({ticker})")
    
    bank_downloads = {
        'total': 0,
        'by_type': {transcript_type: 0 for transcript_type in config['api_settings']['transcript_types']}
    }
    
    try:
        start_date = datetime.strptime(config['stage_0']['sync_start_date'], "%Y-%m-%d").date()
        end_date = datetime.now().date()
        
        api_params = {
            'ids': [ticker],
            'start_date': start_date,
            'end_date': end_date,
            'categories': config['api_settings']['industry_categories'],
            'sort': config['api_settings']['sort_order'],
            'pagination_limit': config['api_settings']['pagination_limit'],
            'pagination_offset': config['api_settings']['pagination_offset']
        }
        
        response = api_instance.get_transcripts_ids(**api_params)
        
        if not response or not hasattr(response, 'data') or not response.data:
            logger.warning(f"No transcripts found for {ticker}")
            return bank_downloads
        
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
        ]
        
        logger.info(f"Found {len(earnings_transcripts)} earnings transcripts for {ticker}")
        
        if not earnings_transcripts:
            return bank_downloads
        
        total_downloaded = 0
        
        for transcript_type in config['api_settings']['transcript_types']:
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
                downloaded_count = 0
                
                for transcript, filename in new_transcripts:
                    if pd.isna(transcript.get('transcripts_link')) or not transcript.get('transcripts_link'):
                        logger.warning(f"No download link for {filename}")
                        continue
                    
                    nas_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", 
                                                transcript_type.lower(), ticker, filename)
                    
                    if download_transcript_with_retry(nas_conn, transcript['transcripts_link'], 
                                                    nas_file_path, filename, configuration):
                        downloaded_count += 1
                        
                    time.sleep(config['api_settings']['request_delay'])
                
                logger.info(f"Successfully downloaded {downloaded_count}/{len(new_transcripts)} {transcript_type} transcripts")
                total_downloaded += downloaded_count
                bank_downloads['by_type'][transcript_type] = downloaded_count
        
        bank_downloads['total'] = total_downloaded
        logger.info(f"Completed {ticker}: Downloaded {total_downloaded} new transcripts total")
        
        return bank_downloads
        
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return bank_downloads

def generate_final_inventory(nas_conn: SMBConnection) -> None:
    """Generate final inventory of all downloaded files from NAS."""
    logger.info("Generating final inventory from NAS")
    
    inventory = {}
    
    for transcript_type in config['api_settings']['transcript_types']:
        inventory[transcript_type] = {}
        
        for ticker in config['monitored_institutions'].keys():
            institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker)
            files = nas_list_files(nas_conn, institution_path)
            inventory[transcript_type][ticker] = len(files)
    
    for transcript_type in config['api_settings']['transcript_types']:
        total_type = sum(count for count in inventory[transcript_type].values())
        logger.info(f"Total {transcript_type}: {total_type} files")
    
    # Save inventory files
    for transcript_type in config['api_settings']['transcript_types']:
        inventory_json = json.dumps(inventory[transcript_type], indent=2)
        inventory_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", 
                                          f"{transcript_type.lower()}_listing.json")
        
        inventory_file_obj = io.BytesIO(inventory_json.encode('utf-8'))
        nas_upload_file(nas_conn, inventory_file_obj, inventory_file_path)
    
    complete_inventory_json = json.dumps(inventory, indent=2)
    complete_inventory_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", 
                                          f"complete_inventory_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    
    complete_inventory_obj = io.BytesIO(complete_inventory_json.encode('utf-8'))
    nas_upload_file(nas_conn, complete_inventory_obj, complete_inventory_path)

def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
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
    """Main function to orchestrate the Stage 0 bulk transcript sync."""
    global config, logger, user, password
    
    logger = setup_logging()
    logger.info("STAGE 0: BULK TRANSCRIPT SYNC")
    
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
    
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=API_USERNAME,
        password=API_PASSWORD,
        proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
        ssl_ca_cert=temp_cert_path
    )
    configuration.get_basic_auth_token()
    logger.info("FactSet API client configured")
    
    # Close and reconnect to NAS for main operations
    nas_conn.close()
    
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to reconnect to NAS - aborting sync")
        return
    
    try:
        create_directory_structure(nas_conn)
        
        start_time = datetime.now()
        total_institutions = len(config['monitored_institutions'])
        total_downloaded = 0
        
        logger.info(f"Starting bulk sync for {total_institutions} institutions")
        logger.info(f"Date range: {config['stage_0']['sync_start_date']} to {datetime.now().date()}")
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            for i, (ticker, institution_info) in enumerate(config['monitored_institutions'].items(), 1):
                logger.info(f"Processing institution {i}/{total_institutions}")
                
                bank_downloads = process_bank(ticker, institution_info, api_instance, nas_conn, configuration)
                total_downloaded += bank_downloads['total']
                
                if i < total_institutions:
                    time.sleep(config['api_settings']['request_delay'])
        
        generate_final_inventory(nas_conn)
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info("STAGE 0 BULK SYNC COMPLETE")
        logger.info(f"Total new transcripts downloaded: {total_downloaded}")
        logger.info(f"Execution time: {execution_time}")
        
        # Upload log file to NAS
        log_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "logs", 
                                    f"stage0_sync_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        with open(logger.temp_log_file, 'rb') as log_file:
            nas_upload_file(nas_conn, log_file, log_file_path)
        
    finally:
        if nas_conn:
            nas_conn.close()
        
        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
            except:
                pass
        
        try:
            os.unlink(logger.temp_log_file)
        except:
            pass

if __name__ == "__main__":
    main()