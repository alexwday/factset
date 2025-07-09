"""
Earnings Transcript Repository Sync v2
This script maintains a local repository of earnings transcripts for monitored financial institutions.
Simplified approach with file-based inventory and robust bank-by-bank processing.
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
from pathlib import Path
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tempfile
import io
from smb.SMBConnection import SMBConnection

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# SSL and Proxy Configuration
SSL_CERT_NAS_PATH = "Inputs/certificate/certificate.cer"  # Path to SSL certificate on NAS
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Email Configuration
SMTP_SERVER = "your.smtp.server.com"  # Replace with your SMTP server
SMTP_PORT = 587  # Replace with your SMTP port (587 for TLS, 465 for SSL, 25 for non-encrypted)
EMAIL_FROM = "your.email@domain.com"  # Replace with sender email
EMAIL_TO = ["recipient1@domain.com", "recipient2@domain.com"]  # Replace with recipient emails
EMAIL_USERNAME = "your.email@domain.com"  # Set to your email username for authentication
EMAIL_PASSWORD = "your_email_password"  # Set to your email password for authentication
USE_TLS = True  # Set to True if your server requires TLS/STARTTLS (most corporate servers do)
SEND_EMAIL_NOTIFICATIONS = True  # Set to False to disable email notifications

# NAS Configuration
NAS_USERNAME = "your_nas_username"  # NAS username
NAS_PASSWORD = "your_nas_password"  # NAS password
NAS_SERVER_IP = "192.168.1.100"  # NAS server IP address
NAS_SERVER_NAME = "NAS-SERVER"  # NAS server name (for NTLM)
NAS_SHARE_NAME = "shared_folder"  # Share name on NAS
NAS_BASE_PATH = "transcript_repository"  # Base path within the share for our files
NAS_PORT = 445  # SMB port (usually 445)
CLIENT_MACHINE_NAME = "SYNC-CLIENT"  # Name of this client machine

# Repository Configuration
SYNC_START_DATE = "2023-01-01"  # From 2023 to present

# Monitored Financial Institutions
MONITORED_INSTITUTIONS = {
    # Major Canadian Banks
    "RY-CA": {"name": "Royal Bank of Canada", "region": "CA", "type": "Bank"},
    "TD-CA": {"name": "Toronto-Dominion Bank", "region": "CA", "type": "Bank"},
    "BNS-CA": {"name": "Bank of Nova Scotia", "region": "CA", "type": "Bank"},
    "BMO-CA": {"name": "Bank of Montreal", "region": "CA", "type": "Bank"},
    "CM-CA": {"name": "Canadian Imperial Bank of Commerce", "region": "CA", "type": "Bank"},
    "NA-CA": {"name": "National Bank of Canada", "region": "CA", "type": "Bank"},
    
    # Major US Banks
    "JPM-US": {"name": "JPMorgan Chase & Co.", "region": "US", "type": "Bank"},
    "BAC-US": {"name": "Bank of America Corporation", "region": "US", "type": "Bank"},
    "WFC-US": {"name": "Wells Fargo & Company", "region": "US", "type": "Bank"},
    "C-US": {"name": "Citigroup Inc.", "region": "US", "type": "Bank"},
    "GS-US": {"name": "Goldman Sachs Group Inc.", "region": "US", "type": "Bank"},
    "MS-US": {"name": "Morgan Stanley", "region": "US", "type": "Bank"},
    
    # Major Insurance Companies
    "MFC-CA": {"name": "Manulife Financial Corporation", "region": "CA", "type": "Insurance"},
    "SLF-CA": {"name": "Sun Life Financial Inc.", "region": "CA", "type": "Insurance"},
    "UNH-US": {"name": "UnitedHealth Group Incorporated", "region": "US", "type": "Insurance"},
}

# Industry Categories to Monitor
INDUSTRY_CATEGORIES = ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"]

# Transcript Types
TRANSCRIPT_TYPES = ["Corrected", "Raw", "NearRealTime"]

# API Configuration
SORT_ORDER = ["-storyDateTime"]
PAGINATION_LIMIT = 1000
PAGINATION_OFFSET = 0
REQUEST_DELAY = 2.0  # Seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # Seconds between retries

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication
user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Configure FactSet API client
configuration = fds.sdk.EventsandTranscripts.Configuration(
    username=API_USERNAME,
    password=API_PASSWORD,
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=SSL_CERT_PATH
)
configuration.get_basic_auth_token()

# =============================================================================
# NAS CONNECTION FUNCTIONS
# =============================================================================

def get_nas_connection():
    """
    Create and return an SMB connection to the NAS
    """
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        # Connect to the NAS
        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            logger.info(f"✓ Connected to NAS: {NAS_SERVER_IP}")
            return conn
        else:
            logger.error(f"✗ Failed to connect to NAS: {NAS_SERVER_IP}")
            return None
            
    except Exception as e:
        logger.error(f"✗ Error connecting to NAS: {e}")
        return None

def nas_path_join(*parts):
    """
    Join path parts for NAS paths (always use forward slashes)
    """
    return '/'.join(str(part) for part in parts if part)

def nas_file_exists(conn, file_path):
    """
    Check if a file exists on the NAS
    """
    try:
        # Try to get file attributes - if it succeeds, file exists
        conn.getAttributes(NAS_SHARE_NAME, file_path)
        return True
    except:
        return False

def nas_create_directory(conn, dir_path):
    """
    Create directory on NAS (creates parent directories as needed)
    """
    try:
        # Try to create the directory
        conn.createDirectory(NAS_SHARE_NAME, dir_path)
        logger.debug(f"Created NAS directory: {dir_path}")
        return True
    except Exception as e:
        # Directory might already exist, check if it exists
        if nas_file_exists(conn, dir_path):
            return True
        else:
            # Try to create parent directories first
            parent_dir = '/'.join(dir_path.split('/')[:-1])
            if parent_dir and parent_dir != dir_path:
                nas_create_directory(conn, parent_dir)
                try:
                    conn.createDirectory(NAS_SHARE_NAME, dir_path)
                    logger.debug(f"Created NAS directory: {dir_path}")
                    return True
                except:
                    logger.error(f"Failed to create NAS directory: {dir_path}")
                    return False
            return False

def nas_upload_file(conn, local_file_obj, nas_file_path):
    """
    Upload a file object to NAS
    """
    try:
        # Ensure parent directory exists
        parent_dir = '/'.join(nas_file_path.split('/')[:-1])
        if parent_dir:
            nas_create_directory(conn, parent_dir)
        
        # Upload the file
        conn.storeFile(NAS_SHARE_NAME, nas_file_path, local_file_obj)
        logger.debug(f"Uploaded file to NAS: {nas_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload file to NAS {nas_file_path}: {e}")
        return False

def nas_download_file(conn, nas_file_path):
    """
    Download a file from NAS and return as bytes
    """
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(NAS_SHARE_NAME, nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        logger.error(f"Failed to download file from NAS {nas_file_path}: {e}")
        return None

def nas_list_files(conn, directory_path, pattern="*.xml"):
    """
    List files in a NAS directory matching a pattern
    """
    try:
        files = conn.listPath(NAS_SHARE_NAME, directory_path)
        matching_files = []
        
        for file_info in files:
            if not file_info.isDirectory and file_info.filename.endswith('.xml'):
                matching_files.append(file_info.filename)
        
        return matching_files
    except Exception as e:
        logger.debug(f"Directory doesn't exist or is empty: {directory_path}")
        return []

# Set up logging
def setup_logging():
    """Set up logging configuration (now uses temporary local file + NAS upload)"""
    # Create temporary log file locally
    temp_log_file = tempfile.NamedTemporaryFile(
        mode='w+', 
        suffix='.log', 
        prefix=f'sync_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
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
    logger.temp_log_file = temp_log_file.name  # Store for later upload
    
    return logger

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_for_filename(text):
    """
    Sanitize text to be safe for filename
    """
    if pd.isna(text) or text is None:
        return "unknown"
    
    # Convert to string and remove problematic characters
    clean_text = str(text)
    clean_text = re.sub(r'[<>:"/\\|?*]', '_', clean_text)
    clean_text = re.sub(r'[^\w\s-]', '', clean_text)
    clean_text = re.sub(r'[-\s]+', '_', clean_text)
    return clean_text.strip('_')

def create_directory_structure(nas_conn):
    """
    Create the directory structure for the transcript repository on NAS
    """
    logger.info("Creating directory structure on NAS...")
    
    # Create main folder structure
    inputs_path = nas_path_join(NAS_BASE_PATH, "Inputs")
    outputs_path = nas_path_join(NAS_BASE_PATH, "Outputs")
    
    nas_create_directory(nas_conn, inputs_path)
    nas_create_directory(nas_conn, outputs_path)
    
    # Create Inputs subdirectories
    certificate_path = nas_path_join(inputs_path, "certificate")
    nas_create_directory(nas_conn, certificate_path)
    
    # Create Outputs subdirectories
    data_path = nas_path_join(outputs_path, "data")
    logs_path = nas_path_join(outputs_path, "logs")
    listing_path = nas_path_join(outputs_path, "listing")
    
    nas_create_directory(nas_conn, data_path)
    nas_create_directory(nas_conn, logs_path)
    nas_create_directory(nas_conn, listing_path)
    
    # Create data subdirectories for each transcript type
    for transcript_type in TRANSCRIPT_TYPES:
        type_path = nas_path_join(data_path, transcript_type.lower())
        nas_create_directory(nas_conn, type_path)
        
        # Create subdirectories for each institution
        for ticker in MONITORED_INSTITUTIONS.keys():
            institution_path = nas_path_join(type_path, ticker)
            nas_create_directory(nas_conn, institution_path)
    
    logger.info(f"✓ Created directory structure on NAS: {NAS_BASE_PATH}")
    logger.info(f"  - Inputs: {inputs_path}")
    logger.info(f"  - Outputs: {outputs_path}")

def create_filename(transcript_data, target_ticker=None):
    """
    Create standardized filename from transcript data
    Format: {primary_id}_{event_date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml
    """
    try:
        # Extract and sanitize components
        # Use target_ticker if provided, otherwise use first primary_id
        if target_ticker:
            primary_id = sanitize_for_filename(target_ticker)
        else:
            primary_id = sanitize_for_filename(transcript_data.get('primary_ids', ['unknown'])[0] if transcript_data.get('primary_ids') else 'unknown')
        
        # Handle event_date
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
        
        filename = f"{primary_id}_{event_date_str}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml"
        
        return filename
        
    except Exception as e:
        logger.error(f"Error creating filename: {e}")
        # Fallback filename
        return f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"

def get_existing_files(nas_conn, ticker, transcript_type):
    """
    Get list of existing transcript files for a specific ticker and type from NAS
    """
    institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker)
    
    existing_files = nas_list_files(nas_conn, institution_path)
    return set(existing_files)

def download_transcript_with_retry(nas_conn, transcript_link, nas_file_path, transcript_id):
    """
    Download transcript with retry logic and upload to NAS
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"  Downloading {transcript_id} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Set up authentication and proxy
            auth = (configuration.username, configuration.password)
            proxies = {
                'http': configuration.proxy,
                'https': configuration.proxy
            }
            
            # Download the transcript
            response = requests.get(
                transcript_link,
                auth=auth,
                proxies=proxies,
                verify=configuration.ssl_ca_cert,
                timeout=30
            )
            
            response.raise_for_status()
            
            # Create file object from response content
            file_obj = io.BytesIO(response.content)
            file_size = len(response.content)
            
            # Upload to NAS
            if nas_upload_file(nas_conn, file_obj, nas_file_path):
                logger.info(f"  ✓ Downloaded and uploaded {transcript_id} ({file_size:,} bytes)")
                return True
            else:
                logger.error(f"  ✗ Failed to upload {transcript_id} to NAS")
                return False
            
        except Exception as e:
            logger.warning(f"  ✗ Download attempt {attempt + 1} failed for {transcript_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"  ✗ Failed to download {transcript_id} after {MAX_RETRIES} attempts")
                return False
    
    return False

def send_email_notification(download_summary, total_downloaded, execution_time):
    """
    Send email notification with download summary
    """
    if not SEND_EMAIL_NOTIFICATIONS:
        logger.info("Email notifications disabled")
        return
    
    try:
        logger.info("Sending email notification...")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = ', '.join(EMAIL_TO)
        msg['Subject'] = f"Earnings Transcript Sync Complete - {total_downloaded} New Transcripts"
        
        # Create email body
        body = f"""
Earnings Transcript Repository Sync Complete

Execution Time: {execution_time}
Total New Transcripts Downloaded: {total_downloaded}

DOWNLOAD SUMMARY BY INSTITUTION:
{'='*50}
"""
        
        if download_summary:
            for ticker, details in download_summary.items():
                if details['total'] > 0:
                    institution_name = MONITORED_INSTITUTIONS[ticker]['name']
                    body += f"\n{institution_name} ({ticker}): {details['total']} new transcripts\n"
                    
                    for transcript_type, count in details['by_type'].items():
                        if count > 0:
                            body += f"  - {transcript_type}: {count}\n"
        else:
            body += "\nNo new transcripts downloaded.\n"
        
        body += f"""
{'='*50}

Repository Location: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}
  - Transcript Data: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}\\Outputs\\data
  - Logs: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}\\Outputs\\logs
  - Inventory: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}\\Outputs\\listing

Sync Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated message from the Earnings Transcript Sync system.
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to server and send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        
        # Enable TLS if configured
        if USE_TLS:
            server.starttls()
        
        # Login only if credentials are provided
        if EMAIL_USERNAME and EMAIL_PASSWORD:
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(EMAIL_FROM, EMAIL_TO, text)
        server.quit()
        
        logger.info(f"✓ Email notification sent to {', '.join(EMAIL_TO)}")
        
    except Exception as e:
        logger.error(f"✗ Failed to send email notification: {e}")

def process_bank(ticker, institution_info, api_instance, nas_conn):
    """
    Process a single bank: query API, check NAS files, download new transcripts
    Returns summary of downloads for email notification
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING: {institution_info['name']} ({ticker})")
    logger.info(f"{'='*60}")
    
    # Initialize download tracking for this bank
    bank_downloads = {
        'total': 0,
        'by_type': {transcript_type: 0 for transcript_type in TRANSCRIPT_TYPES}
    }
    
    try:
        # Step 1: Query API for transcripts
        logger.info(f"Step 1: Querying API for {ticker} transcripts...")
        
        # Calculate date range
        start_date = datetime.strptime(SYNC_START_DATE, "%Y-%m-%d").date()
        end_date = datetime.now().date()
        
        # API parameters
        api_params = {
            'ids': [ticker],
            'start_date': start_date,
            'end_date': end_date,
            'categories': INDUSTRY_CATEGORIES,
            'sort': SORT_ORDER,
            'pagination_limit': PAGINATION_LIMIT,
            'pagination_offset': PAGINATION_OFFSET
        }
        
        # Make API call
        response = api_instance.get_transcripts_ids(**api_params)
        
        if not response or not hasattr(response, 'data') or not response.data:
            logger.warning(f"  No transcripts found for {ticker}")
            return
        
        # Convert to list of dictionaries for easier handling
        all_transcripts = []
        for transcript in response.data:
            transcript_dict = transcript.to_dict()
            all_transcripts.append(transcript_dict)
        
        logger.info(f"  ✓ Found {len(all_transcripts)} total transcripts for {ticker}")
        
        # Filter to only transcripts where our ticker is the ONLY primary ID
        if all_transcripts:
            original_count = len(all_transcripts)
            filtered_transcripts = []
            mixed_ownership_count = 0
            
            for transcript in all_transcripts:
                primary_ids = transcript.get('primary_ids', [])
                
                # Only keep transcripts where our ticker is the ONLY primary ID
                if primary_ids == [ticker]:
                    filtered_transcripts.append(transcript)
                elif ticker in primary_ids:
                    mixed_ownership_count += 1
                    logger.debug(f"    Skipping mixed ownership transcript with primary_ids: {primary_ids}")
            
            if mixed_ownership_count > 0:
                logger.info(f"  ✓ Filtered out {mixed_ownership_count} mixed ownership transcripts")
            
            logger.info(f"  ✓ Filtered to {len(filtered_transcripts)} transcripts where {ticker} is the sole primary company")
            all_transcripts = filtered_transcripts
        
        # Filter for earnings transcripts
        earnings_transcripts = []
        for transcript in all_transcripts:
            event_type = transcript.get('event_type', '')
            if event_type and 'Earnings' in str(event_type):
                earnings_transcripts.append(transcript)
        
        logger.info(f"  ✓ Found {len(earnings_transcripts)} earnings transcripts for {ticker}")
        
        if not earnings_transcripts:
            logger.info(f"  No earnings transcripts to process for {ticker}")
            return
        
        # Step 2: Process by transcript type
        total_downloaded = 0
        
        for transcript_type in TRANSCRIPT_TYPES:
            logger.info(f"\nStep 2: Processing {transcript_type} transcripts for {ticker}...")
            
            # Filter transcripts by type
            type_transcripts = [t for t in earnings_transcripts if t.get('transcript_type') == transcript_type]
            
            if not type_transcripts:
                logger.info(f"  No {transcript_type} transcripts found for {ticker}")
                continue
            
            logger.info(f"  Found {len(type_transcripts)} {transcript_type} transcripts")
            
            # Step 3: Check existing files on NAS
            existing_files = get_existing_files(nas_conn, ticker, transcript_type)
            logger.info(f"  Found {len(existing_files)} existing {transcript_type} files for {ticker}")
            
            # Step 4: Identify new transcripts
            new_transcripts = []
            for transcript in type_transcripts:
                filename = create_filename(transcript, target_ticker=ticker)
                if filename not in existing_files:
                    new_transcripts.append((transcript, filename))
            
            logger.info(f"  Found {len(new_transcripts)} new {transcript_type} transcripts to download")
            
            # Step 5: Download new transcripts and upload to NAS
            if new_transcripts:
                logger.info(f"  Downloading {len(new_transcripts)} new {transcript_type} transcripts...")
                downloaded_count = 0
                
                for transcript, filename in new_transcripts:
                    if pd.isna(transcript.get('transcripts_link')) or not transcript.get('transcripts_link'):
                        logger.warning(f"    No download link for {filename}")
                        continue
                    
                    # Create NAS file path
                    nas_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker, filename)
                    
                    if download_transcript_with_retry(nas_conn, transcript['transcripts_link'], nas_file_path, filename):
                        downloaded_count += 1
                        
                    # Add delay between downloads
                    time.sleep(REQUEST_DELAY)
                
                logger.info(f"  ✓ Successfully downloaded {downloaded_count}/{len(new_transcripts)} {transcript_type} transcripts")
                total_downloaded += downloaded_count
                bank_downloads['by_type'][transcript_type] = downloaded_count
            else:
                logger.info(f"  ✓ No new {transcript_type} transcripts to download")
        
        bank_downloads['total'] = total_downloaded
        logger.info(f"\n✓ COMPLETED {ticker}: Downloaded {total_downloaded} new transcripts total")
        
        return bank_downloads
        
    except Exception as e:
        logger.error(f"✗ Error processing {ticker}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return bank_downloads  # Return empty downloads on error

def generate_final_inventory(nas_conn):
    """
    Generate final inventory of all downloaded files from NAS
    """
    logger.info(f"\n{'='*60}")
    logger.info("GENERATING FINAL INVENTORY FROM NAS")
    logger.info(f"{'='*60}")
    
    inventory = {}
    
    for transcript_type in TRANSCRIPT_TYPES:
        inventory[transcript_type] = {}
        
        for ticker in MONITORED_INSTITUTIONS.keys():
            institution_path = nas_path_join(NAS_BASE_PATH, "Outputs", "data", transcript_type.lower(), ticker)
            files = nas_list_files(nas_conn, institution_path)
            inventory[transcript_type][ticker] = len(files)
    
    # Log summary
    logger.info("\nFINAL INVENTORY SUMMARY:")
    logger.info("-" * 40)
    
    for transcript_type in TRANSCRIPT_TYPES:
        logger.info(f"\n{transcript_type} Transcripts:")
        total_type = 0
        for ticker, count in inventory[transcript_type].items():
            if count > 0:
                institution_name = MONITORED_INSTITUTIONS[ticker]['name']
                logger.info(f"  {ticker} ({institution_name}): {count} files")
                total_type += count
        logger.info(f"  Total {transcript_type}: {total_type} files")
    
    # Save individual inventory files by transcript type to listing folder
    for transcript_type in TRANSCRIPT_TYPES:
        inventory_json = json.dumps(inventory[transcript_type], indent=2)
        inventory_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", f"{transcript_type.lower()}_listing.json")
        
        inventory_file_obj = io.BytesIO(inventory_json.encode('utf-8'))
        if nas_upload_file(nas_conn, inventory_file_obj, inventory_file_path):
            logger.info(f"  {transcript_type} listing saved: {inventory_file_path}")
        else:
            logger.error(f"  Failed to save {transcript_type} listing: {inventory_file_path}")
    
    # Save complete inventory to listing folder
    complete_inventory_json = json.dumps(inventory, indent=2)
    complete_inventory_path = nas_path_join(NAS_BASE_PATH, "Outputs", "listing", f"complete_inventory_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    
    complete_inventory_obj = io.BytesIO(complete_inventory_json.encode('utf-8'))
    if nas_upload_file(nas_conn, complete_inventory_obj, complete_inventory_path):
        logger.info(f"\nComplete inventory saved: {complete_inventory_path}")
    else:
        logger.error(f"Failed to save complete inventory: {complete_inventory_path}")

def setup_ssl_certificate(nas_conn):
    """
    Download SSL certificate from NAS and set up for use
    """
    try:
        logger.info("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, SSL_CERT_NAS_PATH)
        if cert_data:
            # Create temporary file for SSL certificate
            temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()
            
            # Update SSL paths
            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name
            
            logger.info(f"✓ SSL certificate downloaded from NAS: {SSL_CERT_NAS_PATH}")
            return temp_cert.name
        else:
            logger.error(f"✗ Failed to download SSL certificate from NAS: {SSL_CERT_NAS_PATH}")
            return None
    except Exception as e:
        logger.error(f"✗ Error downloading SSL certificate from NAS: {e}")
        return None

def main():
    """
    Main function to orchestrate the transcript repository sync
    """
    logger.info("=" * 80)
    logger.info("EARNINGS TRANSCRIPT REPOSITORY SYNC v2")
    logger.info("=" * 80)
    
    temp_cert_path = None
    
    # Connect to NAS to get SSL certificate first
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting sync")
        return
    
    # Download SSL certificate from NAS
    temp_cert_path = setup_ssl_certificate(nas_conn)
    if not temp_cert_path:
        nas_conn.close()
        return
    
    # Close initial connection and reconnect with SSL certificate
    nas_conn.close()
    
    # Reconnect to NAS with SSL certificate now available
    nas_conn = get_nas_connection()
    if not nas_conn:
        logger.error("Failed to connect to NAS - aborting sync")
        return
    
    try:
        # Create directory structure on NAS
        create_directory_structure(nas_conn)
        
        # Track overall progress
        start_time = datetime.now()
        total_institutions = len(MONITORED_INSTITUTIONS)
        processed_institutions = 0
        download_summary = {}
        total_downloaded = 0
        
        logger.info(f"Starting sync for {total_institutions} institutions")
        logger.info(f"Date range: {SYNC_START_DATE} to {datetime.now().date()}")
        logger.info(f"Industry categories: {INDUSTRY_CATEGORIES}")
        logger.info(f"Transcript types: {TRANSCRIPT_TYPES}")
        logger.info(f"NAS Location: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}")
        
        try:
            with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
                api_instance = transcripts_api.TranscriptsApi(api_client)
                
                # Process each institution
                for i, (ticker, institution_info) in enumerate(MONITORED_INSTITUTIONS.items(), 1):
                    logger.info(f"\nProcessing institution {i}/{total_institutions}")
                    
                    bank_downloads = process_bank(ticker, institution_info, api_instance, nas_conn)
                    download_summary[ticker] = bank_downloads
                    total_downloaded += bank_downloads['total']
                    processed_institutions += 1
                    
                    # Add delay between institutions
                    if i < total_institutions:
                        logger.info(f"Waiting {REQUEST_DELAY} seconds before next institution...")
                        time.sleep(REQUEST_DELAY)
        
        except Exception as e:
            logger.error(f"Critical error during sync: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Generate final inventory on NAS
        generate_final_inventory(nas_conn)
        
        # Final summary
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        logger.info(f"\n{'='*80}")
        logger.info("SYNC COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Processed {processed_institutions}/{total_institutions} institutions")
        logger.info(f"Total new transcripts downloaded: {total_downloaded}")
        logger.info(f"Execution time: {execution_time}")
        logger.info(f"Repository location: \\\\{NAS_SERVER_IP}\\{NAS_SHARE_NAME}\\{NAS_BASE_PATH}")
        
        # Upload log file to NAS
        try:
            log_file_path = nas_path_join(NAS_BASE_PATH, "Outputs", "logs", f"sync_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
            with open(logger.temp_log_file, 'rb') as log_file:
                if nas_upload_file(nas_conn, log_file, log_file_path):
                    logger.info(f"Log file uploaded to NAS: {log_file_path}")
                else:
                    logger.error(f"Failed to upload log file to NAS")
        except Exception as e:
            logger.error(f"Error uploading log file: {e}")
        
        # Send email notification
        send_email_notification(download_summary, total_downloaded, execution_time)
        
    finally:
        # Clean up NAS connection
        if nas_conn:
            nas_conn.close()
            logger.info("NAS connection closed")
        
        # Clean up temporary SSL certificate
        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
                logger.info("Temporary SSL certificate cleaned up")
            except:
                pass
        
        # Clean up temporary log file
        try:
            os.unlink(logger.temp_log_file)
        except:
            pass

if __name__ == "__main__":
    logger = setup_logging()
    main()