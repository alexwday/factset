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

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION VARIABLES
# =============================================================================

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Repository Configuration
REPOSITORY_ROOT = "transcript_repository"
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

# Set up logging
def setup_logging():
    """Set up logging configuration"""
    log_dir = Path(REPOSITORY_ROOT) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"sync_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

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

def create_directory_structure():
    """
    Create the directory structure for the transcript repository
    """
    base_path = Path(REPOSITORY_ROOT)
    
    # Create main directories
    for transcript_type in TRANSCRIPT_TYPES:
        type_dir = base_path / transcript_type.lower()
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each institution
        for ticker in MONITORED_INSTITUTIONS.keys():
            institution_dir = type_dir / ticker
            institution_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logs directory
    (base_path / "logs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure at {base_path}")

def create_filename(transcript_data):
    """
    Create standardized filename from transcript data
    Format: {primary_id}_{event_date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml
    """
    try:
        # Extract and sanitize components
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

def get_existing_files(ticker, transcript_type):
    """
    Get list of existing transcript files for a specific ticker and type
    """
    institution_dir = Path(REPOSITORY_ROOT) / transcript_type.lower() / ticker
    
    if not institution_dir.exists():
        return set()
    
    existing_files = set()
    for file_path in institution_dir.glob("*.xml"):
        existing_files.add(file_path.name)
    
    return existing_files

def download_transcript_with_retry(transcript_link, output_path, transcript_id):
    """
    Download transcript with retry logic
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
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = output_path.stat().st_size
            logger.info(f"  ✓ Downloaded {transcript_id} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.warning(f"  ✗ Download attempt {attempt + 1} failed for {transcript_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"  Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"  ✗ Failed to download {transcript_id} after {MAX_RETRIES} attempts")
                return False
    
    return False

def process_bank(ticker, institution_info, api_instance):
    """
    Process a single bank: query API, check local files, download new transcripts
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING: {institution_info['name']} ({ticker})")
    logger.info(f"{'='*60}")
    
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
            
            # Step 3: Check existing files
            existing_files = get_existing_files(ticker, transcript_type)
            logger.info(f"  Found {len(existing_files)} existing {transcript_type} files for {ticker}")
            
            # Step 4: Identify new transcripts
            new_transcripts = []
            for transcript in type_transcripts:
                filename = create_filename(transcript)
                if filename not in existing_files:
                    new_transcripts.append((transcript, filename))
            
            logger.info(f"  Found {len(new_transcripts)} new {transcript_type} transcripts to download")
            
            # Step 5: Download new transcripts
            if new_transcripts:
                logger.info(f"  Downloading {len(new_transcripts)} new {transcript_type} transcripts...")
                downloaded_count = 0
                
                for transcript, filename in new_transcripts:
                    if pd.isna(transcript.get('transcripts_link')) or not transcript.get('transcripts_link'):
                        logger.warning(f"    No download link for {filename}")
                        continue
                    
                    output_path = Path(REPOSITORY_ROOT) / transcript_type.lower() / ticker / filename
                    
                    if download_transcript_with_retry(transcript['transcripts_link'], output_path, filename):
                        downloaded_count += 1
                        
                    # Add delay between downloads
                    time.sleep(REQUEST_DELAY)
                
                logger.info(f"  ✓ Successfully downloaded {downloaded_count}/{len(new_transcripts)} {transcript_type} transcripts")
                total_downloaded += downloaded_count
            else:
                logger.info(f"  ✓ No new {transcript_type} transcripts to download")
        
        logger.info(f"\n✓ COMPLETED {ticker}: Downloaded {total_downloaded} new transcripts total")
        
    except Exception as e:
        logger.error(f"✗ Error processing {ticker}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def generate_final_inventory():
    """
    Generate final inventory of all downloaded files
    """
    logger.info(f"\n{'='*60}")
    logger.info("GENERATING FINAL INVENTORY")
    logger.info(f"{'='*60}")
    
    inventory = {}
    
    for transcript_type in TRANSCRIPT_TYPES:
        inventory[transcript_type] = {}
        type_dir = Path(REPOSITORY_ROOT) / transcript_type.lower()
        
        if not type_dir.exists():
            continue
            
        for ticker in MONITORED_INSTITUTIONS.keys():
            ticker_dir = type_dir / ticker
            if not ticker_dir.exists():
                inventory[transcript_type][ticker] = 0
                continue
                
            files = list(ticker_dir.glob("*.xml"))
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
    
    # Save inventory to JSON
    inventory_file = Path(REPOSITORY_ROOT) / "logs" / f"inventory_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(inventory_file, 'w') as f:
        json.dump(inventory, f, indent=2)
    
    logger.info(f"\nInventory saved to: {inventory_file}")

def main():
    """
    Main function to orchestrate the transcript repository sync
    """
    logger.info("=" * 80)
    logger.info("EARNINGS TRANSCRIPT REPOSITORY SYNC v2")
    logger.info("=" * 80)
    
    # Check SSL certificate
    if not os.path.exists(SSL_CERT_PATH):
        logger.error(f"SSL certificate not found at {SSL_CERT_PATH}")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Track overall progress
    start_time = datetime.now()
    total_institutions = len(MONITORED_INSTITUTIONS)
    processed_institutions = 0
    
    logger.info(f"Starting sync for {total_institutions} institutions")
    logger.info(f"Date range: {SYNC_START_DATE} to {datetime.now().date()}")
    logger.info(f"Industry categories: {INDUSTRY_CATEGORIES}")
    logger.info(f"Transcript types: {TRANSCRIPT_TYPES}")
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Process each institution
            for i, (ticker, institution_info) in enumerate(MONITORED_INSTITUTIONS.items(), 1):
                logger.info(f"\nProcessing institution {i}/{total_institutions}")
                
                process_bank(ticker, institution_info, api_instance)
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
    
    # Generate final inventory
    generate_final_inventory()
    
    # Final summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    logger.info(f"\n{'='*80}")
    logger.info("SYNC COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Processed {processed_institutions}/{total_institutions} institutions")
    logger.info(f"Execution time: {execution_time}")
    logger.info(f"Repository location: {Path(REPOSITORY_ROOT).absolute()}")

if __name__ == "__main__":
    logger = setup_logging()
    main()