"""
Earnings Transcript Repository Sync
This script maintains a local repository of earnings transcripts for monitored financial institutions.
It supports incremental sync, all transcript types, and robust error handling.
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
BATCH_SIZE = 10  # Process institutions in batches

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
    "USB-US": {"name": "U.S. Bancorp", "region": "US", "type": "Bank"},
    "PNC-US": {"name": "PNC Financial Services Group Inc.", "region": "US", "type": "Bank"},
    "TFC-US": {"name": "Truist Financial Corporation", "region": "US", "type": "Bank"},
    "COF-US": {"name": "Capital One Financial Corporation", "region": "US", "type": "Bank"},
    
    # Major Canadian Insurance Companies
    "MFC-CA": {"name": "Manulife Financial Corporation", "region": "CA", "type": "Insurance"},
    "SLF-CA": {"name": "Sun Life Financial Inc.", "region": "CA", "type": "Insurance"},
    "GWO-CA": {"name": "Great-West Lifeco Inc.", "region": "CA", "type": "Insurance"},
    "FFH-CA": {"name": "Fairfax Financial Holdings Limited", "region": "CA", "type": "Insurance"},
    "IFC-CA": {"name": "Intact Financial Corporation", "region": "CA", "type": "Insurance"},
    
    # Major US Insurance Companies
    "BRK.B-US": {"name": "Berkshire Hathaway Inc.", "region": "US", "type": "Insurance"},
    "UNH-US": {"name": "UnitedHealth Group Incorporated", "region": "US", "type": "Insurance"},
    "PG-US": {"name": "Procter & Gamble Company", "region": "US", "type": "Insurance"},
    "JNJ-US": {"name": "Johnson & Johnson", "region": "US", "type": "Insurance"},
    "PFE-US": {"name": "Pfizer Inc.", "region": "US", "type": "Insurance"},
    "CVX-US": {"name": "Chevron Corporation", "region": "US", "type": "Insurance"},
    "WMT-US": {"name": "Walmart Inc.", "region": "US", "type": "Insurance"},
    "MSFT-US": {"name": "Microsoft Corporation", "region": "US", "type": "Insurance"},
    "AAPL-US": {"name": "Apple Inc.", "region": "US", "type": "Insurance"},
    "GOOGL-US": {"name": "Alphabet Inc.", "region": "US", "type": "Insurance"},
}

# Industry Categories to Monitor
INDUSTRY_CATEGORIES = ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"]

# Event Types to Monitor (earnings only)
EVENT_TYPES = ["Earnings"]

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
    
    log_file = log_dir / f"sync_log_{datetime.now().strftime('%Y-%m-%d')}.log"
    
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

def sanitize_filename(filename):
    """
    Sanitize filename to be path-safe
    """
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'[^\w\s-]', '', sanitized)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.strip('_')

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
    
    # Create listings and logs directories
    (base_path / "listings").mkdir(parents=True, exist_ok=True)
    (base_path / "logs").mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directory structure at {base_path}")

def load_local_inventory(transcript_type):
    """
    Load the local inventory of transcripts for a given type
    """
    listings_path = Path(REPOSITORY_ROOT) / "listings" / f"{transcript_type.lower()}_transcripts.csv"
    
    if listings_path.exists():
        try:
            df = pd.read_csv(listings_path)
            logger.info(f"Loaded {len(df)} existing {transcript_type} transcripts from inventory")
            return df
        except Exception as e:
            logger.error(f"Error loading inventory for {transcript_type}: {e}")
            return pd.DataFrame()
    else:
        logger.info(f"No existing inventory found for {transcript_type}")
        return pd.DataFrame()

def save_local_inventory(transcript_type, df):
    """
    Save the local inventory of transcripts for a given type
    """
    listings_path = Path(REPOSITORY_ROOT) / "listings" / f"{transcript_type.lower()}_transcripts.csv"
    
    try:
        df.to_csv(listings_path, index=False)
        logger.info(f"Saved {len(df)} {transcript_type} transcripts to inventory")
    except Exception as e:
        logger.error(f"Error saving inventory for {transcript_type}: {e}")

def download_transcript_with_retry(transcript_link, output_path, transcript_id):
    """
    Download transcript with retry logic
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading {transcript_id} (attempt {attempt + 1}/{MAX_RETRIES})")
            
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
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Successfully downloaded {transcript_id} ({file_size:,} bytes)")
            return True
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed for {transcript_id}: {e}")
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"Failed to download {transcript_id} after {MAX_RETRIES} attempts")
                return False
    
    return False

def get_transcripts_for_institution(ticker, institution_info, api_instance):
    """
    Get all earnings transcripts for a specific institution
    """
    try:
        logger.info(f"Fetching transcripts for {institution_info['name']} ({ticker})")
        
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
            logger.warning(f"No transcripts found for {ticker}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(response.to_dict()['data'])
        
        # Filter for earnings transcripts only
        if 'event_type' in df.columns:
            earnings_df = df[df['event_type'].isin(EVENT_TYPES)]
            logger.info(f"Found {len(earnings_df)} earnings transcripts for {ticker}")
        else:
            earnings_df = df
            logger.info(f"Found {len(earnings_df)} transcripts for {ticker} (no event type filtering)")
        
        # Add institution metadata
        earnings_df['institution_ticker'] = ticker
        earnings_df['institution_name'] = institution_info['name']
        earnings_df['institution_region'] = institution_info['region']
        earnings_df['institution_type'] = institution_info['type']
        earnings_df['fetch_timestamp'] = datetime.now()
        
        return earnings_df
        
    except Exception as e:
        logger.error(f"Error fetching transcripts for {ticker}: {e}")
        return pd.DataFrame()

def process_transcripts_by_type(all_transcripts_df, transcript_type):
    """
    Process transcripts for a specific type and download missing ones
    """
    logger.info(f"Processing {transcript_type} transcripts")
    
    # Filter for the specific transcript type
    type_df = all_transcripts_df[all_transcripts_df['transcript_type'] == transcript_type].copy()
    
    if type_df.empty:
        logger.info(f"No {transcript_type} transcripts found")
        return
    
    # Load existing inventory
    existing_df = load_local_inventory(transcript_type)
    
    # Find missing transcripts
    if not existing_df.empty:
        # Compare by report_id and version_id
        existing_keys = set(existing_df['report_id'] + '_' + existing_df['version_id'].astype(str))
        current_keys = set(type_df['report_id'] + '_' + type_df['version_id'].astype(str))
        missing_keys = current_keys - existing_keys
        
        if missing_keys:
            missing_df = type_df[
                (type_df['report_id'] + '_' + type_df['version_id'].astype(str)).isin(missing_keys)
            ]
            logger.info(f"Found {len(missing_df)} new {transcript_type} transcripts to download")
        else:
            logger.info(f"No new {transcript_type} transcripts to download")
            return
    else:
        missing_df = type_df
        logger.info(f"No existing inventory. Will download all {len(missing_df)} {transcript_type} transcripts")
    
    # Download missing transcripts
    successful_downloads = []
    
    for _, transcript in missing_df.iterrows():
        if pd.isna(transcript['transcripts_link']):
            logger.warning(f"No download link for transcript {transcript['report_id']}")
            continue
        
        # Create filename
        ticker = transcript['institution_ticker']
        event_date = pd.to_datetime(transcript['event_date']).strftime('%Y-%m-%d') if pd.notna(transcript['event_date']) else 'unknown_date'
        event_id = transcript['event_id']
        headline = sanitize_filename(transcript['headline'][:50]) if pd.notna(transcript['headline']) else 'unknown_title'
        
        filename = f"{ticker}_{event_date}_{event_id}_{transcript_type}_{headline}.xml"
        
        # Create output path
        output_path = Path(REPOSITORY_ROOT) / transcript_type.lower() / ticker / filename
        
        # Download transcript
        if download_transcript_with_retry(transcript['transcripts_link'], output_path, transcript['report_id']):
            successful_downloads.append(transcript)
            
        # Add delay between downloads
        time.sleep(REQUEST_DELAY)
    
    # Update inventory
    if successful_downloads:
        new_entries_df = pd.DataFrame(successful_downloads)
        updated_df = pd.concat([existing_df, new_entries_df], ignore_index=True) if not existing_df.empty else new_entries_df
        save_local_inventory(transcript_type, updated_df)
        logger.info(f"Successfully downloaded {len(successful_downloads)} {transcript_type} transcripts")

def main():
    """
    Main function to orchestrate the transcript repository sync
    """
    logger.info("=" * 60)
    logger.info("EARNINGS TRANSCRIPT REPOSITORY SYNC")
    logger.info("=" * 60)
    
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
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Process institutions in batches
            all_transcripts = []
            
            for i, (ticker, institution_info) in enumerate(MONITORED_INSTITUTIONS.items(), 1):
                logger.info(f"Processing institution {i}/{total_institutions}: {institution_info['name']}")
                
                # Get transcripts for this institution
                transcripts_df = get_transcripts_for_institution(ticker, institution_info, api_instance)
                
                if not transcripts_df.empty:
                    all_transcripts.append(transcripts_df)
                
                processed_institutions += 1
                
                # Add delay between institutions
                if i < total_institutions:
                    time.sleep(REQUEST_DELAY)
            
            # Combine all transcripts
            if all_transcripts:
                combined_df = pd.concat(all_transcripts, ignore_index=True)
                logger.info(f"Total transcripts found: {len(combined_df)}")
                
                # Process each transcript type
                for transcript_type in TRANSCRIPT_TYPES:
                    process_transcripts_by_type(combined_df, transcript_type)
            else:
                logger.warning("No transcripts found for any institution")
    
    except Exception as e:
        logger.error(f"Critical error during sync: {e}")
        raise
    
    # Final summary
    end_time = datetime.now()
    execution_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("SYNC COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {processed_institutions}/{total_institutions} institutions")
    logger.info(f"Execution time: {execution_time}")
    logger.info(f"Repository location: {Path(REPOSITORY_ROOT).absolute()}")

if __name__ == "__main__":
    logger = setup_logging()
    main()