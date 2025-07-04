"""
FactSet Bank Transcripts by Primary ID Puller
This script pulls all available transcripts for specific Canadian banks using their primary IDs.
Uses get_transcripts_ids method instead of date-based filtering to access all transcript versions.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
import os
from urllib.parse import quote
from datetime import datetime
import time

# =============================================================================
# CONFIGURATION VARIABLES (HARDCODED)
# =============================================================================

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Big 6 Canadian Banks Primary IDs
# Note: These may need to be updated with the correct FactSet primary IDs
CANADIAN_BANKS = {
    "Royal Bank of Canada": "RY-CA",
    "Toronto-Dominion Bank": "TD-CA", 
    "Bank of Nova Scotia": "BNS-CA",
    "Bank of Montreal": "BMO-CA",
    "Canadian Imperial Bank of Commerce": "CM-CA",
    "National Bank of Canada": "NA-CA"
}

# Search Configuration  
# Set to True/False for primary_id parameter, or None to omit the parameter entirely
PRIMARY_ID_SEARCH = None  # None=omit parameter, True=primary only, False=mentioned anywhere
SORT_ORDER = ["-storyDateTime"]
PAGINATION_LIMIT = 1000
PAGINATION_OFFSET = 0

# Optional date range (leave None to get ALL transcripts)
START_DATE = None  # Format: "YYYY-MM-DD" or None for no limit
END_DATE = None    # Format: "YYYY-MM-DD" or None for no limit

# Output Configuration
OUTPUT_FILE = "canadian_banks_all_transcripts.csv"

# Request Configuration
REQUEST_DELAY = 2.0  # Seconds between requests to avoid rate limiting

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
# UTILITY FUNCTIONS
# =============================================================================

def get_transcripts_for_bank(bank_name, bank_id, api_instance):
    """
    Get all transcripts for a specific bank using primary ID search
    
    Args:
        bank_name (str): Human-readable bank name
        bank_id (str): Bank primary ID (e.g., "RY-CA")
        api_instance: TranscriptsApi instance
    
    Returns:
        pd.DataFrame or None: DataFrame with transcripts or None if no data
    """
    try:
        print(f"\nFetching transcripts for {bank_name} ({bank_id})...")
        
        # Prepare parameters for API call
        api_params = {
            'ids': [bank_id],
            'sort': SORT_ORDER,
            'pagination_limit': PAGINATION_LIMIT,
            'pagination_offset': PAGINATION_OFFSET
        }
        
        # Add primary_id parameter if enabled (ensuring it's a proper boolean)
        if PRIMARY_ID_SEARCH is not None:
            api_params['primary_id'] = bool(PRIMARY_ID_SEARCH)
        
        # Add date range if specified
        if START_DATE:
            from dateutil.parser import parse as dateutil_parser
            api_params['start_date'] = dateutil_parser(START_DATE).date()
            
        if END_DATE:
            from dateutil.parser import parse as dateutil_parser
            api_params['end_date'] = dateutil_parser(END_DATE).date()
        
        # Debug: Print the API parameters being sent
        print(f"API parameters: {api_params}")
        
        # Make API call
        response = api_instance.get_transcripts_ids(**api_params)
        
        if not response or not hasattr(response, 'data') or not response.data:
            print(f"No transcripts found for {bank_name}")
            return None
        
        # Convert response to DataFrame
        df = pd.DataFrame(response.to_dict()['data'])
        print(f"Found {len(df)} transcripts for {bank_name}")
        
        # Add bank identification columns
        df['bank_name'] = bank_name
        df['bank_id'] = bank_id
        df['fetch_timestamp'] = datetime.now()
        
        return df
        
    except Exception as e:
        print(f"Error fetching transcripts for {bank_name}: {str(e)}")
        return None

def get_all_banks_transcripts():
    """
    Get all transcripts for all Canadian banks and combine them
    
    Returns:
        pd.DataFrame: Combined DataFrame with all transcripts
    """
    print("=" * 60)
    print("FactSet Canadian Banks Transcripts by Primary ID")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  Banks: {len(CANADIAN_BANKS)} Canadian banks")
    print(f"  Primary ID Search: {PRIMARY_ID_SEARCH}")
    print(f"  Date Range: {START_DATE or 'No limit'} to {END_DATE or 'No limit'}")
    print(f"  Output File: {OUTPUT_FILE}")
    
    all_transcripts = []
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        
        for i, (bank_name, bank_id) in enumerate(CANADIAN_BANKS.items(), 1):
            print(f"\nProgress: {i}/{len(CANADIAN_BANKS)}")
            
            # Get transcripts for this bank
            df = get_transcripts_for_bank(bank_name, bank_id, api_instance)
            
            if df is not None and not df.empty:
                all_transcripts.append(df)
            
            # Add delay between requests to avoid rate limiting
            if i < len(CANADIAN_BANKS):  # Don't delay after the last request
                print(f"Waiting {REQUEST_DELAY} seconds...")
                time.sleep(REQUEST_DELAY)
    
    # Combine all transcripts
    if all_transcripts:
        combined_df = pd.concat(all_transcripts, ignore_index=True)
        print(f"\n=== COLLECTION COMPLETE ===")
        print(f"Total transcripts found: {len(combined_df)}")
        return combined_df
    else:
        print("\nNo transcripts found for any banks.")
        return pd.DataFrame()

def analyze_transcript_data(df):
    """
    Analyze the combined transcript data
    
    Args:
        df (pd.DataFrame): Combined transcript data
    """
    if df.empty:
        return
    
    print("\n=== DATA ANALYSIS ===")
    
    # Bank breakdown
    bank_counts = df['bank_name'].value_counts()
    print(f"\nTranscripts by bank:")
    for bank, count in bank_counts.items():
        print(f"  {bank}: {count} transcripts")
    
    # Event type breakdown (if available)
    if 'event_type' in df.columns:
        event_counts = df['event_type'].value_counts()
        print(f"\nTranscripts by event type:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count} transcripts")
    
    # Version information (if available)
    if 'version_id' in df.columns:
        version_info = df['version_id'].value_counts()
        print(f"\nVersion distribution:")
        print(f"  Unique versions: {len(version_info)}")
        if len(version_info) <= 10:
            for version, count in version_info.items():
                print(f"  {version}: {count} transcripts")
    
    # Date range (if available)
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        print(f"\nDate range:")
        print(f"  Earliest: {df['event_date'].min()}")
        print(f"  Latest: {df['event_date'].max()}")
    
    # Categories analysis (if available)
    if 'categories' in df.columns:
        all_categories = []
        for cat_list in df['categories'].dropna():
            if isinstance(cat_list, list):
                all_categories.extend(cat_list)
            elif isinstance(cat_list, str):
                all_categories.append(cat_list)
        unique_categories = list(set(all_categories))
        print(f"\nUnique categories found: {len(unique_categories)}")

def save_transcripts_to_csv(df, output_file):
    """
    Save transcripts DataFrame to CSV file with analysis
    
    Args:
        df (pd.DataFrame): DataFrame with transcripts
        output_file (str): Output CSV file path
    """
    if df.empty:
        print("No data to save.")
        return
    
    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved {len(df)} transcripts to: {output_file}")
        
        # Run analysis
        analyze_transcript_data(df)
        
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

def main():
    """
    Main function to orchestrate the transcript pulling process
    """
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("\nStarting transcript collection...")
    start_time = datetime.now()
    
    try:
        # Get all transcripts for Canadian banks
        combined_df = get_all_banks_transcripts()
        
        # Save to CSV with analysis
        save_transcripts_to_csv(combined_df, OUTPUT_FILE)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nExecution completed in: {execution_time}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()