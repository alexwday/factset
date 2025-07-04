"""
FactSet Industry Transcripts Puller
This script pulls earnings transcripts for specified industries from FactSet Events and Transcripts API
for a specified date range and combines them into a single CSV file.
Currently configured for banks and financial services (IN:BANKS, IN:FNLSVC).
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
from dateutil.parser import parse as dateutil_parser
import os
from urllib.parse import quote
from datetime import datetime, timedelta
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
TIME_ZONE = "America/New_York"

# Date Range Configuration (Format: YYYY-MM-DD)
START_DATE = "2024-05-01"
END_DATE = "2024-05-31"

# Categories Filter (Multiple Industries)
# Add multiple industry codes as needed. Common codes include:
# IN:BANKS (Banks), IN:FNLSVC (Financial Services), IN:INSUR (Insurance), etc.
# Use the factset_categories_puller.py script to get a full list of available codes.
INDUSTRY_FILTERS = ["IN:BANKS", "IN:FNLSVC"]  # Example: Banks and Financial Services
CATEGORIES_FILTER = INDUSTRY_FILTERS

# Event Type Filter (Earnings only)
EVENT_TYPE = "Earnings"

# Optional Ticker-Country Filter
# Major Canadian and US Banks - modify this list as needed
# Leave empty list [] to disable this feature
TICKER_COUNTRY_FLAGS = [
    # Major Canadian Banks ("Big Six")
    "RY-CA",    # Royal Bank of Canada
    "TD-CA",    # Toronto-Dominion Bank
    "BNS-CA",   # Bank of Nova Scotia (Scotiabank)
    "BMO-CA",   # Bank of Montreal
    "CM-CA",    # Canadian Imperial Bank of Commerce (CIBC)
    "NA-CA",    # National Bank of Canada
    
    # Major US Banks
    "JPM-US",   # JPMorgan Chase & Co.
    "BAC-US",   # Bank of America Corporation
    "WFC-US",   # Wells Fargo & Company
    "C-US",     # Citigroup Inc.
    "USB-US",   # U.S. Bancorp
    "PNC-US",   # PNC Financial Services Group
    "TFC-US",   # Truist Financial Corporation
    "COF-US",   # Capital One Financial Corporation
    "MS-US",    # Morgan Stanley
    "GS-US",    # Goldman Sachs Group Inc.
    "BK-US",    # The Bank of New York Mellon Corporation
    "STT-US",   # State Street Corporation
    "AXP-US",   # American Express Company
    "SCHW-US",  # Charles Schwab Corporation
    "BLK-US",   # BlackRock Inc.
    "ALLY-US",  # Ally Financial Inc.
    "RF-US",    # Regions Financial Corporation
    "KEY-US",   # KeyCorp
    "CFG-US",   # Citizens Financial Group Inc.
    "MTB-US",   # M&T Bank Corporation
    "FITB-US",  # Fifth Third Bancorp
    "HBAN-US",  # Huntington Bancshares Incorporated
    "ZION-US",  # Zions Bancorporation
    "CMA-US",   # Comerica Incorporated
]

# Output Configuration
OUTPUT_FILE = "industry_transcripts_combined.csv"
SORT_ORDER = ["-storyDateTime"]
PAGINATION_LIMIT = 1000
PAGINATION_OFFSET = 0

# Request Configuration
REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting

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

def check_ticker_country_flags(row, ticker_list):
    """
    Check if any ticker-country codes are found in headline or all_ids
    
    Args:
        row: DataFrame row
        ticker_list (list): List of ticker-country codes to search for
    
    Returns:
        tuple: (found_tickers, flag_status)
    """
    if not ticker_list:
        return [], False
    
    found_tickers = []
    
    # Check headline
    headline = str(row.get('headline', '')).upper()
    
    # Check all_ids (which may be a list or string)
    all_ids = row.get('all_ids', [])
    if isinstance(all_ids, list):
        all_ids_str = ' '.join([str(id_item).upper() for id_item in all_ids])
    else:
        all_ids_str = str(all_ids).upper()
    
    # Search for each ticker-country code
    for ticker in ticker_list:
        ticker_upper = ticker.upper()
        
        # Check if ticker found in headline or all_ids
        if ticker_upper in headline or ticker_upper in all_ids_str:
            found_tickers.append(ticker)
    
    return found_tickers, len(found_tickers) > 0

def add_ticker_country_flags(df, ticker_list):
    """
    Add ticker-country flag columns to the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with transcript data
        ticker_list (list): List of ticker-country codes to flag
    
    Returns:
        pd.DataFrame: DataFrame with added flag columns
    """
    if df.empty or not ticker_list:
        if not ticker_list:
            print("No ticker-country flags specified, skipping ticker flagging")
        return df
    
    print(f"Adding ticker-country flags for: {', '.join(ticker_list)}")
    
    # Add columns for ticker flags
    df_copy = df.copy()
    
    # Apply ticker checking to each row
    ticker_results = df_copy.apply(lambda row: check_ticker_country_flags(row, ticker_list), axis=1)
    
    # Extract found tickers and flag status
    df_copy['ticker_flags_found'] = [result[0] for result in ticker_results]
    df_copy['has_ticker_flags'] = [result[1] for result in ticker_results]
    
    # Convert ticker_flags_found list to string for CSV compatibility
    df_copy['ticker_flags_found'] = df_copy['ticker_flags_found'].apply(
        lambda tickers: ', '.join(tickers) if tickers else ''
    )
    
    # Count how many transcripts have ticker flags
    flagged_count = df_copy['has_ticker_flags'].sum()
    print(f"Found {flagged_count} transcripts containing specified ticker-country codes")
    
    return df_copy

def filter_transcripts_for_earnings(df):
    """
    Filter transcripts to only include earnings events
    (Industries are already filtered by the API categories parameter)
    
    Args:
        df (pd.DataFrame): DataFrame with transcript data
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only earnings transcripts
    """
    if df.empty:
        return df
    
    # Filter for earnings (event_type = Earnings)
    earnings_mask = df['event_type'].apply(
        lambda event_type: str(event_type).lower() == 'earnings' if event_type is not None 
        else False
    )
    
    # Apply filter
    filtered_df = df[earnings_mask].copy()
    
    return filtered_df

def generate_date_range(start_date_str, end_date_str):
    """
    Generate a list of dates between start_date and end_date (inclusive)
    
    Args:
        start_date_str (str): Start date in YYYY-MM-DD format
        end_date_str (str): End date in YYYY-MM-DD format
    
    Returns:
        list: List of datetime.date objects
    """
    start_date = dateutil_parser(start_date_str).date()
    end_date = dateutil_parser(end_date_str).date()
    
    date_list = []
    current_date = start_date
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    return date_list

def get_transcripts_for_date(target_date, api_instance):
    """
    Get transcripts for a specific date with industry category filtering and earnings event type
    
    Args:
        target_date (datetime.date): Date to fetch transcripts for
        api_instance: TranscriptsApi instance
    
    Returns:
        pd.DataFrame or None: DataFrame with transcripts or None if no data
    """
    try:
        print(f"Fetching transcripts for {target_date}...")
        
        # Use get_transcripts_ids with categories filtering
        # This endpoint supports both date and categories parameters
        response = api_instance.get_transcripts_ids(
            start_date=target_date,
            end_date=target_date,
            categories=CATEGORIES_FILTER,
            sort=SORT_ORDER,
            pagination_limit=PAGINATION_LIMIT,
            pagination_offset=PAGINATION_OFFSET
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            print(f"No transcripts found for {target_date}")
            return None
        
        # Convert response to DataFrame
        df = pd.DataFrame(response.to_dict()['data'])
        print(f"Found {len(df)} industry transcripts on {target_date}")
        
        # Filter for earnings only (industries already filtered by API)
        filtered_df = filter_transcripts_for_earnings(df)
        
        if filtered_df is None or filtered_df.empty:
            print(f"No industry earnings transcripts found for {target_date} after filtering")
            return None
        
        print(f"Found {len(filtered_df)} industry earnings transcripts on {target_date}")
        
        # Add ticker-country flags if specified
        filtered_df = add_ticker_country_flags(filtered_df, TICKER_COUNTRY_FLAGS)
        
        # Add date column for tracking
        filtered_df['fetch_date'] = target_date
        
        return filtered_df
        
    except Exception as e:
        print(f"Error fetching transcripts for {target_date}: {str(e)}")
        return None

def get_all_transcripts_for_date_range(start_date_str, end_date_str):
    """
    Get all industry earnings transcripts for a date range, processing day by day
    
    Args:
        start_date_str (str): Start date in YYYY-MM-DD format
        end_date_str (str): End date in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: Combined DataFrame with all transcripts
    """
    print(f"Fetching industry earnings transcripts from {start_date_str} to {end_date_str}")
    print(f"Filtering for industries: {', '.join(INDUSTRY_FILTERS)} AND event_type: {EVENT_TYPE}")
    
    # Generate date range
    date_list = generate_date_range(start_date_str, end_date_str)
    print(f"Processing {len(date_list)} days...")
    
    all_transcripts = []
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        
        for i, target_date in enumerate(date_list, 1):
            print(f"Progress: {i}/{len(date_list)} - {target_date}")
            
            # Get transcripts for this date
            df = get_transcripts_for_date(target_date, api_instance)
            
            if df is not None and not df.empty:
                all_transcripts.append(df)
            
            # Add delay between requests to avoid rate limiting
            if i < len(date_list):  # Don't delay after the last request
                time.sleep(REQUEST_DELAY)
    
    # Combine all transcripts
    if all_transcripts:
        combined_df = pd.concat(all_transcripts, ignore_index=True)
        print(f"\nTotal transcripts found: {len(combined_df)}")
        return combined_df
    else:
        print("\nNo transcripts found for the specified date range and categories.")
        return pd.DataFrame()

def save_transcripts_to_csv(df, output_file):
    """
    Save transcripts DataFrame to CSV file
    
    Args:
        df (pd.DataFrame): DataFrame with transcripts
        output_file (str): Output CSV file path
    """
    if df.empty:
        print("No data to save.")
        return
    
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} transcripts to: {output_file}")
        
        # Print summary statistics
        print("\n=== SUMMARY ===")
        print(f"Total transcripts: {len(df)}")
        if 'fetch_date' in df.columns:
            print(f"Date range: {df['fetch_date'].min()} to {df['fetch_date'].max()}")
        
        # Ticker flag summary
        if 'has_ticker_flags' in df.columns:
            flagged_count = df['has_ticker_flags'].sum()
            print(f"Transcripts with ticker flags: {flagged_count} ({flagged_count/len(df)*100:.1f}%)")
            
            if flagged_count > 0:
                # Show which tickers were found
                all_found_tickers = []
                for ticker_str in df['ticker_flags_found'].dropna():
                    if ticker_str:
                        all_found_tickers.extend(ticker_str.split(', '))
                unique_found_tickers = list(set(all_found_tickers))
                print(f"Ticker-country codes found: {', '.join(sorted(unique_found_tickers))}")
        
        if 'categories' in df.columns:
            print("Categories found in data:")
            # Flatten categories lists and count unique values
            all_categories = []
            for cat_list in df['categories'].dropna():
                if isinstance(cat_list, list):
                    all_categories.extend(cat_list)
                elif isinstance(cat_list, str):
                    all_categories.append(cat_list)
            unique_categories = list(set(all_categories))
            for cat in sorted(unique_categories):
                print(f"  - {cat}")
        
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

def main():
    """
    Main function to orchestrate the transcript pulling process
    """
    print("=" * 60)
    print("FactSet Industry Transcripts Puller")
    print("=" * 60)
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Date Range: {START_DATE} to {END_DATE}")
    print(f"  Industries: {', '.join(INDUSTRY_FILTERS)}")
    print(f"  Event Type: {EVENT_TYPE}")
    print(f"  Ticker-Country Flags: {', '.join(TICKER_COUNTRY_FLAGS) if TICKER_COUNTRY_FLAGS else 'None'}")
    print(f"  Output File: {OUTPUT_FILE}")
    print(f"  SSL Cert: {SSL_CERT_PATH}")
    print(f"  Proxy: {PROXY_URL}")
    
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("\nStarting transcript collection...")
    start_time = datetime.now()
    
    try:
        # Get all transcripts for the date range
        combined_df = get_all_transcripts_for_date_range(START_DATE, END_DATE)
        
        # Save to CSV
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