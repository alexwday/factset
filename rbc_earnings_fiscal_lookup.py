"""
Royal Bank of Canada Earnings Transcripts with Fiscal Year/Quarter Lookup
This script pulls RBC earnings transcripts, finds the most recent ones,
and retrieves fiscal year and quarter information from the calendar API.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api, calendar_api
import os
from urllib.parse import quote
from datetime import datetime
import time
import requests

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

# Royal Bank of Canada Configuration
RBC_PRIMARY_ID = "RY-CA"
RBC_NAME = "Royal Bank of Canada"

# Search Configuration
SORT_ORDER = ["-event_date"]  # Sort by most recent first
PAGINATION_LIMIT = 50  # Get last 50 transcripts
PAGINATION_OFFSET = 0

# Output Configuration
OUTPUT_FILE = "rbc_earnings_transcripts_with_fiscal_info.csv"
TRANSCRIPT_OUTPUT_DIR = "rbc_transcript_xml"
LATEST_TRANSCRIPT_FILE = "latest_rbc_earnings_transcript.xml"

# Request Configuration
REQUEST_DELAY = 1.0  # Seconds between requests

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

def get_rbc_earnings_transcripts(api_instance):
    """
    Get earnings transcripts for Royal Bank of Canada
    
    Args:
        api_instance: TranscriptsApi instance
    
    Returns:
        pd.DataFrame or None: DataFrame with earnings transcripts
    """
    try:
        print(f"\nFetching earnings transcripts for {RBC_NAME} ({RBC_PRIMARY_ID})...")
        
        # Get transcripts using primary ID
        response = api_instance.get_transcripts_ids(
            ids=[RBC_PRIMARY_ID],
            sort=SORT_ORDER,
            pagination_limit=PAGINATION_LIMIT,
            pagination_offset=PAGINATION_OFFSET
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            print("No transcripts found for RBC")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(response.to_dict()['data'])
        
        # Filter for earnings transcripts only
        if 'event_type' in df.columns:
            earnings_df = df[df['event_type'].str.contains('Earnings', case=False, na=False)]
            print(f"Found {len(earnings_df)} earnings transcripts out of {len(df)} total transcripts")
            return earnings_df
        else:
            print(f"Found {len(df)} transcripts (unable to filter by event type)")
            return df
        
    except Exception as e:
        print(f"Error fetching transcripts: {str(e)}")
        return None

def get_fiscal_info_for_events(event_ids, calendar_api_instance):
    """
    Get fiscal year and quarter information for given event IDs
    
    Args:
        event_ids: List of event IDs
        calendar_api_instance: CalendarApi instance
    
    Returns:
        dict: Dictionary mapping event_id to fiscal info
    """
    fiscal_info = {}
    
    print(f"\nLooking up fiscal information for {len(event_ids)} events...")
    
    for i, event_id in enumerate(event_ids):
        try:
            # Get event details from calendar API
            response = calendar_api_instance.get_company_event(
                company_event_id=event_id
            )
            
            if response and hasattr(response, 'data') and response.data:
                event_data = response.data
                fiscal_info[event_id] = {
                    'fiscal_year': getattr(event_data, 'fiscal_year', None),
                    'fiscal_period': getattr(event_data, 'fiscal_period', None),
                    'event_date': getattr(event_data, 'event_date', None),
                    'event_type': getattr(event_data, 'event_type', None)
                }
                print(f"  Event {event_id}: FY{fiscal_info[event_id]['fiscal_year']} {fiscal_info[event_id]['fiscal_period']}")
            else:
                print(f"  Event {event_id}: No fiscal data found")
                fiscal_info[event_id] = {
                    'fiscal_year': None,
                    'fiscal_period': None,
                    'event_date': None,
                    'event_type': None
                }
            
            # Add delay to avoid rate limiting
            if i < len(event_ids) - 1:
                time.sleep(REQUEST_DELAY)
                
        except Exception as e:
            print(f"  Event {event_id}: Error - {str(e)}")
            fiscal_info[event_id] = {
                'fiscal_year': None,
                'fiscal_period': None,
                'event_date': None,
                'event_type': None
            }
    
    return fiscal_info

def download_transcript_xml(transcript_link, output_path, api_config):
    """
    Download transcript XML file from the provided link
    
    Args:
        transcript_link: URL to download the transcript
        output_path: Local path to save the XML file
        api_config: API configuration object
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nDownloading transcript XML to: {output_path}")
        
        # Set up authentication and proxy
        auth = (api_config.username, api_config.password)
        proxies = {
            'http': api_config.proxy,
            'https': api_config.proxy
        }
        
        # Download the XML file
        response = requests.get(
            transcript_link,
            auth=auth,
            proxies=proxies,
            verify=api_config.ssl_ca_cert,
            timeout=30
        )
        
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(output_path)
        print(f"Successfully downloaded transcript ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        print(f"Error downloading transcript: {str(e)}")
        return False

def main():
    """
    Main function to pull RBC earnings transcripts and get fiscal information
    """
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("=" * 60)
    print("Royal Bank of Canada Earnings Transcripts Fiscal Lookup")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            # Initialize API instances
            transcripts_api_instance = transcripts_api.TranscriptsApi(api_client)
            calendar_api_instance = calendar_api.CalendarApi(api_client)
            
            # Step 1: Get RBC earnings transcripts
            earnings_df = get_rbc_earnings_transcripts(transcripts_api_instance)
            
            if earnings_df is None or earnings_df.empty:
                print("No earnings transcripts found.")
                return
            
            # Step 2: Extract event IDs
            if 'event_id' in earnings_df.columns:
                event_ids = earnings_df['event_id'].dropna().unique().tolist()
                print(f"\nFound {len(event_ids)} unique event IDs")
            else:
                print("No event_id column found in transcripts data")
                return
            
            # Step 3: Get fiscal information from calendar API
            fiscal_info = get_fiscal_info_for_events(event_ids, calendar_api_instance)
            
            # Step 4: Merge fiscal information with transcript data
            fiscal_df = pd.DataFrame.from_dict(fiscal_info, orient='index')
            fiscal_df.reset_index(inplace=True)
            fiscal_df.rename(columns={'index': 'event_id'}, inplace=True)
            
            # Merge with earnings transcripts
            final_df = earnings_df.merge(
                fiscal_df[['event_id', 'fiscal_year', 'fiscal_period']], 
                on='event_id', 
                how='left'
            )
            
            # Add metadata
            final_df['bank_name'] = RBC_NAME
            final_df['bank_id'] = RBC_PRIMARY_ID
            final_df['fetch_timestamp'] = datetime.now()
            
            # Sort by event date (most recent first)
            if 'event_date' in final_df.columns:
                final_df = final_df.sort_values('event_date', ascending=False)
            
            # Save to CSV
            final_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSuccessfully saved {len(final_df)} earnings transcripts to: {OUTPUT_FILE}")
            
            # Download the latest transcript XML
            if not final_df.empty and 'transcripts_link' in final_df.columns:
                latest_transcript = final_df.iloc[0]
                
                if pd.notna(latest_transcript['transcripts_link']):
                    # Create output directory if it doesn't exist
                    os.makedirs(TRANSCRIPT_OUTPUT_DIR, exist_ok=True)
                    
                    # Generate filename with date and event info
                    event_date = latest_transcript.get('event_date', 'unknown_date')
                    fiscal_year = latest_transcript.get('fiscal_year', 'FY_unknown')
                    fiscal_period = latest_transcript.get('fiscal_period', 'Q_unknown')
                    
                    # Create filename
                    xml_filename = f"RBC_earnings_{event_date}_{fiscal_year}_{fiscal_period}.xml"
                    xml_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, xml_filename)
                    
                    # Also save as "latest" for easy access
                    latest_path = os.path.join(TRANSCRIPT_OUTPUT_DIR, LATEST_TRANSCRIPT_FILE)
                    
                    # Download the transcript
                    if download_transcript_xml(latest_transcript['transcripts_link'], xml_path, configuration):
                        # Copy to "latest" file
                        import shutil
                        shutil.copy2(xml_path, latest_path)
                        print(f"Also saved as: {latest_path}")
                        
                        print(f"\nLatest transcript details:")
                        print(f"  Event Date: {event_date}")
                        print(f"  Fiscal Year: {fiscal_year}")
                        print(f"  Fiscal Period: {fiscal_period}")
                        if 'headline' in latest_transcript:
                            print(f"  Headline: {latest_transcript['headline']}")
                else:
                    print("\nNo transcript link available for the latest earnings transcript")
            
            # Display summary of most recent earnings
            print("\n=== MOST RECENT EARNINGS TRANSCRIPTS ===")
            display_cols = ['event_date', 'fiscal_year', 'fiscal_period', 'headline']
            display_cols = [col for col in display_cols if col in final_df.columns]
            
            if display_cols:
                print(final_df[display_cols].head(10).to_string(index=False))
            
            # Calculate statistics
            fiscal_data_count = final_df['fiscal_year'].notna().sum()
            print(f"\n=== STATISTICS ===")
            print(f"Total earnings transcripts: {len(final_df)}")
            print(f"Transcripts with fiscal data: {fiscal_data_count}")
            print(f"Coverage rate: {fiscal_data_count/len(final_df)*100:.1f}%")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your configuration and try again.")
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = end_time - start_time
    print(f"\nExecution completed in: {execution_time}")

if __name__ == "__main__":
    main()