"""
RBC Latest Earnings Transcript Downloader
This script pulls the latest Royal Bank of Canada earnings transcript and saves it locally.
Based on factset_bank_by_id_puller.py
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
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

# Royal Bank of Canada Primary ID
RBC_PRIMARY_ID = "RY-CA"
RBC_NAME = "Royal Bank of Canada"

# Search Configuration
SORT_ORDER = ["-storyDateTime"]  # Sort by most recent first
PAGINATION_LIMIT = 50  # Get last 50 transcripts to ensure we find earnings
PAGINATION_OFFSET = 0

# Output Configuration
OUTPUT_DIR = "rbc_transcripts"
OUTPUT_FILE = "latest_rbc_earnings_transcript.xml"

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
# MAIN FUNCTIONS
# =============================================================================

def download_file(file_link, output_path, file_type="file"):
    """
    Download file from the provided link
    """
    try:
        print(f"\nDownloading {file_type} to: {output_path}")
        
        # Set up authentication and proxy for direct download
        auth = (configuration.username, configuration.password)
        proxies = {
            'http': configuration.proxy,
            'https': configuration.proxy
        }
        
        # Download the file
        response = requests.get(
            file_link,
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
        print(f"Successfully downloaded {file_type} ({file_size:,} bytes)")
        return True
        
    except Exception as e:
        print(f"Error downloading {file_type}: {str(e)}")
        return False

def get_slides_for_event(event_id, event_date, api_instance):
    """
    Get investor slides for a specific event
    """
    try:
        print(f"\nSearching for slides for event {event_id}...")
        
        # Use event date as start/end date for slides search
        if pd.isna(event_date):
            print("No event date available, cannot search for slides")
            return None
            
        event_date_obj = pd.to_datetime(event_date).date()
        
        # Search for slides on the event date
        response = api_instance.get_transcripts_investor_slides(
            start_date=event_date_obj,
            end_date=event_date_obj,
            ids=[RBC_PRIMARY_ID]
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            print("No slides found for this event")
            return None
        
        # Convert to DataFrame for easier handling
        slides_df = pd.DataFrame(response.to_dict()['data'])
        
        # Try to find slides matching the event_id
        matching_slides = slides_df[slides_df['event_id'] == event_id]
        
        if not matching_slides.empty:
            print(f"Found {len(matching_slides)} matching slides for event {event_id}")
            return matching_slides.iloc[0]
        else:
            print(f"No slides found matching event {event_id}")
            # Return the first slide from the same date as a fallback
            if not slides_df.empty:
                print(f"Found {len(slides_df)} slides from the same date")
                return slides_df.iloc[0]
            return None
            
    except Exception as e:
        print(f"Error fetching slides: {str(e)}")
        return None

def main():
    """
    Main function to get and download the latest RBC earnings transcript
    """
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("=" * 60)
    print("RBC Latest Earnings Transcript Downloader")
    print("=" * 60)
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            print(f"\nFetching transcripts for {RBC_NAME} ({RBC_PRIMARY_ID})...")
            
            # Get transcripts for RBC - remove sort to test
            api_params = {
                'ids': [RBC_PRIMARY_ID],
                'pagination_limit': PAGINATION_LIMIT,
                'pagination_offset': PAGINATION_OFFSET
            }
            
            print(f"API parameters: {api_params}")
            print(f"SORT_ORDER value: {SORT_ORDER}")
            print(f"Sort parameter being sent: {api_params.get('sort')}")
            response = api_instance.get_transcripts_ids(**api_params)
            
            if not response or not hasattr(response, 'data') or not response.data:
                print("No transcripts found for RBC")
                return
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(response.to_dict()['data'])
            print(f"Found {len(df)} total transcripts")
            
            # Sort by storyDateTime if available
            if 'story_date_time' in df.columns:
                df = df.sort_values('story_date_time', ascending=False)
                print("Sorted by story_date_time")
            elif 'event_date' in df.columns:
                df = df.sort_values('event_date', ascending=False)
                print("Sorted by event_date")
            
            # Filter for earnings transcripts only
            earnings_df = df
            if 'event_type' in df.columns:
                earnings_df = df[df['event_type'].str.contains('Earnings', case=False, na=False)]
                print(f"Found {len(earnings_df)} earnings transcripts")
            
            if earnings_df.empty:
                print("No earnings transcripts found")
                return
            
            # Get the most recent earnings transcript
            latest_transcript = earnings_df.iloc[0]
            
            print(f"\nLatest earnings transcript:")
            print(f"  Event Date: {latest_transcript.get('event_date', 'N/A')}")
            print(f"  Event Type: {latest_transcript.get('event_type', 'N/A')}")
            print(f"  Headline: {latest_transcript.get('headline', 'N/A')}")
            
            # Check if we have a transcript link
            if 'transcripts_link' not in latest_transcript or pd.isna(latest_transcript['transcripts_link']):
                print("\nNo transcript link available for download")
                return
            
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Generate filename with date
            event_date = latest_transcript.get('event_date', 'unknown_date')
            if pd.notna(event_date):
                event_date = pd.to_datetime(event_date).strftime('%Y-%m-%d')
            
            # Create filename
            output_filename = f"RBC_earnings_{event_date}.xml"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # Also save as "latest" for easy access
            latest_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
            
            # Download the transcript
            transcript_downloaded = False
            if download_file(latest_transcript['transcripts_link'], output_path, "transcript"):
                # Copy to "latest" file
                import shutil
                shutil.copy2(output_path, latest_path)
                print(f"Also saved as: {latest_path}")
                transcript_downloaded = True
            
            # Try to get slides for the same event
            event_id = latest_transcript.get('event_id')
            event_date = latest_transcript.get('event_date')
            
            slides_downloaded = False
            if event_id and event_date:
                slides_info = get_slides_for_event(event_id, event_date, api_instance)
                
                if slides_info and 'slides_link' in slides_info and pd.notna(slides_info['slides_link']):
                    # Create slides filename using the same event_date logic
                    slides_date = event_date
                    if pd.notna(slides_date):
                        slides_date = pd.to_datetime(slides_date).strftime('%Y-%m-%d')
                    slides_filename = f"RBC_earnings_slides_{slides_date}.pdf"
                    slides_path = os.path.join(OUTPUT_DIR, slides_filename)
                    
                    # Also save as "latest" slides
                    latest_slides_path = os.path.join(OUTPUT_DIR, "latest_rbc_earnings_slides.pdf")
                    
                    if download_file(slides_info['slides_link'], slides_path, "slides"):
                        shutil.copy2(slides_path, latest_slides_path)
                        print(f"Also saved as: {latest_slides_path}")
                        slides_downloaded = True
            
            # Summary
            print(f"\n{'='*60}")
            print("DOWNLOAD SUMMARY")
            print(f"{'='*60}")
            if transcript_downloaded:
                print(f"✓ Transcript downloaded: {output_path}")
            if slides_downloaded:
                print(f"✓ Slides downloaded: {slides_path}")
            
            if not transcript_downloaded and not slides_downloaded:
                print("✗ No files downloaded")
            elif transcript_downloaded and not slides_downloaded:
                print("✓ Transcript downloaded (no slides available)")
            elif not transcript_downloaded and slides_downloaded:
                print("✓ Slides downloaded (transcript failed)")
            else:
                print("✓ Both transcript and slides downloaded successfully!")
                
            print(f"Files saved in: {OUTPUT_DIR}")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()