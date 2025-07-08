"""
Test Script for Fiscal Quarter Detection
This script tests the approach of cross-referencing Transcripts API with Calendar Events API 
to extract fiscal quarter and fiscal year information for bank earnings calls.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api, calendar_events_api
from fds.sdk.EventsandTranscripts.model.company_event_request import CompanyEventRequest
from fds.sdk.EventsandTranscripts.model.company_event_request_data import CompanyEventRequestData
from fds.sdk.EventsandTranscripts.model.company_event_request_data_date_time import CompanyEventRequestDataDateTime
from fds.sdk.EventsandTranscripts.model.company_event_request_data_universe import CompanyEventRequestDataUniverse
from dateutil.parser import parse as dateutil_parser
import os
from urllib.parse import quote
from datetime import datetime, timedelta, date
import time
import json
from collections import defaultdict

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

# Major Canadian and US Banks for testing
TEST_BANKS = {
    # Major Canadian Banks ("Big Six") - Include both CA and US listings
    "RY-CA": "Royal Bank of Canada",
    "RY-US": "Royal Bank of Canada",
    "TD-CA": "Toronto-Dominion Bank",
    "TD-US": "Toronto-Dominion Bank",
    "BNS-CA": "Bank of Nova Scotia (Scotiabank)",
    "BNS-US": "Bank of Nova Scotia (Scotiabank)",
    "BMO-CA": "Bank of Montreal",
    "BMO-US": "Bank of Montreal",
    "CM-CA": "Canadian Imperial Bank of Commerce (CIBC)",
    "CM-US": "Canadian Imperial Bank of Commerce (CIBC)",
    "NA-CA": "National Bank of Canada",
    
    # Major US Banks
    "JPM-US": "JPMorgan Chase & Co.",
    "BAC-US": "Bank of America Corporation",
    "WFC-US": "Wells Fargo & Company",
    "C-US": "Citigroup Inc.",
    "USB-US": "U.S. Bancorp",
    "PNC-US": "PNC Financial Services Group",
    "TFC-US": "Truist Financial Corporation",
    "COF-US": "Capital One Financial Corporation",
    "MS-US": "Morgan Stanley",
    "GS-US": "Goldman Sachs Group Inc.",
    "BK-US": "The Bank of New York Mellon Corporation",
    "STT-US": "State Street Corporation",
    "AXP-US": "American Express Company",
    "SCHW-US": "Charles Schwab Corporation",
    "BLK-US": "BlackRock Inc.",
    "ALLY-US": "Ally Financial Inc.",
    "RF-US": "Regions Financial Corporation",
    "KEY-US": "KeyCorp",
    "CFG-US": "Citizens Financial Group Inc.",
    "MTB-US": "M&T Bank Corporation",
    "FITB-US": "Fifth Third Bancorp",
    "HBAN-US": "Huntington Bancshares Incorporated",
    "ZION-US": "Zions Bancorporation",
    "CMA-US": "Comerica Incorporated"
}

# Categories Filter (Banking, Financial Services, Insurance, Securities)
CATEGORIES_FILTER = ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"]

# Event Type Filter (Earnings only)
EVENT_TYPE = "Earnings"

# =============================================================================
# DATE RANGE CONFIGURATION
# =============================================================================

# Date Range Configuration - Modify these for your test
# Format: YYYY-MM-DD strings or use date objects
# For single day test: set both to the same date
START_DATE = "2025-01-07"  # Change this to your desired start date
END_DATE = "2025-01-07"    # Change this to your desired end date

# Convert to date objects for internal use
DEFAULT_START_DATE = dateutil_parser(START_DATE).date()
DEFAULT_END_DATE = dateutil_parser(END_DATE).date()

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication (same as reference scripts)
user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Configure FactSet API client (same pattern as reference scripts)
configuration = fds.sdk.EventsandTranscripts.Configuration(
    username=API_USERNAME,
    password=API_PASSWORD,
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=SSL_CERT_PATH
)
configuration.get_basic_auth_token()

def setup_api_client():
    """Setup FactSet API client with proxy and SSL configuration."""
    return fds.sdk.EventsandTranscripts.ApiClient(configuration)

def get_transcripts_by_date_range(api_client, start_date=None, end_date=None):
    """Get transcripts for specified date range for banking industry."""
    print("=== Testing Transcripts API for Date Range ===")
    
    # Use defaults if no dates provided
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = DEFAULT_END_DATE
    
    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = dateutil_parser(start_date).date()
    if isinstance(end_date, str):
        end_date = dateutil_parser(end_date).date()
    
    try:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        
        print(f"Searching for transcripts from: {start_date} to {end_date}")
        
        # Get transcripts for date range
        response = api_instance.get_transcripts_dates(
            start_date=start_date,
            end_date=end_date,
            categories=CATEGORIES_FILTER,
            event_type=EVENT_TYPE,
            time_zone=TIME_ZONE,
            pagination_limit=1000,
            sort="-storyDateTime"
        )
        
        print(f"Found {len(response.data) if response.data else 0} transcripts")
        
        transcripts_data = []
        if response.data:
            for transcript in response.data:
                # Extract key information
                transcript_info = {
                    'report_id': transcript.report_id,
                    'event_id': transcript.event_id,
                    'event_date': transcript.event_date,
                    'story_date_time': transcript.story_date_time,
                    'primary_ids': transcript.primary_ids,
                    'event_type': transcript.event_type,
                    'headline': transcript.headline,
                    'categories': transcript.categories,
                    'has_fiscal_info': False  # Will be updated after calendar events check
                }
                
                # Check if any primary IDs match our monitored banks
                bank_matches = [pid for pid in transcript.primary_ids if pid in TEST_BANKS.keys()]
                transcript_info['monitored_banks'] = bank_matches
                transcript_info['bank_names'] = [TEST_BANKS.get(pid, pid) for pid in bank_matches]
                
                transcripts_data.append(transcript_info)
                
                print(f"\nTranscript: {transcript.report_id}")
                print(f"  Event ID: {transcript.event_id}")
                print(f"  Event Date: {transcript.event_date}")
                print(f"  Primary IDs: {transcript.primary_ids}")
                print(f"  Monitored Banks: {bank_matches}")
                print(f"  Headline: {transcript.headline}")
                print(f"  Categories: {transcript.categories}")
        
        return transcripts_data
        
    except Exception as e:
        print(f"Error fetching transcripts: {e}")
        return []

def get_calendar_events_for_transcripts(api_client, transcripts_data, start_date=None, end_date=None):
    """Get calendar events data to extract fiscal information."""
    print("\n=== Testing Calendar Events API for Fiscal Information ===")
    
    # Use defaults if no dates provided
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = DEFAULT_END_DATE
    
    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = dateutil_parser(start_date).date()
    if isinstance(end_date, str):
        end_date = dateutil_parser(end_date).date()
    
    try:
        api_instance = calendar_events_api.CalendarEventsApi(api_client)
        
        # Expand the range slightly to catch events that might be related
        search_start = start_date - timedelta(days=2)
        search_end = end_date + timedelta(days=2)
        
        print(f"Searching calendar events from {search_start} to {search_end}")
        
        # Build request for calendar events
        request_data = CompanyEventRequestData(
            universe=CompanyEventRequestDataUniverse(
                identifiers=list(TEST_BANKS.keys())
            ),
            date_time=CompanyEventRequestDataDateTime(
                start=search_start.strftime('%Y-%m-%d'),
                end=search_end.strftime('%Y-%m-%d')
            )
        )
        
        request = CompanyEventRequest(data=request_data)
        
        # Get calendar events
        response = api_instance.get_events(company_event_request=request)
        
        print(f"Found {len(response.data) if response.data else 0} calendar events")
        
        events_data = []
        if response.data:
            for event in response.data:
                event_info = {
                    'event_id': getattr(event, 'event_id', None),
                    'report_id': getattr(event, 'report_id', None),
                    'primary_id': getattr(event, 'primary_id', None),
                    'event_date': getattr(event, 'event_date', None),
                    'event_type': getattr(event, 'event_type', None),
                    'headline': getattr(event, 'headline', None),
                    'fiscal_year': getattr(event, 'fiscal_year', None),
                    'fiscal_period': getattr(event, 'fiscal_period', None),
                    'categories': getattr(event, 'categories', None)
                }
                
                events_data.append(event_info)
                
                print(f"\nCalendar Event: {event_info['event_id']}")
                print(f"  Report ID: {event_info['report_id']}")
                print(f"  Primary ID: {event_info['primary_id']}")
                print(f"  Event Date: {event_info['event_date']}")
                print(f"  Event Type: {event_info['event_type']}")
                print(f"  Fiscal Year: {event_info['fiscal_year']}")
                print(f"  Fiscal Period: {event_info['fiscal_period']}")
                print(f"  Headline: {event_info['headline']}")
        
        return events_data
        
    except Exception as e:
        print(f"Error fetching calendar events: {e}")
        return []

def cross_reference_fiscal_data(transcripts_data, events_data):
    """Cross-reference transcripts with calendar events to extract fiscal information."""
    print("\n=== Cross-Referencing Transcripts with Calendar Events ===")
    
    # Create lookup dictionaries for events
    events_by_event_id = {event['event_id']: event for event in events_data if event['event_id']}
    events_by_report_id = {event['report_id']: event for event in events_data if event['report_id']}
    
    # Create lookup by primary_id and date for fuzzy matching
    events_by_id_date = defaultdict(list)
    for event in events_data:
        if event['primary_id'] and event['event_date']:
            key = f"{event['primary_id']}_{event['event_date']}"
            events_by_id_date[key].append(event)
    
    enhanced_transcripts = []
    
    for transcript in transcripts_data:
        enhanced_transcript = transcript.copy()
        
        # Try to find matching calendar event
        matching_event = None
        match_method = None
        
        # Method 1: Match by event_id
        if transcript['event_id'] and transcript['event_id'] in events_by_event_id:
            matching_event = events_by_event_id[transcript['event_id']]
            match_method = "event_id"
        
        # Method 2: Match by report_id
        elif transcript['report_id'] and transcript['report_id'] in events_by_report_id:
            matching_event = events_by_report_id[transcript['report_id']]
            match_method = "report_id"
        
        # Method 3: Match by primary_id and date
        else:
            for primary_id in transcript['primary_ids']:
                key = f"{primary_id}_{transcript['event_date']}"
                if key in events_by_id_date:
                    matching_event = events_by_id_date[key][0]  # Take first match
                    match_method = "primary_id_date"
                    break
        
        # Extract fiscal information if match found
        if matching_event:
            enhanced_transcript['fiscal_year'] = matching_event['fiscal_year']
            enhanced_transcript['fiscal_period'] = matching_event['fiscal_period']
            enhanced_transcript['has_fiscal_info'] = True
            enhanced_transcript['match_method'] = match_method
            
            print(f"\nMatched Transcript {transcript['report_id']}:")
            print(f"  Match Method: {match_method}")
            print(f"  Fiscal Year: {matching_event['fiscal_year']}")
            print(f"  Fiscal Period: {matching_event['fiscal_period']}")
            print(f"  Bank(s): {transcript['bank_names']}")
        else:
            enhanced_transcript['fiscal_year'] = None
            enhanced_transcript['fiscal_period'] = None
            enhanced_transcript['has_fiscal_info'] = False
            enhanced_transcript['match_method'] = None
            
            print(f"\nNo fiscal match for Transcript {transcript['report_id']}")
            print(f"  Bank(s): {transcript['bank_names']}")
        
        enhanced_transcripts.append(enhanced_transcript)
    
    return enhanced_transcripts

def generate_test_summary(enhanced_transcripts):
    """Generate a summary of the test results."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total_transcripts = len(enhanced_transcripts)
    with_fiscal_info = sum(1 for t in enhanced_transcripts if t['has_fiscal_info'])
    monitored_banks = sum(1 for t in enhanced_transcripts if t['monitored_banks'])
    
    print(f"Total Transcripts Found: {total_transcripts}")
    print(f"Transcripts with Fiscal Info: {with_fiscal_info}")
    print(f"Transcripts from Monitored Banks: {monitored_banks}")
    
    if enhanced_transcripts:
        print(f"\nFiscal Information Success Rate: {with_fiscal_info/total_transcripts*100:.1f}%")
        
        # Group by fiscal period
        fiscal_periods = defaultdict(list)
        for transcript in enhanced_transcripts:
            if transcript['has_fiscal_info']:
                key = f"{transcript['fiscal_year']}_{transcript['fiscal_period']}"
                fiscal_periods[key].append(transcript)
        
        print(f"\nFiscal Periods Found:")
        for period, transcripts in fiscal_periods.items():
            print(f"  {period}: {len(transcripts)} transcripts")
            for transcript in transcripts:
                print(f"    - {transcript['bank_names']} ({transcript['report_id']})")
    
    # Save detailed results to CSV
    if enhanced_transcripts:
        df = pd.DataFrame(enhanced_transcripts)
        output_file = f"fiscal_quarter_test_results_{date.today().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

def main(start_date=None, end_date=None):
    """Main function to run the fiscal quarter detection test.
    
    Args:
        start_date: Start date for search (YYYY-MM-DD string or date object)
        end_date: End date for search (YYYY-MM-DD string or date object)
                 If None, uses start_date (single day search)
    """
    # Use defaults if no dates provided
    if start_date is None:
        start_date = DEFAULT_START_DATE
    if end_date is None:
        end_date = start_date if start_date else DEFAULT_END_DATE
    
    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = dateutil_parser(start_date).date()
    if isinstance(end_date, str):
        end_date = dateutil_parser(end_date).date()
    
    print("Starting Fiscal Quarter Detection Test...")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Categories: {CATEGORIES_FILTER}")
    print(f"Monitored Banks: {len(TEST_BANKS)} banks")
    
    # Setup API client
    api_client = setup_api_client()
    
    try:
        # Step 1: Get transcripts for date range
        transcripts_data = get_transcripts_by_date_range(api_client, start_date, end_date)
        time.sleep(1)  # Rate limiting
        
        # Step 2: Get calendar events for fiscal information
        events_data = get_calendar_events_for_transcripts(api_client, transcripts_data, start_date, end_date)
        time.sleep(1)  # Rate limiting
        
        # Step 3: Cross-reference to extract fiscal information
        enhanced_transcripts = cross_reference_fiscal_data(transcripts_data, events_data)
        
        # Step 4: Generate summary
        generate_test_summary(enhanced_transcripts)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        api_client.close()

def run_single_day_test(test_date=None):
    """Convenience function to run test for a single day."""
    if test_date is None:
        test_date = date.today()
    main(start_date=test_date, end_date=test_date)

def run_date_range_test(start_date, end_date):
    """Convenience function to run test for a date range."""
    main(start_date=start_date, end_date=end_date)

if __name__ == "__main__":
    # Default: Run for today only
    main()
    
    # Example usage:
    # 
    # # Test for a specific single day
    # main(start_date="2024-01-15", end_date="2024-01-15")
    # # or
    # run_single_day_test("2024-01-15")
    # 
    # # Test for a date range
    # main(start_date="2024-01-01", end_date="2024-01-31")
    # # or
    # run_date_range_test("2024-01-01", "2024-01-31")
    #
    # # Test for today (default)
    # main()
    # # or
    # run_single_day_test()