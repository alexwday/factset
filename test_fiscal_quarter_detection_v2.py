"""
Test Script for Fiscal Quarter Detection v2
This script combines the exact patterns from factset_bank_transcripts_puller.py 
and factset_calendar_events_puller.py to test fiscal quarter detection.
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

# =============================================================================
# DATE RANGE CONFIGURATION - MODIFY THESE FOR YOUR TEST
# =============================================================================

# Date Range Configuration (Format: YYYY-MM-DD)
START_DATE = "2025-01-07"
END_DATE = "2025-01-07"

# Categories Filter - Updated as requested
CATEGORIES_FILTER = ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"]

# Event Type Filter (Earnings only)
EVENT_TYPE = "Earnings"

# Major Canadian and US Banks (from both reference scripts)
BANK_PRIMARY_IDS = {
    # Major Canadian Banks ("Big Six") - Include both CA and US listings
    "RY-CA": {"name": "Royal Bank of Canada", "region": "Canada"},
    "RY-US": {"name": "Royal Bank of Canada", "region": "Canada"},
    "TD-CA": {"name": "Toronto-Dominion Bank", "region": "Canada"},
    "TD-US": {"name": "Toronto-Dominion Bank", "region": "Canada"},
    "BNS-CA": {"name": "Bank of Nova Scotia (Scotiabank)", "region": "Canada"},
    "BNS-US": {"name": "Bank of Nova Scotia (Scotiabank)", "region": "Canada"},
    "BMO-CA": {"name": "Bank of Montreal", "region": "Canada"},
    "BMO-US": {"name": "Bank of Montreal", "region": "Canada"},
    "CM-CA": {"name": "Canadian Imperial Bank of Commerce (CIBC)", "region": "Canada"},
    "CM-US": {"name": "Canadian Imperial Bank of Commerce (CIBC)", "region": "Canada"},
    "NA-CA": {"name": "National Bank of Canada", "region": "Canada"},
    
    # Major US Banks
    "JPM-US": {"name": "JPMorgan Chase & Co.", "region": "US"},
    "BAC-US": {"name": "Bank of America Corporation", "region": "US"},
    "WFC-US": {"name": "Wells Fargo & Company", "region": "US"},
    "C-US": {"name": "Citigroup Inc.", "region": "US"},
    "USB-US": {"name": "U.S. Bancorp", "region": "US"},
    "PNC-US": {"name": "PNC Financial Services Group", "region": "US"},
    "TFC-US": {"name": "Truist Financial Corporation", "region": "US"},
    "COF-US": {"name": "Capital One Financial Corporation", "region": "US"},
    "MS-US": {"name": "Morgan Stanley", "region": "US"},
    "GS-US": {"name": "Goldman Sachs Group Inc.", "region": "US"},
    "BK-US": {"name": "The Bank of New York Mellon Corporation", "region": "US"},
    "STT-US": {"name": "State Street Corporation", "region": "US"},
    "AXP-US": {"name": "American Express Company", "region": "US"},
    "SCHW-US": {"name": "Charles Schwab Corporation", "region": "US"},
    "BLK-US": {"name": "BlackRock Inc.", "region": "US"},
    "ALLY-US": {"name": "Ally Financial Inc.", "region": "US"},
    "RF-US": {"name": "Regions Financial Corporation", "region": "US"},
    "KEY-US": {"name": "KeyCorp", "region": "US"},
    "CFG-US": {"name": "Citizens Financial Group Inc.", "region": "US"},
    "MTB-US": {"name": "M&T Bank Corporation", "region": "US"},
    "FITB-US": {"name": "Fifth Third Bancorp", "region": "US"},
    "HBAN-US": {"name": "Huntington Bancshares Incorporated", "region": "US"},
    "ZION-US": {"name": "Zions Bancorporation", "region": "US"},
    "CMA-US": {"name": "Comerica Incorporated", "region": "US"},
}

# Output Configuration
OUTPUT_FILE = "fiscal_quarter_test_results.csv"
SORT_ORDER = ["-storyDateTime"]
PAGINATION_LIMIT = 1000
PAGINATION_OFFSET = 0

# Request Configuration
REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting
BATCH_SIZE = 10  # Process banks in batches of this size

# =============================================================================
# SETUP AND CONFIGURATION (EXACT SAME AS REFERENCE SCRIPTS)
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

def generate_date_list(start_date_str, end_date_str):
    """
    Generate a list of dates between start_date and end_date (inclusive).
    Same pattern as factset_bank_transcripts_puller.py
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
    Get transcripts for a specific date - same pattern as reference script
    """
    try:
        print(f"  Fetching transcripts for {target_date}...")
        
        # Make API call - same pattern as factset_bank_transcripts_puller.py
        response = api_instance.get_transcripts_ids(
            start_date=target_date,
            end_date=target_date,
            categories=CATEGORIES_FILTER,
            sort=SORT_ORDER,
            pagination_limit=PAGINATION_LIMIT,
            pagination_offset=PAGINATION_OFFSET
        )
        
        if response.data:
            # Convert to DataFrame - same pattern as reference script
            df = pd.DataFrame([transcript.to_dict() for transcript in response.data])
            df['fetch_date'] = target_date
            print(f"    Found {len(df)} transcripts")
            return df
        else:
            print(f"    No transcripts found")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"    Error fetching transcripts for {target_date}: {e}")
        return pd.DataFrame()

def get_calendar_events(symbols, start_date, end_date, api_instance):
    """
    Get calendar events for symbols - EXACT same pattern as factset_calendar_events_puller.py
    """
    try:
        print(f"\nFetching calendar events for {len(symbols)} banks...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Create request object with proper ISO 8601 format using dateutil_parser
        request_data_dict = {
            "date_time": CompanyEventRequestDataDateTime(
                start=dateutil_parser(start_date.strftime('%Y-%m-%dT00:00:00Z')),
                end=dateutil_parser(end_date.strftime('%Y-%m-%dT23:59:59Z'))
            ),
            "universe": CompanyEventRequestDataUniverse(
                symbols=symbols,
                type="Tickers"
            )
        }
        
        # Only add event_types if we have specific types to filter
        # In our case, we don't define EVENT_TYPES so it won't be added
        
        request_data = CompanyEventRequestData(**request_data_dict)
        
        request = CompanyEventRequest(data=request_data)
        
        # Make API call
        response = api_instance.get_company_event(request)
        
        if not response or not hasattr(response, 'data') or not response.data:
            print("No events found")
            return []
        
        events = response.data
        print(f"Found {len(events)} total events")
        
        # Filter by industry categories if available
        filtered_events = []
        for event in events:
            # Convert event object to dict for easier handling
            event_dict = event.to_dict() if hasattr(event, 'to_dict') else event
            
            # For now, include all events from our bank list
            # You can add category filtering here if the response includes categories
            filtered_events.append(event_dict)
        
        print(f"Filtered to {len(filtered_events)} relevant events")
        return filtered_events
            
    except Exception as e:
        print(f"Error fetching calendar events: {str(e)}")
        return []

def cross_reference_fiscal_data(transcripts_df, calendar_events):
    """
    Cross-reference transcripts with calendar events to extract fiscal information
    """
    print("\n=== Cross-Referencing Transcripts with Calendar Events ===")
    
    if transcripts_df.empty:
        print("No transcripts to cross-reference")
        return transcripts_df
    
    if not calendar_events:
        print("No calendar events to cross-reference")
        return transcripts_df
    
    # Create lookup dictionaries for events
    events_by_event_id = {}
    events_by_report_id = {}
    events_by_id_date = defaultdict(list)
    
    for event in calendar_events:
        if event.get('event_id'):
            events_by_event_id[event['event_id']] = event
        if event.get('report_id'):
            events_by_report_id[event['report_id']] = event
        if event.get('primary_id') and event.get('event_date'):
            key = f"{event['primary_id']}_{event['event_date']}"
            events_by_id_date[key].append(event)
    
    # Add fiscal information columns
    transcripts_df['fiscal_year'] = None
    transcripts_df['fiscal_period'] = None
    transcripts_df['has_fiscal_info'] = False
    transcripts_df['match_method'] = None
    
    matches_found = 0
    
    for idx, row in transcripts_df.iterrows():
        matching_event = None
        match_method = None
        
        # Method 1: Match by event_id
        if row.get('event_id') and row['event_id'] in events_by_event_id:
            matching_event = events_by_event_id[row['event_id']]
            match_method = "event_id"
        
        # Method 2: Match by report_id
        elif row.get('report_id') and row['report_id'] in events_by_report_id:
            matching_event = events_by_report_id[row['report_id']]
            match_method = "report_id"
        
        # Method 3: Match by primary_id and date
        elif row.get('primary_ids'):
            primary_ids = row['primary_ids'] if isinstance(row['primary_ids'], list) else [row['primary_ids']]
            for primary_id in primary_ids:
                key = f"{primary_id}_{row.get('event_date')}"
                if key in events_by_id_date:
                    matching_event = events_by_id_date[key][0]  # Take first match
                    match_method = "primary_id_date"
                    break
        
        # Extract fiscal information if match found
        if matching_event:
            transcripts_df.at[idx, 'fiscal_year'] = matching_event.get('fiscal_year')
            transcripts_df.at[idx, 'fiscal_period'] = matching_event.get('fiscal_period')
            transcripts_df.at[idx, 'has_fiscal_info'] = True
            transcripts_df.at[idx, 'match_method'] = match_method
            matches_found += 1
    
    print(f"Successfully matched {matches_found} out of {len(transcripts_df)} transcripts")
    
    return transcripts_df

def filter_for_monitored_banks(df):
    """
    Filter transcripts to only include monitored banks
    """
    if df.empty:
        return df
    
    # Add bank information columns
    df['monitored_banks'] = None
    df['bank_names'] = None
    df['is_monitored_bank'] = False
    
    monitored_count = 0
    
    for idx, row in df.iterrows():
        primary_ids = row.get('primary_ids', [])
        if not isinstance(primary_ids, list):
            primary_ids = [primary_ids] if primary_ids else []
        
        # Check if any primary IDs match our monitored banks
        bank_matches = [pid for pid in primary_ids if pid in BANK_PRIMARY_IDS.keys()]
        
        if bank_matches:
            df.at[idx, 'monitored_banks'] = bank_matches
            df.at[idx, 'bank_names'] = [BANK_PRIMARY_IDS[pid]['name'] for pid in bank_matches]
            df.at[idx, 'is_monitored_bank'] = True
            monitored_count += 1
    
    # Filter to only monitored banks
    filtered_df = df[df['is_monitored_bank'] == True].copy()
    
    print(f"Filtered to {len(filtered_df)} transcripts from monitored banks (out of {len(df)} total)")
    
    return filtered_df

def main():
    """
    Main function combining both transcript pulling and calendar events patterns
    """
    print("Starting Fiscal Quarter Detection Test v2...")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Categories: {CATEGORIES_FILTER}")
    print(f"Monitored Banks: {len(BANK_PRIMARY_IDS)} banks")
    
    # Generate date list - same pattern as factset_bank_transcripts_puller.py
    date_list = generate_date_list(START_DATE, END_DATE)
    print(f"Processing {len(date_list)} days...")
    
    # =============================================================================
    # STEP 1: GET TRANSCRIPTS (using transcript puller pattern)
    # =============================================================================
    print("\n=== Step 1: Getting Transcripts ===")
    
    all_transcripts = []
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        
        for i, target_date in enumerate(date_list, 1):
            print(f"Processing day {i}/{len(date_list)}: {target_date}")
            
            # Get transcripts for this date
            daily_transcripts = get_transcripts_for_date(target_date, api_instance)
            
            if not daily_transcripts.empty:
                all_transcripts.append(daily_transcripts)
            
            # Rate limiting
            if i < len(date_list):
                time.sleep(REQUEST_DELAY)
    
    # Combine all transcripts
    if all_transcripts:
        transcripts_df = pd.concat(all_transcripts, ignore_index=True)
        print(f"\nTotal transcripts found: {len(transcripts_df)}")
    else:
        print("\nNo transcripts found for the specified date range")
        return
    
    # Filter for monitored banks
    transcripts_df = filter_for_monitored_banks(transcripts_df)
    
    if transcripts_df.empty:
        print("No transcripts found for monitored banks")
        return
    
    # =============================================================================
    # STEP 2: GET CALENDAR EVENTS (using calendar events puller pattern)
    # =============================================================================
    print("\n=== Step 2: Getting Calendar Events ===")
    
    # Get all bank symbols
    all_symbols = list(BANK_PRIMARY_IDS.keys())
    
    # Calculate date range for calendar events (expand slightly)
    # Convert to datetime objects (not date objects) to match reference script
    start_date = dateutil_parser(START_DATE) - timedelta(days=2)
    end_date = dateutil_parser(END_DATE) + timedelta(days=2)
    
    all_events = []
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = calendar_events_api.CalendarEventsApi(api_client)
        
        # Process banks in batches - same pattern as calendar events script
        for i in range(0, len(all_symbols), BATCH_SIZE):
            batch_symbols = all_symbols[i:i + BATCH_SIZE]
            
            print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(all_symbols) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            batch_events = get_calendar_events(batch_symbols, start_date, end_date, api_instance)
            all_events.extend(batch_events)
            
            # Rate limiting
            if i + BATCH_SIZE < len(all_symbols):
                time.sleep(REQUEST_DELAY)
    
    print(f"\nTotal calendar events found: {len(all_events)}")
    
    # =============================================================================
    # STEP 3: CROSS-REFERENCE DATA
    # =============================================================================
    if all_events:
        enhanced_transcripts = cross_reference_fiscal_data(transcripts_df, all_events)
    else:
        print("No calendar events found - adding empty fiscal info columns")
        transcripts_df['fiscal_year'] = None
        transcripts_df['fiscal_period'] = None
        transcripts_df['has_fiscal_info'] = False
        transcripts_df['match_method'] = None
        enhanced_transcripts = transcripts_df
    
    # =============================================================================
    # STEP 4: GENERATE SUMMARY
    # =============================================================================
    print("\n" + "="*60)
    print("FISCAL QUARTER DETECTION TEST SUMMARY")
    print("="*60)
    
    total_transcripts = len(enhanced_transcripts)
    with_fiscal_info = len(enhanced_transcripts[enhanced_transcripts['has_fiscal_info'] == True])
    
    print(f"Total Transcripts Found: {total_transcripts}")
    print(f"Transcripts with Fiscal Info: {with_fiscal_info}")
    if total_transcripts > 0:
        print(f"Fiscal Information Success Rate: {with_fiscal_info/total_transcripts*100:.1f}%")
    
    # Show fiscal periods found
    if with_fiscal_info > 0:
        # Convert bank_names lists to strings for grouping
        fiscal_transcripts = enhanced_transcripts[enhanced_transcripts['has_fiscal_info'] == True].copy()
        fiscal_transcripts['bank_names_str'] = fiscal_transcripts['bank_names'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        fiscal_summary = fiscal_transcripts.groupby(['fiscal_year', 'fiscal_period', 'bank_names_str']).size().reset_index(name='count')
        print(f"\nFiscal Periods Found:")
        for _, row in fiscal_summary.iterrows():
            print(f"  {row['fiscal_year']} {row['fiscal_period']}: {row['bank_names_str']} ({row['count']} transcripts)")
    
    # Save results
    enhanced_transcripts.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()