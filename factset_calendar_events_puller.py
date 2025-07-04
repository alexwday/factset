"""
FactSet Calendar Events Puller
This script pulls upcoming events for major US and Canadian banks from FactSet Events and Transcripts API
and generates an interactive HTML calendar for the next 3 months.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import calendar_events_api
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
import calendar

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

# Major Canadian and US Banks Primary IDs with regions
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

# Industry codes we're interested in
INDUSTRY_FILTERS = ["IN:BANKS", "IN:FNLSVC", "IN:SECS", "IN:INS"]

# Event types to include - leaving empty to get ALL event types
EVENT_TYPES = []  # Empty list means include all event types

# Output Configuration
OUTPUT_HTML_FILE = "bank_events_calendar.html"
OUTPUT_DATA_FILE = "bank_events_data.csv"

# Calendar Configuration
MONTHS_AHEAD = 3  # Number of months to look ahead

# Request Configuration
REQUEST_DELAY = 1.0  # Seconds between requests to avoid rate limiting
BATCH_SIZE = 10  # Process banks in batches of this size

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

def get_date_range():
    """
    Calculate the date range for the next N months
    
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    start_date = datetime.now()
    # Calculate end date (N months ahead)
    end_date = start_date + timedelta(days=MONTHS_AHEAD * 30)
    
    return start_date, end_date

def check_industry_categories(categories, industry_filters):
    """
    Check if event categories match any of our industry filters
    
    Args:
        categories: List of category codes from the event
        industry_filters: List of industry codes we're looking for
    
    Returns:
        bool: True if matches any industry filter
    """
    if not categories:
        return False
    
    # Convert to list if string
    if isinstance(categories, str):
        categories = [categories]
    
    # Check if any category matches our filters
    for category in categories:
        if any(industry in str(category) for industry in industry_filters):
            return True
    
    return False

def get_calendar_events(symbols, start_date, end_date, api_instance):
    """
    Fetch calendar events for given symbols and date range
    
    Args:
        symbols: List of bank primary IDs
        start_date: Start datetime
        end_date: End datetime
        api_instance: CalendarEventsApi instance
    
    Returns:
        list: List of event objects
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
        if EVENT_TYPES:
            request_data_dict["event_types"] = EVENT_TYPES
            
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

def process_events_by_month(events):
    """
    Organize events by year and month for calendar display
    
    Args:
        events: List of event dictionaries
    
    Returns:
        dict: Nested dict organized by year -> month -> day -> events
    """
    events_by_date = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for event in events:
        # Parse event date
        event_date = event.get('event_date_time')
        if not event_date:
            continue
        
        # Handle different date formats
        if isinstance(event_date, str):
            try:
                event_datetime = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            except:
                continue
        else:
            event_datetime = event_date
        
        year = event_datetime.year
        month = event_datetime.month
        day = event_datetime.day
        
        # Add event to the appropriate date
        events_by_date[year][month][day].append(event)
    
    return events_by_date

def generate_html_calendar(events, start_date, end_date):
    """
    Generate modern interactive HTML calendar with filtering
    
    Args:
        events: List of event dictionaries
        start_date: Start datetime
        end_date: End datetime
    
    Returns:
        str: HTML content
    """
    # Embedded HTML template for self-contained operation
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FactSet Bank Events Calendar - Professional View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary-blue: #1e3a8a;
            --secondary-blue: #3b82f6;
            --accent-blue: #60a5fa;
            --light-blue: #dbeafe;
            --text-dark: #1f2937;
            --text-medium: #4b5563;
            --text-light: #6b7280;
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --border-light: #e2e8f0;
            --border-medium: #cbd5e1;
            --success: #059669;
            --warning: #d97706;
            --error: #dc2626;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-secondary);
            color: var(--text-dark);
            line-height: 1.5;
            overflow-x: hidden;
        }
        
        .app-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Sticky Header */
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--primary-blue);
            color: white;
            padding: 12px 0;
            box-shadow: var(--shadow-md);
            border-bottom: 3px solid var(--accent-blue);
        }
        
        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            padding: 0 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }
        
        .header-title {
            display: flex;
            flex-direction: column;
            min-width: 300px;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 2px;
            letter-spacing: -0.5px;
        }
        
        .header p {
            opacity: 0.85;
            font-size: 13px;
            font-weight: 400;
        }
        
        .header-controls {
            display: flex;
            gap: 16px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        
        /* Main Content Area */
        .main-content {
            flex: 1;
            max-width: 1600px;
            margin: 0 auto;
            padding: 0 24px;
            width: 100%;
        }
        
        /* Controls Section - Now in Header */
        .controls {
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            min-width: 140px;
        }
        
        .control-group label {
            font-weight: 500;
            margin-bottom: 4px;
            color: rgba(255, 255, 255, 0.9);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .control-group select,
        .control-group input {
            padding: 6px 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            font-size: 13px;
            background: rgba(255, 255, 255, 0.95);
            color: var(--text-dark);
            transition: all 0.2s ease;
            min-width: 140px;
        }
        
        .control-group select:focus,
        .control-group input:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.3);
            background: white;
        }
        
        /* Sidebar Layout */
        .content-wrapper {
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 16px;
            padding: 16px 0;
            min-height: calc(100vh - 120px);
        }
        
        .sidebar {
            background: var(--bg-primary);
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
            padding: 20px;
            height: fit-content;
            position: sticky;
            top: 90px;
        }
        
        .sidebar h3 {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-dark);
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-light);
        }
        
        .filter-section {
            margin-bottom: 24px;
        }
        
        .filter-section:last-child {
            margin-bottom: 0;
        }
        
        .filter-section label {
            font-weight: 500;
            margin-bottom: 8px;
            color: var(--text-medium);
            font-size: 13px;
            display: block;
        }
        
        .filter-section select {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid var(--border-light);
            border-radius: 6px;
            font-size: 14px;
            background: white;
            color: var(--text-dark);
            transition: all 0.2s ease;
        }
        
        .filter-section select:focus {
            outline: none;
            border-color: var(--secondary-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .main-calendar-area {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        /* Dashboard Cards */
        .dashboard-section {
            margin-bottom: 16px;
        }
        
        .upcoming-earnings {
            background: var(--bg-primary);
            border: 1px solid var(--border-light);
            border-radius: 8px;
            padding: 20px;
            box-shadow: var(--shadow-sm);
        }
        
        .upcoming-earnings h3 {
            color: var(--text-dark);
            margin-bottom: 16px;
            font-size: 16px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .upcoming-events {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            max-height: 120px;
            overflow-y: auto;
        }
        
        .upcoming-events::-webkit-scrollbar {
            width: 6px;
        }
        
        .upcoming-events::-webkit-scrollbar-track {
            background: var(--bg-tertiary);
            border-radius: 3px;
        }
        
        .upcoming-events::-webkit-scrollbar-thumb {
            background: var(--border-medium);
            border-radius: 3px;
        }
        
        .upcoming-events::-webkit-scrollbar-thumb:hover {
            background: var(--text-light);
        }
        
        .upcoming-event {
            background: var(--light-blue);
            border: 1px solid #bfdbfe;
            padding: 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .upcoming-event:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .upcoming-event strong {
            display: block;
            color: var(--primary-blue);
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        
        /* Month Navigation */
        .month-navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            background: var(--bg-primary);
            padding: 12px 16px;
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-light);
        }
        
        .month-nav-button {
            background: var(--secondary-blue);
            color: white;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .month-nav-button:hover:not(:disabled) {
            background: var(--primary-blue);
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .month-nav-button:disabled {
            background: var(--border-medium);
            cursor: not-allowed;
            transform: none;
            color: var(--text-light);
        }
        
        .current-month {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-dark);
            text-align: center;
            flex: 1;
        }
        
        .nav-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        
        /* Calendar */
        .calendar-wrapper {
            background: var(--bg-primary);
            border-radius: 8px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-light);
            overflow: hidden;
        }
        
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 1px;
            background: var(--border-light);
        }
        
        .day-header {
            text-align: center;
            font-weight: 600;
            padding: 12px 8px;
            background: var(--bg-tertiary);
            color: var(--text-medium);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--border-medium);
        }
        
        .day-cell {
            min-height: 110px;
            padding: 8px;
            background: var(--bg-primary);
            position: relative;
            transition: all 0.2s ease;
            border: none;
        }
        
        .day-cell:hover {
            background: var(--bg-secondary);
            transform: scale(1.02);
            z-index: 10;
            box-shadow: var(--shadow-md);
            border-radius: 4px;
        }
        
        .day-number {
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-dark);
            font-size: 14px;
        }
        
        .today .day-number {
            background: var(--secondary-blue);
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }
        
        .event {
            font-size: 10px;
            padding: 3px 6px;
            margin: 1px 0;
            border-radius: 3px;
            cursor: pointer;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            transition: all 0.2s ease;
            border: 1px solid transparent;
            line-height: 1.2;
            font-weight: 500;
        }
        
        .event:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
            z-index: 5;
        }
        
        .event strong {
            font-weight: 600;
        }
        
        .event-time {
            opacity: 0.8;
            font-size: 9px;
        }
        
        /* Event type colors */
        .event.earnings,
        .event.confirmedearningsrelease,
        .event.projectedearningsrelease {
            background-color: #fee2e2;
            color: #dc2626;
            border-color: #fecaca;
        }
        
        .event.conference {
            background-color: #fed7aa;
            color: #ea580c;
            border-color: #fdba74;
        }
        
        .event.shareholdersmeeting {
            background-color: #d1fae5;
            color: #059669;
            border-color: #a7f3d0;
        }
        
        .event.analystsinvestorsmeeting {
            background-color: #e9d5ff;
            color: #7c3aed;
            border-color: #d8b4fe;
        }
        
        .event.salesrevenuecall,
        .event.salesrevenuerelease {
            background-color: #ccfbf1;
            color: #0d9488;
            border-color: #99f6e4;
        }
        
        .event.guidancecall {
            background-color: #dbeafe;
            color: #2563eb;
            border-color: #bfdbfe;
        }
        
        .event.specialsituation {
            background-color: #e5e7eb;
            color: #4b5563;
            border-color: #d1d5db;
        }
        
        .event.split {
            background-color: #ffedd5;
            color: #ea580c;
            border-color: #fed7aa;
        }
        
        .event.dividend {
            background-color: #d1fae5;
            color: #059669;
            border-color: #86efac;
        }
        
        .event.hidden {
            display: none;
        }
        
        .empty-cell {
            background: var(--bg-tertiary);
            opacity: 0.5;
        }
        
        .more-events {
            font-size: 9px;
            color: var(--text-light);
            font-style: italic;
            margin-top: 2px;
            text-align: center;
            padding: 2px;
            background: var(--bg-secondary);
            border-radius: 2px;
            cursor: pointer;
        }
        
        .more-events:hover {
            background: var(--border-light);
            color: var(--text-medium);
        }
        
        /* Responsive Design */
        @media (max-width: 1200px) {
            .content-wrapper {
                grid-template-columns: 1fr;
            }
            
            .sidebar {
                position: static;
                order: -1;
            }
            
            .header-content {
                flex-direction: column;
                gap: 12px;
            }
            
        }
        
        @media (max-width: 768px) {
            .main-content {
                padding: 0 16px;
            }
            
            .day-cell {
                min-height: 80px;
            }
            
            .event {
                font-size: 9px;
                padding: 2px 4px;
            }
        }
        
        
        /* Modern Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(8px);
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal-content {
            background: var(--bg-primary);
            margin: 3% auto;
            border-radius: 12px;
            width: 95%;
            max-width: 700px;
            max-height: 90vh;
            overflow: hidden;
            box-shadow: var(--shadow-lg);
            animation: slideUpIn 0.4s ease;
            border: 1px solid var(--border-light);
        }
        
        @keyframes slideUpIn {
            from {
                transform: translateY(50px) scale(0.95);
                opacity: 0;
            }
            to {
                transform: translateY(0) scale(1);
                opacity: 1;
            }
        }
        
        .modal-header {
            background: var(--primary-blue);
            color: white;
            padding: 20px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 3px solid var(--accent-blue);
        }
        
        .modal-title {
            font-size: 18px;
            font-weight: 600;
            margin: 0;
        }
        
        .modal-subtitle {
            font-size: 13px;
            opacity: 0.85;
            margin: 2px 0 0 0;
        }
        
        .close {
            background: none;
            border: none;
            color: rgba(255, 255, 255, 0.8);
            font-size: 24px;
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            transition: all 0.2s ease;
            line-height: 1;
        }
        
        .close:hover {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            transform: scale(1.1);
        }
        
        .modal-body {
            padding: 24px;
            max-height: calc(90vh - 120px);
            overflow-y: auto;
        }
        
        .event-details {
            display: grid;
            gap: 20px;
        }
        
        .detail-section {
            background: var(--bg-secondary);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border-light);
        }
        
        .detail-section h4 {
            color: var(--text-dark);
            margin: 0 0 12px 0;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border-light);
        }
        
        .detail-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }
        
        .detail-item {
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-light);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        
        .detail-value {
            font-size: 14px;
            color: var(--text-dark);
            font-weight: 500;
        }
        
        .event-link {
            color: var(--secondary-blue);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.2s ease;
            padding: 8px 12px;
            background: var(--light-blue);
            border-radius: 6px;
            display: inline-block;
            margin-top: 4px;
        }
        
        .event-link:hover {
            background: var(--accent-blue);
            color: white;
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .event-type-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .priority-high {
            background: #fee2e2;
            color: #dc2626;
        }
        
        .priority-medium {
            background: #fed7aa;
            color: #ea580c;
        }
        
        .priority-low {
            background: #d1fae5;
            color: #059669;
        }
        
        /* Print Styles */
        @media print {
            * {
                -webkit-print-color-adjust: exact !important;
                color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
            
            body {
                background: white !important;
                font-size: 12px;
            }
            
            .header {
                position: static !important;
                background: var(--primary-blue) !important;
                color: white !important;
                page-break-inside: avoid;
            }
            
            .sidebar {
                display: none !important;
            }
            
            .content-wrapper {
                grid-template-columns: 1fr !important;
            }
            
            .dashboard-section {
                grid-template-columns: 1fr !important;
                page-break-inside: avoid;
            }
            
            .month-navigation {
                page-break-inside: avoid;
            }
            
            .calendar-wrapper {
                page-break-inside: avoid;
                box-shadow: none !important;
                border: 2px solid var(--border-medium) !important;
            }
            
            .day-cell {
                border: 1px solid var(--border-medium) !important;
                min-height: 80px !important;
                page-break-inside: avoid;
            }
            
            .event {
                background: var(--bg-secondary) !important;
                border: 1px solid var(--border-medium) !important;
                font-size: 9px !important;
            }
            
            .modal {
                display: none !important;
            }
            
            
            @page {
                margin: 0.75in;
                size: landscape;
            }
            
            .print-header {
                display: block;
                text-align: center;
                margin-bottom: 20px;
                font-weight: bold;
                font-size: 16px;
            }
            
            .print-footer {
                display: block;
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                text-align: center;
                font-size: 10px;
                color: var(--text-light);
                padding: 10px;
                border-top: 1px solid var(--border-light);
            }
        }
        
        .print-header,
        .print-footer {
            display: none;
        }
        
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Sticky Header -->
        <div class="header">
            <div class="header-content">
                <div class="header-title">
                    <h1>FactSet Bank Events Calendar</h1>
                    <p>Professional institutional calendar - Generated {{GENERATED_DATE}}</p>
                </div>
                
                <div class="header-controls">
                    <!-- Quick Filters -->
                    <div class="controls">
                        <div class="control-group">
                            <label for="quickRegionFilter">Region</label>
                            <select id="quickRegionFilter" onchange="applyFilters()">
                                <option value="all">All Regions</option>
                                <option value="US">United States</option>
                                <option value="Canada">Canada</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="content-wrapper">
                <!-- Sidebar Filters -->
                <div class="sidebar">
                    <h3>Advanced Filters</h3>
                    
                    <div class="filter-section">
                        <label for="regionFilter">Region</label>
                        <select id="regionFilter" onchange="applyFilters()">
                            <option value="all">All Regions</option>
                            <option value="US">United States</option>
                            <option value="Canada">Canada</option>
                        </select>
                    </div>
                    
                    <div class="filter-section">
                        <label for="bankFilter">Banks</label>
                        <select id="bankFilter" onchange="applyFilters()">
                            <option value="all">All Banks</option>
                            <!-- Bank options will be dynamically added -->
                        </select>
                    </div>
                    
                    <div class="filter-section">
                        <label for="eventTypeFilter">Event Types</label>
                        <select id="eventTypeFilter" onchange="applyFilters()">
                            <option value="all">All Event Types</option>
                            <!-- Event type options will be dynamically added -->
                        </select>
                    </div>
                    
                </div>
                
                <!-- Main Calendar Area -->
                <div class="main-calendar-area">
                    <!-- Dashboard Section -->
                    <div class="dashboard-section">
                        <div class="upcoming-earnings">
                            <h3>📊 Upcoming Key Events</h3>
                            <div class="upcoming-events" id="upcomingEvents">
                                <!-- Upcoming earnings will be dynamically added -->
                            </div>
                        </div>
                    </div>
                    
                    <!-- Month Navigation -->
                    <div class="month-navigation">
                        <div class="nav-group">
                            <button class="month-nav-button" onclick="previousMonth()" id="prevButton">
                                ← Previous
                            </button>
                        </div>
                        
                        <div class="current-month" id="currentMonth">January 2024</div>
                        
                        <div class="nav-group">
                            <button class="month-nav-button" onclick="nextMonth()" id="nextButton">
                                Next →
                            </button>
                        </div>
                    </div>
                    
                    <!-- Calendar -->
                    <div class="calendar-wrapper">
                        <div class="print-header">
                            FactSet Bank Events Calendar - <span id="printMonth"></span>
                        </div>
                        <div class="calendar-grid" id="calendarGrid">
                            <!-- Calendar will be dynamically generated -->
                        </div>
                        <div class="print-footer">
                            Generated on {{GENERATED_DATE}} | FactSet Professional Calendar
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Modal -->
    <div id="eventModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <div id="eventDetails"></div>
        </div>
    </div>
    
    <script>
        // Global variables
        const eventsData = {{EVENTS_DATA}};
        const bankInfo = {{BANK_INFO}};
        const availableMonths = {{AVAILABLE_MONTHS}};
        let currentMonthIndex = 0;
        let selectedBank = 'all';
        let selectedEventType = 'all';
        let selectedRegion = 'all';
        let selectedDateRange = 'all';
        let currentView = 'month';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeBankDropdown();
            initializeEventTypeDropdown();
            updateUpcomingEarnings();
            renderCurrentMonth();
        });
        
        
        
        // Initialize bank dropdown
        function initializeBankDropdown() {
            const dropdown = document.getElementById('bankFilter');
            const banks = Object.entries(bankInfo).sort((a, b) => a[1].name.localeCompare(b[1].name));
            
            banks.forEach(([ticker, info]) => {
                const option = document.createElement('option');
                option.value = ticker;
                option.textContent = `${info.name} (${ticker})`;
                dropdown.appendChild(option);
            });
        }
        
        // Initialize event type dropdown
        function initializeEventTypeDropdown() {
            const dropdown = document.getElementById('eventTypeFilter');
            const eventTypes = [...new Set(eventsData.map(e => e.event_type))].filter(Boolean).sort();
            
            eventTypes.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = formatEventType(type);
                dropdown.appendChild(option);
            });
        }
        
        // Format event type for display
        function formatEventType(type) {
            const typeMap = {
                'Earnings': 'Earnings',
                'Conference': 'Conference',
                'ShareholdersMeeting': 'Shareholders Meeting',
                'AnalystsInvestorsMeeting': 'Analysts Meeting',
                'SalesRevenueCall': 'Sales/Revenue',
                'GuidanceCall': 'Guidance',
                'SpecialSituation': 'Special',
                'Split': 'Split',
                'Dividend': 'Dividend',
                'ConfirmedEarningsRelease': 'Earnings Release',
                'ProjectedEarningsRelease': 'Projected Earnings',
                'SalesRevenueRelease': 'Revenue Release'
            };
            return typeMap[type] || type;
        }
        
        // Get short event type for calendar display
        function getShortEventType(type) {
            const shortMap = {
                'Earnings': 'Earnings',
                'Conference': 'Conf',
                'ShareholdersMeeting': 'SH Mtg',
                'AnalystsInvestorsMeeting': 'Analyst',
                'SalesRevenueCall': 'Sales',
                'GuidanceCall': 'Guidance',
                'SpecialSituation': 'Special',
                'Split': 'Split',
                'Dividend': 'Dividend',
                'ConfirmedEarningsRelease': 'Earnings',
                'ProjectedEarningsRelease': 'Earnings',
                'SalesRevenueRelease': 'Sales'
            };
            return shortMap[type] || type;
        }
        
        // Update upcoming earnings banner
        function updateUpcomingEarnings() {
            const now = new Date();
            const earningsTypes = ['Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease'];
            
            // Find upcoming earnings events with filters applied
            const upcomingEarnings = eventsData
                .filter(event => {
                    const eventDate = new Date(event.event_date_time);
                    const ticker = event.ticker;
                    const region = bankInfo[ticker]?.region;
                    
                    // Check if it's an upcoming earnings event
                    if (!(eventDate >= now && earningsTypes.includes(event.event_type))) {
                        return false;
                    }
                    
                    // Check if ticker exists in our bank info
                    if (!bankInfo[ticker]) {
                        return false;
                    }
                    
                    // Apply region filter
                    if (selectedRegion !== 'all' && region !== selectedRegion) {
                        return false;
                    }
                    
                    // Apply bank filter
                    if (selectedBank !== 'all' && ticker !== selectedBank) {
                        return false;
                    }
                    
                    return true;
                })
                .sort((a, b) => new Date(a.event_date_time) - new Date(b.event_date_time));
            
            const container = document.getElementById('upcomingEvents');
            container.innerHTML = '';
            
            if (upcomingEarnings.length === 0) {
                container.innerHTML = '<div class="upcoming-event">No upcoming earnings events found for current filters</div>';
                return;
            }
            
            upcomingEarnings.forEach(event => {
                const eventEl = document.createElement('div');
                eventEl.className = 'upcoming-event';
                eventEl.onclick = () => showEventDetails(event.event_id);
                eventEl.style.cursor = 'pointer';
                
                const eventDate = new Date(event.event_date_time);
                const bankName = bankInfo[event.ticker]?.name || event.ticker;
                
                // Calculate days until event
                const daysUntil = Math.ceil((eventDate - now) / (1000 * 60 * 60 * 24));
                const timeText = daysUntil === 0 ? 'Today' : 
                                daysUntil === 1 ? 'Tomorrow' : 
                                `${daysUntil} days`;
                
                eventEl.innerHTML = `
                    <strong>${event.ticker}</strong>
                    ${bankName}<br>
                    <small>${timeText} - ${eventDate.toLocaleDateString()}</small><br>
                    ${eventDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                `;
                
                container.appendChild(eventEl);
            });
        }
        
        // Apply all filters
        function applyFilters() {
            // Sync both sets of filter controls
            const regionFilter = document.getElementById('regionFilter');
            const quickRegionFilter = document.getElementById('quickRegionFilter');
            const eventTypeFilter = document.getElementById('eventTypeFilter');
            
            // Determine which filter was changed and sync the other
            if (document.activeElement === quickRegionFilter) {
                regionFilter.value = quickRegionFilter.value;
            } else if (document.activeElement === regionFilter) {
                quickRegionFilter.value = regionFilter.value;
            }
            
            selectedRegion = regionFilter.value;
            selectedBank = document.getElementById('bankFilter').value;
            selectedEventType = eventTypeFilter.value;
            
            updateUpcomingEarnings();
            renderCurrentMonth();
        }
        
        // Check if event should be visible
        function isEventVisible(event) {
            const ticker = event.ticker;
            const eventType = event.event_type;
            const region = bankInfo[ticker]?.region;
            
            // Check if ticker exists in our bank info
            if (!bankInfo[ticker]) {
                return false;
            }
            
            // Check region filter
            if (selectedRegion !== 'all' && region !== selectedRegion) {
                return false;
            }
            
            // Check bank filter
            if (selectedBank !== 'all' && ticker !== selectedBank) {
                return false;
            }
            
            // Check event type filter
            if (selectedEventType !== 'all' && eventType !== selectedEventType) {
                return false;
            }
            
            return true;
        }
        
        // Navigation functions
        function previousMonth() {
            if (currentMonthIndex > 0) {
                currentMonthIndex--;
                renderCurrentMonth();
            }
        }
        
        function nextMonth() {
            if (currentMonthIndex < availableMonths.length - 1) {
                currentMonthIndex++;
                renderCurrentMonth();
            }
        }
        
        // Render current month
        function renderCurrentMonth() {
            const month = availableMonths[currentMonthIndex];
            const year = month.year;
            const monthNum = month.month;
            
            // Update navigation
            document.getElementById('currentMonth').textContent = `${month.name} ${year}`;
            document.getElementById('printMonth').textContent = `${month.name} ${year}`;
            document.getElementById('prevButton').disabled = currentMonthIndex === 0;
            document.getElementById('nextButton').disabled = currentMonthIndex === availableMonths.length - 1;
            
            // Get region filter
            selectedRegion = document.getElementById('regionFilter').value;
            
            // Render calendar
            const grid = document.getElementById('calendarGrid');
            grid.innerHTML = '';
            
            // Add day headers
            const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
            dayNames.forEach(day => {
                const header = document.createElement('div');
                header.className = 'day-header';
                header.textContent = day;
                grid.appendChild(header);
            });
            
            // Get calendar data
            const firstDay = new Date(year, monthNum - 1, 1).getDay();
            const daysInMonth = new Date(year, monthNum, 0).getDate();
            
            // Add empty cells for days before month starts
            for (let i = 0; i < firstDay; i++) {
                const cell = document.createElement('div');
                cell.className = 'day-cell empty-cell';
                grid.appendChild(cell);
            }
            
            // Add days of the month
            for (let day = 1; day <= daysInMonth; day++) {
                const cell = document.createElement('div');
                cell.className = 'day-cell';
                
                const dayNumber = document.createElement('div');
                dayNumber.className = 'day-number';
                dayNumber.textContent = day;
                cell.appendChild(dayNumber);
                
                // Get events for this day
                const dayEvents = eventsData.filter(event => {
                    const eventDate = new Date(event.event_date_time);
                    return eventDate.getFullYear() === year &&
                           eventDate.getMonth() === monthNum - 1 &&
                           eventDate.getDate() === day;
                });
                
                // Filter and display events
                const visibleEvents = dayEvents.filter(event => isEventVisible(event));
                const maxDisplay = 4;
                
                // Check if this is today
                const today = new Date();
                if (today.getFullYear() === year && 
                    today.getMonth() === monthNum - 1 && 
                    today.getDate() === day) {
                    cell.classList.add('today');
                }
                
                visibleEvents.slice(0, maxDisplay).forEach(event => {
                    const eventEl = document.createElement('div');
                    const eventType = (event.event_type || '').toLowerCase().replace(/\s/g, '');
                    eventEl.className = `event ${eventType}`;
                    
                    // Show ticker and abbreviated event type with time
                    const eventTime = new Date(event.event_date_time);
                    const timeStr = eventTime.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    const shortEventType = getShortEventType(event.event_type);
                    
                    eventEl.innerHTML = `
                        <strong>${event.ticker}</strong>
                        <div>${shortEventType}</div>
                        <div class="event-time">${timeStr}</div>
                    `;
                    eventEl.onclick = () => showEventDetails(event.event_id);
                    cell.appendChild(eventEl);
                });
                
                if (visibleEvents.length > maxDisplay) {
                    const more = document.createElement('div');
                    more.className = 'more-events';
                    more.textContent = `+${visibleEvents.length - maxDisplay} more`;
                    cell.appendChild(more);
                }
                
                grid.appendChild(cell);
            }
            
        }
        
        // Show event details
        function showEventDetails(eventId) {
            const event = eventsData.find(e => e.event_id === eventId);
            if (!event) return;
            
            const modal = document.getElementById('eventModal');
            const bankData = bankInfo[event.ticker];
            const eventDate = new Date(event.event_date_time);
            
            // Determine event priority based on type
            const earningsTypes = ['Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease'];
            const priority = earningsTypes.includes(event.event_type) ? 'high' : 
                           event.event_type === 'Conference' ? 'medium' : 'low';
            
            modal.innerHTML = `
                <div class="modal-content">
                    <div class="modal-header">
                        <div>
                            <h3 class="modal-title">${event.company_name || bankData?.name || event.ticker}</h3>
                            <p class="modal-subtitle">${formatEventType(event.event_type)} • ${eventDate.toLocaleDateString()}</p>
                        </div>
                        <button class="close">&times;</button>
                    </div>
                    
                    <div class="modal-body">
                        <div class="event-details">
                            <!-- Key Information -->
                            <div class="detail-section">
                                <h4>Key Information</h4>
                                <div class="detail-grid">
                                    <div class="detail-item">
                                        <span class="detail-label">Company</span>
                                        <span class="detail-value">${event.company_name || bankData?.name || 'Unknown'}</span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label">Ticker</span>
                                        <span class="detail-value">${event.ticker}</span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label">Region</span>
                                        <span class="detail-value">${bankData?.region || 'Unknown'}</span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label">Event Type</span>
                                        <span class="detail-value">
                                            <span class="event-type-badge priority-${priority}">
                                                ${formatEventType(event.event_type)}
                                            </span>
                                        </span>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Date & Time -->
                            <div class="detail-section">
                                <h4>Date & Time</h4>
                                <div class="detail-grid">
                                    <div class="detail-item">
                                        <span class="detail-label">Date</span>
                                        <span class="detail-value">${eventDate.toLocaleDateString('en-US', {
                                            weekday: 'long',
                                            year: 'numeric',
                                            month: 'long',
                                            day: 'numeric'
                                        })}</span>
                                    </div>
                                    <div class="detail-item">
                                        <span class="detail-label">Time</span>
                                        <span class="detail-value">${eventDate.toLocaleTimeString('en-US', {
                                            hour: '2-digit',
                                            minute: '2-digit',
                                            timeZoneName: 'short'
                                        })}</span>
                                    </div>
                                    ${event.market_time_code ? `
                                        <div class="detail-item">
                                            <span class="detail-label">Market Time</span>
                                            <span class="detail-value">${event.market_time_code}</span>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                            
                            ${event.description ? `
                                <div class="detail-section">
                                    <h4>Description</h4>
                                    <p style="margin: 0; color: var(--text-medium); line-height: 1.5;">${event.description}</p>
                                </div>
                            ` : ''}
                            
                            ${event.fiscal_year || event.fiscal_period ? `
                                <div class="detail-section">
                                    <h4>Fiscal Information</h4>
                                    <div class="detail-grid">
                                        ${event.fiscal_year ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Fiscal Year</span>
                                                <span class="detail-value">${event.fiscal_year}</span>
                                            </div>
                                        ` : ''}
                                        ${event.fiscal_period ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Fiscal Period</span>
                                                <span class="detail-value">${event.fiscal_period}</span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${event.webcast_link || event.ir_link ? `
                                <div class="detail-section">
                                    <h4>Links & Resources</h4>
                                    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                                        ${event.webcast_link ? `
                                            <a href="${event.webcast_link}" target="_blank" class="event-link">📺 Join Webcast</a>
                                        ` : ''}
                                        ${event.ir_link ? `
                                            <a href="${event.ir_link}" target="_blank" class="event-link">🔗 Investor Relations</a>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${event.contact_name || event.contact_email || event.contact_phone ? `
                                <div class="detail-section">
                                    <h4>Contact Information</h4>
                                    <div class="detail-grid">
                                        ${event.contact_name ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Contact Name</span>
                                                <span class="detail-value">${event.contact_name}</span>
                                            </div>
                                        ` : ''}
                                        ${event.contact_email ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Email</span>
                                                <span class="detail-value">
                                                    <a href="mailto:${event.contact_email}" class="event-link">${event.contact_email}</a>
                                                </span>
                                            </div>
                                        ` : ''}
                                        ${event.contact_phone ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Phone</span>
                                                <span class="detail-value">
                                                    <a href="tel:${event.contact_phone}" class="event-link">${event.contact_phone}</a>
                                                </span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${event.report_id || event.last_modified_date ? `
                                <div class="detail-section">
                                    <h4>Metadata</h4>
                                    <div class="detail-grid">
                                        ${event.report_id ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Report ID</span>
                                                <span class="detail-value">${event.report_id}</span>
                                            </div>
                                        ` : ''}
                                        ${event.last_modified_date ? `
                                            <div class="detail-item">
                                                <span class="detail-label">Last Modified</span>
                                                <span class="detail-value">${new Date(event.last_modified_date).toLocaleString()}</span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
            
            // Add event listeners
            modal.querySelector('.close').onclick = () => {
                modal.style.display = 'none';
            };
            
            modal.style.display = 'block';
        }
        
        
        // Modal controls - Updated to handle dynamic content
        document.addEventListener('click', function(e) {
            const modal = document.getElementById('eventModal');
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
        
        // Keyboard support
        document.addEventListener('keydown', function(e) {
            const modal = document.getElementById('eventModal');
            if (e.key === 'Escape' && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });
        
        window.onclick = function(event) {
            const modal = document.getElementById('eventModal');
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
    </script>
</body>
</html>"""
    
    # Get available months
    available_months = []
    current = start_date
    while current <= end_date:
        available_months.append({
            'year': current.year,
            'month': current.month,
            'name': calendar.month_name[current.month]
        })
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
    
    # Prepare bank info for JavaScript
    bank_info = {}
    for ticker, info in BANK_PRIMARY_IDS.items():
        bank_info[ticker] = info
    
    # Replace template variables
    html = template.replace('{{GENERATED_DATE}}', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace('{{EVENTS_DATA}}', json.dumps(events, default=str))
    html = html.replace('{{BANK_INFO}}', json.dumps(bank_info))
    html = html.replace('{{AVAILABLE_MONTHS}}', json.dumps(available_months))
    
    return html

def generate_html_calendar_old(events, start_date, end_date):
    """
    Generate an interactive HTML calendar with events
    
    Args:
        events: List of event dictionaries
        start_date: Start datetime
        end_date: End datetime
    
    Returns:
        str: HTML content
    """
    # Process events by date
    events_by_date = process_events_by_month(events)
    
    # Get list of available months
    available_months = []
    current = start_date
    while current <= end_date:
        available_months.append({
            'year': current.year,
            'month': current.month,
            'name': calendar.month_name[current.month]
        })
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)
    
    # Start HTML
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Events Calendar - Interactive View</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .calendar-container {
            display: flex;
            flex-wrap: wrap;
            gap: 30px;
            justify-content: center;
        }
        .month-calendar {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            min-width: 350px;
        }
        .month-header {
            text-align: center;
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
        }
        .day-header {
            text-align: center;
            font-weight: bold;
            padding: 5px;
            background-color: #2c3e50;
            color: white;
            font-size: 0.9em;
        }
        .day-cell {
            min-height: 80px;
            border: 1px solid #ddd;
            padding: 5px;
            position: relative;
            background-color: #fff;
        }
        .day-number {
            font-weight: bold;
            margin-bottom: 3px;
            color: #666;
        }
        .event {
            font-size: 0.75em;
            background-color: #3498db;
            color: white;
            padding: 2px 4px;
            margin: 1px 0;
            border-radius: 3px;
            cursor: pointer;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }
        .event:hover {
            background-color: #2980b9;
        }
        .event.earnings,
        .event.confirmedearningsrelease,
        .event.projectedearningsrelease {
            background-color: #e74c3c;
        }
        .event.conference {
            background-color: #f39c12;
        }
        .event.shareholdersmeeting {
            background-color: #27ae60;
        }
        .event.analystsinvestorsmeeting {
            background-color: #9b59b6;
        }
        .event.salesrevenuecall,
        .event.salesrevenuerelease {
            background-color: #16a085;
        }
        .event.guidancecall {
            background-color: #3498db;
        }
        .event.specialsituation {
            background-color: #95a5a6;
        }
        .event.split {
            background-color: #d35400;
        }
        .event.dividend {
            background-color: #2ecc71;
        }
        .empty-cell {
            background-color: #f9f9f9;
        }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        .modal-content {
            background-color: white;
            margin: 10% auto;
            padding: 20px;
            border-radius: 8px;
            width: 80%;
            max-width: 600px;
            max-height: 70vh;
            overflow-y: auto;
        }
        .close {
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close:hover {
            color: #000;
        }
        .event-details {
            margin-top: 20px;
        }
        .event-details h3 {
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .event-details p {
            margin: 5px 0;
        }
        .event-link {
            color: #3498db;
            text-decoration: none;
        }
        .event-link:hover {
            text-decoration: underline;
        }
        .legend {
            margin: 20px 0;
            text-align: center;
        }
        .legend-item {
            display: inline-block;
            margin: 0 10px;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bank Events Calendar</h1>
        <p>Upcoming events for major US and Canadian banks</p>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <span class="legend-color" style="background-color: #e74c3c;"></span>
            <span>Earnings/Releases</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #f39c12;"></span>
            <span>Conference</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #27ae60;"></span>
            <span>Shareholders Meeting</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #9b59b6;"></span>
            <span>Analysts/Investors Meeting</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #16a085;"></span>
            <span>Sales/Revenue</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #3498db;"></span>
            <span>Guidance</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #2ecc71;"></span>
            <span>Dividend</span>
        </div>
    </div>
    
    <div class="calendar-container">
"""
    
    # Generate calendar for each month
    current_date = start_date
    while current_date <= end_date:
        year = current_date.year
        month = current_date.month
        
        # Get month events
        month_events = events_by_date.get(year, {}).get(month, {})
        
        # Generate month calendar
        html += generate_month_calendar(year, month, month_events)
        
        # Move to next month
        if month == 12:
            current_date = current_date.replace(year=year+1, month=1, day=1)
        else:
            current_date = current_date.replace(month=month+1, day=1)
    
    html += """
    </div>
    
    <!-- Modal for event details -->
    <div id="eventModal" class="modal">
        <div class="modal-content">
            <span class="close">&times;</span>
            <div id="eventDetails"></div>
        </div>
    </div>
    
    <script>
        // Store all events data
        const eventsData = """ + json.dumps(events, default=str) + """;
        
        // Modal functionality
        const modal = document.getElementById('eventModal');
        const span = document.getElementsByClassName('close')[0];
        const eventDetails = document.getElementById('eventDetails');
        
        // Close modal when clicking X
        span.onclick = function() {
            modal.style.display = 'none';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            if (event.target == modal) {
                modal.style.display = 'none';
            }
        }
        
        // Show event details
        function showEventDetails(eventId) {
            const event = eventsData.find(e => e.event_id === eventId);
            if (!event) return;
            
            let detailsHtml = '<div class="event-details">';
            detailsHtml += '<h3>' + (event.company_name || 'Unknown Company') + '</h3>';
            detailsHtml += '<p><strong>Ticker:</strong> ' + (event.ticker || 'N/A') + '</p>';
            detailsHtml += '<p><strong>Event Type:</strong> ' + (event.event_type || 'N/A') + '</p>';
            detailsHtml += '<p><strong>Date & Time:</strong> ' + new Date(event.event_date_time).toLocaleString() + '</p>';
            detailsHtml += '<p><strong>Market Time:</strong> ' + (event.market_time_code || 'N/A') + '</p>';
            
            if (event.description) {
                detailsHtml += '<p><strong>Description:</strong> ' + event.description + '</p>';
            }
            
            if (event.fiscal_year || event.fiscal_period) {
                detailsHtml += '<p><strong>Fiscal Period:</strong> ' + 
                    (event.fiscal_year || '') + ' ' + (event.fiscal_period || '') + '</p>';
            }
            
            if (event.webcast_link) {
                detailsHtml += '<p><strong>Webcast:</strong> <a href="' + event.webcast_link + 
                    '" target="_blank" class="event-link">Join Webcast</a></p>';
            }
            
            if (event.ir_link) {
                detailsHtml += '<p><strong>Investor Relations:</strong> <a href="' + event.ir_link + 
                    '" target="_blank" class="event-link">IR Page</a></p>';
            }
            
            if (event.contact_name || event.contact_email || event.contact_phone) {
                detailsHtml += '<h4>Contact Information:</h4>';
                if (event.contact_name) detailsHtml += '<p><strong>Name:</strong> ' + event.contact_name + '</p>';
                if (event.contact_email) detailsHtml += '<p><strong>Email:</strong> ' + event.contact_email + '</p>';
                if (event.contact_phone) detailsHtml += '<p><strong>Phone:</strong> ' + event.contact_phone + '</p>';
            }
            
            if (event.report_id) {
                detailsHtml += '<p><strong>Report ID:</strong> ' + event.report_id + '</p>';
            }
            
            if (event.last_modified_date) {
                detailsHtml += '<p><em>Last Modified: ' + new Date(event.last_modified_date).toLocaleString() + '</em></p>';
            }
            
            detailsHtml += '</div>';
            
            eventDetails.innerHTML = detailsHtml;
            modal.style.display = 'block';
        }
    </script>
</body>
</html>
"""
    
    return html

def generate_month_calendar(year, month, month_events):
    """
    Generate HTML for a single month calendar
    
    Args:
        year: Year
        month: Month (1-12)
        month_events: Dict of day -> events list
    
    Returns:
        str: HTML for the month
    """
    month_name = calendar.month_name[month]
    cal = calendar.monthcalendar(year, month)
    
    html = f"""
    <div class="month-calendar">
        <div class="month-header">{month_name} {year}</div>
        <div class="calendar-grid">
"""
    
    # Day headers
    for day_name in ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']:
        html += f'<div class="day-header">{day_name}</div>'
    
    # Calendar days
    for week in cal:
        for day in week:
            if day == 0:
                html += '<div class="day-cell empty-cell"></div>'
            else:
                day_events = month_events.get(day, [])
                html += '<div class="day-cell">'
                html += f'<div class="day-number">{day}</div>'
                
                # Add events for this day
                for event in day_events[:3]:  # Show max 3 events per day
                    event_type = event.get('event_type', '').lower().replace(' ', '')
                    event_id = event.get('event_id', '')
                    ticker = event.get('ticker', 'Unknown')
                    
                    html += f'<div class="event {event_type}" onclick="showEventDetails(\'{event_id}\')">'
                    html += f'{ticker}'
                    html += '</div>'
                
                if len(day_events) > 3:
                    html += f'<div style="font-size: 0.7em; color: #666;">+{len(day_events)-3} more</div>'
                
                html += '</div>'
    
    html += """
        </div>
    </div>
"""
    
    return html

def save_events_to_csv(events, output_file):
    """
    Save events data to CSV for further analysis
    
    Args:
        events: List of event dictionaries
        output_file: Output CSV file path
    """
    if not events:
        print("No events to save")
        return
    
    try:
        df = pd.DataFrame(events)
        df.to_csv(output_file, index=False)
        print(f"Events data saved to: {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")

def main():
    """
    Main function to orchestrate the calendar events pulling process
    """
    print("=" * 60)
    print("FactSet Calendar Events Puller")
    print("=" * 60)
    
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Banks: {len(BANK_PRIMARY_IDS)} banks")
    print(f"  Months ahead: {MONTHS_AHEAD}")
    print(f"  Event types: {', '.join(EVENT_TYPES)}")
    print(f"  Industry filters: {', '.join(INDUSTRY_FILTERS)}")
    print(f"  Output HTML: {OUTPUT_HTML_FILE}")
    print(f"  Output CSV: {OUTPUT_DATA_FILE}")
    
    # Get date range
    start_date, end_date = get_date_range()
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    
    # Get bank symbols
    all_symbols = list(BANK_PRIMARY_IDS.keys())
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = calendar_events_api.CalendarEventsApi(api_client)
            
            # Process banks in batches
            all_events = []
            for i in range(0, len(all_symbols), BATCH_SIZE):
                batch_symbols = all_symbols[i:i + BATCH_SIZE]
                print(f"\nProcessing batch {i//BATCH_SIZE + 1}/{(len(all_symbols) + BATCH_SIZE - 1)//BATCH_SIZE}")
                print(f"Banks in batch: {', '.join(batch_symbols[:3])}{'...' if len(batch_symbols) > 3 else ''}")
                
                # Fetch calendar events for this batch
                batch_events = get_calendar_events(batch_symbols, start_date, end_date, api_instance)
                all_events.extend(batch_events)
                
                # Delay between batches
                if i + BATCH_SIZE < len(all_symbols):
                    print(f"Waiting {REQUEST_DELAY} seconds before next batch...")
                    time.sleep(REQUEST_DELAY)
            
            events = all_events
            
            if not events:
                print("\nNo events found for the specified criteria")
                return
            
            # Generate HTML calendar
            print("\nGenerating HTML calendar...")
            html_content = generate_html_calendar(events, start_date, end_date)
            
            # Save HTML file
            with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"HTML calendar saved to: {OUTPUT_HTML_FILE}")
            
            # Save CSV data
            save_events_to_csv(events, OUTPUT_DATA_FILE)
            
            # Print summary
            print("\n=== SUMMARY ===")
            print(f"Total events found: {len(events)}")
            
            # Count by event type
            event_types_count = defaultdict(int)
            for event in events:
                event_type = event.get('event_type', 'Unknown')
                event_types_count[event_type] += 1
            
            print("\nEvents by type:")
            for event_type, count in sorted(event_types_count.items()):
                print(f"  {event_type}: {count}")
            
            # Count by company
            company_count = defaultdict(int)
            for event in events:
                company = event.get('company_name', event.get('ticker', 'Unknown'))
                company_count[company] += 1
            
            print("\nTop companies by event count:")
            for company, count in sorted(company_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {company}: {count}")
            
            print(f"\nExecution completed successfully!")
            print(f"Open {OUTPUT_HTML_FILE} in your browser to view the calendar.")
            
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()