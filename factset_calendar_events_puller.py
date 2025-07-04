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
    # Major Canadian Banks ("Big Six")
    "RY-CA": {"name": "Royal Bank of Canada", "region": "Canada"},
    "TD-CA": {"name": "Toronto-Dominion Bank", "region": "Canada"},
    "BNS-CA": {"name": "Bank of Nova Scotia (Scotiabank)", "region": "Canada"},
    "BMO-CA": {"name": "Bank of Montreal", "region": "Canada"},
    "CM-CA": {"name": "Canadian Imperial Bank of Commerce (CIBC)", "region": "Canada"},
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
    <title>Bank Events Calendar - Interactive View</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Header */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            opacity: 0.9;
        }
        
        /* Controls Section */
        .controls {
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-bottom: 30px;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
        }
        
        .control-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
        }
        
        .control-group select,
        .control-group input {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .control-group select:focus,
        .control-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        /* Month Navigation */
        .month-navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .month-nav-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .month-nav-button:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        }
        
        .month-nav-button:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
            transform: none;
        }
        
        .current-month {
            font-size: 1.5em;
            font-weight: 600;
            color: #333;
        }
        
        /* Calendar */
        .calendar-wrapper {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            padding: 25px;
        }
        
        .calendar-grid {
            display: grid;
            grid-template-columns: repeat(7, 1fr);
            gap: 2px;
        }
        
        .day-header {
            text-align: center;
            font-weight: 600;
            padding: 15px 5px;
            background-color: #f7fafc;
            color: #4a5568;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .day-cell {
            min-height: 120px;
            border: 1px solid #e2e8f0;
            padding: 8px;
            background-color: #fff;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .day-cell:hover {
            background-color: #f7fafc;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .day-number {
            font-weight: 600;
            margin-bottom: 5px;
            color: #4a5568;
        }
        
        .event {
            font-size: 0.8em;
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 6px;
            cursor: pointer;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            transition: all 0.2s ease;
            border: 1px solid transparent;
        }
        
        .event:hover {
            transform: translateX(2px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
            background-color: #f9fafb;
        }
        
        .more-events {
            font-size: 0.75em;
            color: #6b7280;
            font-style: italic;
            margin-top: 4px;
        }
        
        /* Filter Pills */
        .filter-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        
        .filter-pill {
            background: #e5e7eb;
            color: #374151;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            border: 2px solid transparent;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .filter-pill.active {
            background: #667eea;
            color: white;
            border-color: #5a67d8;
        }
        
        .filter-pill:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
        }
        
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 30px;
            border-radius: 16px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                transform: translateY(-50px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }
        
        .close {
            color: #9ca3af;
            float: right;
            font-size: 32px;
            font-weight: bold;
            cursor: pointer;
            transition: color 0.2s ease;
        }
        
        .close:hover {
            color: #374151;
        }
        
        .event-details h3 {
            color: #1f2937;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .event-details p {
            margin: 10px 0;
            color: #4b5563;
        }
        
        .event-details strong {
            color: #374151;
        }
        
        .event-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s ease;
        }
        
        .event-link:hover {
            color: #5a67d8;
            text-decoration: underline;
        }
        
        /* Stats Section */
        .stats {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin-top: 30px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .stat-card {
            text-align: center;
            padding: 20px;
            background: #f7fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        .stat-number {
            font-size: 2em;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            color: #6b7280;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        /* Multi-select dropdown */
        .multi-select {
            position: relative;
        }
        
        .multi-select-toggle {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: white;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .multi-select-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 2px solid #e0e0e0;
            border-top: none;
            border-radius: 0 0 8px 8px;
            max-height: 300px;
            overflow-y: auto;
            display: none;
            z-index: 100;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .multi-select-dropdown.show {
            display: block;
        }
        
        .multi-select-option {
            padding: 10px;
            cursor: pointer;
            transition: background 0.2s ease;
            display: flex;
            align-items: center;
        }
        
        .multi-select-option:hover {
            background: #f7fafc;
        }
        
        .multi-select-option input {
            margin-right: 10px;
        }
        
        .select-all {
            border-bottom: 1px solid #e0e0e0;
            font-weight: 600;
            background: #f7fafc;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Bank Events Calendar</h1>
            <p>Track upcoming events for major US and Canadian banks</p>
            <p>Generated on: {{GENERATED_DATE}}</p>
        </div>
        
        <!-- Controls -->
        <div class="controls">
            <div class="controls-grid">
                <!-- Region Filter -->
                <div class="control-group">
                    <label for="regionFilter">Region</label>
                    <select id="regionFilter" onchange="applyFilters()">
                        <option value="all">All Regions</option>
                        <option value="US">United States</option>
                        <option value="Canada">Canada</option>
                    </select>
                </div>
                
                <!-- Bank Filter -->
                <div class="control-group">
                    <label>Banks</label>
                    <div class="multi-select">
                        <div class="multi-select-toggle" onclick="toggleBankDropdown()">
                            <span id="bankFilterText">All Banks Selected</span>
                            <span>▼</span>
                        </div>
                        <div id="bankDropdown" class="multi-select-dropdown">
                            <div class="multi-select-option select-all" onclick="toggleAllBanks()">
                                <input type="checkbox" id="selectAllBanks" checked>
                                <label for="selectAllBanks">Select All</label>
                            </div>
                            <!-- Bank options will be dynamically added -->
                        </div>
                    </div>
                </div>
                
                <!-- Event Type Filter -->
                <div class="control-group">
                    <label>Event Types</label>
                    <div class="filter-pills" id="eventTypeFilters">
                        <!-- Event type pills will be dynamically added -->
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Month Navigation -->
        <div class="month-navigation">
            <button class="month-nav-button" onclick="previousMonth()" id="prevButton">← Previous</button>
            <div class="current-month" id="currentMonth">January 2024</div>
            <button class="month-nav-button" onclick="nextMonth()" id="nextButton">Next →</button>
        </div>
        
        <!-- Calendar -->
        <div class="calendar-wrapper">
            <div class="calendar-grid" id="calendarGrid">
                <!-- Calendar will be dynamically generated -->
            </div>
        </div>
        
        <!-- Stats -->
        <div class="stats">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number" id="totalEvents">0</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="visibleEvents">0</div>
                    <div class="stat-label">Visible Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="selectedBanks">0</div>
                    <div class="stat-label">Selected Banks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number" id="currentMonthEvents">0</div>
                    <div class="stat-label">Events This Month</div>
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
        let selectedBanks = new Set(Object.keys(bankInfo));
        let selectedEventTypes = new Set();
        let selectedRegion = 'all';
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeBankDropdown();
            initializeEventTypeFilters();
            renderCurrentMonth();
            updateStats();
        });
        
        // Initialize bank dropdown
        function initializeBankDropdown() {
            const dropdown = document.getElementById('bankDropdown');
            const banks = Object.entries(bankInfo).sort((a, b) => a[1].name.localeCompare(b[1].name));
            
            banks.forEach(([ticker, info]) => {
                const option = document.createElement('div');
                option.className = 'multi-select-option';
                option.innerHTML = `
                    <input type="checkbox" id="bank-${ticker}" value="${ticker}" checked onchange="toggleBank('${ticker}')">
                    <label for="bank-${ticker}">${info.name} (${ticker})</label>
                `;
                dropdown.appendChild(option);
            });
        }
        
        // Initialize event type filters
        function initializeEventTypeFilters() {
            const container = document.getElementById('eventTypeFilters');
            const eventTypes = [...new Set(eventsData.map(e => e.event_type))].filter(Boolean).sort();
            
            // Add "All" pill
            const allPill = document.createElement('div');
            allPill.className = 'filter-pill active';
            allPill.textContent = 'All Events';
            allPill.onclick = () => toggleEventType('all');
            container.appendChild(allPill);
            
            // Add individual event type pills
            eventTypes.forEach(type => {
                const pill = document.createElement('div');
                pill.className = 'filter-pill';
                pill.textContent = formatEventType(type);
                pill.onclick = () => toggleEventType(type);
                pill.id = `filter-${type}`;
                container.appendChild(pill);
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
        
        // Toggle bank selection
        function toggleBank(ticker) {
            if (selectedBanks.has(ticker)) {
                selectedBanks.delete(ticker);
            } else {
                selectedBanks.add(ticker);
            }
            updateBankFilterText();
            applyFilters();
        }
        
        // Toggle all banks
        function toggleAllBanks() {
            const selectAll = document.getElementById('selectAllBanks');
            const checkboxes = document.querySelectorAll('#bankDropdown input[type="checkbox"]:not(#selectAllBanks)');
            
            if (selectAll.checked) {
                selectedBanks = new Set(Object.keys(bankInfo));
                checkboxes.forEach(cb => cb.checked = true);
            } else {
                selectedBanks.clear();
                checkboxes.forEach(cb => cb.checked = false);
            }
            
            updateBankFilterText();
            applyFilters();
        }
        
        // Update bank filter text
        function updateBankFilterText() {
            const text = document.getElementById('bankFilterText');
            const count = selectedBanks.size;
            const total = Object.keys(bankInfo).length;
            
            if (count === 0) {
                text.textContent = 'No Banks Selected';
            } else if (count === total) {
                text.textContent = 'All Banks Selected';
            } else {
                text.textContent = `${count} Banks Selected`;
            }
            
            // Update select all checkbox
            document.getElementById('selectAllBanks').checked = count === total;
        }
        
        // Toggle bank dropdown
        function toggleBankDropdown() {
            const dropdown = document.getElementById('bankDropdown');
            dropdown.classList.toggle('show');
        }
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.multi-select')) {
                document.getElementById('bankDropdown').classList.remove('show');
            }
        });
        
        // Toggle event type
        function toggleEventType(type) {
            const pills = document.querySelectorAll('.filter-pill');
            
            if (type === 'all') {
                selectedEventTypes.clear();
                pills.forEach(pill => {
                    pill.classList.toggle('active', pill.textContent === 'All Events');
                });
            } else {
                // Remove "All Events" selection
                pills[0].classList.remove('active');
                
                if (selectedEventTypes.has(type)) {
                    selectedEventTypes.delete(type);
                } else {
                    selectedEventTypes.add(type);
                }
                
                // Update pill states
                pills.forEach(pill => {
                    if (pill.id && pill.id.startsWith('filter-')) {
                        const pillType = pill.id.replace('filter-', '');
                        pill.classList.toggle('active', selectedEventTypes.has(pillType));
                    }
                });
                
                // If no types selected, select all
                if (selectedEventTypes.size === 0) {
                    pills[0].classList.add('active');
                }
            }
            
            applyFilters();
        }
        
        // Apply all filters
        function applyFilters() {
            const regionFilter = document.getElementById('regionFilter').value;
            
            // Filter banks by region first
            let visibleBanks = new Set(selectedBanks);
            if (regionFilter !== 'all') {
                visibleBanks = new Set([...visibleBanks].filter(ticker => 
                    bankInfo[ticker].region === regionFilter
                ));
            }
            
            // Update bank checkboxes based on region
            document.querySelectorAll('#bankDropdown input[type="checkbox"]:not(#selectAllBanks)').forEach(cb => {
                const ticker = cb.value;
                if (regionFilter !== 'all' && bankInfo[ticker].region !== regionFilter) {
                    cb.disabled = true;
                    cb.parentElement.style.opacity = '0.5';
                } else {
                    cb.disabled = false;
                    cb.parentElement.style.opacity = '1';
                }
            });
            
            renderCurrentMonth();
            updateStats();
        }
        
        // Check if event should be visible
        function isEventVisible(event) {
            const ticker = event.ticker;
            const eventType = event.event_type;
            const region = bankInfo[ticker]?.region;
            
            // Check region filter
            if (selectedRegion !== 'all' && region !== selectedRegion) {
                return false;
            }
            
            // Check bank filter
            if (!selectedBanks.has(ticker)) {
                return false;
            }
            
            // Check event type filter
            if (selectedEventTypes.size > 0 && !selectedEventTypes.has(eventType)) {
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
                const maxDisplay = 3;
                
                visibleEvents.slice(0, maxDisplay).forEach(event => {
                    const eventEl = document.createElement('div');
                    const eventType = (event.event_type || '').toLowerCase().replace(/\s/g, '');
                    eventEl.className = `event ${eventType}`;
                    eventEl.textContent = event.ticker || 'Unknown';
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
            
            updateStats();
        }
        
        // Show event details
        function showEventDetails(eventId) {
            const event = eventsData.find(e => e.event_id === eventId);
            if (!event) return;
            
            const modal = document.getElementById('eventModal');
            const details = document.getElementById('eventDetails');
            
            details.innerHTML = `
                <div class="event-details">
                    <h3>${event.company_name || 'Unknown Company'}</h3>
                    <p><strong>Ticker:</strong> ${event.ticker || 'N/A'}</p>
                    <p><strong>Region:</strong> ${bankInfo[event.ticker]?.region || 'Unknown'}</p>
                    <p><strong>Event Type:</strong> ${formatEventType(event.event_type) || 'N/A'}</p>
                    <p><strong>Date & Time:</strong> ${new Date(event.event_date_time).toLocaleString()}</p>
                    ${event.market_time_code ? `<p><strong>Market Time:</strong> ${event.market_time_code}</p>` : ''}
                    ${event.description ? `<p><strong>Description:</strong> ${event.description}</p>` : ''}
                    ${event.fiscal_year || event.fiscal_period ? `<p><strong>Fiscal Period:</strong> ${event.fiscal_year || ''} ${event.fiscal_period || ''}</p>` : ''}
                    ${event.webcast_link ? `<p><strong>Webcast:</strong> <a href="${event.webcast_link}" target="_blank" class="event-link">Join Webcast</a></p>` : ''}
                    ${event.ir_link ? `<p><strong>Investor Relations:</strong> <a href="${event.ir_link}" target="_blank" class="event-link">IR Page</a></p>` : ''}
                    ${event.contact_name || event.contact_email || event.contact_phone ? '<h4>Contact Information:</h4>' : ''}
                    ${event.contact_name ? `<p><strong>Name:</strong> ${event.contact_name}</p>` : ''}
                    ${event.contact_email ? `<p><strong>Email:</strong> ${event.contact_email}</p>` : ''}
                    ${event.contact_phone ? `<p><strong>Phone:</strong> ${event.contact_phone}</p>` : ''}
                    ${event.report_id ? `<p><strong>Report ID:</strong> ${event.report_id}</p>` : ''}
                    ${event.last_modified_date ? `<p><em>Last Modified: ${new Date(event.last_modified_date).toLocaleString()}</em></p>` : ''}
                </div>
            `;
            
            modal.style.display = 'block';
        }
        
        // Update statistics
        function updateStats() {
            const total = eventsData.length;
            const visible = eventsData.filter(e => isEventVisible(e)).length;
            const banksCount = selectedBanks.size;
            
            // Count current month events
            const month = availableMonths[currentMonthIndex];
            const monthEvents = eventsData.filter(event => {
                const eventDate = new Date(event.event_date_time);
                return eventDate.getFullYear() === month.year &&
                       eventDate.getMonth() === month.month - 1 &&
                       isEventVisible(event);
            }).length;
            
            document.getElementById('totalEvents').textContent = total;
            document.getElementById('visibleEvents').textContent = visible;
            document.getElementById('selectedBanks').textContent = banksCount;
            document.getElementById('currentMonthEvents').textContent = monthEvents;
        }
        
        // Modal controls
        document.querySelector('.close').onclick = function() {
            document.getElementById('eventModal').style.display = 'none';
        }
        
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