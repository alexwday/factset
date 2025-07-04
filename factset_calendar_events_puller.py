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

# Major Canadian and US Banks Primary IDs
BANK_PRIMARY_IDS = {
    # Major Canadian Banks ("Big Six")
    "RY-CA": "Royal Bank of Canada",
    "TD-CA": "Toronto-Dominion Bank",
    "BNS-CA": "Bank of Nova Scotia (Scotiabank)",
    "BMO-CA": "Bank of Montreal",
    "CM-CA": "Canadian Imperial Bank of Commerce (CIBC)",
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
    "CMA-US": "Comerica Incorporated",
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
    
    # Start HTML
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Events Calendar</title>
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