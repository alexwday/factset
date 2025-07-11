#!/usr/bin/env python3
"""
Events Calendar Checker
Shows upcoming earnings events for monitored banks from the FactSet Events Calendar API.
Perfect for seeing what events are scheduled for the institutions we're monitoring.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import calendar_events_api
from fds.sdk.EventsandTranscripts.models import CompanyEventRequest, CompanyEventRequestData, CompanyEventRequestDataDateTime, CompanyEventRequestDataUniverse
from dateutil.parser import parse as dateutil_parser
import os
from urllib.parse import quote
from datetime import datetime, timedelta
import tempfile
import io
from smb.SMBConnection import SMBConnection
from typing import Dict, List, Optional, Set, Tuple, Any
import warnings
from dotenv import load_dotenv
import json

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment
API_USERNAME = os.getenv('API_USERNAME')
API_PASSWORD = os.getenv('API_PASSWORD')
PROXY_USER = os.getenv('PROXY_USER')
PROXY_PASSWORD = os.getenv('PROXY_PASSWORD')
PROXY_URL = os.getenv('PROXY_URL')
NAS_USERNAME = os.getenv('NAS_USERNAME')
NAS_PASSWORD = os.getenv('NAS_PASSWORD')
NAS_SERVER_IP = os.getenv('NAS_SERVER_IP')
NAS_SERVER_NAME = os.getenv('NAS_SERVER_NAME')
NAS_SHARE_NAME = os.getenv('NAS_SHARE_NAME')
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH')
NAS_PORT = int(os.getenv('NAS_PORT', 445))
CONFIG_PATH = os.getenv('CONFIG_PATH')
CLIENT_MACHINE_NAME = os.getenv('CLIENT_MACHINE_NAME')
PROXY_DOMAIN = os.getenv('PROXY_DOMAIN', 'MAPLE')

# Validate required environment variables
required_env_vars = [
    'API_USERNAME', 'API_PASSWORD', 'PROXY_USER', 'PROXY_PASSWORD', 'PROXY_URL',
    'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
    'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            print("✅ Connected to NAS successfully")
            return conn
        else:
            print("❌ Failed to connect to NAS")
            return None
            
    except Exception as e:
        print(f"❌ Error connecting to NAS: {e}")
        return None

def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(NAS_SHARE_NAME, nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        print(f"❌ Failed to download file from NAS {nas_file_path}: {e}")
        return None

def load_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load configuration from NAS."""
    try:
        print("📄 Loading configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)
        
        if config_data:
            config = json.loads(config_data.decode('utf-8'))
            print("✅ Successfully loaded configuration from NAS")
            return config
        else:
            print("❌ Config file not found on NAS")
            return None
            
    except Exception as e:
        print(f"❌ Error loading config from NAS: {e}")
        return None

def setup_ssl_certificate(nas_conn: SMBConnection, ssl_cert_path: str) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
    try:
        print("🔒 Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
        if cert_data:
            temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()
            
            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name
            
            print("✅ SSL certificate downloaded from NAS")
            return temp_cert.name
        else:
            print("❌ Failed to download SSL certificate from NAS")
            return None
    except Exception as e:
        print(f"❌ Error downloading SSL certificate from NAS: {e}")
        return None

def check_upcoming_events(api_instance: calendar_events_api.CalendarEventsApi, 
                         monitored_tickers: List[str],
                         days_ahead: int = 30) -> List[Tuple[Dict[str, Any], str]]:
    """Get upcoming earnings events for monitored banks."""
    try:
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=days_ahead)
        
        print(f"🔍 Querying upcoming events from {start_date} to {end_date}")
        
        # Convert dates to datetime objects for API
        start_datetime = dateutil_parser(f"{start_date}T00:00:00Z")
        end_datetime = dateutil_parser(f"{end_date}T23:59:59Z")
        
        # Build the request object for all monitored tickers
        company_event_request = CompanyEventRequest(
            data=CompanyEventRequestData(
                date_time=CompanyEventRequestDataDateTime(
                    start=start_datetime,
                    end=end_datetime,
                ),
                universe=CompanyEventRequestDataUniverse(
                    symbols=monitored_tickers,
                    type="Tickers",
                ),
                event_types=["Earnings"],
            ),
        )
        
        print(f"🏦 Checking events for {len(monitored_tickers)} institutions...")
        
        # Make the API call
        response = api_instance.get_company_event(company_event_request)
        
        upcoming_events = []
        
        if response and hasattr(response, 'data') and response.data:
            events = [event.to_dict() for event in response.data]
            print(f"📊 Found {len(events)} total events")
            
            for event in events:
                # Extract ticker from event data
                ticker = event.get('ticker', 'Unknown')
                if ticker in monitored_tickers:
                    upcoming_events.append((event, ticker))
                    event_datetime = event.get('event_date_time', 'Unknown')
                    event_date_str = event_datetime.strftime('%Y-%m-%d') if hasattr(event_datetime, 'strftime') else str(event_datetime)
                    print(f"  ✅ {ticker}: {event.get('event_type', 'Unknown')} on {event_date_str}")
        else:
            print(f"📭 No events found for the specified date range")
        
        # Sort by event date
        upcoming_events.sort(key=lambda x: x[0].get('event_date_time', datetime.min))
        
        return upcoming_events
        
    except Exception as e:
        print(f"❌ Error querying upcoming events: {e}")
        return []

def format_event_info(event: Dict[str, Any], ticker: str) -> str:
    """Format event information for display."""
    event_datetime = event.get('event_date_time', 'Unknown')
    
    if hasattr(event_datetime, 'strftime'):
        event_date = event_datetime.strftime('%Y-%m-%d')
        event_time = event_datetime.strftime('%H:%M')
    else:
        event_date = 'Unknown'
        event_time = 'Unknown'
    
    event_type = event.get('event_type', 'Unknown')
    event_id = event.get('event_id', 'Unknown')
    webcast_status = event.get('webcast_status', 'Unknown')
    
    return f"  {ticker} | {event_date} {event_time} | {event_type} | Event:{event_id} | Webcast:{webcast_status}"

def main():
    """Main function to check upcoming events."""
    print("\n" + "="*60)
    print("📅 UPCOMING EARNINGS EVENTS CALENDAR CHECKER")
    print("="*60)
    
    # Connect to NAS and load configuration
    nas_conn = get_nas_connection()
    if not nas_conn:
        return
    
    config = load_config(nas_conn)
    if not config:
        nas_conn.close()
        return
    
    # Setup SSL certificate
    temp_cert_path = setup_ssl_certificate(nas_conn, config['ssl_cert_nas_path'])
    if not temp_cert_path:
        nas_conn.close()
        return
    
    # Get monitored tickers
    monitored_tickers = list(config['monitored_institutions'].keys())
    print(f"🏦 Monitoring {len(monitored_tickers)} institutions")
    
    # Configure FactSet API
    user = PROXY_USER
    password = quote(PROXY_PASSWORD)
    
    escaped_domain = quote(PROXY_DOMAIN + '\\' + user)
    proxy_url = f"http://{escaped_domain}:{password}@{PROXY_URL}"
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=API_USERNAME,
        password=API_PASSWORD,
        proxy=proxy_url,
        ssl_ca_cert=temp_cert_path
    )
    configuration.get_basic_auth_token()
    print("✅ FactSet API client configured")
    
    try:
        # Check upcoming events (next 30 days)
        days_ahead = 30
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = calendar_events_api.CalendarEventsApi(api_client)
            
            # Get upcoming events
            upcoming_events = check_upcoming_events(api_instance, monitored_tickers, days_ahead)
            
            print(f"\n📋 UPCOMING EARNINGS EVENTS (Next {days_ahead} days)")
            print("-" * 60)
            
            if upcoming_events:
                # Group by date
                by_date = {}
                for event, ticker in upcoming_events:
                    event_datetime = event.get('event_date_time', 'Unknown')
                    if hasattr(event_datetime, 'strftime'):
                        event_date = event_datetime.strftime('%Y-%m-%d')
                    else:
                        event_date = 'Unknown'
                    
                    if event_date not in by_date:
                        by_date[event_date] = []
                    by_date[event_date].append((event, ticker))
                
                # Sort dates
                sorted_dates = sorted(by_date.keys())
                
                print(f"✅ Found earnings events on {len(sorted_dates)} different dates:")
                print()
                
                for date in sorted_dates:
                    events_on_date = by_date[date]
                    print(f"📅 {date} ({len(events_on_date)} events):")
                    
                    for event, ticker in events_on_date:
                        institution_name = config['monitored_institutions'].get(ticker, {}).get('name', ticker)
                        print(f"  🏦 {institution_name} ({ticker})")
                        print(f"    {format_event_info(event, ticker)}")
                        
                        # Show additional details if available
                        if event.get('event_headline'):
                            print(f"    📝 {event['event_headline']}")
                        if event.get('dial_in_info'):
                            print(f"    📞 Dial-in info available")
                        if event.get('webcast_url'):
                            print(f"    🌐 Webcast URL available")
                        print()
                    
                print("💡 These are the upcoming earnings events for your monitored institutions!")
                print("💡 Run the daily sync script on these dates to capture transcripts.")
                    
            else:
                print("📭 No upcoming earnings events found")
                print("💡 This might indicate no events are scheduled in the next 30 days")
                print("💡 or there might be an issue with the API query")
            
            print("-" * 60)
            print(f"✅ Check complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        
        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
            except:
                pass

if __name__ == "__main__":
    main()