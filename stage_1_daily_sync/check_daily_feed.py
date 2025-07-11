#!/usr/bin/env python3
"""
Daily Feed Checker
Quick check of today's earnings transcripts without downloading anything.
Perfect for testing and seeing what's available.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
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

def check_daily_feed(api_instance: transcripts_api.TranscriptsApi, 
                    check_date: datetime.date, 
                    monitored_tickers: List[str]) -> List[Tuple[Dict[str, Any], str]]:
    """Get all transcripts for check date and filter to monitored institutions."""
    try:
        print(f"🔍 Querying transcripts for date: {check_date}")
        
        # Use date-based API endpoint
        response = api_instance.get_transcripts_dates(
            start_date=check_date,
            end_date=check_date,
            sort=["-storyDateTime"],
            pagination_limit=1000
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            print(f"📭 No transcripts found for date: {check_date}")
            return []
        
        all_transcripts = [transcript.to_dict() for transcript in response.data]
        print(f"📊 Found {len(all_transcripts)} total transcripts for date: {check_date}")
        
        # Filter to monitored institutions and earnings transcripts
        filtered_transcripts = []
        earnings_transcripts = []
        
        for transcript in all_transcripts:
            primary_ids = transcript.get('primary_ids', [])
            
            # Validate that primary_ids is actually a list
            if not isinstance(primary_ids, list):
                continue
            
            # Check if exactly one primary ID and it's one of our monitored tickers
            if len(primary_ids) == 1 and primary_ids[0] in monitored_tickers:
                filtered_transcripts.append((transcript, primary_ids[0]))
                
                # Also check if it's an earnings transcript
                if transcript.get('event_type', '') and 'Earnings' in str(transcript.get('event_type', '')):
                    earnings_transcripts.append((transcript, primary_ids[0]))
        
        print(f"🏦 Found {len(filtered_transcripts)} transcripts for monitored institutions")
        print(f"💰 Found {len(earnings_transcripts)} earnings transcripts for monitored institutions")
        
        return earnings_transcripts
        
    except Exception as e:
        print(f"❌ Error querying transcripts for date {check_date}: {e}")
        return []

def format_transcript_info(transcript: Dict[str, Any], ticker: str) -> str:
    """Format transcript information for display."""
    event_date = transcript.get('event_date', 'Unknown')
    event_type = transcript.get('event_type', 'Unknown')
    transcript_type = transcript.get('transcript_type', 'Unknown')
    event_id = transcript.get('event_id', 'Unknown')
    version_id = transcript.get('version_id', 'Unknown')
    
    return f"  {ticker} | {event_date} | {event_type} | {transcript_type} | Event:{event_id} | Version:{version_id}"

def main():
    """Main function to check daily feed."""
    print("\n" + "="*60)
    print("📡 DAILY EARNINGS TRANSCRIPT FEED CHECKER")
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
    print(f"🏦 Monitoring {len(monitored_tickers)} institutions: {', '.join(monitored_tickers[:5])}{'...' if len(monitored_tickers) > 5 else ''}")
    
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
        # Check today's feed
        check_date = datetime.now().date()
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Get today's earnings transcripts
            earnings_transcripts = check_daily_feed(api_instance, check_date, monitored_tickers)
            
            print(f"\n📋 RESULTS FOR {check_date}")
            print("-" * 60)
            
            if earnings_transcripts:
                # Group by ticker
                by_ticker = {}
                for transcript, ticker in earnings_transcripts:
                    if ticker not in by_ticker:
                        by_ticker[ticker] = []
                    by_ticker[ticker].append(transcript)
                
                print(f"✅ Found earnings transcripts for {len(by_ticker)} monitored banks:")
                print()
                
                for ticker, transcripts in by_ticker.items():
                    institution_name = config['monitored_institutions'][ticker]['name']
                    print(f"🏦 {institution_name} ({ticker}) - {len(transcripts)} transcript(s):")
                    
                    for transcript in transcripts:
                        print(format_transcript_info(transcript, ticker))
                    print()
            else:
                print("📭 No earnings transcripts found for monitored banks today")
                print("💡 This is normal on non-earnings days")
            
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