#!/usr/bin/env python3
"""
Big 6 Canadian Banks - Headlines Summary

This script retrieves headlines for the Big 6 Canadian banks and displays:
- Headlines with bank ticker and date
- Clean, focused output for quick scanning

Author: Generated with Claude Code
Date: 2024-07-15
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from urllib.parse import quote
import logging

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from smb.SMBConnection import SMBConnection

# FactSet SDK imports
import fds.sdk.StreetAccountNews as streetaccount
from fds.sdk.StreetAccountNews.api import headlines_api
from fds.sdk.StreetAccountNews.model.headlines_request import HeadlinesRequest
from fds.sdk.StreetAccountNews.model.headlines_request_data import HeadlinesRequestData
from fds.sdk.StreetAccountNews.model.headlines_request_meta import HeadlinesRequestMeta
from fds.sdk.StreetAccountNews.model.headlines_request_meta_pagination import HeadlinesRequestMetaPagination
from fds.sdk.StreetAccountNews.model.headlines_request_tickers_object import HeadlinesRequestTickersObject

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Big 6 Canadian Banks
BIG_6_BANKS = {
    "RY-CA": "Royal Bank of Canada",
    "TD-CA": "Toronto-Dominion Bank", 
    "BNS-CA": "Bank of Nova Scotia",
    "BMO-CA": "Bank of Montreal",
    "CM-CA": "Canadian Imperial Bank of Commerce",
    "NA-CA": "National Bank of Canada"
}

def load_environment_variables() -> bool:
    """Load and validate environment variables."""
    load_dotenv()
    
    required_vars = [
        'API_USERNAME', 'API_PASSWORD',
        'PROXY_USER', 'PROXY_PASSWORD', 'PROXY_URL',
        'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
        'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    logger.info("✅ All required environment variables loaded")
    return True

def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return NAS connection."""
    try:
        nas_username = os.getenv('NAS_USERNAME')
        nas_password = os.getenv('NAS_PASSWORD')
        nas_server_ip = os.getenv('NAS_SERVER_IP')
        nas_server_name = os.getenv('NAS_SERVER_NAME')
        client_machine_name = os.getenv('CLIENT_MACHINE_NAME')
        nas_port = int(os.getenv('NAS_PORT', 445))
        
        conn = SMBConnection(
            username=nas_username,
            password=nas_password,
            my_name=client_machine_name,
            remote_name=nas_server_name,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if conn.connect(nas_server_ip, nas_port):
            logger.info("✅ Connected to NAS")
            return conn
        else:
            logger.error("❌ Failed to connect to NAS")
            return None
            
    except Exception as e:
        logger.error(f"❌ NAS connection error: {e}")
        return None

def nas_download_file(conn: SMBConnection, nas_path: str) -> Optional[bytes]:
    """Download file from NAS."""
    try:
        share_name = os.getenv('NAS_SHARE_NAME')
        
        with tempfile.NamedTemporaryFile() as temp_file:
            conn.retrieveFile(share_name, nas_path, temp_file)
            temp_file.seek(0)
            return temp_file.read()
            
    except Exception as e:
        logger.error(f"❌ Failed to download {nas_path}: {e}")
        return None

def setup_ssl_certificate(nas_conn: SMBConnection, config: Dict[str, Any]) -> Optional[str]:
    """Download and setup SSL certificate."""
    try:
        ssl_cert_nas_path = config.get('ssl_cert_nas_path')
        if not ssl_cert_nas_path:
            logger.error("❌ SSL certificate path not found in config")
            return None
            
        cert_data = nas_download_file(nas_conn, ssl_cert_nas_path)
        if not cert_data:
            return None
            
        temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
        temp_cert.write(cert_data)
        temp_cert.close()
        
        # Set environment variables for SSL verification
        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
        os.environ["SSL_CERT_FILE"] = temp_cert.name
        
        logger.info("✅ SSL certificate configured")
        return temp_cert.name
        
    except Exception as e:
        logger.error(f"❌ SSL certificate setup failed: {e}")
        return None

def load_config(nas_conn: SMBConnection) -> Optional[Dict[str, Any]]:
    """Load configuration from NAS."""
    try:
        config_path = os.getenv('CONFIG_PATH')
        config_data = nas_download_file(nas_conn, config_path)
        
        if not config_data:
            return None
            
        config = json.loads(config_data.decode('utf-8'))
        logger.info("✅ Configuration loaded from NAS")
        return config
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return None

def setup_proxy_authentication() -> str:
    """Setup corporate proxy with NTLM authentication."""
    user = os.getenv('PROXY_USER')
    password = quote(os.getenv('PROXY_PASSWORD'))
    proxy_url = os.getenv('PROXY_URL')
    proxy_domain = os.getenv('PROXY_DOMAIN', 'MAPLE')
    
    escaped_domain = quote(proxy_domain + '\\\\' + user)
    full_proxy_url = f"http://{escaped_domain}:{password}@{proxy_url}"
    
    logger.info("✅ Proxy configured")
    return full_proxy_url

def get_big6_headlines() -> Optional[List[Dict[str, Any]]]:
    """Retrieve headlines for Big 6 Canadian banks."""
    nas_conn = None
    temp_cert_path = None
    
    try:
        # Setup connections
        nas_conn = get_nas_connection()
        if not nas_conn:
            return None
            
        config = load_config(nas_conn)
        if not config:
            return None
            
        # Setup SSL
        temp_cert_path = setup_ssl_certificate(nas_conn, config)
        if not temp_cert_path:
            return None
            
        # Setup API configuration
        proxy_url = setup_proxy_authentication()
        
        configuration = streetaccount.Configuration(
            username=os.getenv('API_USERNAME'),
            password=os.getenv('API_PASSWORD'),
            proxy=proxy_url,
            ssl_ca_cert=temp_cert_path
        )
        
        # Create ticker objects for Big 6 banks
        bank_tickers = [
            HeadlinesRequestTickersObject(value=ticker, type="Equity")
            for ticker in BIG_6_BANKS.keys()
        ]
        
        # Prepare request for Big 6 bank headlines
        headlines_request = HeadlinesRequest(
            data=HeadlinesRequestData(
                tickers=bank_tickers,
                categories=["Earnings", "Corporate Actions", "Company News", "Guidance"],
                sectors=["Financial"],
                regions=["North America"],
                is_primary=True,
                predefined_range="today"
            ),
            meta=HeadlinesRequestMeta(
                pagination=HeadlinesRequestMetaPagination(
                    limit=200,
                    offset=0
                ),
                attributes=["headlines"]
            )
        )
        
        headlines_data = []
        
        # Get headlines using API client
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            logger.info("📰 Retrieving Big 6 Canadian bank headlines...")
            
            try:
                response = api_instance.get_street_account_headlines(headlines_request)
                
                if response and response.data:
                    for headline_item in response.data:
                        # Extract relevant fields
                        headline_data = {
                            'headline': getattr(headline_item, 'headline', 'No headline'),
                            'publish_time': getattr(headline_item, 'publish_time', None),
                            'tickers': getattr(headline_item, 'tickers', []),
                            'story_id': getattr(headline_item, 'story_id', None),
                            'categories': getattr(headline_item, 'categories', []),
                            'source': getattr(headline_item, 'source', 'StreetAccount')
                        }
                        headlines_data.append(headline_data)
                    
                    logger.info(f"✅ Retrieved {len(headlines_data)} headlines")
                else:
                    logger.warning("⚠️  No headlines found for Big 6 banks today")
                    
            except Exception as e:
                logger.error(f"❌ Headlines retrieval failed: {e}")
                return None
        
        return headlines_data
        
    except Exception as e:
        logger.error(f"❌ Big 6 headlines retrieval failed: {e}")
        return None
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def display_headlines_summary(headlines: List[Dict[str, Any]]) -> None:
    """Display clean headlines summary for Big 6 banks."""
    print("\\n" + "="*100)
    print("🏦 BIG 6 CANADIAN BANKS - HEADLINES SUMMARY")
    print("="*100)
    
    if not headlines:
        print("❌ No headlines found for Big 6 banks today")
        return
    
    # Group headlines by bank
    bank_headlines = {}
    for headline in headlines:
        tickers = headline.get('tickers', [])
        for ticker in tickers:
            ticker_symbol = ticker if isinstance(ticker, str) else ticker.get('symbol', 'Unknown')
            if ticker_symbol in BIG_6_BANKS:
                if ticker_symbol not in bank_headlines:
                    bank_headlines[ticker_symbol] = []
                bank_headlines[ticker_symbol].append(headline)
    
    # Display by bank
    for ticker in BIG_6_BANKS.keys():
        bank_name = BIG_6_BANKS[ticker]
        bank_news = bank_headlines.get(ticker, [])
        
        print(f"\\n🏦 {bank_name} ({ticker}) - {len(bank_news)} headlines")
        print("-" * 60)
        
        if not bank_news:
            print("   📰 No news found for today")
            continue
            
        # Sort by publish time (newest first)
        bank_news.sort(key=lambda x: x.get('publish_time', ''), reverse=True)
        
        for i, headline in enumerate(bank_news, 1):
            headline_text = headline.get('headline', 'No headline')
            publish_time = headline.get('publish_time', 'Unknown time')
            categories = headline.get('categories', [])
            
            # Format publish time
            if publish_time and publish_time != 'Unknown time':
                try:
                    dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                    time_str = dt.strftime('%Y-%m-%d %H:%M AST')
                except:
                    time_str = publish_time
            else:
                time_str = 'Unknown time'
            
            # Format categories
            categories_str = ', '.join(categories) if categories else 'General'
            
            print(f"   {i:2d}. {headline_text}")
            print(f"       ⏰ {time_str} | 📂 {categories_str}")
            print()
    
    # Summary statistics
    total_headlines = len(headlines)
    banks_with_news = len(bank_headlines)
    
    print("\\n" + "="*60)
    print("📊 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Headlines: {total_headlines}")
    print(f"Banks with News: {banks_with_news}/6")
    print(f"Average Headlines per Bank: {total_headlines/6:.1f}")
    
    if bank_headlines:
        most_active = max(bank_headlines.items(), key=lambda x: len(x[1]))
        print(f"Most Active: {BIG_6_BANKS[most_active[0]]} ({len(most_active[1])} headlines)")

def main():
    """Main execution function."""
    print("🚀 Starting Big 6 Canadian Banks Headlines Summary...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Get headlines data
    headlines = get_big6_headlines()
    if headlines is None:
        logger.error("❌ Failed to retrieve headlines data")
        sys.exit(1)
    
    # Display summary
    display_headlines_summary(headlines)
    
    print("\\n" + "="*80)
    print("✅ Big 6 Canadian Banks Headlines Summary Complete!")
    print("="*80)

if __name__ == "__main__":
    main()