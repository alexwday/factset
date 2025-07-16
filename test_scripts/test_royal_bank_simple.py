#!/usr/bin/env python3
"""
Royal Bank of Canada - Simple News Test

This script uses minimal filtering to retrieve ALL available news for Royal Bank 
of Canada (RY-CA) to see what's actually available in the StreetAccountNews feed.

Author: Generated with Claude Code
Date: 2024-07-16
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

def test_royal_bank_news() -> None:
    """Test different approaches to get Royal Bank news."""
    nas_conn = None
    temp_cert_path = None
    
    try:
        # Setup connections
        nas_conn = get_nas_connection()
        if not nas_conn:
            return
            
        config = load_config(nas_conn)
        if not config:
            return
            
        # Setup SSL
        temp_cert_path = setup_ssl_certificate(nas_conn, config)
        if not temp_cert_path:
            return
            
        # Setup API configuration
        proxy_url = setup_proxy_authentication()
        
        configuration = streetaccount.Configuration(
            username=os.getenv('API_USERNAME'),
            password=os.getenv('API_PASSWORD'),
            proxy=proxy_url,
            ssl_ca_cert=temp_cert_path
        )
        
        print("\n" + "="*80)
        print("🏦 ROYAL BANK OF CANADA (RY-CA) - NEWS DISCOVERY TESTS")
        print("="*80)
        
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            # Test 1: Just ticker, no other filters
            print("\n📰 TEST 1: Minimal filtering - just RY-CA ticker")
            print("-" * 60)
            
            try:
                minimal_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        tickers=[HeadlinesRequestTickersObject(value="RY-CA", type="Equity")]
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=50)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=minimal_request)
                
                if response and response.data:
                    print(f"✅ Found {len(response.data)} headlines with minimal filtering")
                    for i, item in enumerate(response.data[:5], 1):
                        headline = getattr(item, 'headlines', 'No headline')
                        story_time = getattr(item, 'story_time', 'No time')
                        symbols = getattr(item, 'symbols', [])
                        primary_symbols = getattr(item, 'primary_symbols', [])
                        print(f"   {i}. {headline[:80]}...")
                        print(f"      Time: {story_time} | Symbols: {symbols} | Primary: {primary_symbols}")
                else:
                    print("❌ No headlines found with minimal filtering")
                    
            except Exception as e:
                print(f"❌ Minimal filtering failed: {e}")
            
            # Test 2: Add time range (last 3 months to get 90 days)
            print("\n📰 TEST 2: RY-CA + last 3 months (90 days)")
            print("-" * 60)
            
            try:
                time_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        tickers=[HeadlinesRequestTickersObject(value="RY-CA", type="Equity")],
                        predefined_range="threeMonths"
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=50)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=time_request)
                
                if response and response.data:
                    print(f"✅ Found {len(response.data)} headlines with 3-month filtering")
                    for i, item in enumerate(response.data[:10], 1):
                        headline = getattr(item, 'headlines', 'No headline')
                        story_time = getattr(item, 'story_time', 'No time')
                        symbols = getattr(item, 'symbols', [])
                        primary_symbols = getattr(item, 'primary_symbols', [])
                        print(f"   {i:2d}. {headline[:60]}...")
                        print(f"       Time: {story_time}")
                        print(f"       Symbols: {symbols}")
                        print(f"       Primary: {primary_symbols}")
                        print()
                else:
                    print("❌ No headlines found with time filtering")
                    
            except Exception as e:
                print(f"❌ Time filtering failed: {e}")
            
            # Test 3: No ticker at all, just search for "Royal Bank"
            print("\n📰 TEST 3: Text search for 'Royal Bank'")
            print("-" * 60)
            
            try:
                # Note: This might not be available in the API, but let's see
                text_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        # Note: Checking if there's a text search field
                        predefined_range="oneWeek"
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=100)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=text_request)
                
                if response and response.data:
                    # Search through headlines for Royal Bank mentions
                    royal_bank_headlines = []
                    for item in response.data:
                        headline = getattr(item, 'headline', '')
                        if 'royal bank' in headline.lower() or 'ry-ca' in headline.lower():
                            royal_bank_headlines.append(item)
                    
                    print(f"✅ Found {len(royal_bank_headlines)} Royal Bank mentions in {len(response.data)} headlines")
                    for i, item in enumerate(royal_bank_headlines[:5], 1):
                        headline = getattr(item, 'headline', 'No headline')
                        print(f"   {i}. {headline[:80]}...")
                else:
                    print("❌ No headlines found for text search")
                    
            except Exception as e:
                print(f"❌ Text search failed: {e}")
            
            # Test 4: Different ticker variations
            print("\n📰 TEST 4: Testing different ticker formats")
            print("-" * 60)
            
            ticker_variations = ["RY-CA", "RY.TO", "RY", "ROYAL BANK"]
            
            for ticker in ticker_variations:
                try:
                    var_request = HeadlinesRequest(
                        data=HeadlinesRequestData(
                            tickers=[HeadlinesRequestTickersObject(value=ticker, type="Equity")],
                            predefined_range="oneWeek"
                        ),
                        meta=HeadlinesRequestMeta(
                            pagination=HeadlinesRequestMetaPagination(limit=10)
                        )
                    )
                    
                    response = api_instance.get_street_account_headlines(headlines_request=var_request)
                    
                    if response and response.data:
                        print(f"✅ {ticker:10} - Found {len(response.data)} headlines")
                    else:
                        print(f"❌ {ticker:10} - No headlines found")
                        
                except Exception as e:
                    print(f"❌ {ticker:10} - Error: {str(e)[:50]}...")
            
            # Test 5: Financial sector without specific ticker
            print("\n📰 TEST 5: Financial sector headlines (to see data structure)")
            print("-" * 60)
            
            try:
                sector_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        sectors=["Financial"],
                        regions=["North America"],
                        predefined_range="today"
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=10)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=sector_request)
                
                if response and response.data:
                    print(f"✅ Found {len(response.data)} financial sector headlines")
                    print("\nSample headline structure:")
                    sample = response.data[0]
                    print(f"   Headline: {getattr(sample, 'headline', 'N/A')}")
                    print(f"   Tickers: {getattr(sample, 'tickers', 'N/A')}")
                    print(f"   Categories: {getattr(sample, 'categories', 'N/A')}")
                    print(f"   Regions: {getattr(sample, 'regions', 'N/A')}")
                    print(f"   Sectors: {getattr(sample, 'sectors', 'N/A')}")
                    print(f"   Available attributes: {[attr for attr in dir(sample) if not attr.startswith('_')][:10]}...")
                else:
                    print("❌ No financial sector headlines found")
                    
            except Exception as e:
                print(f"❌ Financial sector test failed: {e}")
        
        print("\n" + "="*80)
        print("🎯 SUMMARY & RECOMMENDATIONS")
        print("="*80)
        print("Based on the tests above, we can determine:")
        print("1. Which ticker format works for Royal Bank")
        print("2. Whether the issue is ticker matching or filtering")
        print("3. What data structure the headlines actually have")
        print("4. How to reliably get Royal Bank news")
        
    except Exception as e:
        logger.error(f"❌ Royal Bank test failed: {e}")
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def main():
    """Main execution function."""
    print("🚀 Starting Royal Bank of Canada News Discovery...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Run tests
    test_royal_bank_news()
    
    print("\n" + "="*80)
    print("✅ Royal Bank News Discovery Complete!")
    print("="*80)

if __name__ == "__main__":
    main()