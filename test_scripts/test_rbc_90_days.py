#!/usr/bin/env python3
"""
Royal Bank of Canada - 90 Days News (Minimal Filtering)

This script retrieves ALL available Royal Bank of Canada news from the last 90 days
with minimal filtering - just the RBC ticker to see everything available.

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

def get_rbc_90_days_news() -> Optional[List[Dict[str, Any]]]:
    """Get Royal Bank of Canada news from last 90 days with minimal filtering."""
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
        
        # Request RBC news with minimal filtering (just ticker + 3 months)
        rbc_request = HeadlinesRequest(
            data=HeadlinesRequestData(
                tickers=[HeadlinesRequestTickersObject(value="RY-CA", type="Equity")],
                predefined_range="threeMonths"  # Closest to 90 days
            ),
            meta=HeadlinesRequestMeta(
                pagination=HeadlinesRequestMetaPagination(
                    limit=500,  # Get up to 500 stories
                    offset=0
                )
            )
        )
        
        news_data = []
        
        # Get news using API client
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            logger.info("📰 Retrieving Royal Bank of Canada news (last 3 months)...")
            
            try:
                response = api_instance.get_street_account_headlines(headlines_request=rbc_request)
                
                if response and response.data:
                    for item in response.data:
                        # Extract all available fields using correct API field names
                        news_item = {
                            'headline': getattr(item, 'headlines', 'No headline'),
                            'story_body': getattr(item, 'story_body', None),
                            'story_time': getattr(item, 'story_time', None),
                            'story_id': getattr(item, 'id', None),
                            'symbols': getattr(item, 'symbols', []),
                            'primary_symbols': getattr(item, 'primary_symbols', []),
                            'subjects': getattr(item, 'subjects', []),
                            'reference_uris': getattr(item, 'reference_uris', None),
                            'url': getattr(item, 'url', None)
                        }
                        news_data.append(news_item)
                    
                    logger.info(f"✅ Retrieved {len(news_data)} news items")
                else:
                    logger.warning("⚠️  No news found for Royal Bank of Canada")
                    
            except Exception as e:
                logger.error(f"❌ News retrieval failed: {e}")
                return None
        
        return news_data
        
    except Exception as e:
        logger.error(f"❌ RBC 90-day news retrieval failed: {e}")
        return None
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def display_rbc_news_summary(news_items: List[Dict[str, Any]]) -> None:
    """Display Royal Bank news summary."""
    print("\\n" + "="*100)
    print("🏦 ROYAL BANK OF CANADA - NEWS SUMMARY (LAST 90 DAYS)")
    print("="*100)
    
    if not news_items:
        print("❌ No news found for Royal Bank of Canada in the last 90 days")
        return
    
    print(f"\\n📊 OVERVIEW:")
    print(f"   Total News Items: {len(news_items)}")
    
    # Count items with story body
    items_with_body = sum(1 for item in news_items if item.get('story_body'))
    print(f"   Items with Full Story: {items_with_body}")
    
    # Count unique subjects
    all_subjects = []
    for item in news_items:
        subjects = item.get('subjects', [])
        if isinstance(subjects, list):
            all_subjects.extend(subjects)
    unique_subjects = list(set(all_subjects))
    print(f"   Unique Subjects/Categories: {len(unique_subjects)}")
    if unique_subjects:
        print(f"   Sample Subjects: {', '.join(unique_subjects[:5])}...")
    
    # Show recent headlines
    print(f"\\n📰 RECENT HEADLINES (First 15):")
    print("-" * 80)
    
    for i, item in enumerate(news_items[:15], 1):
        headline = item.get('headline', 'No headline')
        story_time = item.get('story_time', 'Unknown time')
        symbols = item.get('symbols', [])
        primary_symbols = item.get('primary_symbols', [])
        subjects = item.get('subjects', [])
        
        # Format time
        time_str = str(story_time)[:19] if story_time else 'Unknown time'
        
        # Format subjects
        subjects_str = ', '.join(subjects[:3]) if subjects else 'General'
        if len(subjects) > 3:
            subjects_str += f" (+{len(subjects)-3} more)"
        
        print(f"\\n{i:2d}. {headline}")
        print(f"    ⏰ {time_str}")
        print(f"    🏷️  {subjects_str}")
        if symbols or primary_symbols:
            all_symbols = list(set((symbols or []) + (primary_symbols or [])))
            print(f"    📈 Symbols: {', '.join(all_symbols)}")
    
    # Time distribution analysis
    print(f"\\n📅 TIME DISTRIBUTION:")
    print("-" * 40)
    
    # Group by month
    monthly_counts = {}
    for item in news_items:
        story_time = item.get('story_time')
        if story_time:
            try:
                # Extract month-year from timestamp
                time_str = str(story_time)
                if len(time_str) >= 7:
                    month_year = time_str[:7]  # YYYY-MM
                    monthly_counts[month_year] = monthly_counts.get(month_year, 0) + 1
            except:
                pass
    
    for month, count in sorted(monthly_counts.items(), reverse=True):
        print(f"   {month}: {count} items")
    
    print(f"\\n✅ Royal Bank news retrieval complete!")

def main():
    """Main execution function."""
    print("🚀 Starting Royal Bank of Canada 90-Day News Retrieval...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Get news data
    news_items = get_rbc_90_days_news()
    if news_items is None:
        logger.error("❌ Failed to retrieve news data")
        sys.exit(1)
    
    # Display summary
    display_rbc_news_summary(news_items)
    
    print("\\n" + "="*80)
    print("✅ Royal Bank 90-Day News Retrieval Complete!")
    print("="*80)

if __name__ == "__main__":
    main()