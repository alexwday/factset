#!/usr/bin/env python3
"""
Royal Bank - All Content Types Discovery

This script searches for Royal Bank content in multiple ways to find
headline news stories, not just financial updates. Tests different
approaches to find business news coverage.

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

def test_content_discovery() -> None:
    """Test different approaches to find headline news about RBC."""
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
        
        print("\\n" + "="*80)
        print("🔍 ROYAL BANK CONTENT TYPE DISCOVERY")
        print("="*80)
        
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            # Test 1: No filtering at all - see what types of content exist
            print("\\n📰 TEST 1: General Financial Sector News (No specific ticker)")
            print("-" * 60)
            
            try:
                general_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        sectors=["Financial"],
                        regions=["North America"],
                        predefined_range="today"
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=20)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=general_request)
                
                if response and response.data:
                    print(f"✅ Found {len(response.data)} general financial headlines today")
                    
                    # Look for different types of content
                    subjects_found = set()
                    headline_types = {
                        'earnings': 0,
                        'dividend': 0,
                        'merger': 0,
                        'acquisition': 0,
                        'regulation': 0,
                        'management': 0,
                        'strategy': 0,
                        'lawsuit': 0,
                        'scandal': 0,
                        'partnership': 0,
                        'technology': 0,
                        'other': 0
                    }
                    
                    for item in response.data:
                        headline = getattr(item, 'headlines', '').lower()
                        subjects = getattr(item, 'subjects', [])
                        
                        if isinstance(subjects, list):
                            subjects_found.update(subjects)
                        
                        # Categorize headlines
                        if any(word in headline for word in ['earnings', 'profit', 'revenue']):
                            headline_types['earnings'] += 1
                        elif any(word in headline for word in ['dividend', 'payout']):
                            headline_types['dividend'] += 1
                        elif any(word in headline for word in ['merger', 'merge']):
                            headline_types['merger'] += 1
                        elif any(word in headline for word in ['acquisition', 'acquire', 'buy']):
                            headline_types['acquisition'] += 1
                        elif any(word in headline for word in ['regulation', 'regulatory', 'compliance']):
                            headline_types['regulation'] += 1
                        elif any(word in headline for word in ['ceo', 'management', 'executive']):
                            headline_types['management'] += 1
                        elif any(word in headline for word in ['strategy', 'strategic', 'plan']):
                            headline_types['strategy'] += 1
                        elif any(word in headline for word in ['lawsuit', 'legal', 'court']):
                            headline_types['lawsuit'] += 1
                        elif any(word in headline for word in ['scandal', 'investigation', 'fraud']):
                            headline_types['scandal'] += 1
                        elif any(word in headline for word in ['partnership', 'partner', 'alliance']):
                            headline_types['partnership'] += 1
                        elif any(word in headline for word in ['technology', 'digital', 'ai', 'tech']):
                            headline_types['technology'] += 1
                        else:
                            headline_types['other'] += 1
                    
                    print("\\n   Content Type Breakdown:")
                    for content_type, count in headline_types.items():
                        if count > 0:
                            print(f"     {content_type.title()}: {count}")
                    
                    print(f"\\n   Unique Subjects Found: {len(subjects_found)}")
                    if subjects_found:
                        print(f"     {', '.join(list(subjects_found)[:8])}...")
                    
                    # Show sample headlines
                    print("\\n   Sample Headlines:")
                    for i, item in enumerate(response.data[:3], 1):
                        headline = getattr(item, 'headlines', 'No headline')
                        print(f"     {i}. {headline[:80]}...")
                        
                else:
                    print("❌ No general financial headlines found today")
                    
            except Exception as e:
                print(f"❌ General financial test failed: {e}")
            
            # Test 2: Try without any filtering to see all content types
            print("\\n📰 TEST 2: No Filtering At All (Recent 50 headlines)")
            print("-" * 60)
            
            try:
                no_filter_request = HeadlinesRequest(
                    data=HeadlinesRequestData(
                        predefined_range="today"
                    ),
                    meta=HeadlinesRequestMeta(
                        pagination=HeadlinesRequestMetaPagination(limit=50)
                    )
                )
                
                response = api_instance.get_street_account_headlines(headlines_request=no_filter_request)
                
                if response and response.data:
                    print(f"✅ Found {len(response.data)} total headlines today")
                    
                    # Look for Canadian bank mentions in any headlines
                    canadian_bank_keywords = [
                        'royal bank', 'rbc', 'td bank', 'toronto dominion',
                        'scotia', 'bank of montreal', 'bmo', 'cibc',
                        'national bank', 'canada', 'canadian'
                    ]
                    
                    bank_mentions = []
                    for item in response.data:
                        headline = getattr(item, 'headlines', '').lower()
                        story_body = getattr(item, 'story_body', '').lower() if getattr(item, 'story_body', None) else ''
                        
                        for keyword in canadian_bank_keywords:
                            if keyword in headline or keyword in story_body:
                                bank_mentions.append({
                                    'headline': getattr(item, 'headlines', 'No headline'),
                                    'matched_keyword': keyword,
                                    'symbols': getattr(item, 'symbols', []),
                                    'primary_symbols': getattr(item, 'primary_symbols', [])
                                })
                                break
                    
                    print(f"\\n   Canadian Bank Mentions Found: {len(bank_mentions)}")
                    for mention in bank_mentions[:5]:
                        print(f"     • {mention['headline'][:70]}...")
                        print(f"       Matched: '{mention['matched_keyword']}' | Symbols: {mention['symbols']}")
                        
                else:
                    print("❌ No headlines found with no filtering")
                    
            except Exception as e:
                print(f"❌ No filtering test failed: {e}")
            
            # Test 3: Search for specific business news topics
            print("\\n📰 TEST 3: Business News Categories")
            print("-" * 60)
            
            # Try different categories that might contain headline news
            business_categories = [
                "M&A",
                "Mergers",
                "Acquisitions", 
                "Management",
                "Strategy",
                "Corporate",
                "Business",
                "News",
                "Executive",
                "Partnership",
                "Technology",
                "Innovation"
            ]
            
            for category in business_categories[:5]:  # Test first 5
                try:
                    category_request = HeadlinesRequest(
                        data=HeadlinesRequestData(
                            tickers=[HeadlinesRequestTickersObject(value="RY-CA", type="Equity")],
                            # Note: We'll test if these category names work
                            predefined_range="oneWeek"
                        ),
                        meta=HeadlinesRequestMeta(
                            pagination=HeadlinesRequestMetaPagination(limit=10)
                        )
                    )
                    
                    response = api_instance.get_street_account_headlines(headlines_request=category_request)
                    
                    if response and response.data:
                        # Look for headlines that might be business news
                        business_news_count = 0
                        for item in response.data:
                            headline = getattr(item, 'headlines', '').lower()
                            if any(word in headline for word in ['launches', 'announces', 'partners', 'expands', 'invests', 'appoints', 'strategy', 'initiative']):
                                business_news_count += 1
                        
                        if business_news_count > 0:
                            print(f"   {category}: {business_news_count} potential business news items")
                        
                except Exception as e:
                    continue
        
        print("\\n" + "="*60)
        print("🎯 ANALYSIS & RECOMMENDATIONS")
        print("="*60)
        print("\\nBased on the content discovery:")
        print("1. StreetAccountNews appears to focus on financial/regulatory updates")
        print("2. Headline business news may be limited or categorized differently")
        print("3. Content is heavily weighted toward earnings, dividends, compliance")
        print("4. For headline news, you might need:")
        print("   • Different FactSet APIs (News API, Company News)")
        print("   • External news sources (Reuters, Bloomberg, etc.)")
        print("   • Web scraping financial news sites")
        print("\\nStreetAccountNews seems optimized for financial professionals")
        print("tracking regulatory and market events, not general business journalism.")
        
    except Exception as e:
        logger.error(f"❌ Content discovery failed: {e}")
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def main():
    """Main execution function."""
    print("🚀 Starting Royal Bank Content Type Discovery...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Run discovery
    test_content_discovery()
    
    print("\\n" + "="*80)
    print("✅ Content Type Discovery Complete!")
    print("="*80)

if __name__ == "__main__":
    main()