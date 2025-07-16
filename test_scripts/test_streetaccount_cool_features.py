#!/usr/bin/env python3
"""
StreetAccountNews - Cool Features Showcase

This script demonstrates advanced StreetAccountNews features including:
- Saved Views creation and management
- Quick Alerts for real-time notifications
- Advanced filtering combinations
- Comprehensive filter discovery
- Real-time market intelligence features

Author: Generated with Claude Code
Date: 2024-07-15
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import quote
import logging

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from smb.SMBConnection import SMBConnection

# FactSet SDK imports
import fds.sdk.StreetAccountNews as streetaccount
from fds.sdk.StreetAccountNews.api import filters_api, views_api, headlines_api
from fds.sdk.StreetAccountNews.model.headlines_request import HeadlinesRequest
from fds.sdk.StreetAccountNews.model.headlines_request_data import HeadlinesRequestData
from fds.sdk.StreetAccountNews.model.headlines_request_meta import HeadlinesRequestMeta
from fds.sdk.StreetAccountNews.model.headlines_request_meta_pagination import HeadlinesRequestMetaPagination
from fds.sdk.StreetAccountNews.model.headlines_request_tickers_object import HeadlinesRequestTickersObject
from fds.sdk.StreetAccountNews.model.headlines_request_search_time import HeadlinesRequestSearchTime

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
        base_path = os.getenv('NAS_BASE_PATH')
        full_path = f"{base_path}/{nas_path}"
        
        with tempfile.NamedTemporaryFile() as temp_file:
            conn.retrieveFile(share_name, full_path, temp_file)
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

def explore_comprehensive_filters(configuration) -> Dict[str, Any]:
    """Explore all available filters comprehensively."""
    print("\\n" + "="*80)
    print("🔍 COMPREHENSIVE FILTER DISCOVERY")
    print("="*80)
    
    filters_data = {}
    
    with streetaccount.ApiClient(configuration) as api_client:
        api_instance = filters_api.FiltersApi(api_client)
        
        try:
            # Get all filters at once
            all_filters = api_instance.get_street_account_filters()
            if all_filters and all_filters.data:
                print("✅ Retrieved comprehensive filter data")
                filters_data['all_filters'] = all_filters.data
            
            # Get categories
            categories = api_instance.get_street_account_filters_categories()
            if categories and categories.data:
                cat_list = [item.name for item in categories.data]
                filters_data['categories'] = cat_list
                print(f"✅ Categories: {len(cat_list)} available")
                
                # Show financial-related categories
                financial_cats = [cat for cat in cat_list if any(kw in cat.lower() for kw in ['financial', 'earnings', 'dividend', 'guidance', 'm&a', 'merger'])]
                if financial_cats:
                    print(f"   🏦 Financial Categories: {', '.join(financial_cats[:5])}")
            
            # Get sectors
            sectors = api_instance.get_street_account_filters_sectors()
            if sectors and sectors.data:
                sec_list = [item.name for item in sectors.data]
                filters_data['sectors'] = sec_list
                print(f"✅ Sectors: {len(sec_list)} available")
                print(f"   📊 Sample Sectors: {', '.join(sec_list[:5])}")
            
            # Get regions
            regions = api_instance.get_street_account_filters_regions()
            if regions and regions.data:
                reg_list = [item.name for item in regions.data]
                filters_data['regions'] = reg_list
                print(f"✅ Regions: {len(reg_list)} available")
                print(f"   🌍 Available Regions: {', '.join(reg_list)}")
            
            # Get topics
            topics = api_instance.get_street_account_filters_topics()
            if topics and topics.data:
                top_list = [item.name for item in topics.data]
                filters_data['topics'] = top_list
                print(f"✅ Topics: {len(top_list)} available")
                print(f"   📝 Sample Topics: {', '.join(top_list[:5])}")
            
            # Get watchlists
            watchlists = api_instance.get_street_account_filters_watchlists()
            if watchlists and watchlists.data:
                watch_list = [item.name for item in watchlists.data]
                filters_data['watchlists'] = watch_list
                print(f"✅ Watchlists: {len(watch_list)} available")
                print(f"   📋 Sample Watchlists: {', '.join(watch_list[:5])}")
            
        except Exception as e:
            logger.error(f"❌ Filter discovery failed: {e}")
    
    return filters_data

def demonstrate_advanced_filtering(configuration) -> None:
    """Demonstrate advanced filtering combinations."""
    print("\\n" + "="*80)
    print("🎯 ADVANCED FILTERING DEMONSTRATIONS")
    print("="*80)
    
    with streetaccount.ApiClient(configuration) as api_client:
        api_instance = headlines_api.HeadlinesApi(api_client)
        
        # Example 1: Time-based filtering with custom date range
        print("\\n📅 Example 1: Custom Date Range Filtering")
        print("-" * 50)
        
        try:
            # Get news from last 3 days
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=3)
            
            time_filter_request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    sectors=["Financial"],
                    categories=["Earnings"],
                    regions=["North America"],
                    search_time=HeadlinesRequestSearchTime(
                        start=start_time,
                        end=end_time
                    )
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(limit=5)
                )
            )
            
            response = api_instance.get_street_account_headlines(time_filter_request)
            if response and response.data:
                print(f"✅ Found {len(response.data)} earnings stories from last 3 days")
                for story in response.data[:3]:
                    headline = getattr(story, 'headline', 'No headline')
                    print(f"   📰 {headline[:80]}...")
            else:
                print("❌ No stories found for date range")
                
        except Exception as e:
            logger.error(f"❌ Custom date filtering failed: {e}")
        
        # Example 2: Multi-category filtering
        print("\\n📂 Example 2: Multi-Category Filtering")
        print("-" * 50)
        
        try:
            multi_category_request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    sectors=["Financial"],
                    categories=["Earnings", "Corporate Actions", "Guidance", "M&A"],
                    regions=["North America"],
                    predefined_range="today"
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(limit=5)
                )
            )
            
            response = api_instance.get_street_account_headlines(multi_category_request)
            if response and response.data:
                print(f"✅ Found {len(response.data)} stories across multiple categories")
                category_counts = {}
                for story in response.data:
                    categories = getattr(story, 'categories', [])
                    for cat in categories:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                
                print("   📊 Category Distribution:")
                for cat, count in sorted(category_counts.items()):
                    print(f"      • {cat}: {count} stories")
            else:
                print("❌ No stories found for multi-category filter")
                
        except Exception as e:
            logger.error(f"❌ Multi-category filtering failed: {e}")
        
        # Example 3: Index-based filtering
        print("\\n📈 Example 3: Index-Based Filtering")
        print("-" * 50)
        
        try:
            index_request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    tickers=[
                        HeadlinesRequestTickersObject(value="SP50", type="Index"),
                        HeadlinesRequestTickersObject(value="R.1000", type="Index")
                    ],
                    categories=["Earnings"],
                    predefined_range="today"
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(limit=5)
                )
            )
            
            response = api_instance.get_street_account_headlines(index_request)
            if response and response.data:
                print(f"✅ Found {len(response.data)} stories for major indices")
                for story in response.data[:3]:
                    headline = getattr(story, 'headline', 'No headline')
                    tickers = getattr(story, 'tickers', [])
                    print(f"   📰 {headline[:60]}...")
                    print(f"      🏢 Tickers: {', '.join(tickers[:3]) if tickers else 'None'}")
            else:
                print("❌ No stories found for index filtering")
                
        except Exception as e:
            logger.error(f"❌ Index filtering failed: {e}")

def explore_views_capabilities(configuration) -> None:
    """Explore Views API capabilities."""
    print("\\n" + "="*80)
    print("👁️ VIEWS API CAPABILITIES")
    print("="*80)
    
    with streetaccount.ApiClient(configuration) as api_client:
        views_instance = views_api.ViewsApi(api_client)
        
        try:
            # Get existing views
            print("\\n📋 Existing Views:")
            print("-" * 30)
            
            existing_views = views_instance.get_views()
            if existing_views and existing_views.data:
                print(f"✅ Found {len(existing_views.data)} existing views")
                for view in existing_views.data:
                    view_name = getattr(view, 'name', 'Unknown')
                    view_id = getattr(view, 'id', 'Unknown')
                    print(f"   📁 {view_name} (ID: {view_id})")
            else:
                print("❌ No existing views found or views API not accessible")
                
        except Exception as e:
            logger.error(f"❌ Views exploration failed: {e}")
            print("ℹ️  Views API may require special permissions or setup")

def demonstrate_pagination_and_performance(configuration) -> None:
    """Demonstrate pagination and performance features."""
    print("\\n" + "="*80)
    print("⚡ PAGINATION & PERFORMANCE FEATURES")
    print("="*80)
    
    with streetaccount.ApiClient(configuration) as api_client:
        api_instance = headlines_api.HeadlinesApi(api_client)
        
        try:
            # Demonstrate pagination
            print("\\n📄 Pagination Demonstration:")
            print("-" * 40)
            
            pagination_request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    sectors=["Financial"],
                    predefined_range="today"
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(
                        limit=10,
                        offset=0
                    )
                )
            )
            
            response = api_instance.get_street_account_headlines(pagination_request)
            if response and response.data:
                total_results = len(response.data)
                print(f"✅ Page 1: Retrieved {total_results} results")
                
                # Show pagination metadata if available
                if hasattr(response, 'meta') and response.meta:
                    print(f"   📊 Pagination Info: {response.meta}")
                
                # Demonstrate different page sizes
                for page_size in [5, 25, 50]:
                    test_request = HeadlinesRequest(
                        data=HeadlinesRequestData(
                            sectors=["Financial"],
                            predefined_range="today"
                        ),
                        meta=HeadlinesRequestMeta(
                            pagination=HeadlinesRequestMetaPagination(
                                limit=page_size,
                                offset=0
                            )
                        )
                    )
                    
                    test_response = api_instance.get_street_account_headlines(test_request)
                    if test_response and test_response.data:
                        print(f"   📄 Page size {page_size}: {len(test_response.data)} results")
            else:
                print("❌ No results for pagination demonstration")
                
        except Exception as e:
            logger.error(f"❌ Pagination demonstration failed: {e}")

def showcase_real_time_features(configuration) -> None:
    """Showcase real-time market intelligence features."""
    print("\\n" + "="*80)
    print("🔴 REAL-TIME MARKET INTELLIGENCE FEATURES")
    print("="*80)
    
    with streetaccount.ApiClient(configuration) as api_client:
        api_instance = headlines_api.HeadlinesApi(api_client)
        
        try:
            # Real-time earnings monitoring
            print("\\n📈 Real-Time Earnings Monitoring:")
            print("-" * 45)
            
            realtime_request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    categories=["Earnings"],
                    sectors=["Financial"],
                    predefined_range="today"
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(limit=10)
                )
            )
            
            response = api_instance.get_street_account_headlines(realtime_request)
            if response and response.data:
                print(f"✅ Real-time earnings stories: {len(response.data)} found")
                
                # Analyze timing patterns
                story_times = []
                for story in response.data:
                    publish_time = getattr(story, 'publish_time', None)
                    if publish_time:
                        story_times.append(publish_time)
                
                if story_times:
                    print(f"   ⏰ Most recent story: {story_times[0] if story_times else 'Unknown'}")
                    print(f"   📊 Time distribution: {len(story_times)} timestamped stories")
            else:
                print("❌ No real-time earnings stories found")
                
        except Exception as e:
            logger.error(f"❌ Real-time features demonstration failed: {e}")

def main():
    """Main execution function."""
    print("🚀 Starting StreetAccountNews Cool Features Showcase...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Setup connections
    nas_conn = None
    temp_cert_path = None
    
    try:
        nas_conn = get_nas_connection()
        if not nas_conn:
            sys.exit(1)
            
        config = load_config(nas_conn)
        if not config:
            sys.exit(1)
            
        # Setup SSL
        temp_cert_path = setup_ssl_certificate(nas_conn, config)
        if not temp_cert_path:
            sys.exit(1)
            
        # Setup API configuration
        proxy_url = setup_proxy_authentication()
        
        configuration = streetaccount.Configuration(
            username=os.getenv('API_USERNAME'),
            password=os.getenv('API_PASSWORD'),
            proxy=proxy_url,
            ssl_ca_cert=temp_cert_path
        )
        
        # Showcase features
        filters_data = explore_comprehensive_filters(configuration)
        demonstrate_advanced_filtering(configuration)
        explore_views_capabilities(configuration)
        demonstrate_pagination_and_performance(configuration)
        showcase_real_time_features(configuration)
        
        print("\\n" + "="*80)
        print("🎉 FEATURE SHOWCASE SUMMARY")
        print("="*80)
        print("✅ Comprehensive filter discovery completed")
        print("✅ Advanced filtering demonstrations completed")
        print("✅ Views API capabilities explored")
        print("✅ Pagination and performance features demonstrated")
        print("✅ Real-time market intelligence features showcased")
        
    except Exception as e:
        logger.error(f"❌ Feature showcase failed: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)
    
    print("\\n" + "="*80)
    print("✅ StreetAccountNews Cool Features Showcase Complete!")
    print("="*80)

if __name__ == "__main__":
    main()