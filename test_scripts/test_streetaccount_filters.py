#!/usr/bin/env python3
"""
StreetAccountNews Filters Test Script

This script explores and validates available filters for StreetAccountNews API,
specifically focusing on financial services and banking sector filters.

Author: Generated with Claude Code
Date: 2024-07-15
"""

import os
import sys
import json
import time
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
from fds.sdk.StreetAccountNews.api import filters_api
from fds.sdk.StreetAccountNews.model.filter_categories_response import FilterCategoriesResponse
from fds.sdk.StreetAccountNews.model.filter_sectors_response import FilterSectorsResponse
from fds.sdk.StreetAccountNews.model.filter_regions_response import FilterRegionsResponse
from fds.sdk.StreetAccountNews.model.filter_topic_response import FilterTopicResponse

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streetaccount_filters_test.log')
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

def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    if '..' in path or path.startswith('/'):
        logger.error(f"Invalid file path: {path}")
        return False
    return True

def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    if not path.startswith(('Inputs/', 'Outputs/')):
        logger.error(f"Invalid NAS path: {path}")
        return False
    return True

def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    if '@' in url:
        return url.split('@')[1]
    return url

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
            logger.info(f"✅ Connected to NAS: {sanitize_url_for_logging(nas_server_name)}")
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
        if not validate_nas_path(nas_path):
            return None
            
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

def setup_ssl_certificate(nas_conn: SMBConnection, ssl_cert_path: str) -> Optional[str]:
    """Download and setup SSL certificate."""
    try:
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
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
    
    logger.info(f"✅ Proxy configured: {sanitize_url_for_logging(full_proxy_url)}")
    return full_proxy_url

def get_streetaccount_filters() -> Optional[Dict[str, Any]]:
    """Retrieve and analyze all available StreetAccountNews filters."""
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
        ssl_cert_path = config.get('ssl_cert_nas_path')
        temp_cert_path = setup_ssl_certificate(nas_conn, ssl_cert_path)
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
        
        filters_data = {}
        
        # Get filters using API client
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = filters_api.FiltersApi(api_client)
            
            logger.info("🔍 Retrieving StreetAccountNews filters...")
            
            # Get Categories
            try:
                categories_response = api_instance.get_street_account_filters_categories()
                if categories_response and categories_response.data:
                    filters_data['categories'] = [item.name for item in categories_response.data]
                    logger.info(f"✅ Categories retrieved: {len(filters_data['categories'])}")
                else:
                    filters_data['categories'] = []
                    logger.warning("⚠️  No categories data received")
            except Exception as e:
                logger.error(f"❌ Categories retrieval failed: {e}")
                filters_data['categories'] = []
            
            # Get Sectors  
            try:
                sectors_response = api_instance.get_street_account_filters_sectors()
                if sectors_response and sectors_response.data:
                    filters_data['sectors'] = [item.name for item in sectors_response.data]
                    logger.info(f"✅ Sectors retrieved: {len(filters_data['sectors'])}")
                else:
                    filters_data['sectors'] = []
                    logger.warning("⚠️  No sectors data received")
            except Exception as e:
                logger.error(f"❌ Sectors retrieval failed: {e}")
                filters_data['sectors'] = []
            
            # Get Regions
            try:
                regions_response = api_instance.get_street_account_filters_regions()
                if regions_response and regions_response.data:
                    filters_data['regions'] = [item.name for item in regions_response.data]
                    logger.info(f"✅ Regions retrieved: {len(filters_data['regions'])}")
                else:
                    filters_data['regions'] = []
                    logger.warning("⚠️  No regions data received")
            except Exception as e:
                logger.error(f"❌ Regions retrieval failed: {e}")
                filters_data['regions'] = []
            
            # Get Topics
            try:
                topics_response = api_instance.get_street_account_filters_topics()
                if topics_response and topics_response.data:
                    filters_data['topics'] = [item.name for item in topics_response.data]
                    logger.info(f"✅ Topics retrieved: {len(filters_data['topics'])}")
                else:
                    filters_data['topics'] = []
                    logger.warning("⚠️  No topics data received")
            except Exception as e:
                logger.error(f"❌ Topics retrieval failed: {e}")
                filters_data['topics'] = []
        
        return filters_data
        
    except Exception as e:
        logger.error(f"❌ Filter retrieval failed: {e}")
        return None
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def analyze_financial_filters(filters_data: Dict[str, Any]) -> None:
    """Analyze and display financial services related filters."""
    print("\\n" + "="*60)
    print("📊 STREETACCOUNT NEWS FILTERS ANALYSIS")
    print("="*60)
    
    # Financial keywords to search for
    financial_keywords = [
        'financial', 'banking', 'bank', 'insurance', 'asset', 'investment',
        'credit', 'loan', 'mortgage', 'capital', 'fund', 'wealth', 'trading',
        'securities', 'brokerage', 'fintech', 'payment', 'earnings', 'guidance'
    ]
    
    for filter_type, filter_list in filters_data.items():
        print(f"\\n🔍 {filter_type.upper()} ({len(filter_list)} available):")
        
        # Find financial-related filters
        financial_filters = []
        for filter_item in filter_list:
            filter_lower = filter_item.lower()
            if any(keyword in filter_lower for keyword in financial_keywords):
                financial_filters.append(filter_item)
        
        if financial_filters:
            print(f"   🏦 Financial-related {filter_type}: {len(financial_filters)} found")
            for f in sorted(financial_filters):
                print(f"      • {f}")
        else:
            print(f"   ⚠️  No financial-related {filter_type} identified")
        
        # Show sample of all filters
        print(f"   📋 Sample {filter_type}:")
        for f in sorted(filter_list)[:5]:
            print(f"      • {f}")
        if len(filter_list) > 5:
            print(f"      ... and {len(filter_list) - 5} more")

def export_filters_analysis(filters_data: Dict[str, Any]) -> None:
    """Export filters analysis to JSON file."""
    try:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"streetaccount_filters_analysis_{timestamp}.json"
        
        export_data = {
            'timestamp': timestamp,
            'analysis_date': datetime.now(timezone.utc).isoformat(),
            'filters_summary': {
                filter_type: {
                    'total_count': len(filter_list),
                    'items': sorted(filter_list)
                }
                for filter_type, filter_list in filters_data.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\\n💾 Filters analysis exported to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")

def main():
    """Main execution function."""
    print("🚀 Starting StreetAccountNews Filters Analysis...")
    print("="*60)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Get filters data
    filters_data = get_streetaccount_filters()
    if not filters_data:
        logger.error("❌ Failed to retrieve filters data")
        sys.exit(1)
    
    # Analyze and display results
    analyze_financial_filters(filters_data)
    
    # Export analysis
    export_filters_analysis(filters_data)
    
    print("\\n" + "="*60)
    print("✅ StreetAccountNews Filters Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main()