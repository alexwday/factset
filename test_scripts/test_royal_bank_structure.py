#!/usr/bin/env python3
"""
Royal Bank Data Structure Inspector

This script gets Royal Bank headlines and inspects the actual data structure
to understand how to properly access headline, time, ticker fields.

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

def deep_inspect_object(obj, name="object", max_depth=3, current_depth=0):
    """Deep inspection of object structure."""
    if current_depth >= max_depth:
        return f"[Max depth {max_depth} reached]"
    
    result = []
    result.append(f"\n{'  ' * current_depth}🔍 {name} ({type(obj).__name__}):")
    
    # Check if it's a basic type
    if isinstance(obj, (str, int, float, bool, type(None))):
        result.append(f"{'  ' * current_depth}   Value: {repr(obj)}")
        return '\n'.join(result)
    
    # Check if it's a list
    if isinstance(obj, list):
        result.append(f"{'  ' * current_depth}   Length: {len(obj)}")
        if obj:
            result.append(deep_inspect_object(obj[0], f"{name}[0]", max_depth, current_depth + 1))
        return '\n'.join(result)
    
    # Check if it's a dict
    if isinstance(obj, dict):
        result.append(f"{'  ' * current_depth}   Keys: {list(obj.keys())}")
        for key, value in list(obj.items())[:3]:  # Show first 3 items
            result.append(deep_inspect_object(value, f"{name}['{key}']", max_depth, current_depth + 1))
        return '\n'.join(result)
    
    # For objects, check attributes
    attrs = [attr for attr in dir(obj) if not attr.startswith('_')]
    result.append(f"{'  ' * current_depth}   Attributes: {attrs[:10]}{'...' if len(attrs) > 10 else ''}")
    
    # Try to access some common attributes
    common_attrs = ['headline', 'title', 'text', 'content', 'publish_time', 'date', 'timestamp', 'tickers', 'symbols']
    for attr in common_attrs:
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                result.append(f"{'  ' * current_depth}   .{attr}: {repr(value)[:100]}{'...' if len(repr(value)) > 100 else ''}")
            except Exception as e:
                result.append(f"{'  ' * current_depth}   .{attr}: Error accessing - {e}")
    
    # Try to convert to dict if possible
    try:
        if hasattr(obj, 'to_dict'):
            dict_repr = obj.to_dict()
            result.append(f"{'  ' * current_depth}   .to_dict(): {str(dict_repr)[:200]}{'...' if len(str(dict_repr)) > 200 else ''}")
    except Exception as e:
        result.append(f"{'  ' * current_depth}   .to_dict(): Error - {e}")
    
    return '\n'.join(result)

def inspect_royal_bank_data() -> None:
    """Inspect Royal Bank data structure in detail."""
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
        print("🔬 ROYAL BANK DATA STRUCTURE DEEP INSPECTION")
        print("="*80)
        
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            # Get some Royal Bank headlines
            request = HeadlinesRequest(
                data=HeadlinesRequestData(
                    tickers=[HeadlinesRequestTickersObject(value="RY-CA", type="Equity")],
                    predefined_range="oneMonth"
                ),
                meta=HeadlinesRequestMeta(
                    pagination=HeadlinesRequestMetaPagination(limit=5)
                )
            )
            
            response = api_instance.get_street_account_headlines(headlines_request=request)
            
            if response and response.data:
                print(f"✅ Got {len(response.data)} headlines for inspection\n")
                
                # Inspect the response object itself
                print("📋 RESPONSE OBJECT STRUCTURE:")
                print(deep_inspect_object(response, "response"))
                
                # Inspect the first headline in detail
                if response.data:
                    first_headline = response.data[0]
                    print("\n" + "="*60)
                    print("📰 FIRST HEADLINE DETAILED INSPECTION:")
                    print("="*60)
                    print(deep_inspect_object(first_headline, "headline", max_depth=4))
                    
                    # Try different ways to access data
                    print("\n" + "="*60)
                    print("🧪 ALTERNATIVE ACCESS METHODS:")
                    print("="*60)
                    
                    # Method 1: Direct attribute access
                    print("\n1. Direct attribute access:")
                    for attr in ['headline', 'title', 'text', 'content', 'publish_time', 'date', 'timestamp']:
                        try:
                            value = getattr(first_headline, attr, None)
                            print(f"   .{attr}: {repr(value)}")
                        except Exception as e:
                            print(f"   .{attr}: Error - {e}")
                    
                    # Method 2: Dictionary-style access
                    print("\n2. Dictionary-style access:")
                    try:
                        if hasattr(first_headline, '__getitem__'):
                            for key in ['headline', 'title', 'text', 'publishTime', 'publish_time']:
                                try:
                                    value = first_headline[key]
                                    print(f"   ['{key}']: {repr(value)}")
                                except Exception as e:
                                    print(f"   ['{key}']: Error - {e}")
                    except Exception as e:
                        print(f"   Dictionary access not supported: {e}")
                    
                    # Method 3: to_dict() method
                    print("\n3. to_dict() method:")
                    try:
                        if hasattr(first_headline, 'to_dict'):
                            data_dict = first_headline.to_dict()
                            print(f"   Keys in dict: {list(data_dict.keys())}")
                            for key, value in list(data_dict.items())[:5]:
                                print(f"   {key}: {repr(value)[:100]}{'...' if len(repr(value)) > 100 else ''}")
                    except Exception as e:
                        print(f"   to_dict() error: {e}")
                    
                    # Method 4: Check for nested objects
                    print("\n4. Nested object inspection:")
                    for attr in dir(first_headline):
                        if not attr.startswith('_') and not callable(getattr(first_headline, attr, None)):
                            try:
                                value = getattr(first_headline, attr)
                                if value is not None and not isinstance(value, (str, int, float, bool)):
                                    print(f"   .{attr} (complex): {type(value).__name__}")
                                    if hasattr(value, 'to_dict'):
                                        nested_dict = value.to_dict()
                                        print(f"     Keys: {list(nested_dict.keys())}")
                            except Exception as e:
                                print(f"   .{attr}: Error accessing - {e}")
                
            else:
                print("❌ No headlines returned for inspection")
        
    except Exception as e:
        logger.error(f"❌ Data structure inspection failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def main():
    """Main execution function."""
    print("🚀 Starting Royal Bank Data Structure Inspection...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Run inspection
    inspect_royal_bank_data()
    
    print("\n" + "="*80)
    print("✅ Data Structure Inspection Complete!")
    print("="*80)

if __name__ == "__main__":
    main()