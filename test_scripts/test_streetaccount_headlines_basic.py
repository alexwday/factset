#!/usr/bin/env python3
"""
StreetAccountNews Basic Headlines Test Script

This script retrieves and displays today's financial services headlines using
basic filtering capabilities of the StreetAccountNews API.

Author: Generated with Claude Code
Date: 2024-07-15
"""

import os
import sys
import json
import time
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import quote
import logging

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from pysmb.SMBConnection import SMBConnection

# FactSet SDK imports
import fds.sdk.StreetAccountNews as streetaccount
from fds.sdk.StreetAccountNews.api import headlines_api
from fds.sdk.StreetAccountNews.model.headlines_request import HeadlinesRequest
from fds.sdk.StreetAccountNews.model.headlines_request_data import HeadlinesRequestData
from fds.sdk.StreetAccountNews.model.headlines_request_meta import HeadlinesRequestMeta
from fds.sdk.StreetAccountNews.model.headlines_request_meta_pagination import HeadlinesRequestMetaPagination

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streetaccount_headlines_basic_test.log')
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

def sanitize_headline_for_logging(headline: str) -> str:
    """Remove sensitive information from headlines before logging."""
    return headline[:100] + "..." if len(headline) > 100 else headline

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

def get_financial_headlines_today() -> Optional[List[Dict[str, Any]]]:
    """Retrieve today's financial services headlines."""
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
        
        # Prepare request for financial headlines
        headlines_request = HeadlinesRequest(
            data=HeadlinesRequestData(
                sectors=["Financial"],  # Focus on financial sector
                categories=["Earnings", "Corporate Actions", "Company News"],
                regions=["North America"],  # Focus on North American companies
                is_primary=True,  # Only primary company news
                predefined_range="today"  # Today's headlines only
            ),
            meta=HeadlinesRequestMeta(
                pagination=HeadlinesRequestMetaPagination(
                    limit=100,  # Get up to 100 headlines
                    offset=0
                ),
                attributes=["headlines"]  # Include headline content
            )
        )
        
        headlines_data = []
        
        # Get headlines using API client
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            logger.info("📰 Retrieving today's financial services headlines...")
            
            try:
                response = api_instance.get_street_account_news_headlines(headlines_request)
                
                if response and response.data:
                    for headline_item in response.data:
                        headline_data = {
                            'headline': getattr(headline_item, 'headline', 'No headline'),
                            'publish_time': getattr(headline_item, 'publish_time', None),
                            'tickers': getattr(headline_item, 'tickers', []),
                            'categories': getattr(headline_item, 'categories', []),
                            'sectors': getattr(headline_item, 'sectors', []),
                            'regions': getattr(headline_item, 'regions', []),
                            'story_id': getattr(headline_item, 'story_id', None),
                            'story_body': getattr(headline_item, 'story_body', '')
                        }
                        headlines_data.append(headline_data)
                    
                    logger.info(f"✅ Retrieved {len(headlines_data)} headlines")
                else:
                    logger.warning("⚠️  No headlines data received")
                    
            except Exception as e:
                logger.error(f"❌ Headlines retrieval failed: {e}")
                return None
        
        return headlines_data
        
    except Exception as e:
        logger.error(f"❌ Financial headlines retrieval failed: {e}")
        return None
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def analyze_headlines_by_category(headlines: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze headlines by category."""
    category_counts = {}
    
    for headline in headlines:
        categories = headline.get('categories', [])
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    return category_counts

def analyze_headlines_by_sector(headlines: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze headlines by sector."""
    sector_counts = {}
    
    for headline in headlines:
        sectors = headline.get('sectors', [])
        for sector in sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    return sector_counts

def get_top_tickers(headlines: List[Dict[str, Any]], top_n: int = 10) -> List[tuple]:
    """Get top mentioned tickers."""
    ticker_counts = {}
    
    for headline in headlines:
        tickers = headline.get('tickers', [])
        for ticker in tickers:
            ticker_symbol = ticker if isinstance(ticker, str) else ticker.get('symbol', 'Unknown')
            ticker_counts[ticker_symbol] = ticker_counts.get(ticker_symbol, 0) + 1
    
    return sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

def display_headlines_summary(headlines: List[Dict[str, Any]]) -> None:
    """Display comprehensive headlines summary."""
    print("\\n" + "="*70)
    print("📰 FINANCIAL SERVICES HEADLINES - Today")
    print("="*70)
    
    if not headlines:
        print("❌ No headlines found for today")
        return
    
    print(f"✅ Total Headlines Found: {len(headlines)}")
    
    # Category analysis
    category_counts = analyze_headlines_by_category(headlines)
    if category_counts:
        print("\\n📊 Headlines by Category:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   📈 {category}: {count} headlines")
    
    # Sector analysis
    sector_counts = analyze_headlines_by_sector(headlines)
    if sector_counts:
        print("\\n🏢 Headlines by Sector:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   🏦 {sector}: {count} headlines")
    
    # Top tickers
    top_tickers = get_top_tickers(headlines, 10)
    if top_tickers:
        print("\\n🔝 Most Mentioned Tickers:")
        for ticker, count in top_tickers:
            print(f"   💹 {ticker}: {count} mentions")
    
    # Recent headlines sample
    print("\\n📋 Recent Headlines Sample:")
    recent_headlines = sorted(headlines, 
                            key=lambda x: x.get('publish_time', ''), 
                            reverse=True)[:10]
    
    for i, headline in enumerate(recent_headlines, 1):
        headline_text = headline.get('headline', 'No headline')
        publish_time = headline.get('publish_time', 'Unknown time')
        tickers = headline.get('tickers', [])
        
        # Format tickers
        ticker_str = ', '.join(tickers[:3]) if tickers else 'No tickers'
        if len(tickers) > 3:
            ticker_str += f" (+{len(tickers) - 3} more)"
        
        print(f"   {i:2d}. {headline_text[:80]}...")
        print(f"       🏦 {ticker_str} | ⏰ {publish_time}")
        print()

def export_headlines_analysis(headlines: List[Dict[str, Any]]) -> None:
    """Export headlines analysis to JSON file."""
    try:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        filename = f"streetaccount_headlines_basic_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            'timestamp': timestamp,
            'analysis_date': datetime.now(timezone.utc).isoformat(),
            'total_headlines': len(headlines),
            'category_analysis': analyze_headlines_by_category(headlines),
            'sector_analysis': analyze_headlines_by_sector(headlines),
            'top_tickers': get_top_tickers(headlines, 20),
            'headlines_sample': headlines[:20]  # Export first 20 headlines
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\\n💾 Headlines analysis exported to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")

def main():
    """Main execution function."""
    print("🚀 Starting StreetAccountNews Basic Headlines Analysis...")
    print("="*70)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Get headlines data
    headlines = get_financial_headlines_today()
    if not headlines:
        logger.error("❌ Failed to retrieve headlines data")
        sys.exit(1)
    
    # Display analysis
    display_headlines_summary(headlines)
    
    # Export analysis
    export_headlines_analysis(headlines)
    
    print("\\n" + "="*70)
    print("✅ StreetAccountNews Basic Headlines Analysis Complete!")
    print("="*70)

if __name__ == "__main__":
    main()