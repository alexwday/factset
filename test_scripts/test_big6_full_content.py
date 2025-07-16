#!/usr/bin/env python3
"""
Big 6 Canadian Banks - Full Article Content

This script retrieves full story content for Big 6 Canadian banks and displays:
- Complete article content/story body
- Analysis of content structure and features
- Examples of what data is available

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
from fds.sdk.StreetAccountNews.api import headlines_api
from fds.sdk.StreetAccountNews.model.headlines_request import HeadlinesRequest
from fds.sdk.StreetAccountNews.model.headlines_request_data import HeadlinesRequestData
from fds.sdk.StreetAccountNews.model.headlines_request_meta import HeadlinesRequestMeta
from fds.sdk.StreetAccountNews.model.headlines_request_meta_pagination import HeadlinesRequestMetaPagination
from fds.sdk.StreetAccountNews.model.headlines_request_tickers_object import HeadlinesRequestTickersObject
from fds.sdk.StreetAccountNews.model.headlines_request_data_search_time import HeadlinesRequestDataSearchTime

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

def get_big6_full_content() -> Optional[List[Dict[str, Any]]]:
    """Retrieve full article content for Big 6 Canadian banks."""
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
        
        # Calculate 60-day date range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=60)
        
        # Prepare request for Big 6 bank full content (last 60 days)
        headlines_request = HeadlinesRequest(
            data=HeadlinesRequestData(
                tickers=bank_tickers,
                categories=["Earnings", "Corporate Actions"],
                sectors=["Financial"],
                regions=["North America"],
                is_primary=True,
                search_time=HeadlinesRequestDataSearchTime(
                    start=start_time,
                    end=end_time
                )
            ),
            meta=HeadlinesRequestMeta(
                pagination=HeadlinesRequestMetaPagination(
                    limit=10,  # Limit to 10 for full content analysis
                    offset=0
                ),
                # Note: NOT specifying attributes means we get ALL available fields including full story body
            )
        )
        
        articles_data = []
        
        # Get full content using API client
        with streetaccount.ApiClient(configuration) as api_client:
            api_instance = headlines_api.HeadlinesApi(api_client)
            
            logger.info("📰 Retrieving Big 6 Canadian bank full article content...")
            
            try:
                response = api_instance.get_street_account_headlines(headlines_request=headlines_request)
                
                if response and response.data:
                    for article_item in response.data:
                        # Extract ALL available fields using correct API field names
                        article_data = {
                            'headline': getattr(article_item, 'headlines', 'No headline'),
                            'story_body': getattr(article_item, 'story_body', 'No story body'),
                            'publish_time': getattr(article_item, 'story_time', None),
                            'story_id': getattr(article_item, 'id', None),
                            'tickers': getattr(article_item, 'symbols', []),
                            'primary_tickers': getattr(article_item, 'primary_symbols', []),
                            'categories': getattr(article_item, 'subjects', []),
                            'reference_uris': getattr(article_item, 'reference_uris', None),
                            'url': getattr(article_item, 'url', None),
                            'all_attributes': dir(article_item)  # Get all available attributes
                        }
                        articles_data.append(article_data)
                    
                    logger.info(f"✅ Retrieved {len(articles_data)} full articles")
                else:
                    logger.warning("⚠️  No articles found for Big 6 banks today")
                    
            except Exception as e:
                logger.error(f"❌ Full content retrieval failed: {e}")
                return None
        
        return articles_data
        
    except Exception as e:
        logger.error(f"❌ Big 6 full content retrieval failed: {e}")
        return None
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
        if temp_cert_path and os.path.exists(temp_cert_path):
            os.unlink(temp_cert_path)

def analyze_content_structure(articles: List[Dict[str, Any]]) -> None:
    """Analyze the structure and available fields in article content."""
    print("\\n" + "="*80)
    print("🔍 CONTENT STRUCTURE ANALYSIS")
    print("="*80)
    
    if not articles:
        print("❌ No articles to analyze")
        return
    
    # Analyze first article in detail
    first_article = articles[0]
    
    print("\\n📊 Available Data Fields:")
    print("-" * 40)
    
    field_analysis = {
        'headline': 'News headline/title',
        'story_body': 'Full article content',
        'publish_time': 'Publication timestamp',
        'story_id': 'Unique story identifier',
        'tickers': 'Associated stock tickers',
        'categories': 'Content categories',
        'sectors': 'Industry sectors',
        'regions': 'Geographic regions',
        'source': 'News source',
        'author': 'Article author',
        'summary': 'Article summary',
        'sentiment': 'Sentiment analysis',
        'relevance_score': 'Relevance scoring',
        'word_count': 'Article word count'
    }
    
    for field, description in field_analysis.items():
        value = first_article.get(field)
        if value is not None and value != [] and value != 'No story body':
            if isinstance(value, str) and len(value) > 100:
                display_value = f"{value[:100]}..."
            else:
                display_value = str(value)
            print(f"✅ {field:15} | {description:25} | {display_value}")
        else:
            print(f"❌ {field:15} | {description:25} | Not available")
    
    # Show all available attributes
    if 'all_attributes' in first_article:
        print("\\n🔧 All Available Attributes:")
        print("-" * 40)
        attributes = [attr for attr in first_article['all_attributes'] if not attr.startswith('_')]
        for attr in sorted(attributes):
            print(f"   • {attr}")

def display_full_content_examples(articles: List[Dict[str, Any]]) -> None:
    """Display examples of full article content."""
    print("\\n" + "="*100)
    print("📰 BIG 6 CANADIAN BANKS - FULL ARTICLE CONTENT EXAMPLES")
    print("="*100)
    
    if not articles:
        print("❌ No articles found for Big 6 banks")
        return
    
    for i, article in enumerate(articles[:3], 1):  # Show first 3 articles
        headline = article.get('headline', 'No headline')
        story_body = article.get('story_body', 'No story body')
        publish_time = article.get('publish_time', 'Unknown time')
        tickers = article.get('tickers', [])
        categories = article.get('categories', [])
        
        # Format publish time
        if publish_time and publish_time != 'Unknown time':
            try:
                dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                time_str = dt.strftime('%Y-%m-%d %H:%M AST')
            except:
                time_str = publish_time
        else:
            time_str = 'Unknown time'
        
        # Format tickers
        ticker_str = ', '.join(tickers) if tickers else 'No tickers'
        
        # Format categories
        categories_str = ', '.join(categories) if categories else 'General'
        
        print(f"\\n📄 ARTICLE {i}")
        print("=" * 80)
        print(f"📰 Headline: {headline}")
        print(f"🏦 Tickers: {ticker_str}")
        print(f"📂 Categories: {categories_str}")
        print(f"⏰ Published: {time_str}")
        print("\\n📝 Full Story Content:")
        print("-" * 50)
        
        if story_body and story_body != 'No story body':
            # Show first 1000 characters of story body
            if len(story_body) > 1000:
                print(f"{story_body[:1000]}...")
                print(f"\\n[Content truncated - Full article is {len(story_body)} characters]")
            else:
                print(story_body)
        else:
            print("❌ No story body content available")
        
        print("\\n" + "="*80)
    
    # Content statistics
    print("\\n📊 CONTENT STATISTICS")
    print("="*50)
    
    total_articles = len(articles)
    articles_with_body = sum(1 for a in articles if a.get('story_body') and a.get('story_body') != 'No story body')
    
    if articles_with_body > 0:
        avg_length = sum(len(a.get('story_body', '')) for a in articles if a.get('story_body')) / articles_with_body
        max_length = max(len(a.get('story_body', '')) for a in articles if a.get('story_body'))
        min_length = min(len(a.get('story_body', '')) for a in articles if a.get('story_body'))
        
        print(f"Total Articles: {total_articles}")
        print(f"Articles with Full Content: {articles_with_body}")
        print(f"Average Content Length: {avg_length:.0f} characters")
        print(f"Longest Article: {max_length} characters")
        print(f"Shortest Article: {min_length} characters")
    else:
        print("❌ No articles with full content found")

def main():
    """Main execution function."""
    print("🚀 Starting Big 6 Canadian Banks Full Content Analysis...")
    print("="*80)
    
    # Load environment
    if not load_environment_variables():
        sys.exit(1)
    
    # Get full content data
    articles = get_big6_full_content()
    if articles is None:
        logger.error("❌ Failed to retrieve full content data")
        sys.exit(1)
    
    # Analyze content structure
    analyze_content_structure(articles)
    
    # Display full content examples
    display_full_content_examples(articles)
    
    print("\\n" + "="*80)
    print("✅ Big 6 Canadian Banks Full Content Analysis Complete!")
    print("="*80)

if __name__ == "__main__":
    main()