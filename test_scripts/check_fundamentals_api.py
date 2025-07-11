#!/usr/bin/env python3
"""
FactSet Fundamentals API Test Script
Comprehensive test of all available fundamental data for RY-CA (Royal Bank of Canada).
Shows ALL categories, metrics, and data points available for business evaluation.
"""

import pandas as pd
import fds.sdk.FactSetFundamentals
from fds.sdk.FactSetFundamentals.api import data_items_api, fact_set_fundamentals_api
from fds.sdk.FactSetFundamentals.models import *
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
import time
from pathlib import Path

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

# Test configuration
TEST_TICKER = "RY-CA"  # Royal Bank of Canada
TEST_PERIODS = ["QTR", "QTR_R", "ANN", "ANN_R", "LTM"]  # Different periodicities to test
TEST_CURRENCIES = ["CAD", "USD", "LOCAL"]  # Different currencies to test

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

def get_available_metrics(data_api: data_items_api.DataItemsApi) -> Dict[str, List[Dict[str, Any]]]:
    """Get all available metrics by category."""
    print("📊 Discovering all available fundamental metrics...")
    
    categories = [
        "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "RATIOS",
        "FINANCIAL_SERVICES", "INDUSTRY_METRICS", "PENSION_AND_POSTRETIREMENT",
        "MARKET_DATA", "MISCELLANEOUS", "DATES"
    ]
    
    all_metrics = {}
    
    for category in categories:
        try:
            print(f"  🔍 Fetching {category} metrics...")
            response = data_api.get_fds_fundamentals_metrics(category=category)
            
            if response and hasattr(response, 'data') and response.data:
                metrics = [metric.to_dict() for metric in response.data]
                all_metrics[category] = metrics
                print(f"    ✅ Found {len(metrics)} {category} metrics")
            else:
                print(f"    ⚠️  No metrics found for {category}")
                all_metrics[category] = []
                
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"    ❌ Error fetching {category} metrics: {e}")
            all_metrics[category] = []
    
    return all_metrics

def get_fundamental_data(fund_api: fact_set_fundamentals_api.FactSetFundamentalsApi, 
                        ticker: str, 
                        metrics: List[str], 
                        periodicity: str = "QTR",
                        currency: str = "CAD") -> Optional[List[Dict[str, Any]]]:
    """Get fundamental data for specific metrics."""
    try:
        print(f"  📈 Fetching {len(metrics)} metrics for {ticker} ({periodicity}, {currency})")
        
        # Add date range for recent data (last 3 years)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=3*365)
        
        response = fund_api.get_fds_fundamentals(
            ids=[ticker],
            metrics=metrics,
            periodicity=periodicity,
            fiscal_period_start=start_date.strftime('%Y-%m-%d'),
            fiscal_period_end=end_date.strftime('%Y-%m-%d'),
            currency=currency,
            restated="RP"
        )
        
        if response and hasattr(response, 'data') and response.data:
            data = [item.to_dict() for item in response.data]
            print(f"    ✅ Retrieved {len(data)} data points")
            return data
        else:
            print(f"    ⚠️  No data returned for {ticker}")
            return None
            
    except Exception as e:
        print(f"    ❌ Error fetching fundamental data: {e}")
        return None

def analyze_data_coverage(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze data coverage and completeness."""
    if not data:
        return {"total_points": 0, "metrics_with_data": 0, "date_range": None}
    
    # Group by metric
    metrics_data = {}
    all_dates = set()
    
    for item in data:
        metric = item.get('metric')
        value = item.get('value')
        fiscal_end_date = item.get('fiscal_end_date')
        
        if metric and value is not None:
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append({
                'value': value,
                'date': fiscal_end_date,
                'fiscal_year': item.get('fiscal_year'),
                'fiscal_period': item.get('fiscal_period')
            })
            if fiscal_end_date:
                all_dates.add(fiscal_end_date)
    
    # Calculate coverage stats
    date_range = None
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = f"{min_date} to {max_date}"
    
    return {
        "total_points": len(data),
        "metrics_with_data": len(metrics_data),
        "unique_metrics": list(metrics_data.keys()),
        "date_range": date_range,
        "metrics_data": metrics_data
    }

def format_metric_info(metric: Dict[str, Any]) -> str:
    """Format metric information for display."""
    metric_name = metric.get('metric', 'Unknown')
    description = metric.get('description', 'No description')
    data_type = metric.get('data_type', 'Unknown')
    
    # Truncate long descriptions
    if len(description) > 80:
        description = description[:77] + "..."
    
    return f"    {metric_name:<20} | {data_type:<8} | {description}"

def display_sample_data(metrics_data: Dict[str, List[Dict[str, Any]]], max_metrics: int = 10):
    """Display sample data for top metrics."""
    print(f"\n📊 Sample Data (showing up to {max_metrics} metrics with most data points):")
    print("-" * 100)
    
    # Sort metrics by number of data points
    sorted_metrics = sorted(metrics_data.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (metric, values) in enumerate(sorted_metrics[:max_metrics]):
        print(f"\n🔹 {metric} ({len(values)} data points):")
        
        # Show most recent 5 data points
        recent_values = sorted(values, key=lambda x: x.get('date', ''), reverse=True)[:5]
        
        for val in recent_values:
            date = val.get('date', 'Unknown')
            value = val.get('value', 'N/A')
            fy = val.get('fiscal_year', 'N/A')
            fp = val.get('fiscal_period', 'N/A')
            
            # Format value based on type
            if isinstance(value, (int, float)):
                if abs(value) >= 1000000:
                    formatted_value = f"{value/1000000:.1f}M"
                elif abs(value) >= 1000:
                    formatted_value = f"{value/1000:.1f}K"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            print(f"    {date} | FY{fy} Q{fp} | {formatted_value}")

def generate_html_report(all_metrics: Dict[str, List[Dict[str, Any]]], 
                        category_results: Dict[str, Dict[str, Any]], 
                        ticker: str) -> str:
    """Generate comprehensive HTML report."""
    
    # Calculate overall statistics
    total_metrics = sum(len(metrics) for metrics in all_metrics.values())
    total_data_points = sum(
        max((result.get('total_points', 0) for result in cat_data.values()), default=0)
        for cat_data in category_results.values()
    )
    categories_with_data = sum(
        1 for cat_data in category_results.values() 
        if any(result.get('total_points', 0) > 0 for result in cat_data.values())
    )
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FactSet Fundamentals API Analysis - {ticker}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .summary {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-bottom: 30px;
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .category-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-left: 4px solid #667eea;
            }}
            .category-header {{
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #667eea;
            }}
            .metric-count {{
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
                margin-bottom: 10px;
            }}
            .status-good {{ color: #4CAF50; }}
            .status-warning {{ color: #ff9800; }}
            .status-error {{ color: #f44336; }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .data-table th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
            }}
            .data-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            .data-table tr:hover {{
                background: #f8f9fa;
            }}
            .conclusion {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin-top: 30px;
            }}
            .conclusion.good {{
                border-left: 4px solid #4CAF50;
            }}
            .conclusion.warning {{
                border-left: 4px solid #ff9800;
            }}
            .conclusion.error {{
                border-left: 4px solid #f44336;
            }}
            .timestamp {{
                color: #666;
                font-size: 0.9em;
                text-align: center;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>FactSet Fundamentals API Analysis</h1>
            <h2>{ticker} - Royal Bank of Canada</h2>
            <p>Comprehensive evaluation of available fundamental data</p>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metrics-grid">
                <div class="category-card">
                    <div class="metric-count">{total_metrics}</div>
                    <div>Total Available Metrics</div>
                </div>
                <div class="category-card">
                    <div class="metric-count">{categories_with_data}/10</div>
                    <div>Categories with Data</div>
                </div>
                <div class="category-card">
                    <div class="metric-count">{total_data_points:,}</div>
                    <div>Data Points Retrieved</div>
                </div>
            </div>
        </div>
        
        <div class="summary">
            <h2>Category Analysis</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Available Metrics</th>
                        <th>Data Points</th>
                        <th>Coverage</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add category rows
    for category, metrics in all_metrics.items():
        cat_data = category_results.get(category, {})
        metrics_available = len(metrics)
        
        if cat_data:
            best_result = max(cat_data.values(), key=lambda x: x.get('total_points', 0))
            data_points = best_result.get('total_points', 0)
            metrics_with_data = best_result.get('metrics_with_data', 0)
            
            if data_points > 0:
                coverage = f"{(metrics_with_data / metrics_available * 100):.1f}%" if metrics_available > 0 else "0%"
                status = "✅ Excellent" if metrics_with_data > metrics_available * 0.5 else "⚠️ Moderate"
                status_class = "status-good" if metrics_with_data > metrics_available * 0.5 else "status-warning"
            else:
                coverage = "0%"
                status = "❌ No Data"
                status_class = "status-error"
        else:
            data_points = 0
            coverage = "0%"
            status = "❌ No Access"
            status_class = "status-error"
        
        html_content += f"""
                    <tr>
                        <td><strong>{category.replace('_', ' ')}</strong></td>
                        <td>{metrics_available}</td>
                        <td>{data_points:,}</td>
                        <td>{coverage}</td>
                        <td class="{status_class}">{status}</td>
                    </tr>
        """
    
    # Business conclusion
    if total_data_points > 100:
        conclusion_class = "good"
        conclusion_text = "✅ EXCELLENT - FactSet Fundamentals API provides comprehensive data coverage"
        recommendation = "Recommend proceeding with full implementation for internal reporting"
    elif total_data_points > 50:
        conclusion_class = "warning"
        conclusion_text = "⚠️ MODERATE - FactSet Fundamentals API provides partial data coverage"
        recommendation = "Consider supplementary data sources for complete reporting"
    else:
        conclusion_class = "error"
        conclusion_text = "❌ LIMITED - FactSet Fundamentals API provides limited data coverage"
        recommendation = "Investigate access permissions or consider alternative data sources"
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <div class="conclusion {conclusion_class}">
            <h2>Business Recommendation</h2>
            <h3>{conclusion_text}</h3>
            <p><strong>Recommendation:</strong> {recommendation}</p>
            <p><strong>Key Findings:</strong></p>
            <ul>
                <li>Total metrics available: {total_metrics:,}</li>
                <li>Categories with data: {categories_with_data}/10</li>
                <li>Maximum data points: {total_data_points:,}</li>
                <li>Tested across multiple periods: QTR, ANN, LTM</li>
                <li>Tested across multiple currencies: CAD, USD, LOCAL</li>
            </ul>
        </div>
        
        <div class="timestamp">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main function to test FactSet Fundamentals API."""
    print("\n" + "="*80)
    print("🏦 FACTSET FUNDAMENTALS API COMPREHENSIVE TEST")
    print("="*80)
    print(f"🎯 Testing institution: {TEST_TICKER} (Royal Bank of Canada)")
    print(f"📅 Testing periods: {', '.join(TEST_PERIODS)}")
    print(f"💰 Testing currencies: {', '.join(TEST_CURRENCIES)}")
    print("="*80)
    
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
    
    # Configure FactSet API
    user = PROXY_USER
    password = quote(PROXY_PASSWORD)
    
    escaped_domain = quote(PROXY_DOMAIN + '\\' + user)
    proxy_url = f"http://{escaped_domain}:{password}@{PROXY_URL}"
    configuration = fds.sdk.FactSetFundamentals.Configuration(
        username=API_USERNAME,
        password=API_PASSWORD,
        proxy=proxy_url,
        ssl_ca_cert=temp_cert_path
    )
    configuration.get_basic_auth_token()
    print("✅ FactSet Fundamentals API client configured")
    
    try:
        with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
            # Initialize API instances
            data_api = data_items_api.DataItemsApi(api_client)
            fund_api = fact_set_fundamentals_api.FactSetFundamentalsApi(api_client)
            
            # Phase 1: Discover all available metrics
            print("\n🔍 PHASE 1: DISCOVERING ALL AVAILABLE METRICS")
            print("="*80)
            
            all_metrics = get_available_metrics(data_api)
            
            # Display metrics summary
            print(f"\n📋 METRICS SUMMARY BY CATEGORY:")
            print("-" * 80)
            total_metrics = 0
            
            for category, metrics in all_metrics.items():
                total_metrics += len(metrics)
                print(f"🔹 {category:<25} | {len(metrics):>4} metrics")
                
                if metrics:
                    print(f"   Sample metrics:")
                    for metric in metrics[:3]:  # Show first 3 metrics
                        print(f"     {format_metric_info(metric)}")
                    if len(metrics) > 3:
                        print(f"     ... and {len(metrics) - 3} more")
                print()
            
            print(f"📊 TOTAL AVAILABLE METRICS: {total_metrics}")
            
            # Phase 2: Test data retrieval for each category
            print(f"\n🔍 PHASE 2: TESTING DATA RETRIEVAL FOR {TEST_TICKER}")
            print("="*80)
            
            category_results = {}
            
            for category, metrics in all_metrics.items():
                if not metrics:
                    continue
                    
                print(f"\n🔹 Testing {category} ({len(metrics)} metrics available)")
                print("-" * 60)
                
                # Get metric codes for this category
                metric_codes = [m.get('metric') for m in metrics if m.get('metric')]
                
                if not metric_codes:
                    print(f"  ⚠️  No valid metric codes found for {category}")
                    continue
                
                # Test with different periodicities and currencies
                category_data = {}
                
                for periodicity in TEST_PERIODS:
                    for currency in TEST_CURRENCIES:
                        print(f"  🔍 Testing {periodicity} data in {currency}...")
                        
                        # Limit to first 50 metrics to avoid hitting API limits
                        test_metrics = metric_codes[:50]
                        
                        data = get_fundamental_data(
                            fund_api, TEST_TICKER, test_metrics, periodicity, currency
                        )
                        
                        if data:
                            analysis = analyze_data_coverage(data)
                            category_data[f"{periodicity}_{currency}"] = analysis
                            
                            print(f"    ✅ {analysis['total_points']} data points, "
                                  f"{analysis['metrics_with_data']} metrics with data")
                        else:
                            print(f"    ❌ No data returned")
                        
                        time.sleep(1)  # Rate limiting
                
                category_results[category] = category_data
                
                # Show best results for this category
                if category_data:
                    best_result = max(category_data.values(), key=lambda x: x['total_points'])
                    if best_result['total_points'] > 0:
                        print(f"  🎯 Best coverage: {best_result['total_points']} data points")
                        print(f"     Date range: {best_result['date_range']}")
                        print(f"     Metrics: {best_result['metrics_with_data']} with data")
                        
                        # Show sample data
                        if best_result.get('metrics_data'):
                            display_sample_data(best_result['metrics_data'], max_metrics=5)
            
            # Phase 3: Comprehensive summary
            print(f"\n🔍 PHASE 3: COMPREHENSIVE SUMMARY")
            print("="*80)
            
            # Overall statistics
            total_data_points = sum(
                max((result.get('total_points', 0) for result in cat_data.values()), default=0)
                for cat_data in category_results.values()
            )
            
            categories_with_data = sum(
                1 for cat_data in category_results.values() 
                if any(result.get('total_points', 0) > 0 for result in cat_data.values())
            )
            
            print(f"📊 OVERALL RESULTS FOR {TEST_TICKER}:")
            print(f"  🎯 Total available metrics: {total_metrics}")
            print(f"  📈 Categories with data: {categories_with_data}/{len(all_metrics)}")
            print(f"  💾 Maximum data points retrieved: {total_data_points}")
            print()
            
            # Category-by-category results
            print("📋 CATEGORY-BY-CATEGORY RESULTS:")
            print("-" * 80)
            
            for category, cat_data in category_results.items():
                if not cat_data:
                    print(f"🔹 {category:<25} | ❌ No data available")
                    continue
                
                best_result = max(cat_data.values(), key=lambda x: x['total_points'])
                metrics_available = len(all_metrics.get(category, []))
                
                if best_result['total_points'] > 0:
                    print(f"🔹 {category:<25} | ✅ {best_result['total_points']:>4} data points "
                          f"| {best_result['metrics_with_data']:>3}/{metrics_available} metrics")
                else:
                    print(f"🔹 {category:<25} | ⚠️  Available but no data for {TEST_TICKER}")
            
            # Business recommendations
            print(f"\n💡 BUSINESS EVALUATION RECOMMENDATIONS:")
            print("-" * 80)
            
            key_categories = ['INCOME_STATEMENT', 'BALANCE_SHEET', 'CASH_FLOW', 'RATIOS', 'FINANCIAL_SERVICES']
            
            for category in key_categories:
                if category in category_results:
                    best_result = max(category_results[category].values(), key=lambda x: x['total_points'])
                    if best_result['total_points'] > 0:
                        coverage = (best_result['metrics_with_data'] / len(all_metrics.get(category, []))) * 100
                        print(f"✅ {category}: {coverage:.1f}% metric coverage - EXCELLENT for internal reporting")
                    else:
                        print(f"⚠️  {category}: No data available - May need alternative sources")
                else:
                    print(f"❌ {category}: Category not accessible - API limitation")
            
            print(f"\n🎯 CONCLUSION:")
            if total_data_points > 100:
                print("✅ FactSet Fundamentals API provides COMPREHENSIVE data coverage for RY-CA")
                print("✅ Suitable for internal reporting and analysis")
                print("✅ Recommend proceeding with full implementation")
            elif total_data_points > 50:
                print("⚠️  FactSet Fundamentals API provides MODERATE data coverage for RY-CA")
                print("⚠️  May need supplementary data sources for complete reporting")
            else:
                print("❌ FactSet Fundamentals API provides LIMITED data coverage for RY-CA")
                print("❌ Consider alternative data sources or investigate access permissions")
            
            print("-" * 80)
            print(f"✅ Test complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🎯 Tested {TEST_TICKER} across {len(TEST_PERIODS)} periods and {len(TEST_CURRENCIES)} currencies")
            print(f"📊 Evaluated {total_metrics} available metrics across {len(all_metrics)} categories")
            
            # Generate HTML report
            print(f"\n📄 GENERATING HTML REPORT...")
            html_report = generate_html_report(all_metrics, category_results, TEST_TICKER)
            
            # Save HTML report
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            html_filename = f"factset_fundamentals_analysis_{TEST_TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            html_path = output_dir / html_filename
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            print(f"✅ HTML report saved: {html_path}")
            print(f"🌐 Open the file in your browser to view the formatted report")
            print(f"📧 Share the HTML file with your team for review")
            
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