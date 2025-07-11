#!/usr/bin/env python3
"""
FactSet Segments API Test Script
Explores segment-level data for RY-CA (Royal Bank of Canada).
Shows business unit breakdowns like Wealth Management, Capital Markets, P&CB.
"""

import pandas as pd
import fds.sdk.FactSetFundamentals
from fds.sdk.FactSetFundamentals.api import segments_api
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
TEST_PERIOD = "QTR"    # Latest quarter
TEST_CURRENCY = "CAD"  # Canadian dollars

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

def explore_segments_api(seg_api: segments_api.SegmentsApi, ticker: str) -> Dict[str, Any]:
    """Explore what segments-related methods are available."""
    print(f"🔍 Exploring Segments API for {ticker}...")
    
    # Get all available methods in the segments API
    api_methods = [method for method in dir(seg_api) if not method.startswith('_')]
    print(f"📋 Available Segments API methods: {api_methods}")
    
    results = {}
    
    # Try different methods to discover segments
    for method_name in api_methods:
        if 'segment' in method_name.lower():
            try:
                method = getattr(seg_api, method_name)
                print(f"🧪 Testing method: {method_name}")
                
                # Try calling with minimal parameters
                if 'get' in method_name.lower():
                    try:
                        # Try different parameter combinations
                        if 'ids' in method.__code__.co_varnames:
                            response = method(ids=[ticker])
                        elif 'id' in method.__code__.co_varnames:
                            response = method(id=ticker)
                        else:
                            response = method()
                        
                        if response:
                            print(f"  ✅ {method_name} returned data")
                            results[method_name] = response
                        else:
                            print(f"  ⚠️  {method_name} returned no data")
                            
                    except Exception as e:
                        print(f"  ❌ {method_name} error: {e}")
                        
            except Exception as e:
                print(f"  ❌ Error accessing {method_name}: {e}")
    
    return results

def test_segments_data(seg_api: segments_api.SegmentsApi, ticker: str) -> Optional[Dict[str, Any]]:
    """Test getting segments data for the ticker."""
    print(f"📊 Testing segments data retrieval for {ticker}...")
    
    try:
        # Try the most likely segments API call patterns
        test_methods = [
            # Method 1: Basic segments call
            lambda: seg_api.get_segments(ids=[ticker]),
            
            # Method 2: Segments with additional parameters
            lambda: seg_api.get_segments(ids=[ticker], periodicity="QTR"),
            
            # Method 3: Try with different parameter name
            lambda: seg_api.get_fds_segments(ids=[ticker]),
            
            # Method 4: Try segments for list
            lambda: seg_api.get_segments_for_list(ids=[ticker]),
        ]
        
        for i, method in enumerate(test_methods, 1):
            try:
                print(f"  🧪 Testing method {i}...")
                response = method()
                
                if response:
                    print(f"    ✅ Method {i} succeeded!")
                    
                    # Try to unwrap if needed
                    if hasattr(response, 'get_response_200'):
                        response = response.get_response_200()
                    
                    if hasattr(response, 'data') and response.data:
                        print(f"    📊 Found {len(response.data)} segments")
                        return response.data
                    else:
                        print(f"    📊 Response structure: {type(response)}")
                        print(f"    📊 Response attributes: {dir(response)}")
                        return response
                else:
                    print(f"    ⚠️  Method {i} returned no data")
                    
            except Exception as e:
                print(f"    ❌ Method {i} error: {e}")
                
    except Exception as e:
        print(f"❌ Error testing segments data: {e}")
        
    return None

def analyze_segments_data(segments_data: Any) -> Dict[str, Any]:
    """Analyze and format segments data."""
    print("📈 Analyzing segments data structure...")
    
    if not segments_data:
        return {"error": "No segments data to analyze"}
    
    analysis = {
        "data_type": str(type(segments_data)),
        "segments_found": 0,
        "segments_details": []
    }
    
    try:
        # Handle different data structures
        if isinstance(segments_data, list):
            analysis["segments_found"] = len(segments_data)
            
            for i, segment in enumerate(segments_data[:10]):  # Limit to first 10 for display
                segment_info = {
                    "index": i,
                    "type": str(type(segment)),
                    "attributes": dir(segment) if hasattr(segment, '__dict__') else "N/A"
                }
                
                # Try to extract common segment attributes
                if hasattr(segment, 'to_dict'):
                    segment_dict = segment.to_dict()
                    segment_info["data"] = segment_dict
                elif hasattr(segment, '__dict__'):
                    segment_info["data"] = segment.__dict__
                
                analysis["segments_details"].append(segment_info)
                
        elif hasattr(segments_data, 'to_dict'):
            segment_dict = segments_data.to_dict()
            analysis["segments_found"] = 1
            analysis["segments_details"] = [segment_dict]
            
        else:
            analysis["raw_data"] = str(segments_data)
            
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def generate_segments_report(ticker: str, segments_analysis: Dict[str, Any]) -> str:
    """Generate HTML report for segments analysis."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FactSet Segments Analysis - {ticker}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.5;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            .header {{
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                text-align: center;
            }}
            .section {{
                background: #fff;
                margin-bottom: 20px;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section-header {{
                background: #495057;
                color: white;
                padding: 15px 20px;
                font-size: 1.1em;
                font-weight: bold;
            }}
            .content {{
                padding: 20px;
            }}
            .segment-item {{
                border-bottom: 1px solid #dee2e6;
                padding: 15px 0;
            }}
            .segment-item:last-child {{
                border-bottom: none;
            }}
            .code {{
                font-family: 'Monaco', 'Consolas', monospace;
                background: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.9em;
            }}
            .error {{
                color: #dc3545;
                background: #f8d7da;
                padding: 10px;
                border-radius: 4px;
            }}
            .success {{
                color: #155724;
                background: #d4edda;
                padding: 10px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>FactSet Segments Analysis</h1>
            <h2>{ticker} - Business Segment Data</h2>
        </div>
        
        <div class="section">
            <div class="section-header">Analysis Summary</div>
            <div class="content">
                <p><strong>Data Type:</strong> <span class="code">{segments_analysis.get('data_type', 'Unknown')}</span></p>
                <p><strong>Segments Found:</strong> {segments_analysis.get('segments_found', 0)}</p>
                
                {f'<div class="error">Error: {segments_analysis["error"]}</div>' if segments_analysis.get('error') else ''}
                {f'<div class="success">Successfully found segment data for {ticker}</div>' if segments_analysis.get('segments_found', 0) > 0 else ''}
            </div>
        </div>
    """
    
    # Add segments details
    if segments_analysis.get('segments_details'):
        html_content += """
        <div class="section">
            <div class="section-header">Segments Details</div>
            <div class="content">
        """
        
        for segment in segments_analysis['segments_details']:
            html_content += f"""
            <div class="segment-item">
                <h4>Segment {segment.get('index', 'Unknown')}</h4>
                <p><strong>Type:</strong> <span class="code">{segment.get('type', 'Unknown')}</span></p>
                
                {f'<p><strong>Data:</strong></p><pre class="code">{json.dumps(segment.get("data", {}), indent=2)}</pre>' if segment.get('data') else ''}
            </div>
            """
        
        html_content += """
            </div>
        </div>
        """
    
    html_content += f"""
        <div class="section">
            <div class="section-header">Raw Analysis Data</div>
            <div class="content">
                <pre class="code">{json.dumps(segments_analysis, indent=2, default=str)}</pre>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #6c757d;">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    """Main function to test FactSet Segments API."""
    print("\n" + "="*80)
    print("🏦 FACTSET SEGMENTS API EXPLORATION")
    print("="*80)
    print(f"🎯 Testing ticker: {TEST_TICKER}")
    print(f"📅 Period: {TEST_PERIOD}")
    print(f"💰 Currency: {TEST_CURRENCY}")
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
    print("✅ FactSet Segments API client configured")
    
    try:
        with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
            # Initialize Segments API
            seg_api = segments_api.SegmentsApi(api_client)
            
            # Phase 1: Explore available segments methods
            print(f"\n🔍 PHASE 1: EXPLORING SEGMENTS API METHODS")
            print("="*80)
            
            api_exploration = explore_segments_api(seg_api, TEST_TICKER)
            
            # Phase 2: Test segments data retrieval
            print(f"\n🔍 PHASE 2: TESTING SEGMENTS DATA RETRIEVAL")
            print("="*80)
            
            segments_data = test_segments_data(seg_api, TEST_TICKER)
            
            # Phase 3: Analyze segments data
            print(f"\n🔍 PHASE 3: ANALYZING SEGMENTS DATA")
            print("="*80)
            
            segments_analysis = analyze_segments_data(segments_data)
            
            # Display results
            print(f"\n📊 RESULTS SUMMARY:")
            print("-" * 80)
            print(f"Segments found: {segments_analysis.get('segments_found', 0)}")
            print(f"Data type: {segments_analysis.get('data_type', 'Unknown')}")
            
            if segments_analysis.get('error'):
                print(f"❌ Error: {segments_analysis['error']}")
            
            if segments_analysis.get('segments_found', 0) > 0:
                print("✅ Successfully found segments data!")
                
                # Show sample segment details
                for segment in segments_analysis.get('segments_details', [])[:3]:
                    print(f"\n📋 Sample Segment:")
                    if segment.get('data'):
                        for key, value in list(segment['data'].items())[:5]:
                            print(f"  {key}: {value}")
            else:
                print("⚠️  No segments data found - may need different API approach")
            
            # Generate HTML report
            print(f"\n📄 GENERATING SEGMENTS ANALYSIS REPORT...")
            html_report = generate_segments_report(TEST_TICKER, segments_analysis)
            
            # Save HTML report
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            html_filename = f"factset_segments_analysis_{TEST_TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            html_path = output_dir / html_filename
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            print(f"✅ Segments analysis report saved: {html_path}")
            print(f"🌐 Open the file in your browser to view the detailed analysis")
            
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