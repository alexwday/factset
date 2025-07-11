#!/usr/bin/env python3
"""
FactSet Segments API Test Script
Explores segment-level data for RY-CA (Royal Bank of Canada).
Shows business unit breakdowns like Wealth Management, Capital Markets, P&CB.
"""

import pandas as pd
import fds.sdk.FactSetFundamentals
from fds.sdk.FactSetFundamentals.api import segments_api, metrics_api
from fds.sdk.FactSetFundamentals.models import *
from fds.sdk.FactSetFundamentals.model.segments_request import SegmentsRequest
from fds.sdk.FactSetFundamentals.model.segment_request_body import SegmentRequestBody
from fds.sdk.FactSetFundamentals.model.ids_batch_max30000 import IdsBatchMax30000
from fds.sdk.FactSetFundamentals.model.segments_periodicity import SegmentsPeriodicity
from fds.sdk.FactSetFundamentals.model.segment_type import SegmentType
from fds.sdk.FactSetFundamentals.model.fiscal_period import FiscalPeriod
from fds.sdk.FactSetFundamentals.model.batch import Batch
import os
from urllib.parse import quote
from datetime import datetime, timedelta, date
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

def discover_all_metrics(data_api: metrics_api.MetricsApi) -> Tuple[List[str], Dict[str, str]]:
    """Discover all available metrics from the metrics API and return metrics list and descriptions map."""
    print("🔍 Discovering all available metrics...")
    
    categories = [
        "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "RATIOS",
        "FINANCIAL_SERVICES", "INDUSTRY_METRICS", "PENSION_AND_POSTRETIREMENT",
        "MARKET_DATA", "MISCELLANEOUS", "DATES"
    ]
    
    all_metrics = []
    metric_descriptions = {}
    
    for category in categories:
        try:
            print(f"  📊 Fetching {category} metrics...")
            response = data_api.get_fds_fundamentals_metrics(category=category)
            
            if response and hasattr(response, 'data') and response.data:
                category_metrics = []
                for metric in response.data:
                    if hasattr(metric, 'metric') and metric.metric:
                        metric_code = metric.metric
                        metric_desc = getattr(metric, 'description', 'No description available')
                        
                        category_metrics.append(metric_code)
                        metric_descriptions[metric_code] = metric_desc
                
                all_metrics.extend(category_metrics)
                print(f"    ✅ Found {len(category_metrics)} {category} metrics")
            else:
                print(f"    ⚠️  No metrics found for {category}")
                
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"    ❌ Error fetching {category} metrics: {e}")
    
    # Remove duplicates and sort
    unique_metrics = sorted(list(set(all_metrics)))
    print(f"📊 Total unique metrics discovered: {len(unique_metrics)}")
    print(f"📊 Metric descriptions captured: {len(metric_descriptions)}")
    
    return unique_metrics, metric_descriptions

def explore_segments_api(seg_api: segments_api.SegmentsApi, ticker: str) -> Dict[str, Any]:
    """Explore what segments-related methods are available."""
    print(f"🔍 Exploring Segments API for {ticker}...")
    
    # Get all available methods in the segments API
    api_methods = [method for method in dir(seg_api) if not method.startswith('_')]
    print(f"📋 Available Segments API methods: {api_methods}")
    
    # Look specifically for the correct method
    target_method = 'get_fds_segments_for_list'
    if target_method in api_methods:
        print(f"✅ Found correct method: {target_method}")
        
        # Get method signature
        method = getattr(seg_api, target_method)
        print(f"📋 Method signature: {method.__doc__}")
        
        return {"methods": api_methods, "target_method": target_method, "available": True}
    else:
        print(f"❌ Target method {target_method} not found")
        
        # Show available methods that might be relevant
        relevant_methods = [m for m in api_methods if 'segment' in m.lower() or 'get' in m.lower()]
        print(f"🔍 Relevant methods found: {relevant_methods}")
        
        return {"methods": api_methods, "target_method": target_method, "available": False, "relevant": relevant_methods}

def test_segments_data(seg_api: segments_api.SegmentsApi, ticker: str, available_metrics: List[str], metric_descriptions: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Test getting segments data for the ticker."""
    print(f"📊 Testing segments data retrieval for {ticker}...")
    
    try:
        # Create date range for recent data (last 6 months to get latest quarterly data)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        # Create request object with proper model class wrapping
        ids_instance = IdsBatchMax30000([ticker])
        
        # Use discovered metrics (limit to first 20 for initial testing)
        test_metrics = available_metrics[:20]  
        print(f"📊 Testing with {len(test_metrics)} discovered metrics: {test_metrics[:5]}{'...' if len(test_metrics) > 5 else ''}")
        
        # Test both quarterly and annual data to see what's available
        test_configs = [
            {
                "name": "Business Segments - Quarterly",
                "segment_type": SegmentType("BUS"),
                "periodicity": SegmentsPeriodicity("QTR"),
                "metrics": test_metrics
            },
            {
                "name": "Business Segments - Annual",
                "segment_type": SegmentType("BUS"),
                "periodicity": SegmentsPeriodicity("ANN"),
                "metrics": test_metrics
            }
        ]
        
        for config in test_configs:
            try:
                print(f"  🧪 Testing {config['name']}...")
                
                # Create fiscal period
                fiscal_period_instance = FiscalPeriod(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                # Create batch instance
                batch_instance = Batch("N")
                
                # Test each metric individually (segments API might only support one metric at a time)
                successful_metrics = []
                all_segment_data = []
                
                for metric in config["metrics"]:
                    try:
                        print(f"    📊 Testing metric: {metric} ({config['periodicity']})")
                        
                        # Create request body for single metric
                        segment_request_body = SegmentRequestBody(
                            ids=ids_instance,
                            metrics=metric,  # Test one metric at a time
                            periodicity=config["periodicity"],
                            fiscal_period=fiscal_period_instance,
                            segment_type=config["segment_type"],
                            batch=batch_instance
                        )
                        
                        # Create request
                        segments_request = SegmentsRequest(data=segment_request_body)
                        
                        # Make API call
                        response_wrapper = seg_api.get_fds_segments_for_list(segments_request)
                        
                        # Unwrap response
                        if hasattr(response_wrapper, 'get_response_200'):
                            response = response_wrapper.get_response_200()
                        else:
                            response = response_wrapper
                        
                        if response and hasattr(response, 'data') and response.data:
                            successful_metrics.append(metric)
                            all_segment_data.extend(response.data)
                            print(f"      ✅ {metric}: {len(response.data)} data points")
                        else:
                            print(f"      ❌ {metric}: No data")
                            
                    except Exception as e:
                        print(f"      ❌ {metric}: Error - {e}")
                        continue
                
                # Report results for this configuration
                if successful_metrics:
                    print(f"    ✅ {config['name']} succeeded! Found {len(successful_metrics)} working metrics: {successful_metrics}")
                    print(f"    📊 Total data points: {len(all_segment_data)}")
                    
                    # Return data from first successful configuration
                    if all_segment_data:
                        return all_segment_data
                else:
                    print(f"    ⚠️  {config['name']} - no working metrics found")
                        
            except Exception as e:
                print(f"    ❌ {config['name']} error: {e}")
                print(f"    🔍 Error type: {type(e).__name__}")
                
    except Exception as e:
        print(f"❌ Error testing segments data: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        
    return None

def convert_dates_to_strings(obj):
    """Convert date objects to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_dates_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates_to_strings(item) for item in obj]
    elif isinstance(obj, (datetime, date, timedelta)):
        return obj.isoformat()
    elif hasattr(obj, 'date') and callable(getattr(obj, 'date')):
        try:
            return obj.date().isoformat()
        except:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj

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
                    # Convert any date objects to strings for JSON serialization
                    segment_info["data"] = convert_dates_to_strings(segment_dict)
                elif hasattr(segment, '__dict__'):
                    segment_info["data"] = convert_dates_to_strings(segment.__dict__)
                
                analysis["segments_details"].append(segment_info)
                
        elif hasattr(segments_data, 'to_dict'):
            segment_dict = segments_data.to_dict()
            analysis["segments_found"] = 1
            analysis["segments_details"] = [convert_dates_to_strings(segment_dict)]
            
        else:
            analysis["raw_data"] = str(segments_data)
            
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def generate_interactive_html_table(df: pd.DataFrame, ticker: str) -> str:
    """Generate interactive HTML table with filtering, sorting, and expandable descriptions."""
    
    def format_value(value):
        """Format a single value as float with commas."""
        try:
            num_value = pd.to_numeric(value, errors='coerce')
            if pd.notna(num_value):
                return f"{num_value:,.2f}"
            else:
                return str(value) if value != 'N/A' else "N/A"
        except:
            return str(value)
    
    def truncate_description(desc):
        """Truncate description for initial display."""
        if len(str(desc)) > 50:
            return str(desc).split('.')[0] + '...'
        else:
            return str(desc)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FactSet Segments Data - {ticker}</title>
        
        <!-- DataTables CSS -->
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
        <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
        
        <!-- jQuery and DataTables JS -->
        <script type="text/javascript" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
        <script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
        <script type="text/javascript" src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
        
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
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
            .table-container {{
                background: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                overflow-x: auto;
            }}
            .description-cell {{
                max-width: 200px;
                cursor: pointer;
                position: relative;
            }}
            .description-short {{
                color: #007bff;
                text-decoration: underline;
            }}
            .description-full {{
                position: absolute;
                top: 100%;
                left: 0;
                background: #fff;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                z-index: 1000;
                width: 300px;
                display: none;
            }}
            .value-cell {{
                text-align: right;
                font-family: 'Monaco', 'Consolas', monospace;
                font-weight: bold;
            }}
            table.dataTable {{
                border-collapse: collapse !important;
            }}
            table.dataTable thead th {{
                background-color: #495057;
                color: white;
                font-weight: bold;
            }}
            table.dataTable tbody tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            table.dataTable tbody tr:hover {{
                background-color: #e9ecef;
            }}
            .dt-buttons {{
                margin-bottom: 10px;
            }}
            .dt-button {{
                background: #007bff;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                margin-right: 5px;
                cursor: pointer;
            }}
            .dt-button:hover {{
                background: #0056b3;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>FactSet Segments Data</h1>
            <h2>{ticker} - Interactive Analysis Table</h2>
            <p>Click on descriptions to expand • Use column filters to search • Export data with buttons above table</p>
        </div>
        
        <div class="table-container">
            <table id="segmentsTable" class="display" style="width:100%">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Segment</th>
                        <th>Date</th>
                        <th>Metric</th>
                        <th>Description</th>
                        <th>Value</th>
                        <th>FSYM ID</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add table rows using original DataFrame
    for _, row in df.iterrows():
        formatted_value = format_value(row['Value'])
        truncated_desc = truncate_description(row['Description'])
        
        html_content += f"""
                    <tr>
                        <td>{row['Ticker']}</td>
                        <td>{row['Segment']}</td>
                        <td>{row['Date']}</td>
                        <td>{row['Metric']}</td>
                        <td class="description-cell">
                            <span class="description-short" onclick="toggleDescription(this)">
                                {truncated_desc}
                            </span>
                            <div class="description-full">
                                {row['Description']}
                            </div>
                        </td>
                        <td class="value-cell">{formatted_value}</td>
                        <td>{row['FSYM_ID']}</td>
                    </tr>
        """
    
    html_content += f"""
                </tbody>
            </table>
        </div>
        
        <script>
            $(document).ready(function() {{
                $('#segmentsTable').DataTable({{
                    dom: 'Bfrtip',
                    buttons: [
                        'copy', 'csv', 'excel'
                    ],
                    pageLength: 25,
                    responsive: true,
                    columnDefs: [
                        {{ 
                            targets: [5], // Value column
                            type: 'num-fmt'
                        }}
                    ],
                    order: [[1, 'asc'], [3, 'asc']], // Sort by Segment, then Metric
                    initComplete: function () {{
                        // Add individual column search
                        this.api().columns().every(function () {{
                            var column = this;
                            var title = column.header().textContent;
                            
                            if (title !== 'Description' && title !== 'Value') {{
                                var select = $('<select><option value="">All ' + title + '</option></select>')
                                    .appendTo($(column.header()))
                                    .on('change', function () {{
                                        var val = $.fn.dataTable.util.escapeRegex(
                                            $(this).val()
                                        );
                                        column
                                            .search(val ? '^' + val + '$' : '', true, false)
                                            .draw();
                                    }});
                                
                                column.data().unique().sort().each(function (d, j) {{
                                    if (d) {{
                                        select.append('<option value="' + d + '">' + d + '</option>');
                                    }}
                                }});
                            }}
                        }});
                    }}
                }});
            }});
            
            function toggleDescription(element) {{
                var fullDesc = element.nextElementSibling;
                if (fullDesc.style.display === 'block') {{
                    fullDesc.style.display = 'none';
                }} else {{
                    // Hide all other open descriptions
                    document.querySelectorAll('.description-full').forEach(function(desc) {{
                        desc.style.display = 'none';
                    }});
                    fullDesc.style.display = 'block';
                }}
            }}
            
            // Hide description on click outside
            document.addEventListener('click', function(event) {{
                if (!event.target.closest('.description-cell')) {{
                    document.querySelectorAll('.description-full').forEach(function(desc) {{
                        desc.style.display = 'none';
                    }});
                }}
            }});
        </script>
        
        <div style="text-align: center; margin-top: 20px; color: #6c757d; font-size: 0.9em;">
            Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
            Data contains {len(df)} segment records for {ticker}
        </div>
    </body>
    </html>
    """
    
    return html_content

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
                
                {f'<p><strong>Data:</strong></p><pre class="code">{json.dumps(segment.get("data", {}), indent=2, default=str)}</pre>' if segment.get('data') else ''}
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
            # Initialize APIs
            seg_api = segments_api.SegmentsApi(api_client)
            data_api = metrics_api.MetricsApi(api_client)
            
            # Phase 1: Discover all available metrics
            print(f"\n🔍 PHASE 1: DISCOVERING ALL AVAILABLE METRICS")
            print("="*80)
            
            available_metrics, metric_descriptions = discover_all_metrics(data_api)
            
            # Phase 2: Explore available segments methods
            print(f"\n🔍 PHASE 2: EXPLORING SEGMENTS API METHODS")
            print("="*80)
            
            api_exploration = explore_segments_api(seg_api, TEST_TICKER)
            
            # Phase 3: Test segments data retrieval with discovered metrics
            print(f"\n🔍 PHASE 3: TESTING SEGMENTS DATA RETRIEVAL")
            print("="*80)
            
            segments_data = test_segments_data(seg_api, TEST_TICKER, available_metrics, metric_descriptions)
            
            # Phase 4: Generate table output
            print(f"\n📊 GENERATING SEGMENTS DATA TABLE")
            print("="*80)
            
            if segments_data:
                # First, let's examine the actual structure of the segments data
                print(f"🔍 EXAMINING SEGMENTS DATA STRUCTURE:")
                print("-" * 80)
                
                if segments_data:
                    sample_segment = segments_data[0]
                    if hasattr(sample_segment, 'to_dict'):
                        sample_dict = sample_segment.to_dict()
                        print(f"📋 Sample segment structure:")
                        for key, value in sample_dict.items():
                            print(f"  {key}: {value}")
                        print(f"📋 All available fields: {list(sample_dict.keys())}")
                    else:
                        print(f"📋 Sample segment type: {type(sample_segment)}")
                        print(f"📋 Sample segment: {sample_segment}")
                
                # Create table format
                table_data = []
                
                for segment in segments_data:
                    if hasattr(segment, 'to_dict'):
                        segment_dict = segment.to_dict()
                    else:
                        segment_dict = segment
                    
                    # Debug: print all available fields for the first few segments
                    if len(table_data) < 3:
                        print(f"\n🔍 Segment {len(table_data)+1} fields: {list(segment_dict.keys())}")
                    
                    # Use the correct field names based on actual API response structure
                    segment_name = segment_dict.get('label', 'Unknown')
                    date_value = segment_dict.get('date', 'Unknown')
                    
                    # Get metric description from the mapping
                    metric_code = segment_dict.get('metric', 'Unknown')
                    metric_description = metric_descriptions.get(metric_code, 'No description available')
                    
                    # Extract key fields for table (based on actual API response structure)
                    table_row = {
                        'Ticker': segment_dict.get('request_id', TEST_TICKER),
                        'Segment': segment_name,
                        'Date': date_value,
                        'Metric': metric_code,
                        'Description': metric_description,
                        'Value': segment_dict.get('value', 'N/A'),
                        'FSYM_ID': segment_dict.get('fsym_id', 'N/A')
                    }
                    table_data.append(table_row)
                
                # Create DataFrame and display
                if table_data:
                    df = pd.DataFrame(table_data)
                    
                    # Sort by Segment, then by Metric for better organization
                    df = df.sort_values(['Segment', 'Metric'], ascending=[True, True])
                    
                    print(f"📋 SEGMENTS DATA TABLE ({len(df)} rows):")
                    print("-" * 150)
                    print(df.to_string(index=False))
                    
                    # Generate interactive HTML table
                    print(f"\n📄 GENERATING INTERACTIVE HTML TABLE...")
                    html_content = generate_interactive_html_table(df, TEST_TICKER)
                    
                    # Save HTML and CSV
                    output_dir = Path(__file__).parent / "output"
                    output_dir.mkdir(exist_ok=True)
                    
                    # Save HTML
                    html_filename = f"factset_segments_table_{TEST_TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    html_path = output_dir / html_filename
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    # Save CSV
                    csv_filename = f"factset_segments_data_{TEST_TICKER}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    csv_path = output_dir / csv_filename
                    df.to_csv(csv_path, index=False)
                    
                    print(f"✅ Interactive HTML table saved: {html_path}")
                    print(f"✅ CSV data saved: {csv_path}")
                    print(f"📊 Table contains {len(df)} rows with segment data for {TEST_TICKER}")
                    print(f"🌐 Open the HTML file in your browser for interactive filtering and sorting")
                else:
                    print("❌ No segment data found to create table")
            else:
                print("⚠️  No segments data found")
            
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