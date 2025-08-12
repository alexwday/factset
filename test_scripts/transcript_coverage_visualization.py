#!/usr/bin/env python3
"""
Transcript Coverage Visualization Tool

Generates an interactive HTML visualization showing transcript coverage across:
- Fiscal years and quarters (columns)
- Institution categories and names (rows)
- Transcript types (Raw, Corrected, NearRealTime) with counts in cells
- Color coding: Green = has transcripts, Red = no transcripts

Updated to use current monitored institutions from config.yaml (91 institutions)
"""

import os
import yaml
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from pathlib import Path
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
import re
import io
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Authentication and connection settings from environment
NAS_SHARE_NAME = os.getenv('NAS_SHARE_NAME')

def nas_download_file(nas_conn: SMBConnection, file_path: str) -> bytes:
    """Download a file from NAS and return its contents."""
    file_obj = io.BytesIO()
    nas_conn.retrieveFile(NAS_SHARE_NAME, file_path, file_obj)
    return file_obj.getvalue()

def load_config() -> Dict:
    """Load configuration from local config.yaml file."""
    try:
        logger.info("Loading configuration from config.yaml...")
        config_path = Path(__file__).parent.parent / "database_refresh" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

class NASTranscriptScanner:
    """Scans NAS for transcript files and builds coverage data"""
    
    def __init__(self, config: Dict):
        # Load NAS credentials
        self.nas_username = os.getenv('NAS_USERNAME')
        self.nas_password = os.getenv('NAS_PASSWORD')
        self.nas_server_ip = os.getenv('NAS_SERVER_IP')
        self.nas_server_name = os.getenv('NAS_SERVER_NAME')
        self.nas_share_name = os.getenv('NAS_SHARE_NAME')
        
        # Client machine name
        self.client_machine_name = os.getenv('CLIENT_MACHINE_NAME', 'DESKTOP')
        
        # Store configuration
        self.config = config
        self.monitored_institutions = config.get('monitored_institutions', {})
        
        # Build institution categories from config
        self.institution_categories = self._build_categories_from_config()
        
        # Build a ticker to category mapping for quick lookups
        self.ticker_to_category = {}
        for ticker, info in self.monitored_institutions.items():
            inst_type = info.get('type', 'Other')
            # Map the type to display name
            type_display_mapping = {
                'Canadian_Banks': 'Canadian Banks',
                'US_Banks': 'US Banks',
                'European_Banks': 'European Banks',
                'US_Boutiques': 'US Boutiques',
                'Canadian_Asset_Managers': 'Canadian Asset Managers',
                'US_Regionals': 'US Regionals',
                'US_Wealth_Asset_Managers': 'US Wealth & Asset Managers',
                'UK_Wealth_Asset_Managers': 'UK Wealth & Asset Managers',
                'Nordic_Banks': 'Nordic Banks',
                'Canadian_Insurers': 'Canadian Insurers',
                'Canadian_Monoline_Lenders': 'Canadian Monoline Lenders',
                'Australian_Banks': 'Australian Banks',
                'Trusts': 'Trusts'
            }
            display_type = type_display_mapping.get(inst_type, inst_type.replace('_', ' '))
            self.ticker_to_category[ticker] = display_type
        
        # Data structure to hold coverage information
        self.coverage_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "Raw": 0,
            "Corrected": 0,
            "NearRealTime": 0
        })))
        
        # Store companies with transcripts in ignore list
        self.companies_in_ignore_list = set()
        
        # Store count of ignored transcripts per company
        self.ignored_transcript_counts = {}
        
        # Store earnings transcript counts separately
        self.earnings_transcript_counts = defaultdict(lambda: defaultdict(int))
        
        self.years_quarters = set()
    
    def _build_categories_from_config(self) -> Dict:
        """Build institution categories from configuration."""
        categories = defaultdict(lambda: {"institutions": [], "names": {}})
        
        # Create a mapping of institution type to display name
        type_display_mapping = {
            'Canadian_Banks': 'Canadian Banks',
            'US_Banks': 'US Banks',
            'European_Banks': 'European Banks',
            'US_Boutiques': 'US Boutiques',
            'Canadian_Asset_Managers': 'Canadian Asset Managers',
            'US_Regionals': 'US Regionals',
            'US_Wealth_Asset_Managers': 'US Wealth & Asset Managers',
            'UK_Wealth_Asset_Managers': 'UK Wealth & Asset Managers',
            'Nordic_Banks': 'Nordic Banks',
            'Canadian_Insurers': 'Canadian Insurers',
            'Canadian_Monoline_Lenders': 'Canadian Monoline Lenders',
            'Australian_Banks': 'Australian Banks',
            'Trusts': 'Trusts'
        }
        
        for ticker, info in self.monitored_institutions.items():
            inst_type = info.get('type', 'Other')
            display_type = type_display_mapping.get(inst_type, inst_type.replace('_', ' '))
            
            categories[display_type]["institutions"].append(ticker)
            categories[display_type]["names"][ticker] = info.get('name', ticker)
        
        # Sort institutions within each category
        for category in categories.values():
            category["institutions"].sort()
        
        # Sort categories by name for consistent display order
        sorted_categories = dict(sorted(categories.items()))
        
        return sorted_categories
        
    def connect_to_nas(self) -> SMBConnection:
        """Establish connection to NAS"""
        logger.info("Connecting to NAS...")
        conn = SMBConnection(
            self.nas_username,
            self.nas_password,
            self.client_machine_name,
            self.nas_server_name,
            use_ntlm_v2=True
        )
        
        if not conn.connect(self.nas_server_ip, 139):
            raise ConnectionError("Failed to connect to NAS")
            
        logger.info("Successfully connected to NAS")
        return conn
    
    def load_ignore_list(self, conn: SMBConnection):
        """Load the invalid transcript list from NAS to identify companies with ignored transcripts"""
        try:
            # Path to the ignore list Excel file
            base_path = os.getenv('NAS_BASE_PATH', 'Finance Data and Analytics/DSA/Earnings Call Transcripts')
            ignore_list_path = f"{base_path}/Outputs/Data/InvalidTranscripts/invalid_transcripts.xlsx"
            
            logger.info(f"Loading ignore list from: {ignore_list_path}")
            
            # Download the Excel file
            file_obj = io.BytesIO()
            conn.retrieveFile(self.nas_share_name, ignore_list_path, file_obj)
            file_obj.seek(0)
            
            # Read Excel file
            df = pd.read_excel(file_obj)
            
            # Extract unique tickers from the ignore list and count entries per ticker
            if 'ticker' in df.columns:
                unique_tickers = df['ticker'].unique()
                self.companies_in_ignore_list = set(unique_tickers)
                
                # Count transcripts per ticker in the ignore list
                ticker_counts = df.groupby('ticker').size()
                self.ignored_transcript_counts = ticker_counts.to_dict()
                
                logger.info(f"Loaded ignore list with {len(df)} entries covering {len(self.companies_in_ignore_list)} unique companies")
            else:
                logger.warning("No 'ticker' column found in ignore list")
                
        except Exception as e:
            logger.warning(f"Could not load ignore list (may not exist yet): {e}")
            self.companies_in_ignore_list = set()
    
    def scan_transcript_folders(self, conn: SMBConnection):
        """Scan the NAS folder structure for transcripts"""
        # Get the output data path from config (matches Stage 00 configuration)
        base_path = self.config.get('stage_00_download_historical', {}).get('output_data_path', 
                                    'Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Data')
        
        logger.info(f"Scanning for transcripts in base path: {base_path}")
        
        try:
            # List year folders
            year_folders = conn.listPath(self.nas_share_name, base_path)
            logger.info(f"Found {len([y for y in year_folders if y.isDirectory and y.filename not in ['.', '..']])} year folders")
            
            for year_entry in year_folders:
                if year_entry.isDirectory and year_entry.filename.isdigit() and len(year_entry.filename) == 4:
                    year = year_entry.filename
                    year_path = f"{base_path}/{year}"
                    
                    # List quarter folders
                    try:
                        quarter_folders = conn.listPath(self.nas_share_name, year_path)
                        
                        for quarter_entry in quarter_folders:
                            if quarter_entry.isDirectory and quarter_entry.filename.startswith('Q'):
                                quarter = quarter_entry.filename
                                quarter_path = f"{year_path}/{quarter}"
                                year_quarter = f"{year}-{quarter}"
                                self.years_quarters.add(year_quarter)
                                
                                # List institution type folders
                                try:
                                    type_folders = conn.listPath(self.nas_share_name, quarter_path)
                                    
                                    for type_entry in type_folders:
                                        if type_entry.isDirectory and type_entry.filename not in ['.', '..']:
                                            inst_type = type_entry.filename
                                            type_path = f"{quarter_path}/{inst_type}"
                                            logger.debug(f"Found institution type folder: {inst_type} in {year}/{quarter}")
                                            
                                            # List institution folders
                                            try:
                                                inst_folders = conn.listPath(self.nas_share_name, type_path)
                                                
                                                for inst_entry in inst_folders:
                                                    if inst_entry.isDirectory:
                                                        # Extract ticker from folder name (e.g., "RY-CA_Royal_Bank_of_Canada")
                                                        ticker_match = re.match(r'^([A-Z0-9\.-]+)_', inst_entry.filename)
                                                        if ticker_match:
                                                            ticker = ticker_match.group(1)
                                                            
                                                            # Use our ticker to category mapping
                                                            category_found = self.ticker_to_category.get(ticker)
                                                            
                                                            if category_found:
                                                                inst_path = f"{type_path}/{inst_entry.filename}"
                                                                
                                                                # Stage 00 stores XML files directly in the company folder
                                                                # Files are named: ticker_quarter_year_transcripttype_eventid_versionid.xml
                                                                try:
                                                                    files = conn.listPath(self.nas_share_name, inst_path)
                                                                    
                                                                    # Count XML files by transcript type from filename
                                                                    raw_count = 0
                                                                    corrected_count = 0
                                                                    nearreal_count = 0
                                                                    
                                                                    for f in files:
                                                                        if f.filename.endswith('.xml') and not f.isDirectory:
                                                                            # Parse transcript type from filename
                                                                            parts = f.filename[:-4].split('_')  # Remove .xml and split
                                                                            if len(parts) >= 4:
                                                                                transcript_type = parts[3]  # 4th part is transcript type
                                                                                if transcript_type == 'Raw':
                                                                                    raw_count += 1
                                                                                elif transcript_type == 'Corrected':
                                                                                    corrected_count += 1
                                                                                elif transcript_type == 'NearRealTime':
                                                                                    nearreal_count += 1
                                                                    
                                                                    # Store counts if any files found
                                                                    if raw_count > 0:
                                                                        self.coverage_data[category_found][ticker][year_quarter]["Raw"] = raw_count
                                                                    if corrected_count > 0:
                                                                        self.coverage_data[category_found][ticker][year_quarter]["Corrected"] = corrected_count
                                                                    if nearreal_count > 0:
                                                                        self.coverage_data[category_found][ticker][year_quarter]["NearRealTime"] = nearreal_count
                                                                    
                                                                    # Track earnings transcripts (all downloaded transcripts are earnings)
                                                                    total_earnings = raw_count + corrected_count + nearreal_count
                                                                    if total_earnings > 0:
                                                                        self.earnings_transcript_counts[ticker][year_quarter] = total_earnings
                                                                    
                                                                    total_files = raw_count + corrected_count + nearreal_count
                                                                    if total_files > 0:
                                                                        logger.debug(f"Found {total_files} files for {ticker} in {year_quarter}: R={raw_count}, C={corrected_count}, N={nearreal_count}")
                                                                        
                                                                except Exception as e:
                                                                    logger.debug(f"Error accessing {inst_path}: {str(e)[:50]}")
                                                            else:
                                                                logger.debug(f"Ticker {ticker} not in monitored institutions")
                                                                    
                                            except Exception as e:
                                                logger.warning(f"Error listing institution folders in {type_path}: {e}")
                                                
                                except Exception as e:
                                    logger.warning(f"Error listing type folders in {quarter_path}: {e}")
                                    
                    except Exception as e:
                        logger.warning(f"Error listing quarter folders in {year_path}: {e}")
                        
        except Exception as e:
            logger.error(f"Error scanning transcript folders: {e}")
            raise
    
    def generate_html_report(self, output_file: str = "transcript_coverage_report.html"):
        """Generate interactive HTML report with coverage visualization"""
        
        # Sort years and quarters
        sorted_year_quarters = sorted(self.years_quarters)
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Transcript Coverage Report - 91 Monitored Institutions</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }
        
        h1 {
            color: #003d82;
            text-align: center;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .timestamp {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 8px rgba(0,61,130,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        
        th, td {
            border: 1px solid #e1e5e9;
            padding: 6px 8px;
            text-align: center;
        }
        
        th {
            background-color: #003d82;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
            font-weight: 600;
            font-size: 13px;
        }
        
        .category-header {
            cursor: pointer;
            user-select: none;
            font-weight: 600;
            background-color: #f8f9fa;
        }
        
        .category-header td:first-child {
            background-color: #e8f0fe;
            border-left: 4px solid #003d82;
        }
        
        .category-header:hover td:first-child {
            background-color: #d2e3fc;
        }
        
        .institution-row {
            display: none;
        }
        
        .institution-name {
            text-align: left;
            padding-left: 25px;
            font-size: 12px;
            padding-top: 4px;
            padding-bottom: 4px;
        }
        
        .has-transcripts {
            background-color: #e8f5e8;
            color: #2e7d32;
        }
        
        .no-transcripts {
            background-color: #ffebee;
            color: #c62828;
        }
        
        .transcript-display {
            font-size: 14px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 3px;
            color: white;
        }
        
        .corrected {
            background-color: #2e7d32;
        }
        
        .raw {
            background-color: #f57c00;
        }
        
        .nearreal {
            background-color: #1976d2;
        }
        
        .expand-icon {
            display: inline-block;
            width: 15px;
            transition: transform 0.2s;
            color: #003d82;
        }
        
        .expanded .expand-icon {
            transform: rotate(90deg);
        }
        
        .summary {
            margin: 20px 0;
            padding: 20px;
            background: linear-gradient(135deg, #003d82, #0066cc);
            color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,61,130,0.2);
        }
        
        .summary h3 {
            margin-top: 0;
            color: white;
        }
        
        .legend {
            margin: 15px 0;
            padding: 12px;
            background-color: white;
            border: 1px solid #e1e5e9;
            border-radius: 6px;
            font-size: 13px;
        }
        
        .legend-item {
            display: inline-block;
            margin: 0 12px 0 0;
        }
        
        .legend-box {
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 4px;
            vertical-align: middle;
            border: 1px solid #999;
            border-radius: 2px;
        }
        
        .controls {
            margin: 15px 0;
            text-align: center;
        }
        
        .btn {
            background-color: #003d82;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 4px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .btn:hover {
            background-color: #0066cc;
        }
        
        .btn-secondary {
            background-color: #6c757d;
        }
        
        .btn-secondary:hover {
            background-color: #5a6268;
        }
        
        .category-summary {
            font-size: 12px;
            color: #666;
            font-weight: normal;
        }
    </style>
    <script>
        function toggleCategory(categoryId) {
            const header = document.getElementById(categoryId);
            const rows = document.getElementsByClassName(categoryId + '-inst');
            
            header.classList.toggle('expanded');
            
            for (let row of rows) {
                row.style.display = row.style.display === 'table-row' ? 'none' : 'table-row';
            }
        }
        
        function expandCategory(categoryName) {
            const header = document.getElementById(categoryName);
            const rows = document.getElementsByClassName(categoryName + '-inst');
            
            if (header) {
                header.classList.add('expanded');
                for (let row of rows) {
                    row.style.display = 'table-row';
                }
            }
        }
        
        function expandAll() {
            const headers = document.getElementsByClassName('category-header');
            const rows = document.getElementsByClassName('institution-row');
            
            for (let header of headers) {
                header.classList.add('expanded');
            }
            
            for (let row of rows) {
                row.style.display = 'table-row';
            }
        }
        
        function collapseAll() {
            const headers = document.getElementsByClassName('category-header');
            const rows = document.getElementsByClassName('institution-row');
            
            for (let header of headers) {
                header.classList.remove('expanded');
            }
            
            for (let row of rows) {
                row.style.display = 'none';
            }
        }
        
        // Initialize with Canadian Banks and US Banks expanded
        window.onload = function() {
            expandCategory('Canadian_Banks');
            expandCategory('US_Banks');
        }
    </script>
</head>
<body>
    <h1>Transcript Coverage Report - 91 Monitored Institutions</h1>
    <div class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    
    <div class="legend">
        <strong>Legend:</strong>
        <span class="legend-item">
            <span class="legend-box has-transcripts"></span> Has Valid Transcripts
        </span>
        <span class="legend-item">
            <span class="legend-box no-transcripts"></span> No Valid Transcripts
        </span>
        <span class="legend-item">
            <span class="legend-box" style="background-color: #e3f2fd; border-color: #1976d2;"></span> Total Transcripts (includes ignored)
        </span>
        <span class="legend-item">
            <span class="legend-box" style="background-color: #e8f5e8; border-color: #2e7d32;"></span> Earnings Transcripts Only
        </span>
        <span class="legend-item">
            <span class="transcript-display corrected" style="font-size: 12px;">C</span> Corrected
        </span>
        <span class="legend-item">
            <span class="transcript-display raw" style="font-size: 12px;">R</span> Raw
        </span>
        <span class="legend-item">
            <span class="transcript-display nearreal" style="font-size: 12px;">N</span> NearRealTime
        </span>
    </div>
    
    <div class="controls">
        <button class="btn" onclick="expandCategory('Canadian_Banks')">Canadian Banks</button>
        <button class="btn" onclick="expandCategory('US_Banks')">US Banks</button>
        <button class="btn" onclick="expandCategory('European_Banks')">European Banks</button>
        <button class="btn" onclick="expandCategory('US_Regionals')">US Regionals</button>
        <button class="btn btn-secondary" onclick="expandAll()">Expand All</button>
        <button class="btn btn-secondary" onclick="collapseAll()">Collapse All</button>
    </div>
    
    <table>
        <thead>
            <tr>
                <th style="text-align: left; min-width: 300px;">Institution</th>
                <th style="min-width: 100px; background-color: #1976d2; color: white;">Total Transcripts</th>
                <th style="min-width: 100px; background-color: #2e7d32; color: white;">Earnings Transcripts</th>
"""
        
        # Add year-quarter headers
        for yq in sorted_year_quarters:
            html_content += f"                <th>{yq}</th>\n"
        
        html_content += """            </tr>
        </thead>
        <tbody>
"""
        
        # Generate rows for each category and institution
        for category, cat_info in self.institution_categories.items():
            # Create safe category ID by replacing spaces and special chars
            safe_category_id = category.replace(' ', '_').replace('&', 'and')
            
            # Category header row
            html_content += f"""            <tr id="{safe_category_id}" class="category-header" onclick="toggleCategory('{safe_category_id}')">
                <td style="text-align: left;">
                    <span class="expand-icon">▶</span> {category} ({len(cat_info['institutions'])} institutions)
                </td>
"""
            
            # Calculate total and earnings transcript counts for this category
            cat_total_transcripts = 0
            cat_earnings_transcripts = 0
            
            for ticker in cat_info['institutions']:
                # Count earnings transcripts (downloaded)
                for yq_data in self.earnings_transcript_counts.get(ticker, {}).values():
                    cat_earnings_transcripts += yq_data
                
                # Count ignored transcripts
                ignored_count = self.ignored_transcript_counts.get(ticker, 0)
                cat_total_transcripts += ignored_count
            
            # Add earnings to total
            cat_total_transcripts += cat_earnings_transcripts
            
            # Display total transcripts
            if cat_total_transcripts > 0:
                html_content += f'                <td style="background-color: #e3f2fd; color: #1976d2; font-weight: bold;">{cat_total_transcripts}</td>\n'
            else:
                html_content += '                <td style="background-color: #f8f8f8;">-</td>\n'
            
            # Display earnings transcripts
            if cat_earnings_transcripts > 0:
                html_content += f'                <td style="background-color: #e8f5e8; color: #2e7d32; font-weight: bold;">{cat_earnings_transcripts}</td>\n'
            else:
                html_content += '                <td style="background-color: #f8f8f8;">-</td>\n'
            
            # Calculate category totals for each year-quarter
            for yq in sorted_year_quarters:
                total_count = 0
                for ticker in cat_info['institutions']:
                    if ticker in self.coverage_data.get(category, {}):
                        counts = self.coverage_data[category][ticker].get(yq, {})
                        total_count += sum(counts.values())
                
                cell_class = "has-transcripts" if total_count > 0 else "no-transcripts"
                tooltip = f"Total files for all {category} institutions in {yq}"
                html_content += f'                <td class="{cell_class}" title="{tooltip}"><strong>{total_count}</strong></td>\n'
            
            html_content += "            </tr>\n"
            
            # Institution rows
            for ticker in cat_info['institutions']:
                inst_name = cat_info['names'].get(ticker, ticker)
                html_content += f"""            <tr class="{safe_category_id}-inst institution-row">
                <td class="institution-name">{ticker} - {inst_name}</td>
"""
                
                # Calculate total transcripts for this institution
                inst_earnings_total = sum(self.earnings_transcript_counts.get(ticker, {}).values())
                inst_ignored_total = self.ignored_transcript_counts.get(ticker, 0)
                inst_total = inst_earnings_total + inst_ignored_total
                
                # Total transcripts column
                if inst_total > 0:
                    html_content += f'                <td style="background-color: #e3f2fd; color: #1976d2; text-align: center;">{inst_total}</td>\n'
                else:
                    html_content += '                <td style="background-color: #f8f8f8; text-align: center;">-</td>\n'
                
                # Earnings transcripts column  
                if inst_earnings_total > 0:
                    html_content += f'                <td style="background-color: #e8f5e8; color: #2e7d32; text-align: center;">{inst_earnings_total}</td>\n'
                else:
                    html_content += '                <td style="background-color: #f8f8f8; text-align: center;">-</td>\n'
                
                for yq in sorted_year_quarters:
                    counts = self.coverage_data.get(category, {}).get(ticker, {}).get(yq, {})
                    raw_count = counts.get('Raw', 0)
                    corrected_count = counts.get('Corrected', 0)
                    nearreal_count = counts.get('NearRealTime', 0)
                    total = raw_count + corrected_count + nearreal_count
                    
                    cell_class = "has-transcripts" if total > 0 else "no-transcripts"
                    
                    if total > 0:
                        # Priority-based display: C > R > N
                        if corrected_count > 0:
                            cell_content = f'<span class="transcript-display corrected">C:{corrected_count}</span>'
                        elif raw_count > 0:
                            cell_content = f'<span class="transcript-display raw">R:{raw_count}</span>'
                        else:
                            cell_content = f'<span class="transcript-display nearreal">N:{nearreal_count}</span>'
                        
                        # Add tooltip with all counts
                        tooltip = f"Raw:{raw_count}, Corrected:{corrected_count}, NearRealTime:{nearreal_count}"
                        html_content += f'                <td class="{cell_class}" title="{tooltip}">{cell_content}</td>\n'
                    else:
                        html_content += f'                <td class="{cell_class}">-</td>\n'
                
                html_content += "            </tr>\n"
        
        html_content += """        </tbody>
    </table>
    
    <div class="summary">
        <h3>Summary Statistics</h3>
"""
        
        # Calculate comprehensive summary statistics
        total_institutions = sum(len(cat['institutions']) for cat in self.institution_categories.values())
        
        # Count institutions with earnings transcripts
        institutions_with_earnings = set()
        total_earnings_transcripts = 0
        earnings_by_type = {"Raw": 0, "Corrected": 0, "NearRealTime": 0}
        
        for category, cat_data in self.coverage_data.items():
            for ticker, ticker_data in cat_data.items():
                institutions_with_earnings.add(ticker)
                for yq, counts in ticker_data.items():
                    for transcript_type, count in counts.items():
                        total_earnings_transcripts += count
                        earnings_by_type[transcript_type] += count
        
        # Count institutions with any transcripts (earnings or ignored)
        institutions_with_any_transcripts = institutions_with_earnings.union(self.companies_in_ignore_list)
        
        # Count institutions with no transcripts at all
        all_tickers = set()
        for cat_info in self.institution_categories.values():
            all_tickers.update(cat_info['institutions'])
        institutions_with_no_coverage = all_tickers - institutions_with_any_transcripts
        
        # Calculate total ignored transcripts
        total_ignored_transcripts = sum(self.ignored_transcript_counts.values())
        
        # Calculate total transcripts (earnings + ignored)
        total_all_transcripts = total_earnings_transcripts + total_ignored_transcripts
        
        # Calculate percentages
        pct_with_any = (len(institutions_with_any_transcripts) / total_institutions * 100) if total_institutions > 0 else 0
        pct_with_earnings = (len(institutions_with_earnings) / total_institutions * 100) if total_institutions > 0 else 0
        pct_no_coverage = (len(institutions_with_no_coverage) / total_institutions * 100) if total_institutions > 0 else 0
        
        html_content += f"""        <h4>Transcript Counts:</h4>
        <p><strong>Total Transcripts (All Types):</strong> {total_all_transcripts:,}</p>
        <p style="margin-left: 20px;">• <strong>Earnings Call Transcripts:</strong> {total_earnings_transcripts:,}</p>
        <p style="margin-left: 40px;">- Raw: {earnings_by_type['Raw']:,}</p>
        <p style="margin-left: 40px;">- Corrected: {earnings_by_type['Corrected']:,}</p>
        <p style="margin-left: 40px;">- NearRealTime: {earnings_by_type['NearRealTime']:,}</p>
        <p style="margin-left: 20px;">• <strong>Ignored Transcripts (Non-Earnings):</strong> {total_ignored_transcripts:,}</p>
        
        <h4>Institution Coverage:</h4>
        <p><strong>Total Institutions Monitored:</strong> {total_institutions}</p>
        <p><strong>Institutions with Any Transcripts:</strong> {len(institutions_with_any_transcripts)} ({pct_with_any:.1f}%)</p>
        <p><strong>Institutions with Earnings Call Transcripts:</strong> {len(institutions_with_earnings)} ({pct_with_earnings:.1f}%)</p>
        <p><strong>Institutions with No Coverage:</strong> {len(institutions_with_no_coverage)} ({pct_no_coverage:.1f}%)</p>
        
        <h4>Date Range:</h4>
        <p><strong>Coverage Period:</strong> {sorted_year_quarters[0] if sorted_year_quarters else 'N/A'} to {sorted_year_quarters[-1] if sorted_year_quarters else 'N/A'}</p>
    </div>
    
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        print(f"\nReport generated: {output_file}")
        print(f"\n=== TRANSCRIPT COUNTS ===")
        print(f"Total Transcripts (All Types): {total_all_transcripts:,}")
        print(f"  • Earnings Call Transcripts: {total_earnings_transcripts:,}")
        print(f"    - Raw: {earnings_by_type['Raw']:,}")
        print(f"    - Corrected: {earnings_by_type['Corrected']:,}")
        print(f"    - NearRealTime: {earnings_by_type['NearRealTime']:,}")
        print(f"  • Ignored Transcripts (Non-Earnings): {total_ignored_transcripts:,}")
        print(f"\n=== INSTITUTION COVERAGE ===")
        print(f"Total Institutions Monitored: {total_institutions}")
        print(f"Institutions with Any Transcripts: {len(institutions_with_any_transcripts)} ({pct_with_any:.1f}%)")
        print(f"Institutions with Earnings Call Transcripts: {len(institutions_with_earnings)} ({pct_with_earnings:.1f}%)")
        print(f"Institutions with No Coverage: {len(institutions_with_no_coverage)} ({pct_no_coverage:.1f}%)")

def main():
    """Main execution function"""
    try:
        logger.info("Starting transcript coverage analysis...")
        
        # Load configuration from local config.yaml
        config = load_config()
        
        # Create scanner instance with config
        scanner = NASTranscriptScanner(config)
        
        # Connect to NAS for scanning
        conn = scanner.connect_to_nas()
        
        try:
            # Load ignore list first
            logger.info("Loading ignore list from NAS...")
            scanner.load_ignore_list(conn)
            
            # Scan transcript folders
            logger.info("Scanning transcript folders on NAS...")
            scanner.scan_transcript_folders(conn)
            
            # Generate HTML report
            logger.info("Generating HTML report...")
            scanner.generate_html_report()
            
        finally:
            # Close NAS connection
            conn.close()
            logger.info("NAS connection closed")
        
        logger.info("Coverage analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during coverage analysis: {e}")
        raise

if __name__ == "__main__":
    main()