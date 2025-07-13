#!/usr/bin/env python3
"""
Transcript Coverage Visualization Tool

Generates an interactive HTML visualization showing transcript coverage across:
- Fiscal years and quarters (columns)
- Institution categories and names (rows)
- Transcript types (Raw, Corrected, NearRealTime) with counts in cells
- Color coding: Green = has transcripts, Red = no transcripts

Output: Interactive HTML file with expandable categories
"""

import os
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from pathlib import Path
import pysmb
from pysmb.SMBConnection import SMBConnection
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Institution categories from CLAUDE.md
INSTITUTION_CATEGORIES = {
    "Canadian": {
        "institutions": ["RY-CA", "TD-CA", "BNS-CA", "BMO-CA", "CM-CA", "NA-CA", "LB-CA", "CWB-CA"],
        "names": {
            "RY-CA": "Royal Bank of Canada",
            "TD-CA": "Toronto-Dominion Bank", 
            "BNS-CA": "Bank of Nova Scotia",
            "BMO-CA": "Bank of Montreal",
            "CM-CA": "Canadian Imperial Bank of Commerce",
            "NA-CA": "National Bank of Canada",
            "LB-CA": "Laurentian Bank",
            "CWB-CA": "Canadian Western Bank"
        }
    },
    "US": {
        "institutions": ["JPM-US", "BAC-US", "WFC-US", "C-US", "GS-US", "MS-US", "USB-US", "PNC-US", "TFC-US", "COF-US", "SCHW-US"],
        "names": {
            "JPM-US": "JPMorgan Chase",
            "BAC-US": "Bank of America",
            "WFC-US": "Wells Fargo",
            "C-US": "Citigroup",
            "GS-US": "Goldman Sachs",
            "MS-US": "Morgan Stanley",
            "USB-US": "U.S. Bancorp",
            "PNC-US": "PNC Financial Services",
            "TFC-US": "Truist Financial",
            "COF-US": "Capital One",
            "SCHW-US": "Charles Schwab"
        }
    },
    "European": {
        "institutions": ["BCS-GB", "LLOY-GB", "RBS-GB", "HSBA-GB", "STAN-GB", "DBK-DE", "CBK-DE", "BNP-FR", "ACA-FR", "GLE-FR", "SAN-ES", "BBVA-ES", "ISP-IT", "UCG-IT", "ING-NL", "UBS-CH", "CSGN-CH"],
        "names": {
            "BCS-GB": "Barclays",
            "LLOY-GB": "Lloyds Banking Group",
            "RBS-GB": "NatWest Group",
            "HSBA-GB": "HSBC",
            "STAN-GB": "Standard Chartered",
            "DBK-DE": "Deutsche Bank",
            "CBK-DE": "Commerzbank",
            "BNP-FR": "BNP Paribas",
            "ACA-FR": "Credit Agricole",
            "GLE-FR": "Societe Generale",
            "SAN-ES": "Banco Santander",
            "BBVA-ES": "BBVA",
            "ISP-IT": "Intesa Sanpaolo",
            "UCG-IT": "UniCredit",
            "ING-NL": "ING Group",
            "UBS-CH": "UBS",
            "CSGN-CH": "Credit Suisse"
        }
    },
    "Insurance": {
        "institutions": ["MFC-CA", "SLF-CA", "GWO-CA", "IFC-CA", "FFH-CA", "UNH-US", "BRK.A-US", "AIG-US", "TRV-US", "PGR-US"],
        "names": {
            "MFC-CA": "Manulife Financial",
            "SLF-CA": "Sun Life Financial",
            "GWO-CA": "Great-West Lifeco",
            "IFC-CA": "Intact Financial",
            "FFH-CA": "Fairfax Financial",
            "UNH-US": "UnitedHealth Group",
            "BRK.A-US": "Berkshire Hathaway",
            "AIG-US": "American International Group",
            "TRV-US": "Travelers",
            "PGR-US": "Progressive"
        }
    },
    "Other": {
        "institutions": ["PDO-CA", "AKBM-NO"],
        "names": {
            "PDO-CA": "Premium Brands Holdings",
            "AKBM-NO": "Aker BioMarine"
        }
    }
}

class NASTranscriptScanner:
    """Scans NAS for transcript files and builds coverage data"""
    
    def __init__(self):
        # Load NAS credentials
        self.nas_username = os.getenv('NAS_USERNAME')
        self.nas_password = os.getenv('NAS_PASSWORD')
        self.nas_server_ip = os.getenv('NAS_SERVER_IP')
        self.nas_server_name = os.getenv('NAS_SERVER_NAME')
        self.nas_share_name = os.getenv('NAS_SHARE_NAME')
        self.nas_base_path = os.getenv('NAS_BASE_PATH', '')
        
        # Client machine name
        self.client_machine_name = os.getenv('CLIENT_MACHINE_NAME', 'DESKTOP')
        
        # Data structure to hold coverage information
        self.coverage_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            "Raw": 0,
            "Corrected": 0,
            "NearRealTime": 0
        })))
        
        self.years_quarters = set()
        
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
    
    def scan_transcript_folders(self, conn: SMBConnection):
        """Scan the NAS folder structure for transcripts"""
        base_path = f"{self.nas_base_path}/Outputs/Data"
        
        try:
            # List year folders
            year_folders = conn.listPath(self.nas_share_name, base_path)
            
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
                                        if type_entry.isDirectory and type_entry.filename in INSTITUTION_CATEGORIES:
                                            inst_type = type_entry.filename
                                            type_path = f"{quarter_path}/{inst_type}"
                                            
                                            # List institution folders
                                            try:
                                                inst_folders = conn.listPath(self.nas_share_name, type_path)
                                                
                                                for inst_entry in inst_folders:
                                                    if inst_entry.isDirectory:
                                                        # Extract ticker from folder name (e.g., "RY-CA_Royal_Bank_of_Canada")
                                                        ticker_match = re.match(r'^([A-Z0-9\.-]+)_', inst_entry.filename)
                                                        if ticker_match:
                                                            ticker = ticker_match.group(1)
                                                            inst_path = f"{type_path}/{inst_entry.filename}"
                                                            
                                                            # Check transcript type folders
                                                            for transcript_type in ["Raw", "Corrected", "NearRealTime"]:
                                                                type_folder_path = f"{inst_path}/{transcript_type}"
                                                                
                                                                try:
                                                                    files = conn.listPath(self.nas_share_name, type_folder_path)
                                                                    # Count XML files
                                                                    xml_count = sum(1 for f in files if f.filename.endswith('.xml') and not f.isDirectory)
                                                                    
                                                                    if xml_count > 0:
                                                                        self.coverage_data[inst_type][ticker][year_quarter][transcript_type] = xml_count
                                                                        
                                                                except Exception:
                                                                    # Folder doesn't exist or error accessing it
                                                                    pass
                                                                    
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
    <title>Transcript Coverage Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        
        h1 {
            color: #333;
            text-align: center;
        }
        
        .timestamp {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        
        th {
            background-color: #4CAF50;
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .category-header {
            background-color: #e0e0e0;
            font-weight: bold;
            cursor: pointer;
            user-select: none;
        }
        
        .category-header:hover {
            background-color: #d0d0d0;
        }
        
        .institution-row {
            display: none;
        }
        
        .institution-name {
            text-align: left;
            padding-left: 30px;
            font-size: 14px;
        }
        
        .has-transcripts {
            background-color: #c8e6c9;
        }
        
        .no-transcripts {
            background-color: #ffcdd2;
        }
        
        .transcript-counts {
            font-size: 11px;
            line-height: 1.3;
        }
        
        .count-type {
            font-weight: bold;
        }
        
        .expand-icon {
            display: inline-block;
            width: 15px;
            transition: transform 0.2s;
        }
        
        .expanded .expand-icon {
            transform: rotate(90deg);
        }
        
        .summary {
            margin: 20px 0;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        
        .legend {
            margin: 20px 0;
            padding: 10px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        .legend-item {
            display: inline-block;
            margin: 0 15px;
        }
        
        .legend-box {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #999;
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
    </script>
</head>
<body>
    <h1>Transcript Coverage Report</h1>
    <div class="timestamp">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
    
    <div class="legend">
        <strong>Legend:</strong>
        <span class="legend-item">
            <span class="legend-box has-transcripts"></span> Has Transcripts
        </span>
        <span class="legend-item">
            <span class="legend-box no-transcripts"></span> No Transcripts
        </span>
        <span class="legend-item">
            <strong>R:</strong> Raw, <strong>C:</strong> Corrected, <strong>N:</strong> NearRealTime
        </span>
    </div>
    
    <div style="margin: 10px 0;">
        <button onclick="expandAll()">Expand All</button>
        <button onclick="collapseAll()">Collapse All</button>
    </div>
    
    <table>
        <thead>
            <tr>
                <th style="text-align: left; min-width: 250px;">Institution</th>
"""
        
        # Add year-quarter headers
        for yq in sorted_year_quarters:
            html_content += f"                <th>{yq}</th>\n"
        
        html_content += """            </tr>
        </thead>
        <tbody>
"""
        
        # Generate rows for each category and institution
        for category, cat_info in INSTITUTION_CATEGORIES.items():
            # Category header row
            html_content += f"""            <tr id="{category}" class="category-header" onclick="toggleCategory('{category}')">
                <td style="text-align: left;">
                    <span class="expand-icon">▶</span> {category} ({len(cat_info['institutions'])} institutions)
                </td>
"""
            
            # Calculate category totals for each year-quarter
            for yq in sorted_year_quarters:
                total_count = 0
                for ticker in cat_info['institutions']:
                    if ticker in self.coverage_data.get(category, {}):
                        counts = self.coverage_data[category][ticker].get(yq, {})
                        total_count += sum(counts.values())
                
                cell_class = "has-transcripts" if total_count > 0 else "no-transcripts"
                html_content += f'                <td class="{cell_class}"><strong>{total_count}</strong></td>\n'
            
            html_content += "            </tr>\n"
            
            # Institution rows
            for ticker in cat_info['institutions']:
                inst_name = cat_info['names'].get(ticker, ticker)
                html_content += f"""            <tr class="{category}-inst institution-row">
                <td class="institution-name">{ticker} - {inst_name}</td>
"""
                
                for yq in sorted_year_quarters:
                    counts = self.coverage_data.get(category, {}).get(ticker, {}).get(yq, {})
                    raw_count = counts.get('Raw', 0)
                    corrected_count = counts.get('Corrected', 0)
                    nearreal_count = counts.get('NearRealTime', 0)
                    total = raw_count + corrected_count + nearreal_count
                    
                    cell_class = "has-transcripts" if total > 0 else "no-transcripts"
                    
                    if total > 0:
                        cell_content = f"""<div class="transcript-counts">
                            <span class="count-type">R:</span>{raw_count}<br>
                            <span class="count-type">C:</span>{corrected_count}<br>
                            <span class="count-type">N:</span>{nearreal_count}
                        </div>"""
                    else:
                        cell_content = "-"
                    
                    html_content += f'                <td class="{cell_class}">{cell_content}</td>\n'
                
                html_content += "            </tr>\n"
        
        html_content += """        </tbody>
    </table>
    
    <div class="summary">
        <h3>Summary Statistics</h3>
"""
        
        # Calculate summary statistics
        total_institutions = sum(len(cat['institutions']) for cat in INSTITUTION_CATEGORIES.values())
        institutions_with_data = set()
        total_transcripts = 0
        
        for category, cat_data in self.coverage_data.items():
            for ticker, ticker_data in cat_data.items():
                institutions_with_data.add(ticker)
                for yq, counts in ticker_data.items():
                    total_transcripts += sum(counts.values())
        
        html_content += f"""        <p><strong>Total Institutions Monitored:</strong> {total_institutions}</p>
        <p><strong>Institutions with Transcripts:</strong> {len(institutions_with_data)} ({len(institutions_with_data)/total_institutions*100:.1f}%)</p>
        <p><strong>Total Transcripts Found:</strong> {total_transcripts:,}</p>
        <p><strong>Date Range:</strong> {sorted_year_quarters[0] if sorted_year_quarters else 'N/A'} to {sorted_year_quarters[-1] if sorted_year_quarters else 'N/A'}</p>
    </div>
    
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report generated: {output_file}")
        print(f"\nReport generated: {output_file}")
        print(f"Total institutions monitored: {total_institutions}")
        print(f"Institutions with transcripts: {len(institutions_with_data)}")
        print(f"Total transcripts found: {total_transcripts:,}")

def main():
    """Main execution function"""
    try:
        logger.info("Starting transcript coverage analysis...")
        
        # Create scanner instance
        scanner = NASTranscriptScanner()
        
        # Connect to NAS
        conn = scanner.connect_to_nas()
        
        try:
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