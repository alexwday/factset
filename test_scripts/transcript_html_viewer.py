#!/usr/bin/env python3
"""
Enhanced FactSet Transcript XML to HTML and PDF Viewer
Professional-grade financial document viewer with:
- WCAG 2.1 AA compliance
- Modern typography system
- Search functionality
- Print/export capabilities
- Enhanced color system
- Paragraph-level copy functionality
- Clean PDF generation

Part of the FactSet Earnings Transcript Pipeline test suite.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
import html
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

def load_monitored_institutions() -> Dict:
    """Load monitored institutions from config file."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('monitored_institutions', {})
    except FileNotFoundError:
        print("Warning: config.json not found. Logo integration will be limited.")
        return {}

def get_company_logo_url(company_name: str, institutions: Dict) -> str:
    """Get logo URL for a company based on monitored institutions."""
    # Try to match company name to ticker
    for ticker, info in institutions.items():
        if company_name.lower() in info['name'].lower() or info['name'].lower() in company_name.lower():
            # For demo purposes, using a placeholder service
            # In production, you'd integrate with a logo service or maintain local logo files
            return f"https://logo.clearbit.com/{ticker.split('-')[0].lower()}.com"
    
    # Default placeholder for unknown companies
    return "https://via.placeholder.com/120x60/003366/FFFFFF?text=LOGO"

def detect_primary_company(transcript_data: Dict, institutions: Dict) -> Tuple[str, str]:
    """Detect the primary company from transcript data."""
    companies = transcript_data.get('companies', [])
    if not companies:
        return "Unknown Company", "https://via.placeholder.com/120x60/003366/FFFFFF?text=LOGO"
    
    primary_company = companies[0]
    logo_url = get_company_logo_url(primary_company, institutions)
    
    return primary_company, logo_url

def parse_transcript_xml(xml_path: str) -> Dict:
    """Parse FactSet transcript XML and extract structured data."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Handle namespace if present
    namespace = ""
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'
    
    # Helper function to handle namespaced tags
    def ns_tag(tag):
        return f"{namespace}{tag}" if namespace else tag
    
    # Extract metadata
    meta = root.find(ns_tag('meta'))
    if meta is None:
        # Debug: print available elements
        print(f"Root tag: {root.tag}")
        print(f"Root children: {[child.tag for child in root]}")
        raise ValueError("No meta section found in XML")
    
    # Extract title and date
    title = meta.find(ns_tag('title'))
    title_text = title.text if title is not None else "Untitled Transcript"
    
    date = meta.find(ns_tag('date'))
    date_text = date.text if date is not None else "Unknown Date"
    
    # Extract companies
    companies = []
    companies_elem = meta.find(ns_tag('companies'))
    if companies_elem is not None:
        for company in companies_elem.findall(ns_tag('company')):
            companies.append(company.text if company.text else "")
    
    # Extract participants and create speaker mapping
    participants = {}
    participants_elem = meta.find(ns_tag('participants'))
    if participants_elem is not None:
        for participant in participants_elem.findall(ns_tag('participant')):
            p_id = participant.get('id')
            if p_id:
                participants[p_id] = {
                    'type': participant.get('type', ''),
                    'affiliation': participant.get('affiliation', ''),
                    'affiliation_entity': participant.get('affiliation_entity', ''),
                    'title': participant.get('title', ''),
                    'entity': participant.get('entity', ''),
                    'name': participant.text.strip() if participant.text else 'Unknown Speaker'
                }
    
    # Extract body content
    body = root.find(ns_tag('body'))
    if body is None:
        raise ValueError("No body section found in XML")
    
    sections = []
    for section in body.findall(ns_tag('section')):
        section_name = section.get('name', 'Unnamed Section')
        speakers = []
        
        for speaker in section.findall(ns_tag('speaker')):
            speaker_id = speaker.get('id')
            speaker_type = speaker.get('type', '')  # 'q' or 'a' for Q&A sections
            
            # Extract paragraphs from plist or directly from speaker
            paragraphs = []
            plist = speaker.find(ns_tag('plist'))
            if plist is not None:
                # Standard structure: <speaker><plist><p>content</p></plist></speaker>
                for p in plist.findall(ns_tag('p')):
                    if p.text:
                        paragraphs.append(p.text.strip())
            else:
                # Alternative structure (NRT): <speaker><p>content</p></speaker>
                for p in speaker.findall(ns_tag('p')):
                    if p.text:
                        paragraphs.append(p.text.strip())
            
            speakers.append({
                'id': speaker_id,
                'type': speaker_type,
                'paragraphs': paragraphs
            })
        
        sections.append({
            'name': section_name,
            'speakers': speakers
        })
    
    return {
        'title': title_text,
        'date': date_text,
        'companies': companies,
        'participants': participants,
        'sections': sections
    }

def generate_html(transcript_data: Dict) -> str:
    """Generate enhanced HTML from parsed transcript data."""
    
    # Load monitored institutions for logo integration
    institutions = load_monitored_institutions()
    primary_company, logo_url = detect_primary_company(transcript_data, institutions)
    
    # Enhanced HTML template with comprehensive improvements
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* Enhanced Financial Document Styling - WCAG 2.1 AA Compliant */
        :root {{
            /* Primary Brand Colors - High Contrast */
            --primary-color: #003366;
            --primary-light: #0066CC;
            --primary-dark: #001B33;
            
            /* Financial Status Colors - Accessible */
            --positive-color: #00A651;
            --negative-color: #E31C23;
            --neutral-color: #6C757D;
            --warning-color: #FF8C00;
            
            /* Accessible Background Colors */
            --bg-primary: #FFFFFF;
            --bg-secondary: #F8F9FA;
            --bg-tertiary: #E9ECEF;
            --bg-accent: #F0F8FF;
            
            /* High-Contrast Text Colors */
            --text-primary: #212529;
            --text-secondary: #495057;
            --text-muted: #6C757D;
            --text-inverse: #FFFFFF;
            
            /* Enhanced Color System */
            --border-color: #DEE2E6;
            --shadow-color: rgba(0, 0, 0, 0.15);
            --focus-color: #0066CC;
            --hover-color: #E8F4FD;
            
            /* Typography System */
            --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            --font-mono: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', monospace;
            
            /* Responsive Font Scaling */
            --font-size-base: 17px;
            --font-size-large: 19px;
            --font-size-small: 15px;
            --font-size-xl: 24px;
            --font-size-xxl: 32px;
            
            /* Line Height Ratios */
            --line-height-base: 1.6;
            --line-height-heading: 1.3;
            --line-height-caption: 1.4;
            
            /* Spacing System */
            --spacing-xs: 4px;
            --spacing-sm: 8px;
            --spacing-md: 16px;
            --spacing-lg: 24px;
            --spacing-xl: 32px;
            --spacing-xxl: 48px;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: var(--font-primary);
            font-size: var(--font-size-base);
            line-height: var(--line-height-base);
            color: var(--text-primary);
            background: var(--bg-secondary);
            min-height: 100vh;
            padding: var(--spacing-lg);
            margin: 0;
            -webkit-text-size-adjust: 100%;
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--bg-primary);
            border-radius: 12px;
            box-shadow: 0 8px 32px var(--shadow-color);
            overflow: hidden;
            position: relative;
        }}
        
        /* Skip Links for Accessibility */
        .skip-links {{
            position: absolute;
            top: -40px;
            left: 6px;
            z-index: 1000;
        }}
        
        .skip-link {{
            position: absolute;
            top: 0;
            left: 0;
            background: var(--primary-color);
            color: var(--text-inverse);
            padding: var(--spacing-sm) var(--spacing-md);
            text-decoration: none;
            font-weight: 600;
            border-radius: 4px;
            transform: translateY(-100%);
            transition: transform 0.3s ease;
            font-size: var(--font-size-small);
        }}
        
        .skip-link:focus {{
            transform: translateY(0);
            outline: 2px solid var(--focus-color);
            outline-offset: 2px;
        }}
        
        /* Header Section */
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
            color: var(--text-inverse);
            padding: var(--spacing-xl) var(--spacing-xxl);
            position: relative;
            overflow: hidden;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: rotate(45deg);
        }}
        
        .header-content {{
            position: relative;
            z-index: 1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: var(--spacing-lg);
        }}
        
        .company-logo {{
            max-width: 120px;
            max-height: 60px;
            background: var(--bg-primary);
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .title-section h1 {{
            font-size: var(--font-size-xxl);
            font-weight: 700;
            margin-bottom: var(--spacing-sm);
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            line-height: var(--line-height-heading);
        }}
        
        .title-section .company-name {{
            font-size: var(--font-size-large);
            opacity: 0.9;
            font-weight: 300;
            line-height: var(--line-height-caption);
        }}
        
        /* Search and Navigation Bar */
        .search-nav {{
            background: var(--bg-accent);
            padding: var(--spacing-md) var(--spacing-xxl);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
            flex-wrap: wrap;
        }}
        
        .search-container {{
            flex: 1;
            min-width: 300px;
            position: relative;
        }}
        
        .search-input {{
            width: 100%;
            padding: var(--spacing-sm) var(--spacing-md);
            padding-left: 40px;
            border: 2px solid var(--border-color);
            border-radius: 6px;
            font-size: var(--font-size-base);
            font-family: var(--font-primary);
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .search-input:focus {{
            outline: none;
            border-color: var(--focus-color);
            box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1);
        }}
        
        .search-icon {{
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 16px;
        }}
        
        .export-buttons {{
            display: flex;
            gap: var(--spacing-sm);
            flex-wrap: wrap;
        }}
        
        .export-btn {{
            padding: var(--spacing-sm) var(--spacing-md);
            background: var(--primary-color);
            color: var(--text-inverse);
            border: none;
            border-radius: 6px;
            font-size: var(--font-size-small);
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }}
        
        .export-btn:hover {{
            background: var(--primary-light);
            transform: translateY(-1px);
        }}
        
        .export-btn:focus {{
            outline: 2px solid var(--focus-color);
            outline-offset: 2px;
        }}
        
        /* Metadata Section */
        .metadata {{
            background: var(--bg-secondary);
            padding: var(--spacing-lg) var(--spacing-xxl);
            border-bottom: 1px solid var(--border-color);
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-lg);
            margin-bottom: var(--spacing-lg);
        }}
        
        .metadata-item {{
            background: var(--bg-primary);
            padding: var(--spacing-md);
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .metadata-item strong {{
            color: var(--primary-color);
            font-weight: 600;
        }}
        
        /* Expandable Sections */
        .expandable-section {{
            margin-bottom: var(--spacing-lg);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .section-toggle {{
            background: var(--primary-color);
            color: var(--text-inverse);
            padding: var(--spacing-md) var(--spacing-lg);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: var(--font-size-large);
            transition: all 0.3s ease;
            border: none;
            width: 100%;
            text-align: left;
        }}
        
        .section-toggle:focus {{
            outline: 2px solid var(--focus-color);
            outline-offset: 2px;
        }}
        
        .section-toggle:hover {{
            background: var(--primary-light);
            transform: translateY(-1px);
        }}
        
        .section-toggle.active {{
            background: var(--primary-light);
        }}
        
        .toggle-icon {{
            transition: transform 0.3s ease;
            font-size: 1.2em;
        }}
        
        .section-toggle.active .toggle-icon {{
            transform: rotate(180deg);
        }}
        
        .section-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: var(--bg-primary);
        }}
        
        .section-content.active {{
            max-height: none;
            overflow: visible;
            padding: var(--spacing-lg) var(--spacing-lg);
        }}
        
        /* Participants Section - Compact */
        .participants-list {{
            display: flex;
            flex-direction: column;
            gap: var(--spacing-sm);
        }}
        
        .participant-item {{
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: var(--spacing-sm) var(--spacing-md);
            font-size: var(--font-size-small);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .participant-info {{
            flex: 1;
        }}
        
        .participant-name {{
            font-weight: 600;
            color: var(--primary-color);
        }}
        
        .participant-title {{
            color: var(--text-muted);
            font-size: 0.85em;
        }}
        
        .participant-type {{
            background: var(--primary-color);
            color: var(--text-inverse);
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 500;
        }}
        
        /* Transcript Content with Margin Layout */
        .transcript-section {{
            margin-bottom: var(--spacing-xl);
        }}
        
        .speaker-block {{
            margin-bottom: var(--spacing-lg);
            display: flex;
            gap: var(--spacing-lg);
            align-items: flex-start;
        }}
        
        .speaker-margin {{
            flex: 0 0 200px;
            padding: var(--spacing-md);
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
            position: sticky;
            top: 20px;
        }}
        
        .speaker-name {{
            font-weight: 600;
            color: var(--primary-color);
            font-size: var(--font-size-base);
            margin-bottom: var(--spacing-xs);
        }}
        
        .speaker-title {{
            color: var(--text-muted);
            font-size: 0.85em;
            margin-bottom: var(--spacing-sm);
            line-height: 1.3;
        }}
        
        .interaction-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-block;
        }}
        
        .question-badge {{
            background: var(--warning-color);
            color: var(--text-primary);
        }}
        
        .answer-badge {{
            background: var(--positive-color);
            color: var(--text-inverse);
        }}
        
        .speaker-content {{
            flex: 1;
            background: var(--bg-primary);
            border-radius: 8px;
            padding: var(--spacing-lg);
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            position: relative;
        }}
        
        .speaker-content p {{
            margin-bottom: var(--spacing-md);
            text-align: justify;
            line-height: var(--line-height-base);
            font-size: var(--font-size-base);
            position: relative;
            padding-right: 40px;
        }}
        
        .speaker-content p:last-child {{
            margin-bottom: 0;
        }}
        
        .copy-btn {{
            position: absolute;
            right: 8px;
            top: 8px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0;
            transition: all 0.3s ease;
            color: var(--text-secondary);
            font-weight: 500;
        }}
        
        .speaker-content p:hover .copy-btn {{
            opacity: 1;
        }}
        
        .copy-btn:hover {{
            background: var(--hover-color);
            border-color: var(--primary-color);
            color: var(--primary-color);
        }}
        
        .copy-btn:focus {{
            outline: 2px solid var(--focus-color);
            outline-offset: 2px;
            opacity: 1;
        }}
        
        .copy-btn.copied {{
            background: var(--positive-color);
            color: var(--text-inverse);
            border-color: var(--positive-color);
        }}
        
        /* Search Highlighting */
        .search-highlight {{
            background: #FFEB3B;
            padding: 1px 2px;
            border-radius: 2px;
            font-weight: 600;
        }}
        
        .search-results {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 6px 6px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        }}
        
        .search-result-item {{
            padding: var(--spacing-sm) var(--spacing-md);
            cursor: pointer;
            border-bottom: 1px solid var(--border-color);
            font-size: var(--font-size-small);
        }}
        
        .search-result-item:hover {{
            background: var(--hover-color);
        }}
        
        .search-result-item:last-child {{
            border-bottom: none;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            body {{
                padding: var(--spacing-sm);
            }}
            
            .header {{
                padding: var(--spacing-lg);
            }}
            
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}
            
            .title-section h1 {{
                font-size: var(--font-size-xl);
            }}
            
            .metadata {{
                padding: var(--spacing-lg);
            }}
            
            .metadata-grid {{
                grid-template-columns: 1fr;
            }}
            
            .section-content.active {{
                padding: var(--spacing-md);
            }}
            
            .speaker-block {{
                flex-direction: column;
                gap: var(--spacing-md);
            }}
            
            .speaker-margin {{
                flex: none;
                position: static;
            }}
            
            .search-nav {{
                padding: var(--spacing-md);
            }}
            
            .search-container {{
                min-width: 100%;
            }}
            
            .export-buttons {{
                width: 100%;
                justify-content: center;
            }}
            
            .speaker-content p {{
                padding-right: 0;
            }}
            
            .copy-btn {{
                position: static;
                display: block;
                margin-top: var(--spacing-sm);
                opacity: 1;
                width: fit-content;
            }}
        }}
        
        /* Print Styles */
        @media print {{
            body {{
                background: white;
                padding: 0;
                font-size: 12pt;
                line-height: 1.4;
            }}
            
            .container {{
                box-shadow: none;
                border-radius: 0;
                max-width: none;
            }}
            
            .header {{
                background: white !important;
                color: black !important;
                padding: 20pt;
                border-bottom: 2pt solid black;
            }}
            
            .search-nav {{
                display: none;
            }}
            
            .section-toggle {{
                display: none;
            }}
            
            .section-content {{
                max-height: none !important;
                overflow: visible !important;
                padding: 0 !important;
            }}
            
            .expandable-section {{
                page-break-inside: avoid;
                margin-bottom: 20pt;
            }}
            
            .speaker-block {{
                page-break-inside: avoid;
                margin-bottom: 15pt;
            }}
            
            .copy-btn {{
                display: none;
            }}
            
            .participant-item {{
                page-break-inside: avoid;
            }}
            
            .speaker-content p {{
                text-align: left;
                padding-right: 0;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Skip Links for Accessibility -->
        <div class="skip-links">
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <a href="#search" class="skip-link">Skip to search</a>
        </div>
        
        <!-- Header Section -->
        <header class="header" role="banner">
            <div class="header-content">
                <div class="title-section">
                    <h1>{title}</h1>
                    <div class="company-name">{primary_company}</div>
                </div>
                <img src="{logo_url}" alt="{primary_company} Logo" class="company-logo" 
                     onerror="this.src='https://via.placeholder.com/120x60/003366/FFFFFF?text=LOGO'">
            </div>
        </header>
        
        <!-- Search and Navigation -->
        <nav class="search-nav" role="navigation" aria-label="Search and Export Tools">
            <div class="search-container">
                <div style="position: relative;">
                    <span class="search-icon">🔍</span>
                    <input type="text" id="search" class="search-input" placeholder="Search transcript content..." 
                           aria-label="Search transcript content" autocomplete="off">
                    <div class="search-results" id="search-results" role="listbox" aria-hidden="true"></div>
                </div>
            </div>
            <div class="export-buttons">
                <button class="export-btn" onclick="window.print()" aria-label="Print transcript">
                    🖨️ Print
                </button>
                <button class="export-btn" onclick="exportToPDF()" aria-label="Export to PDF">
                    📄 Export PDF
                </button>
                <button class="export-btn" onclick="exportToWord()" aria-label="Export to Word">
                    📝 Export Word
                </button>
            </div>
        </nav>
        
        <!-- Metadata Section -->
        <section class="metadata" role="region" aria-labelledby="metadata-heading">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Date:</strong> {date}
                </div>
            </div>
            
            <!-- Participants Section -->
            <div class="expandable-section">
                <button class="section-toggle" onclick="toggleSection('participants')" 
                        aria-expanded="false" aria-controls="participants-content">
                    <span>Participants ({participants_count})</span>
                    <span class="toggle-icon">▼</span>
                </button>
                <div class="section-content" id="participants-content" role="region" 
                     aria-labelledby="participants-heading">
                    <div class="participants-list">
                        {participants_html}
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Transcript Sections -->
        <main id="main-content" style="padding: var(--spacing-lg);" role="main">
            {sections_html}
        </main>
    </div>
    
    <script>
        // Enhanced JavaScript with accessibility and functionality
        
        // Search functionality
        let searchTimeout;
        const searchInput = document.getElementById('search');
        const searchResults = document.getElementById('search-results');
        let allTextContent = [];
        
        // Initialize search data
        function initializeSearch() {{
            const speakers = document.querySelectorAll('.speaker-block');
            speakers.forEach((speaker, index) => {{
                const speakerName = speaker.querySelector('.speaker-name').textContent;
                const speakerTitle = speaker.querySelector('.speaker-title').textContent;
                const paragraphs = speaker.querySelectorAll('.speaker-content p');
                
                paragraphs.forEach((p, pIndex) => {{
                    allTextContent.push({{
                        element: p,
                        text: p.textContent,
                        speaker: speakerName,
                        title: speakerTitle,
                        id: `speaker-${{index}}-p-${{pIndex}}`
                    }});
                }});
            }});
        }}
        
        // Search function
        function performSearch(query) {{
            if (query.length < 2) {{
                searchResults.style.display = 'none';
                clearHighlights();
                return;
            }}
            
            const results = allTextContent.filter(item => 
                item.text.toLowerCase().includes(query.toLowerCase())
            );
            
            if (results.length > 0) {{
                showSearchResults(results, query);
                highlightSearchTerm(query);
            }} else {{
                searchResults.style.display = 'none';
                clearHighlights();
            }}
        }}
        
        // Show search results
        function showSearchResults(results, query) {{
            searchResults.innerHTML = '';
            results.slice(0, 5).forEach(result => {{
                const item = document.createElement('div');
                item.className = 'search-result-item';
                item.innerHTML = `<strong>${{result.speaker}}</strong>: ${{result.text.substring(0, 100)}}...`;
                item.onclick = () => {{
                    result.element.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    searchResults.style.display = 'none';
                    result.element.style.background = 'var(--bg-accent)';
                    setTimeout(() => {{
                        result.element.style.background = '';
                    }}, 2000);
                }};
                searchResults.appendChild(item);
            }});
            searchResults.style.display = 'block';
        }}
        
        // Highlight search terms
        function highlightSearchTerm(query) {{
            clearHighlights();
            const regex = new RegExp(`(${{query}})`, 'gi');
            
            allTextContent.forEach(item => {{
                if (item.text.toLowerCase().includes(query.toLowerCase())) {{
                    const highlighted = item.text.replace(regex, '<span class="search-highlight">$1</span>');
                    item.element.innerHTML = highlighted + item.element.querySelector('.copy-btn').outerHTML;
                }}
            }});
        }}
        
        // Clear highlights
        function clearHighlights() {{
            document.querySelectorAll('.search-highlight').forEach(el => {{
                el.outerHTML = el.textContent;
            }});
        }}
        
        // Search input event handlers
        searchInput.addEventListener('input', function() {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                performSearch(this.value);
            }}, 300);
        }});
        
        searchInput.addEventListener('blur', function() {{
            setTimeout(() => {{
                searchResults.style.display = 'none';
            }}, 200);
        }});
        
        // Copy functionality
        function copyToClipboard(speakerData, paragraphText) {{
            const formattedText = `${{speakerData.name}}\\n${{speakerData.title}}, ${{speakerData.affiliation}}\\n\\n${{paragraphText}}`;
            
            navigator.clipboard.writeText(formattedText).then(() => {{
                // Show success feedback
                const button = event.target;
                const originalText = button.textContent;
                button.textContent = '✓ Copied';
                button.classList.add('copied');
                
                setTimeout(() => {{
                    button.textContent = originalText;
                    button.classList.remove('copied');
                }}, 2000);
            }}).catch(err => {{
                console.error('Failed to copy text: ', err);
                alert('Failed to copy text. Please try again.');
            }});
        }}
        
        // Section toggle functionality
        function toggleSection(sectionId) {{
            const content = document.getElementById(sectionId + '-content');
            const button = content.previousElementSibling;
            const isActive = content.classList.contains('active');
            
            if (isActive) {{
                content.classList.remove('active');
                button.classList.remove('active');
                button.setAttribute('aria-expanded', 'false');
            }} else {{
                content.classList.add('active');
                button.classList.add('active');
                button.setAttribute('aria-expanded', 'true');
            }}
        }}
        
        // Export functions
        function exportToPDF() {{
            window.print();
        }}
        
        function exportToWord() {{
            const content = document.querySelector('.container').innerHTML;
            const blob = new Blob([content], {{
                type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcript.doc';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {{
            initializeSearch();
            
            // Add keyboard navigation
            document.addEventListener('keydown', function(e) {{
                if (e.ctrlKey && e.key === 'f') {{
                    e.preventDefault();
                    searchInput.focus();
                }}
            }});
        }});
    </script>
</body>
</html>"""
    
    # Generate participants HTML
    participants_html = ""
    participants_count = len(transcript_data['participants'])
    
    for p_id, p_data in transcript_data['participants'].items():
        participant_type = "Company" if p_data['type'] == 'C' else "Analyst" if p_data['type'] == 'A' else p_data['type']
        participants_html += f"""
        <div class="participant-item">
            <div class="participant-info">
                <div class="participant-name">{html.escape(p_data['name'])}</div>
                <div class="participant-title">{html.escape(p_data['title'])}, {html.escape(p_data['affiliation'])}</div>
            </div>
            <span class="participant-type">{participant_type}</span>
        </div>"""
    
    # Generate sections HTML with enhanced functionality
    sections_html = ""
    for section_index, section in enumerate(transcript_data['sections']):
        section_id = f"section-{section_index}"
        section_html = f"""
        <section class="expandable-section" role="region" aria-labelledby="{section_id}-heading">
            <button class="section-toggle" onclick="toggleSection('{section_id}')" 
                    aria-expanded="false" aria-controls="{section_id}-content" id="{section_id}-heading">
                <span>{html.escape(section['name'])}</span>
                <span class="toggle-icon">▼</span>
            </button>
            <div class="section-content" id="{section_id}-content">
                <div class="transcript-section">
        """
        
        for speaker_index, speaker in enumerate(section['speakers']):
            speaker_id = speaker['id']
            speaker_data = transcript_data['participants'].get(speaker_id, {})
            speaker_name = speaker_data.get('name', 'Unknown Speaker')
            speaker_title = speaker_data.get('title', '')
            speaker_affiliation = speaker_data.get('affiliation', '')
            
            # Determine interaction type and styling
            interaction_badge = ""
            if speaker['type'] == 'q':
                interaction_badge = '<span class="interaction-badge question-badge">Question</span>'
            elif speaker['type'] == 'a':
                interaction_badge = '<span class="interaction-badge answer-badge">Answer</span>'
            
            # Build speaker block with enhanced copy functionality
            speaker_html = f"""
                    <div class="speaker-block">
                        <div class="speaker-margin">
                            <div class="speaker-name">{html.escape(speaker_name)}</div>
                            <div class="speaker-title">{html.escape(speaker_title)}</div>
                            {interaction_badge}
                        </div>
                        <div class="speaker-content">
            """
            
            # Add content paragraphs with copy buttons
            for p_index, paragraph in enumerate(speaker['paragraphs']):
                speaker_json = json.dumps({
                    'name': speaker_name,
                    'title': speaker_title,
                    'affiliation': speaker_affiliation
                }).replace('"', '&quot;')
                
                paragraph_escaped = html.escape(paragraph)
                copy_btn_id = f"copy-{section_index}-{speaker_index}-{p_index}"
                
                speaker_html += f"""<p>
                    {paragraph_escaped}
                    <button class="copy-btn" onclick="copyToClipboard({speaker_json}, '{paragraph_escaped.replace("'", "\\'")}')" 
                            aria-label="Copy paragraph and speaker details" title="Copy to clipboard">
                        📋 Copy
                    </button>
                </p>"""
            
            speaker_html += """
                        </div>
                    </div>
            """
            
            section_html += speaker_html
        
        section_html += """
                </div>
            </div>
        </section>
        """
        sections_html += section_html
    
    # Fill in the template
    html_output = html_template.format(
        title=html.escape(transcript_data['title']),
        date=html.escape(transcript_data['date']),
        primary_company=html.escape(primary_company),
        logo_url=logo_url,
        participants_count=participants_count,
        participants_html=participants_html,
        sections_html=sections_html
    )
    
    return html_output

def generate_pdf(transcript_data: Dict, output_path: str) -> None:
    """Generate a clean PDF version of the transcript."""
    
    # Create PDF document
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Define custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=HexColor('#003366'),
        fontName='Helvetica-Bold'
    )
    
    company_style = ParagraphStyle(
        'CompanyName',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=HexColor('#495057'),
        fontName='Helvetica'
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=24,
        textColor=HexColor('#003366'),
        fontName='Helvetica-Bold'
    )
    
    speaker_style = ParagraphStyle(
        'SpeakerHeader',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
        textColor=HexColor('#003366'),
        fontName='Helvetica-Bold'
    )
    
    speaker_title_style = ParagraphStyle(
        'SpeakerTitle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        textColor=HexColor('#6C757D'),
        fontName='Helvetica-Oblique'
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        textColor=HexColor('#212529'),
        fontName='Helvetica',
        leftIndent=0.25*inch
    )
    
    # Build PDF content
    story = []
    
    # Add title
    story.append(Paragraph(transcript_data['title'], title_style))
    
    # Add primary company
    institutions = load_monitored_institutions()
    primary_company, _ = detect_primary_company(transcript_data, institutions)
    story.append(Paragraph(primary_company, company_style))
    
    # Add date
    story.append(Paragraph(f"Date: {transcript_data['date']}", company_style))
    
    # Add spacer
    story.append(Spacer(1, 24))
    
    # Add participants section
    if transcript_data['participants']:
        story.append(Paragraph("Participants", section_style))
        
        for p_id, p_data in transcript_data['participants'].items():
            participant_type = "Company" if p_data['type'] == 'C' else "Analyst" if p_data['type'] == 'A' else p_data['type']
            participant_info = f"<b>{p_data['name']}</b> - {p_data['title']}"
            if p_data['affiliation']:
                participant_info += f", {p_data['affiliation']}"
            participant_info += f" ({participant_type})"
            story.append(Paragraph(participant_info, content_style))
        
        story.append(PageBreak())
    
    # Add transcript sections
    for section in transcript_data['sections']:
        story.append(Paragraph(section['name'], section_style))
        
        for speaker in section['speakers']:
            speaker_id = speaker['id']
            speaker_data = transcript_data['participants'].get(speaker_id, {})
            speaker_name = speaker_data.get('name', 'Unknown Speaker')
            speaker_title = speaker_data.get('title', '')
            speaker_affiliation = speaker_data.get('affiliation', '')
            
            # Add speaker header
            story.append(Paragraph(speaker_name, speaker_style))
            
            # Add speaker title and affiliation
            if speaker_title or speaker_affiliation:
                title_text = speaker_title
                if speaker_affiliation:
                    title_text += f", {speaker_affiliation}" if title_text else speaker_affiliation
                
                # Add interaction type if available
                if speaker['type'] == 'q':
                    title_text += " [QUESTION]"
                elif speaker['type'] == 'a':
                    title_text += " [ANSWER]"
                
                story.append(Paragraph(title_text, speaker_title_style))
            
            # Add speaker content
            for paragraph in speaker['paragraphs']:
                story.append(Paragraph(paragraph, content_style))
            
            # Add spacer between speakers
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)

def main():
    """Main function to parse XML and generate enhanced HTML and clean PDF."""
    if len(sys.argv) != 2:
        print("Usage: python transcript_html_viewer.py <xml_file_path>")
        sys.exit(1)
    
    xml_path = sys.argv[1]
    if not os.path.exists(xml_path):
        print(f"Error: File '{xml_path}' not found")
        sys.exit(1)
    
    try:
        # Parse XML
        print(f"Parsing XML file: {xml_path}")
        transcript_data = parse_transcript_xml(xml_path)
        
        # Generate HTML
        print("Generating enhanced HTML...")
        html_content = generate_html(transcript_data)
        
        # Write HTML output file
        html_output_path = xml_path.rsplit('.', 1)[0] + '_output.html'
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Generate PDF
        print("Generating clean PDF...")
        pdf_output_path = xml_path.rsplit('.', 1)[0] + '_output.pdf'
        generate_pdf(transcript_data, pdf_output_path)
        
        print(f"Success! Output files generated:")
        print(f"  HTML: {html_output_path}")
        print(f"  PDF:  {pdf_output_path}")
        print(f"Features included:")
        print(f"  ✓ WCAG 2.1 AA compliant HTML design")
        print(f"  ✓ Modern typography system")
        print(f"  ✓ Full-text search functionality")
        print(f"  ✓ Print and export capabilities")
        print(f"  ✓ Enhanced color system")
        print(f"  ✓ Paragraph-level copy functionality")
        print(f"  ✓ Keyboard navigation support")
        print(f"  ✓ Mobile-responsive design")
        print(f"  ✓ Clean PDF generation with professional layout")
        print(f"Open the HTML file in a web browser to view the enhanced transcript.")
        print(f"The PDF file provides a clean, print-ready version of the transcript.")
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()