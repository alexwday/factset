#!/usr/bin/env python3
"""
Enhanced FactSet Transcript XML to HTML Viewer
Converts FactSet XML transcripts into modern, interactive HTML format with:
- Company logo integration
- Expandable sections
- Professional financial document styling
- Responsive design
- Q&A formatting

Part of the FactSet Earnings Transcript Pipeline test suite.
"""

import xml.etree.ElementTree as ET
from datetime import datetime
import html
import os
import sys
import json
from typing import Dict, List, Tuple, Optional

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
    return "https://via.placeholder.com/120x60/0066CC/FFFFFF?text=LOGO"

def detect_primary_company(transcript_data: Dict, institutions: Dict) -> Tuple[str, str]:
    """Detect the primary company from transcript data."""
    companies = transcript_data.get('companies', [])
    if not companies:
        return "Unknown Company", "https://via.placeholder.com/120x60/0066CC/FFFFFF?text=LOGO"
    
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
            
            # Extract paragraphs from plist
            paragraphs = []
            plist = speaker.find(ns_tag('plist'))
            if plist is not None:
                for p in plist.findall(ns_tag('p')):
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
    """Generate HTML from parsed transcript data."""
    
    # Load monitored institutions for logo integration
    institutions = load_monitored_institutions()
    primary_company, logo_url = detect_primary_company(transcript_data, institutions)
    
    # Modern, professional HTML template with embedded CSS and JavaScript
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        /* Modern Financial Document Styling */
        :root {{
            --primary-color: #0066CC;
            --secondary-color: #004499;
            --accent-color: #FF6B35;
            --success-color: #28A745;
            --warning-color: #FFC107;
            --danger-color: #DC3545;
            --dark-color: #2C3E50;
            --light-color: #F8F9FA;
            --gray-color: #6C757D;
            --white: #FFFFFF;
            --border-color: #E9ECEF;
            --shadow-color: rgba(0, 0, 0, 0.1);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: var(--white);
            border-radius: 12px;
            box-shadow: 0 8px 32px var(--shadow-color);
            overflow: hidden;
        }}
        
        /* Header Section */
        .header {{
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: var(--white);
            padding: 30px 40px;
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
            gap: 20px;
        }}
        
        .company-logo {{
            max-width: 120px;
            max-height: 60px;
            background: var(--white);
            border-radius: 6px;
            padding: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .title-section h1 {{
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .title-section .company-name {{
            font-size: 1.1em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        /* Metadata Section */
        .metadata {{
            background: var(--light-color);
            padding: 25px 40px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metadata-item {{
            background: var(--white);
            padding: 15px;
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
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .section-toggle {{
            background: var(--primary-color);
            color: var(--white);
            padding: 15px 25px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            font-size: 1.1em;
            transition: all 0.3s ease;
            border: none;
            width: 100%;
            text-align: left;
        }}
        
        .section-toggle:hover {{
            background: var(--secondary-color);
            transform: translateY(-1px);
        }}
        
        .section-toggle.active {{
            background: var(--secondary-color);
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
            background: var(--white);
        }}
        
        .section-content.active {{
            max-height: 2000px;
            padding: 20px 25px;
        }}
        
        /* Participants Section */
        .participants-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }}
        
        .participant-card {{
            background: var(--white);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }}
        
        .participant-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .participant-name {{
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: 5px;
        }}
        
        .participant-title {{
            color: var(--gray-color);
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .participant-type {{
            display: inline-block;
            background: var(--primary-color);
            color: var(--white);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }}
        
        /* Transcript Content */
        .transcript-section {{
            margin-bottom: 30px;
        }}
        
        .speaker-block {{
            margin-bottom: 20px;
            background: var(--white);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
        }}
        
        .speaker-block:hover {{
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }}
        
        .speaker-header {{
            padding: 15px 20px;
            background: var(--light-color);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .speaker-info {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .speaker-name {{
            font-weight: 600;
            color: var(--primary-color);
            font-size: 1.1em;
        }}
        
        .speaker-title {{
            color: var(--gray-color);
            font-size: 0.9em;
        }}
        
        .interaction-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .question-badge {{
            background: var(--warning-color);
            color: var(--dark-color);
        }}
        
        .answer-badge {{
            background: var(--success-color);
            color: var(--white);
        }}
        
        .speaker-content {{
            padding: 20px;
        }}
        
        .speaker-content p {{
            margin-bottom: 15px;
            text-align: justify;
            line-height: 1.7;
        }}
        
        .speaker-content p:last-child {{
            margin-bottom: 0;
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            
            .header {{
                padding: 20px;
            }}
            
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}
            
            .title-section h1 {{
                font-size: 1.8em;
            }}
            
            .metadata {{
                padding: 20px;
            }}
            
            .metadata-grid {{
                grid-template-columns: 1fr;
            }}
            
            .section-content.active {{
                padding: 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <div class="header-content">
                <div class="title-section">
                    <h1>{title}</h1>
                    <div class="company-name">{primary_company}</div>
                </div>
                <img src="{logo_url}" alt="{primary_company} Logo" class="company-logo" 
                     onerror="this.src='https://via.placeholder.com/120x60/0066CC/FFFFFF?text=LOGO'">
            </div>
        </div>
        
        <!-- Metadata Section -->
        <div class="metadata">
            <div class="metadata-grid">
                <div class="metadata-item">
                    <strong>Date:</strong> {date}
                </div>
                <div class="metadata-item">
                    <strong>Companies:</strong> {companies}
                </div>
            </div>
            
            <!-- Participants Section -->
            <div class="expandable-section">
                <button class="section-toggle" onclick="toggleSection('participants')">
                    <span>Participants ({participants_count})</span>
                    <span class="toggle-icon">▼</span>
                </button>
                <div class="section-content" id="participants-content">
                    <div class="participants-grid">
                        {participants_html}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Transcript Sections -->
        <div style="padding: 20px;">
            {sections_html}
        </div>
    </div>
    
    <script>
        function toggleSection(sectionId) {{
            const content = document.getElementById(sectionId + '-content');
            const button = content.previousElementSibling;
            const isActive = content.classList.contains('active');
            
            if (isActive) {{
                content.classList.remove('active');
                button.classList.remove('active');
            }} else {{
                content.classList.add('active');
                button.classList.add('active');
            }}
        }}
        
        // Initialize: collapse all sections by default
        document.addEventListener('DOMContentLoaded', function() {{
            // Optionally auto-expand the first section
            const firstSection = document.querySelector('.expandable-section .section-content');
            if (firstSection && firstSection.id === 'participants-content') {{
                toggleSection('participants');
            }}
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
        <div class="participant-card">
            <div class="participant-name">{html.escape(p_data['name'])}</div>
            <div class="participant-title">{html.escape(p_data['title'])}</div>
            <div class="participant-title">{html.escape(p_data['affiliation'])}</div>
            <span class="participant-type">{participant_type}</span>
        </div>"""
    
    # Generate sections HTML with expandable functionality
    sections_html = ""
    for section_index, section in enumerate(transcript_data['sections']):
        section_id = f"section-{section_index}"
        section_html = f"""
        <div class="expandable-section">
            <button class="section-toggle" onclick="toggleSection('{section_id}')">
                <span>{html.escape(section['name'])}</span>
                <span class="toggle-icon">▼</span>
            </button>
            <div class="section-content" id="{section_id}-content">
                <div class="transcript-section">
        """
        
        for speaker in section['speakers']:
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
            
            # Build speaker block
            speaker_html = f"""
                    <div class="speaker-block">
                        <div class="speaker-header">
                            <div class="speaker-info">
                                <span class="speaker-name">{html.escape(speaker_name)}</span>
                                <span class="speaker-title">{html.escape(speaker_title)}</span>
                            </div>
                            {interaction_badge}
                        </div>
                        <div class="speaker-content">
            """
            
            # Add content paragraphs
            for paragraph in speaker['paragraphs']:
                speaker_html += f'<p>{html.escape(paragraph)}</p>\n'
            
            speaker_html += """
                        </div>
                    </div>
            """
            
            section_html += speaker_html
        
        section_html += """
                </div>
            </div>
        </div>
        """
        sections_html += section_html
    
    # Fill in the template
    html_output = html_template.format(
        title=html.escape(transcript_data['title']),
        date=html.escape(transcript_data['date']),
        companies=html.escape(', '.join(transcript_data['companies'])),
        primary_company=html.escape(primary_company),
        logo_url=logo_url,
        participants_count=participants_count,
        participants_html=participants_html,
        sections_html=sections_html
    )
    
    return html_output

def main():
    """Main function to parse XML and generate HTML."""
    if len(sys.argv) != 2:
        print("Usage: python parse_transcript_to_html.py <xml_file_path>")
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
        print("Generating HTML...")
        html_content = generate_html(transcript_data)
        
        # Write output file
        output_path = xml_path.rsplit('.', 1)[0] + '_output.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Success! HTML output written to: {output_path}")
        print(f"Open the file in a web browser to view the formatted transcript.")
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()