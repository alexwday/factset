#!/usr/bin/env python3
"""
FactSet Transcript XML to HTML Parser
Converts FactSet XML transcripts into readable HTML format with speaker mapping
"""

import xml.etree.ElementTree as ET
from datetime import datetime
import html
import os
import sys
from typing import Dict, List, Tuple, Optional

def parse_transcript_xml(xml_path: str) -> Dict:
    """Parse FactSet transcript XML and extract structured data."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract metadata
    meta = root.find('meta')
    if meta is None:
        raise ValueError("No meta section found in XML")
    
    # Extract title and date
    title = meta.find('title')
    title_text = title.text if title is not None else "Untitled Transcript"
    
    date = meta.find('date')
    date_text = date.text if date is not None else "Unknown Date"
    
    # Extract companies
    companies = []
    companies_elem = meta.find('companies')
    if companies_elem is not None:
        for company in companies_elem.findall('company'):
            companies.append(company.text if company.text else "")
    
    # Extract participants and create speaker mapping
    participants = {}
    participants_elem = meta.find('participants')
    if participants_elem is not None:
        for participant in participants_elem.findall('participant'):
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
    body = root.find('body')
    if body is None:
        raise ValueError("No body section found in XML")
    
    sections = []
    for section in body.findall('section'):
        section_name = section.get('name', 'Unnamed Section')
        speakers = []
        
        for speaker in section.findall('speaker'):
            speaker_id = speaker.get('id')
            speaker_type = speaker.get('type', '')  # 'q' or 'a' for Q&A sections
            
            # Extract paragraphs from plist
            paragraphs = []
            plist = speaker.find('plist')
            if plist is not None:
                for p in plist.findall('p'):
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
    
    # HTML template with embedded CSS
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-header {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .speaker-block {{
            margin-bottom: 25px;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            border-radius: 0 5px 5px 0;
        }}
        .speaker-info {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .speaker-name {{
            font-size: 1.1em;
            color: #3498db;
        }}
        .speaker-title {{
            font-size: 0.9em;
            color: #7f8c8d;
            font-weight: normal;
        }}
        .question {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .answer {{
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
        }}
        .interaction-type {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .question-label {{
            background-color: #ffc107;
            color: #856404;
        }}
        .answer-label {{
            background-color: #17a2b8;
            color: white;
        }}
        .content p {{
            margin: 10px 0;
            text-align: justify;
        }}
        .participants-list {{
            margin-top: 15px;
        }}
        .participant-item {{
            padding: 8px;
            background-color: white;
            margin: 5px 0;
            border-radius: 3px;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        
        <div class="metadata">
            <p><strong>Date:</strong> {date}</p>
            <p><strong>Companies:</strong> {companies}</p>
            
            <div class="participants-list">
                <p><strong>Participants:</strong></p>
                {participants_html}
            </div>
        </div>
        
        {sections_html}
    </div>
</body>
</html>"""
    
    # Generate participants HTML
    participants_html = ""
    for p_id, p_data in transcript_data['participants'].items():
        participant_type = "Company" if p_data['type'] == 'C' else "Analyst" if p_data['type'] == 'A' else p_data['type']
        participants_html += f"""
        <div class="participant-item">
            <strong>{html.escape(p_data['name'])}</strong> - 
            {html.escape(p_data['title'])} at {html.escape(p_data['affiliation'])} 
            <em>({participant_type})</em>
        </div>"""
    
    # Generate sections HTML
    sections_html = ""
    for section in transcript_data['sections']:
        section_html = f'<div class="section">\n<div class="section-header">{html.escape(section["name"])}</div>\n'
        
        for speaker in section['speakers']:
            speaker_id = speaker['id']
            speaker_data = transcript_data['participants'].get(speaker_id, {})
            speaker_name = speaker_data.get('name', 'Unknown Speaker')
            speaker_title = speaker_data.get('title', '')
            speaker_affiliation = speaker_data.get('affiliation', '')
            
            # Determine interaction type and styling
            interaction_class = ""
            interaction_label = ""
            if speaker['type'] == 'q':
                interaction_class = "question"
                interaction_label = '<span class="interaction-type question-label">QUESTION</span>'
            elif speaker['type'] == 'a':
                interaction_class = "answer"
                interaction_label = '<span class="interaction-type answer-label">ANSWER</span>'
            
            # Build speaker block
            speaker_html = f'<div class="speaker-block {interaction_class}">\n'
            speaker_html += f'<div class="speaker-info">'
            speaker_html += f'<span class="speaker-name">{html.escape(speaker_name)}</span>'
            if speaker_title:
                speaker_html += f'<span class="speaker-title"> - {html.escape(speaker_title)}'
                if speaker_affiliation:
                    speaker_html += f', {html.escape(speaker_affiliation)}'
                speaker_html += '</span>'
            speaker_html += interaction_label
            speaker_html += '</div>\n'
            
            # Add content paragraphs
            speaker_html += '<div class="content">\n'
            for paragraph in speaker['paragraphs']:
                speaker_html += f'<p>{html.escape(paragraph)}</p>\n'
            speaker_html += '</div>\n</div>\n'
            
            section_html += speaker_html
        
        section_html += '</div>\n'
        sections_html += section_html
    
    # Fill in the template
    html_output = html_template.format(
        title=html.escape(transcript_data['title']),
        date=html.escape(transcript_data['date']),
        companies=html.escape(', '.join(transcript_data['companies'])),
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