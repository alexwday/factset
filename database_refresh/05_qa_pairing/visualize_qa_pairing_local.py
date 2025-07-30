"""
Local version of the Q&A Pairing visualization tool.
Reads from a local JSON file instead of NAS for easier testing and development.

Usage:
    python visualize_qa_pairing_local.py /path/to/stage_05_qa_paired_content.json
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List
from collections import defaultdict


def count_words(text: str) -> int:
    """Count words in a text string."""
    if not text:
        return 0
    # Simple word count - split by whitespace and filter empty strings
    words = text.split()
    return len(words)


def analyze_transcript_structure(transcript_records: List[Dict]) -> Dict:
    """
    Analyze a transcript's structure and identify issues.
    Returns detailed structure information including word counts and issues.
    """
    # Group by speaker blocks first
    speaker_blocks = defaultdict(list)
    for record in transcript_records:
        block_id = record.get("speaker_block_id")
        speaker_blocks[block_id].append(record)
    
    # Sort blocks by ID
    sorted_block_ids = sorted(speaker_blocks.keys())
    
    # Analyze each block
    blocks_analysis = []
    total_words = 0
    management_words = 0
    qa_words = 0
    
    for block_id in sorted_block_ids:
        block_records = speaker_blocks[block_id]
        
        # Get block info from first record
        first_record = block_records[0]
        section = first_record.get("section_name", "Unknown")
        qa_id = first_record.get("qa_group_id")
        speaker = first_record.get("speaker", "Unknown")
        
        # Count words in block
        block_text = " ".join([r.get("paragraph_content", "") for r in block_records])
        block_words = count_words(block_text)
        total_words += block_words
        
        if section == "Management Discussion":
            management_words += block_words
        elif section == "Q&A":
            qa_words += block_words
        
        blocks_analysis.append({
            "block_id": block_id,
            "section": section,
            "qa_id": qa_id,
            "speaker": speaker,
            "word_count": block_words,
            "paragraph_count": len(block_records)
        })
    
    # Identify issues
    issues = []
    
    # 1. Check for gaps in Q&A sections (Q&A blocks without qa_id)
    qa_blocks = [b for b in blocks_analysis if b["section"] == "Q&A"]
    for i, block in enumerate(qa_blocks):
        if block["qa_id"] is None:
            issues.append({
                "type": "qa_gap",
                "block_id": block["block_id"],
                "position": "start" if i == 0 else ("end" if i == len(qa_blocks) - 1 else "middle"),
                "message": f"Q&A block {block['block_id']} has no QA ID assigned"
            })
    
    # 2. Check QA ID ordering
    qa_ids_sequence = []
    prev_qa_id = None
    for block in qa_blocks:
        if block["qa_id"] is not None and block["qa_id"] != prev_qa_id:
            qa_ids_sequence.append(block["qa_id"])
            prev_qa_id = block["qa_id"]
    
    # Check if QA IDs are consecutive
    if qa_ids_sequence:
        for i in range(1, len(qa_ids_sequence)):
            if qa_ids_sequence[i] != qa_ids_sequence[i-1] + 1:
                issues.append({
                    "type": "qa_order",
                    "qa_ids": [qa_ids_sequence[i-1], qa_ids_sequence[i]],
                    "message": f"Non-consecutive QA IDs: {qa_ids_sequence[i-1]} → {qa_ids_sequence[i]}"
                })
    
    # Get unique QA groups with word counts
    qa_groups = defaultdict(lambda: {"word_count": 0, "block_count": 0})
    for block in qa_blocks:
        if block["qa_id"] is not None:
            qa_groups[block["qa_id"]]["word_count"] += block["word_count"]
            qa_groups[block["qa_id"]]["block_count"] += 1
    
    return {
        "total_words": total_words,
        "management_words": management_words,
        "qa_words": qa_words,
        "blocks": blocks_analysis,
        "qa_groups": dict(qa_groups),
        "issues": issues,
        "block_count": len(blocks_analysis),
        "qa_group_count": len(qa_groups)
    }


def generate_transcript_bar(transcript_id: str, analysis: Dict, max_words: int) -> str:
    """
    Generate HTML for a single transcript visualization bar.
    """
    blocks = analysis["blocks"]
    total_words = analysis["total_words"]
    issues = analysis["issues"]
    
    # Calculate scale factor (bar will be 800px max width)
    scale_factor = 800 / max_words if max_words > 0 else 1
    bar_width = total_words * scale_factor
    
    # Build the bar segments
    segments_html = []
    current_position = 0
    
    # Color palette
    colors = {
        "Management Discussion": "#4a90e2",  # Blue
        "Q&A": "#f5a623",  # Orange base
        "qa_gap": "#e74c3c",  # Red for gaps
    }
    
    # QA ID colors (cycling through a palette)
    qa_colors = ["#f39c12", "#e67e22", "#d68910", "#ca6f1e", "#ba4a00", "#a04000", "#873600", 
                 "#6e2c00", "#5b2c00", "#4a2300", "#391a00", "#281100"]
    
    for block in blocks:
        block_width = block["word_count"] * scale_factor
        
        if block_width > 0:  # Only show blocks with content
            # Determine color
            if block["section"] == "Management Discussion":
                color = colors["Management Discussion"]
                label = "MD"
            elif block["section"] == "Q&A":
                if block["qa_id"] is None:
                    color = colors["qa_gap"]
                    label = "GAP"
                else:
                    # Use cycling colors for QA IDs
                    color = qa_colors[(block["qa_id"] - 1) % len(qa_colors)]
                    label = f"Q{block['qa_id']}"
            else:
                color = "#cccccc"
                label = "?"
            
            # Check if this block has issues
            block_issues = [i for i in issues if i.get("block_id") == block["block_id"]]
            border_style = "border: 2px solid red;" if block_issues else ""
            
            # Make very small segments visible
            min_width = 3 if block_width < 3 else block_width
            
            segment_html = f'''
                <div class="segment" style="width: {min_width}px; background-color: {color}; {border_style}"
                     title="Block {block['block_id']}: {block['speaker']} ({block['word_count']} words)">
                    <span class="label" style="font-size: {10 if min_width > 20 else 8}px;">{label if min_width > 15 else ""}</span>
                </div>
            '''
            segments_html.append(segment_html)
    
    # Build issues list
    issues_html = ""
    if issues:
        issue_items = [f"<li class='issue'>{issue['message']}</li>" for issue in issues]
        issues_html = f"<ul class='issues'>{''.join(issue_items)}</ul>"
    
    # Calculate percentages
    mgmt_pct = (analysis["management_words"] / total_words * 100) if total_words > 0 else 0
    qa_pct = (analysis["qa_words"] / total_words * 100) if total_words > 0 else 0
    
    # Build the complete transcript row
    transcript_html = f'''
    <div class="transcript-row">
        <div class="transcript-info">
            <h3>{transcript_id}</h3>
            <div class="stats">
                <span>Total: {total_words:,} words</span> | 
                <span>MD: {analysis["management_words"]:,} ({mgmt_pct:.1f}%)</span> | 
                <span>Q&A: {analysis["qa_words"]:,} ({qa_pct:.1f}%)</span> | 
                <span>QA Groups: {analysis["qa_group_count"]}</span>
                {f'<span class="error-count">Issues: {len(issues)}</span>' if issues else ''}
            </div>
        </div>
        <div class="bar-container">
            <div class="bar" style="width: {bar_width}px;">
                {''.join(segments_html)}
            </div>
        </div>
        {issues_html}
    </div>
    '''
    
    return transcript_html


def generate_html_visualization(transcripts_data: Dict[str, List[Dict]]) -> str:
    """
    Generate complete HTML visualization file.
    """
    # Analyze all transcripts
    analyses = {}
    max_words = 0
    
    print("Analyzing transcripts...")
    for transcript_id, records in transcripts_data.items():
        analysis = analyze_transcript_structure(records)
        analyses[transcript_id] = analysis
        max_words = max(max_words, analysis["total_words"])
    
    # Generate HTML
    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Stage 5 Q&A Pairing Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #ccc;
        }}
        .transcript-row {{
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .transcript-info {{
            margin-bottom: 10px;
        }}
        .transcript-info h3 {{
            margin: 0 0 5px 0;
            color: #333;
        }}
        .stats {{
            font-size: 14px;
            color: #666;
        }}
        .error-count {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .bar-container {{
            overflow-x: auto;
            margin: 10px 0;
        }}
        .bar {{
            display: flex;
            height: 40px;
            border: 1px solid #ccc;
            border-radius: 3px;
            overflow: hidden;
        }}
        .segment {{
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 10px;
            font-weight: bold;
            overflow: hidden;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .segment:hover {{
            opacity: 0.8;
            transform: scaleY(1.1);
            z-index: 10;
        }}
        .segment .label {{
            text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
        }}
        .issues {{
            margin: 10px 0 0 0;
            padding: 0 0 0 20px;
            font-size: 12px;
        }}
        .issue {{
            color: #e74c3c;
            margin: 2px 0;
        }}
        .summary {{
            margin: 20px 0;
            padding: 20px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .scale-note {{
            text-align: center;
            font-size: 12px;
            color: #666;
            margin: 10px 0;
        }}
        .filter-controls {{
            margin: 20px 0;
            padding: 15px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .filter-controls label {{
            margin-right: 15px;
        }}
        .hidden {{
            display: none;
        }}
    </style>
</head>
<body>
    <h1>Stage 5 Q&A Pairing Visualization</h1>
    
    <div class="legend">
        <div class="legend-item">
            <span class="legend-color" style="background-color: #4a90e2;"></span>
            Management Discussion
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #f5a623;"></span>
            Q&A Sections (numbered by QA ID)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #e74c3c;"></span>
            Gap/Issue
        </div>
        <div class="legend-item">
            <span style="border: 2px solid red; padding: 2px;">Red Border</span>
            Block with Issue
        </div>
    </div>
    
    <div class="filter-controls">
        <label><input type="checkbox" id="showOnlyIssues"> Show only transcripts with issues</label>
        <label><input type="text" id="searchTranscript" placeholder="Search transcript ID..."></label>
    </div>
    
    <div class="scale-note">
        Bar width is proportional to word count. Max width = {max_words:,} words
    </div>
    
    <div class="transcripts">
'''
    
    # Add each transcript
    total_issues = 0
    for transcript_id in sorted(analyses.keys()):
        analysis = analyses[transcript_id]
        html_content += generate_transcript_bar(transcript_id, analysis, max_words)
        total_issues += len(analysis["issues"])
    
    # Add summary
    html_content += f'''
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Transcripts: {len(analyses)}</p>
        <p>Total Issues Found: {total_issues}</p>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <script>
        // Add click to copy transcript ID functionality
        document.querySelectorAll('.transcript-info h3').forEach(function(h3) {{
            h3.style.cursor = 'pointer';
            h3.title = 'Click to copy transcript ID';
            h3.addEventListener('click', function() {{
                navigator.clipboard.writeText(this.textContent);
                this.style.color = '#27ae60';
                setTimeout(() => {{ this.style.color = '#333'; }}, 1000);
            }});
        }});
        
        // Filter functionality
        document.getElementById('showOnlyIssues').addEventListener('change', function() {{
            const showOnlyIssues = this.checked;
            document.querySelectorAll('.transcript-row').forEach(function(row) {{
                const hasIssues = row.querySelector('.issues');
                if (showOnlyIssues && !hasIssues) {{
                    row.classList.add('hidden');
                }} else {{
                    row.classList.remove('hidden');
                }}
            }});
        }});
        
        // Search functionality
        document.getElementById('searchTranscript').addEventListener('input', function() {{
            const searchTerm = this.value.toLowerCase();
            document.querySelectorAll('.transcript-row').forEach(function(row) {{
                const transcriptId = row.querySelector('h3').textContent.toLowerCase();
                if (searchTerm && !transcriptId.includes(searchTerm)) {{
                    row.classList.add('hidden');
                }} else if (!document.getElementById('showOnlyIssues').checked) {{
                    row.classList.remove('hidden');
                }}
            }});
        }});
    </script>
</body>
</html>
'''
    
    return html_content


def group_records_by_transcript(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by transcript."""
    transcripts = defaultdict(list)
    
    for record in records:
        # Create transcript key using filename or event identifiers
        transcript_key = record.get("filename", f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
        transcripts[transcript_key].append(record)
    
    return dict(transcripts)


def main():
    """Main function to generate visualization."""
    if len(sys.argv) < 2:
        print("Usage: python visualize_qa_pairing_local.py /path/to/stage_05_qa_paired_content.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print("=== Stage 5 Q&A Pairing Visualization Tool (Local Version) ===")
    
    try:
        # Load Stage 5 output
        print(f"\nLoading Stage 5 output from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        print(f"Loaded {len(records)} records")
        
        # Group by transcript
        transcripts = group_records_by_transcript(records)
        print(f"Found {len(transcripts)} transcripts")
        
        # Generate visualization
        print("\nGenerating HTML visualization...")
        html_content = generate_html_visualization(transcripts)
        
        # Save HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"qa_pairing_visualization_{timestamp}.html"
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nVisualization saved to: {html_filename}")
        print("Open in a web browser to view")
        
        # Summary of issues found
        total_issues = sum(len(analyze_transcript_structure(records)["issues"]) 
                          for records in transcripts.values())
        print(f"\nTotal issues found across all transcripts: {total_issues}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()