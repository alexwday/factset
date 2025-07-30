"""
Standalone visualization tool for Stage 5 Q&A Pairing output.
Creates an HTML visualization showing transcript structure with colored bars.

Visual Features:
- Management Discussion sections in one color
- Q&A sections in another color with numbered QA IDs
- Red highlighting for issues (gaps, non-consecutive IDs)
- Bar length proportional to word count
- Transcript-level scaling for comparison
"""

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import io

import yaml
from dotenv import load_dotenv
from smb.SMBConnection import SMBConnection

# Load environment variables
load_dotenv()


def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    try:
        conn = SMBConnection(
            username=os.getenv("NAS_USERNAME"),
            password=os.getenv("NAS_PASSWORD"),
            my_name=os.getenv("CLIENT_MACHINE_NAME"),
            remote_name=os.getenv("NAS_SERVER_NAME"),
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )
        
        nas_port = int(os.getenv("NAS_PORT", 445))
        if conn.connect(os.getenv("NAS_SERVER_IP"), nas_port):
            print("NAS connection established successfully")
            return conn
        else:
            print("Failed to establish NAS connection")
            return None
            
    except Exception as e:
        print(f"Error creating NAS connection: {e}")
        return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        print(f"Failed to download file from NAS: {nas_file_path} - {e}")
        return None


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    try:
        # Ensure parent directory exists
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        print(f"Failed to upload file to NAS: {nas_file_path} - {e}")
        return False


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load configuration from NAS."""
    try:
        config_path = os.getenv("CONFIG_PATH")
        print(f"Loading config from: {config_path}")
        
        config_data = nas_download_file(nas_conn, config_path)
        if not config_data:
            print(f"Configuration file not found at {config_path}")
            return None
        
        # Parse YAML configuration
        config = yaml.safe_load(config_data.decode("utf-8"))
        return config
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


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
    qa_colors = ["#f39c12", "#e67e22", "#d68910", "#ca6f1e", "#ba4a00", "#a04000", "#873600"]
    
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
            
            segment_html = f'''
                <div class="segment" style="width: {block_width}px; background-color: {color}; {border_style}"
                     title="Block {block['block_id']}: {block['speaker']} ({block['word_count']} words)">
                    <span class="label">{label}</span>
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


def generate_html_visualization(transcripts_data: Dict[str, List[Dict]], output_path: str) -> None:
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
        }}
        .segment:hover {{
            opacity: 0.8;
            transform: scaleY(1.1);
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
    print("=== Stage 5 Q&A Pairing Visualization Tool ===")
    
    # Connect to NAS
    print("\nConnecting to NAS...")
    nas_conn = get_nas_connection()
    if not nas_conn:
        print("Failed to connect to NAS")
        return
    
    try:
        # Load configuration
        config = load_config_from_nas(nas_conn)
        if not config:
            print("Failed to load configuration")
            return
        
        # Get output path from config
        output_path = config["stage_05_qa_pairing"]["output_data_path"]
        output_filename = "stage_05_qa_paired_content.json"
        output_file_path = f"{output_path}/{output_filename}"
        
        print(f"\nLoading Stage 5 output from: {output_file_path}")
        content_data = nas_download_file(nas_conn, output_file_path)
        
        if not content_data:
            print("Failed to load Stage 5 output file")
            return
        
        # Parse JSON
        records = json.loads(content_data.decode('utf-8'))
        print(f"Loaded {len(records)} records")
        
        # Group by transcript
        transcripts = group_records_by_transcript(records)
        print(f"Found {len(transcripts)} transcripts")
        
        # Generate visualization
        print("\nGenerating HTML visualization...")
        html_content = generate_html_visualization(transcripts, output_path)
        
        # Save HTML file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_filename = f"qa_pairing_visualization_{timestamp}.html"
        html_path = f"{output_path}/{html_filename}"
        
        html_bytes = io.BytesIO(html_content.encode('utf-8'))
        html_bytes.seek(0)
        
        if nas_upload_file(nas_conn, html_bytes, html_path):
            print(f"\nVisualization saved to: {html_filename}")
            print("Download and open in a web browser to view")
        else:
            # Try saving locally as fallback
            local_path = f"./{html_filename}"
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\nVisualization saved locally to: {local_path}")
        
        # Summary of issues found
        total_issues = sum(len(analyze_transcript_structure(records)["issues"]) 
                          for records in transcripts.values())
        print(f"\nTotal issues found across all transcripts: {total_issues}")
        
    finally:
        # Close NAS connection
        if nas_conn:
            nas_conn.close()
            print("\nNAS connection closed")


if __name__ == "__main__":
    main()