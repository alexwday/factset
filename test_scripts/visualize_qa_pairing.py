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
from typing import Dict, List, Tuple, Optional, Any
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
    
    # Sort blocks by ID (ensuring numeric sort)
    sorted_block_ids = sorted(speaker_blocks.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    
    # Analyze each block
    blocks_analysis = []
    total_words = 0
    management_words = 0
    qa_words = 0
    
    for block_id in sorted_block_ids:
        block_records = speaker_blocks[block_id]
        
        # Get block info from first record
        first_record = block_records[0]
        # Check both possible field names for compatibility
        # Stage 5 copies section_name from Stage 4, not section_type
        section = first_record.get("section_name") or first_record.get("section_type", "Unknown")
        qa_id = first_record.get("qa_group_id")
        speaker = first_record.get("speaker", "Unknown")
        
        # Count words in block
        block_text = " ".join([r.get("paragraph_content", "") for r in block_records])
        block_words = count_words(block_text)
        total_words += block_words
        
        # Check for both possible section names from Stage 4
        if section == "Management Discussion" or section == "MD":
            management_words += block_words
        elif section == "Investor Q&A" or section == "Q&A":
            qa_words += block_words
        
        # Get sample text for gaps (first 100 chars)
        sample_text = block_text[:200] + "..." if len(block_text) > 200 else block_text
        
        blocks_analysis.append({
            "block_id": block_id,
            "section": section,
            "qa_id": qa_id,
            "speaker": speaker,
            "word_count": block_words,
            "paragraph_count": len(block_records),
            "sample_text": sample_text
        })
    
    # Identify issues
    issues = []
    
    # 1. Check for gaps in Q&A sections (Q&A blocks without qa_id)
    qa_blocks = [b for b in blocks_analysis if b["section"] == "Investor Q&A" or b["section"] == "Q&A"]
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
    Consolidates Management Discussion by speaker and Q&A by QA ID.
    """
    blocks = analysis["blocks"]
    total_words = analysis["total_words"]
    issues = analysis["issues"]
    qa_groups = analysis["qa_groups"]
    
    # Calculate scale factor (bar will be 800px max width)
    scale_factor = 800 / max_words if max_words > 0 else 1
    bar_width = total_words * scale_factor
    
    # Build consolidated segments
    segments_html = []
    
    # Professional bank-inspired color palette
    mgmt_color = "#003087"  # RBC deep blue
    qa_colors = [
        "#0051A5",  # TD Bank blue
        "#0079C1",  # BMO blue
        "#00539B",  # Scotia blue
        "#6B0F24",  # CIBC burgundy
        "#004B87",  # National Bank blue
        "#0066CC",  # Medium blue
        "#4B7BA7",  # Light steel blue
        "#1E3A5F",  # Navy blue
        "#2E5090",  # Royal blue
        "#3A5998",  # Corporate blue
    ]
    gap_color = "#8B0000"  # Dark red for gaps
    
    # Build segments in order
    consolidated_segments = []
    
    # First pass: identify segment boundaries
    i = 0
    while i < len(blocks):
        block = blocks[i]
        
        if block["section"] == "Management Discussion" or block["section"] == "MD":
            # Consolidate MD blocks by speaker
            speaker = block["speaker"]
            segment_blocks = [block]
            j = i + 1
            
            # Collect consecutive blocks from same speaker in same section type
            while j < len(blocks) and (blocks[j]["section"] == "Management Discussion" or blocks[j]["section"] == "MD") and blocks[j]["speaker"] == speaker:
                segment_blocks.append(blocks[j])
                j += 1
            
            # Create consolidated segment
            total_segment_words = sum(b["word_count"] for b in segment_blocks)
            if total_segment_words > 0:
                consolidated_segments.append({
                    "type": "md",
                    "speaker": speaker,
                    "word_count": total_segment_words,
                    "block_ids": [b["block_id"] for b in segment_blocks],
                    "position": i
                })
            i = j
            
        elif block["section"] == "Investor Q&A" or block["section"] == "Q&A":
            if block["qa_id"] is None:
                # Gap block
                if block["word_count"] > 0:
                    consolidated_segments.append({
                        "type": "gap",
                        "block": block,
                        "position": i
                    })
                i += 1
            else:
                # Q&A block - collect all blocks with same qa_id
                qa_id = block["qa_id"]
                qa_blocks = [block]
                j = i + 1
                
                # Look ahead for more blocks with same qa_id
                while j < len(blocks) and blocks[j].get("qa_id") == qa_id:
                    qa_blocks.append(blocks[j])
                    j += 1
                
                total_qa_words = sum(b["word_count"] for b in qa_blocks)
                if total_qa_words > 0:
                    consolidated_segments.append({
                        "type": "qa",
                        "qa_id": qa_id,
                        "word_count": total_qa_words,
                        "block_count": len(qa_blocks),
                        "position": i
                    })
                i = j
        else:
            i += 1
    
    # Now render segments in order
    for segment in consolidated_segments:
        if segment["type"] == "md":
            segment_width = segment["word_count"] * scale_factor
            if segment_width > 0:
                speaker_name = segment["speaker"].replace("Executives", "Exec").replace("Operator", "Op")
                block_range = f"{min(segment['block_ids'])}-{max(segment['block_ids'])}" if len(segment['block_ids']) > 1 else str(segment['block_ids'][0])
                
                md_segment = f'''
                    <div class="segment md" style="width: {segment_width}px; background-color: {mgmt_color};"
                         title="Management Discussion: {segment['speaker']} ({segment['word_count']:,} words, blocks {block_range})">
                        <span class="label">{speaker_name}</span>
                    </div>
                '''
                segments_html.append(md_segment)
                
        elif segment["type"] == "gap":
            block = segment["block"]
            gap_width = block["word_count"] * scale_factor
            if gap_width > 0:
                sample_text = block.get("sample_text", "").replace('"', '&quot;').replace("'", '&#39;')
                speaker_label = block['speaker'].replace("Executives", "Exec").replace("Operator", "Op")
                
                gap_segment = f'''
                    <div class="segment gap" style="width: {gap_width}px; background-color: {gap_color}; border: 2px solid #5a0000;"
                         title="GAP - Block {block['block_id']} - {speaker_label}: {sample_text} ({block['word_count']} words)">
                        <span class="label">GAP</span>
                    </div>
                '''
                segments_html.append(gap_segment)
                
        elif segment["type"] == "qa":
            qa_width = segment["word_count"] * scale_factor
            if qa_width > 0:
                qa_id = segment["qa_id"]
                color = qa_colors[(qa_id - 1) % len(qa_colors)]
                
                qa_segment = f'''
                    <div class="segment qa" style="width: {qa_width}px; background-color: {color};"
                         title="Q&A Group {qa_id} ({segment['word_count']:,} words, {segment['block_count']} speaker blocks)">
                        <span class="label">Q{qa_id}</span>
                    </div>
                '''
                segments_html.append(qa_segment)
    
    # Build issues list with gap text
    issues_html = ""
    if issues:
        issue_items = []
        for issue in issues:
            if issue["type"] == "qa_gap":
                # Find the block to get sample text
                gap_block = next((b for b in blocks if b["block_id"] == issue["block_id"]), None)
                if gap_block and gap_block.get("sample_text"):
                    sample = gap_block["sample_text"][:100] + "..." if len(gap_block.get("sample_text", "")) > 100 else gap_block.get("sample_text", "")
                    issue_text = f"{issue['message']}<br><small style='color: #666;'>'{sample}'</small>"
                else:
                    issue_text = issue['message']
            else:
                issue_text = issue['message']
            issue_items.append(f"<li class='issue'>{issue_text}</li>")
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
    
    <div class="explanation" style="margin: 20px auto; max-width: 900px; padding: 20px; background: #e8f4fd; border-left: 4px solid #003087; border-radius: 5px;">
        <h2 style="color: #003087; margin-top: 0;">What This Visualization Shows</h2>
        <p><strong>Purpose:</strong> This tool validates the Q&A pairing algorithm from Stage 5, which automatically detects conversation boundaries between analysts and company executives during earnings calls.</p>
        
        <h3 style="color: #003087; font-size: 1.1em;">What We're Checking:</h3>
        <ul style="line-height: 1.6;">
            <li><strong>Completeness:</strong> Every Q&A section paragraph should be assigned to a Q&A group (no gaps)</li>
            <li><strong>Continuity:</strong> Q&A group IDs should be consecutive (1, 2, 3... not 1, 3, 5...)</li>
            <li><strong>Accuracy:</strong> Each Q&A group should contain one complete analyst-executive conversation</li>
            <li><strong>Structure:</strong> Management Discussion should appear before Q&A sections</li>
            <li><strong>Speaker Tracking:</strong> MD sections are consolidated by speaker to show conversation flow</li>
        </ul>
        
        <h3 style="color: #003087; font-size: 1.1em;">Why This Matters:</h3>
        <ul style="line-height: 1.6;">
            <li>Proper Q&A pairing enables accurate attribution of questions to specific analysts</li>
            <li>Clean conversation boundaries allow for focused analysis of individual topics</li>
            <li>Detecting gaps helps identify edge cases where the algorithm needs improvement</li>
            <li>Validated pairings ensure downstream analysis (sentiment, topics) is correctly scoped</li>
        </ul>
    </div>
    
    <div class="legend">
        <div class="legend-item">
            <span class="legend-color" style="background-color: #003087;"></span>
            Management Discussion
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #0051A5;"></span>
            Q&A Sections (numbered by QA ID)
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: #8B0000;"></span>
            Gap/Issue
        </div>
        <div class="legend-item">
            <span style="border: 2px solid #8B0000; padding: 2px;">Red Border</span>
            Block with Issue
        </div>
    </div>
    
    <div style="margin: 20px 0; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="display: flex; gap: 20px; align-items: center;">
            <div style="flex: 1;">
                <input type="text" id="searchInput" placeholder="Search transcript IDs..." 
                       style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
            </div>
            <div>
                <select id="filterSelect" style="padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    <option value="all">All Transcripts</option>
                    <option value="perfect">Perfect (No Issues)</option>
                    <option value="gaps">Has Gaps</option>
                    <option value="order">Has Order Issues</option>
                    <option value="any-issue">Any Issue</option>
                </select>
            </div>
            <button onclick="resetFilters()" style="padding: 8px 16px; background: #003087; color: white; border: none; border-radius: 4px; cursor: pointer;">Reset</button>
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
    
    # Calculate additional statistics
    total_qa_groups = sum(a["qa_group_count"] for a in analyses.values())
    transcripts_with_gaps = sum(1 for a in analyses.values() if any(i["type"] == "qa_gap" for i in a["issues"]))
    transcripts_with_order_issues = sum(1 for a in analyses.values() if any(i["type"] == "qa_order" for i in a["issues"]))
    avg_qa_groups = total_qa_groups / len(analyses) if analyses else 0
    
    # Calculate MD speaker breakdown
    md_speaker_stats = defaultdict(lambda: {"count": 0, "words": 0})
    for analysis in analyses.values():
        for block in analysis["blocks"]:
            if block["section"] in ["Management Discussion", "MD"]:
                speaker = block["speaker"]
                md_speaker_stats[speaker]["count"] += 1
                md_speaker_stats[speaker]["words"] += block["word_count"]
    
    gap_issues = [i for a in analyses.values() for i in a["issues"] if i["type"] == "qa_gap"]
    order_issues = [i for a in analyses.values() for i in a["issues"] if i["type"] == "qa_order"]
    
    # Quality metrics
    perfect_transcripts = len(analyses) - len([a for a in analyses.values() if a["issues"]])
    quality_score = (perfect_transcripts / len(analyses) * 100) if analyses else 0
    
    # Calculate additional useful metrics
    total_gap_words = sum(b["word_count"] for a in analyses.values() 
                         for b in a["blocks"] 
                         if b["section"] == "Investor Q&A" and b["qa_id"] is None)
    
    # Get distribution of Q&A group sizes
    qa_group_sizes = []
    for analysis in analyses.values():
        for qa_id, group_info in analysis["qa_groups"].items():
            qa_group_sizes.append(group_info["word_count"])
    
    avg_qa_group_size = sum(qa_group_sizes) / len(qa_group_sizes) if qa_group_sizes else 0
    min_qa_group_size = min(qa_group_sizes) if qa_group_sizes else 0
    max_qa_group_size = max(qa_group_sizes) if qa_group_sizes else 0
    
    # Calculate MD/Q&A split
    total_md_words = sum(a["management_words"] for a in analyses.values())
    total_qa_words = sum(a["qa_words"] for a in analyses.values())
    total_all_words = sum(a["total_words"] for a in analyses.values())
    
    # Add enhanced summary
    html_content += f'''
    </div>
    
    <div class="summary">
        <h2>Overall Statistics</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0;">
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #003087;">
                <h3 style="margin: 0; color: #003087;">Transcript Overview</h3>
                <p style="font-size: 24px; margin: 10px 0;">{len(analyses)}</p>
                <p style="color: #666; margin: 0;">Total Transcripts</p>
            </div>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0051A5;">
                <h3 style="margin: 0; color: #0051A5;">Q&A Groups</h3>
                <p style="font-size: 24px; margin: 10px 0;">{total_qa_groups}</p>
                <p style="color: #666; margin: 0;">Total ({avg_qa_groups:.1f} avg per transcript)</p>
            </div>
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; border-left: 4px solid #ff9800;">
                <h3 style="margin: 0; color: #ff9800;">Gap Issues</h3>
                <p style="font-size: 24px; margin: 10px 0;">{len(gap_issues)}</p>
                <p style="color: #666; margin: 0;">In {transcripts_with_gaps} transcripts</p>
                <p style="color: #856404; margin: 5px 0; font-size: 12px;">{total_gap_words:,} words in gaps</p>
            </div>
            <div style="background: #f8d7da; padding: 15px; border-radius: 5px; border-left: 4px solid #8B0000;">
                <h3 style="margin: 0; color: #8B0000;">Order Issues</h3>
                <p style="font-size: 24px; margin: 10px 0;">{len(order_issues)}</p>
                <p style="color: #666; margin: 0;">In {transcripts_with_order_issues} transcripts</p>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div style="background: #e8f4fd; padding: 20px; border-radius: 5px;">
                <h3 style="color: #003087; margin-top: 0;">Quality Score</h3>
                <div style="background: #003087; height: 30px; border-radius: 15px; overflow: hidden;">
                    <div style="background: #27ae60; height: 100%; width: {quality_score}%; transition: width 0.5s ease;"></div>
                </div>
                <p style="text-align: center; margin: 10px 0; font-size: 18px;">
                    <strong>{quality_score:.1f}%</strong> ({perfect_transcripts}/{len(analyses)} transcripts without issues)
                </p>
            </div>
            
            <div style="background: #f0f8ff; padding: 20px; border-radius: 5px;">
                <h3 style="color: #0051A5; margin-top: 0;">Content Distribution</h3>
                <div style="margin: 10px 0;">
                    <div style="display: flex; margin-bottom: 10px;">
                        <span style="width: 150px;">Management Discussion:</span>
                        <span style="font-weight: bold;">{(total_md_words/total_all_words*100 if total_all_words else 0):.1f}%</span>
                    </div>
                    <div style="display: flex; margin-bottom: 10px;">
                        <span style="width: 150px;">Q&A Section:</span>
                        <span style="font-weight: bold;">{(total_qa_words/total_all_words*100 if total_all_words else 0):.1f}%</span>
                    </div>
                    <div style="font-size: 12px; color: #666; margin-top: 10px;">
                        Total: {total_all_words:,} words
                    </div>
                </div>
            </div>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #333; margin-top: 0;">Q&A Group Size Distribution</h3>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
                <div>
                    <p style="color: #666; margin: 5px 0;">Average Size</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{avg_qa_group_size:,.0f} words</p>
                </div>
                <div>
                    <p style="color: #666; margin: 5px 0;">Smallest Group</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{min_qa_group_size:,} words</p>
                </div>
                <div>
                    <p style="color: #666; margin: 5px 0;">Largest Group</p>
                    <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{max_qa_group_size:,} words</p>
                </div>
            </div>
        </div>
        
        <div style="background: #e8f4fd; padding: 20px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #003087; margin-top: 0;">Management Discussion Speaker Breakdown</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">'''
    
    # Add MD speaker stats
    for speaker, stats in sorted(md_speaker_stats.items(), key=lambda x: x[1]["words"], reverse=True):
        if stats["words"] > 0:  # Only show speakers with actual content
            html_content += f'''
                <div style="background: white; padding: 10px; border-radius: 3px; border-left: 3px solid #003087;">
                    <p style="font-weight: bold; margin: 5px 0; color: #003087;">{speaker}</p>
                    <p style="margin: 5px 0;">{stats["words"]:,} words</p>
                    <p style="font-size: 12px; color: #666; margin: 5px 0;">{stats["count"]} blocks</p>
                </div>'''
    
    html_content += '''
            </div>
        </div>
        
        <p style="text-align: center; color: #666; margin-top: 20px;">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </p>
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
        
        // Search and filter functionality
        function filterTranscripts() {{
            const searchValue = document.getElementById('searchInput').value.toLowerCase();
            const filterValue = document.getElementById('filterSelect').value;
            const transcriptRows = document.querySelectorAll('.transcript-row');
            
            transcriptRows.forEach(row => {{
                const transcriptId = row.querySelector('h3').textContent.toLowerCase();
                const hasIssues = row.querySelector('.issues') !== null;
                const issuesList = row.querySelector('.issues');
                
                let hasGaps = false;
                let hasOrderIssues = false;
                if (issuesList) {{
                    const issueTexts = issuesList.textContent;
                    hasGaps = issueTexts.includes('has no QA ID');
                    hasOrderIssues = issueTexts.includes('Non-consecutive');
                }}
                
                // Apply search filter
                const matchesSearch = transcriptId.includes(searchValue);
                
                // Apply issue filter
                let matchesFilter = true;
                switch(filterValue) {{
                    case 'perfect':
                        matchesFilter = !hasIssues;
                        break;
                    case 'gaps':
                        matchesFilter = hasGaps;
                        break;
                    case 'order':
                        matchesFilter = hasOrderIssues;
                        break;
                    case 'any-issue':
                        matchesFilter = hasIssues;
                        break;
                }}
                
                // Show/hide based on both filters
                row.style.display = (matchesSearch && matchesFilter) ? 'block' : 'none';
            }});
        }}
        
        // Reset filters
        function resetFilters() {{
            document.getElementById('searchInput').value = '';
            document.getElementById('filterSelect').value = 'all';
            filterTranscripts();
        }}
        
        // Attach event listeners
        document.getElementById('searchInput').addEventListener('input', filterTranscripts);
        document.getElementById('filterSelect').addEventListener('change', filterTranscripts);
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
        data = json.loads(content_data.decode('utf-8'))
        
        # Handle both list and dict formats (Stage 5 might output either)
        if isinstance(data, dict) and 'records' in data:
            records = data['records']
            print(f"Loaded {len(records)} records from dict format")
        elif isinstance(data, list):
            records = data
            print(f"Loaded {len(records)} records from list format")
        else:
            print("Unexpected data format in Stage 5 output")
            return
        
        # Detect which field name is being used for sections
        if records:
            first_record = records[0]
            section_field = None
            if 'section_type' in first_record:
                section_field = 'section_type'
                print("Detected field name: section_type")
            elif 'section_name' in first_record:
                section_field = 'section_name'
                print("Detected field name: section_name")
            else:
                print("WARNING: No section field found in records")
                print(f"Available fields: {list(first_record.keys())}")
        
        print(f"Total records loaded: {len(records)}")
        
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