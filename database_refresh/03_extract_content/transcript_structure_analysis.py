"""
Transcript Structure Analysis for Stage 3 Output
Analyzes stage_03_extracted_content.json to understand section distribution, lengths, and content patterns.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional
import statistics
import io

from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
logger = None
config = {}


def setup_logging() -> logging.Logger:
    """Set up console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def log_console(message: str, level: str = "INFO"):
    """Log message to console."""
    global logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


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
            log_console("NAS connection established successfully")
            return conn
        else:
            log_console("Failed to establish NAS connection", "ERROR")
            return None

    except Exception as e:
        log_console(f"Error creating NAS connection: {e}", "ERROR")
        return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_console(f"Failed to download file from NAS: {nas_file_path}", "ERROR")
        return None


def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    try:
        # Ensure parent directory exists
        parent_dir = "/".join(nas_file_path.split("/")[:-1])
        if parent_dir:
            nas_create_directory_recursive(conn, parent_dir)

        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file_obj)
        return True
    except Exception as e:
        log_console(f"Failed to upload file to NAS: {nas_file_path}", "ERROR")
        return False


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""
    normalized_path = dir_path.strip("/").rstrip("/")
    if not normalized_path:
        return False

    path_parts = [part for part in normalized_path.split("/") if part]
    if not path_parts:
        return False

    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part

        if nas_file_exists(conn, current_path):
            continue

        try:
            conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
        except Exception as e:
            if not nas_file_exists(conn, current_path):
                return False

    return True


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception:
        return False


def load_stage3_output(nas_conn: SMBConnection) -> Optional[List[Dict[str, Any]]]:
    """Load Stage 3 extracted transcript sections from NAS."""
    try:
        # Standard Stage 3 output path
        output_path = "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_03_extracted_content.json"
        
        log_console(f"Loading Stage 3 output from: {output_path}")
        
        data = nas_download_file(nas_conn, output_path)
        if not data:
            log_console("Stage 3 output file not found", "ERROR")
            return None
        
        records = json.loads(data.decode("utf-8"))
        log_console(f"Loaded {len(records)} paragraph records from Stage 3 output")
        return records
        
    except json.JSONDecodeError as e:
        log_console(f"Invalid JSON in Stage 3 output: {e}", "ERROR")
        return None
    except Exception as e:
        log_console(f"Error loading Stage 3 output: {e}", "ERROR")
        return None


def analyze_section_distribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze section count distribution across transcripts."""
    log_console("📊 Analyzing section distribution...")
    
    # Group records by transcript (using file_path as unique identifier)
    transcripts = defaultdict(list)
    for record in records:
        file_path = record.get("file_path", "unknown")
        transcripts[file_path].append(record)
    
    section_counts = {}
    section_distributions = defaultdict(list)
    
    for file_path, transcript_records in transcripts.items():
        # Get unique sections in this transcript
        sections = set()
        for record in transcript_records:
            section_id = record.get("section_id")
            section_name = record.get("section_name", "")
            if section_id:
                sections.add((section_id, section_name))
        
        section_count = len(sections)
        section_counts[file_path] = {
            "count": section_count,
            "sections": list(sections)
        }
        section_distributions[section_count].append(file_path)
    
    # Calculate statistics
    counts = [info["count"] for info in section_counts.values()]
    
    analysis = {
        "total_transcripts": len(transcripts),
        "section_count_distribution": dict(section_distributions),
        "statistics": {
            "min_sections": min(counts) if counts else 0,
            "max_sections": max(counts) if counts else 0,
            "mean_sections": statistics.mean(counts) if counts else 0,
            "median_sections": statistics.median(counts) if counts else 0
        },
        "detailed_counts": section_counts
    }
    
    return analysis


def analyze_section_names(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze section names across all transcripts."""
    log_console("📋 Analyzing section names...")
    
    # Group by transcript and section position
    transcripts = defaultdict(lambda: defaultdict(list))
    for record in records:
        file_path = record.get("file_path", "unknown")
        section_id = record.get("section_id", 0)
        section_name = record.get("section_name", "")
        
        transcripts[file_path][section_id].append(section_name)
    
    # Analyze first and second section names
    first_section_names = []
    second_section_names = []
    other_section_names = defaultdict(list)
    
    for file_path, sections in transcripts.items():
        sorted_sections = sorted(sections.keys())
        
        if len(sorted_sections) >= 1:
            first_name = sections[sorted_sections[0]][0] if sections[sorted_sections[0]] else ""
            first_section_names.append(first_name)
        
        if len(sorted_sections) >= 2:
            second_name = sections[sorted_sections[1]][0] if sections[sorted_sections[1]] else ""
            second_section_names.append(second_name)
        
        # Collect other section names for transcripts with >2 sections
        if len(sorted_sections) > 2:
            for i, section_id in enumerate(sorted_sections[2:], 3):
                section_name = sections[section_id][0] if sections[section_id] else ""
                other_section_names[f"Section {i}"].append((file_path, section_name))
    
    analysis = {
        "first_section_names": {
            "unique_names": list(set(first_section_names)),
            "name_counts": Counter(first_section_names),
            "total": len(first_section_names)
        },
        "second_section_names": {
            "unique_names": list(set(second_section_names)),
            "name_counts": Counter(second_section_names),
            "total": len(second_section_names)
        },
        "other_sections": dict(other_section_names)
    }
    
    return analysis


def analyze_content_lengths(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content lengths at section, speaker block, and paragraph levels."""
    log_console("📏 Analyzing content lengths...")
    
    # Group by transcript, section, and speaker block
    transcripts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for record in records:
        file_path = record.get("file_path", "unknown")
        section_id = record.get("section_id", 0)
        speaker_block_id = record.get("speaker_block_id", 0)
        paragraph_content = record.get("paragraph_content", "")
        
        transcripts[file_path][section_id][speaker_block_id].append(paragraph_content)
    
    # Analyze lengths
    section_lengths = []  # Total characters per section
    speaker_block_lengths = []  # Total characters per speaker block
    paragraph_lengths = []  # Characters per paragraph
    
    section_paragraph_counts = []  # Number of paragraphs per section
    speaker_block_paragraph_counts = []  # Number of paragraphs per speaker block
    
    for file_path, sections in transcripts.items():
        for section_id, speaker_blocks in sections.items():
            section_total_chars = 0
            section_total_paragraphs = 0
            
            for speaker_block_id, paragraphs in speaker_blocks.items():
                speaker_block_total_chars = 0
                
                for paragraph in paragraphs:
                    paragraph_length = len(paragraph)
                    paragraph_lengths.append(paragraph_length)
                    speaker_block_total_chars += paragraph_length
                    section_total_chars += paragraph_length
                
                speaker_block_lengths.append(speaker_block_total_chars)
                speaker_block_paragraph_counts.append(len(paragraphs))
                section_total_paragraphs += len(paragraphs)
            
            section_lengths.append(section_total_chars)
            section_paragraph_counts.append(section_total_paragraphs)
    
    def calculate_stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "std_dev": 0}
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    analysis = {
        "section_character_lengths": calculate_stats(section_lengths),
        "speaker_block_character_lengths": calculate_stats(speaker_block_lengths),
        "paragraph_character_lengths": calculate_stats(paragraph_lengths),
        "section_paragraph_counts": calculate_stats(section_paragraph_counts),
        "speaker_block_paragraph_counts": calculate_stats(speaker_block_paragraph_counts),
        "raw_data": {
            "section_lengths": section_lengths,
            "speaker_block_lengths": speaker_block_lengths,
            "paragraph_lengths": paragraph_lengths,
            "section_paragraph_counts": section_paragraph_counts,
            "speaker_block_paragraph_counts": speaker_block_paragraph_counts
        }
    }
    
    return analysis


def identify_outliers(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Identify outliers in content lengths."""
    log_console("🎯 Identifying outliers...")
    
    outliers = {}
    
    for metric_name, raw_data_key in [
        ("section_character_lengths", "section_lengths"),
        ("speaker_block_character_lengths", "speaker_block_lengths"),
        ("paragraph_character_lengths", "paragraph_lengths")
    ]:
        values = analysis["raw_data"][raw_data_key]
        if not values:
            continue
        
        # Calculate IQR for outlier detection
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        q1 = sorted_values[n // 4] if n > 4 else sorted_values[0]
        q3 = sorted_values[3 * n // 4] if n > 4 else sorted_values[-1]
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_values = [v for v in values if v < lower_bound or v > upper_bound]
        
        outliers[metric_name] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outlier_values),
            "outlier_values": sorted(outlier_values),
            "outlier_percentage": (len(outlier_values) / len(values)) * 100 if values else 0
        }
    
    return outliers


def generate_detailed_transcript_analysis(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate detailed analysis for specific transcript categories."""
    log_console("🔍 Generating detailed transcript analysis...")
    
    # Group by transcript
    transcripts = defaultdict(list)
    for record in records:
        file_path = record.get("file_path", "unknown")
        transcripts[file_path].append(record)
    
    # Categorize transcripts by section count
    one_section_transcripts = []
    two_section_transcripts = []
    multi_section_transcripts = []
    
    for file_path, transcript_records in transcripts.items():
        # Get unique sections
        sections = set()
        for record in transcript_records:
            section_id = record.get("section_id")
            section_name = record.get("section_name", "")
            if section_id:
                sections.add((section_id, section_name))
        
        section_count = len(sections)
        transcript_info = {
            "file_path": file_path,
            "filename": file_path.split("/")[-1] if "/" in file_path else file_path,
            "ticker": transcript_records[0].get("ticker", ""),
            "company_name": transcript_records[0].get("company_name", ""),
            "transcript_type": transcript_records[0].get("transcript_type", ""),
            "sections": list(sections),
            "total_paragraphs": len(transcript_records)
        }
        
        if section_count == 1:
            one_section_transcripts.append(transcript_info)
        elif section_count == 2:
            two_section_transcripts.append(transcript_info)
        else:
            multi_section_transcripts.append(transcript_info)
    
    return {
        "one_section_transcripts": one_section_transcripts,
        "two_section_transcripts": two_section_transcripts,
        "multi_section_transcripts": multi_section_transcripts,
        "counts": {
            "one_section": len(one_section_transcripts),
            "two_section": len(two_section_transcripts),
            "multi_section": len(multi_section_transcripts)
        }
    }


def save_analysis_report(nas_conn: SMBConnection, analysis_data: Dict[str, Any]) -> bool:
    """Save comprehensive analysis report to NAS."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
        output_filename = f"transcript_structure_analysis_{timestamp}.json"
        output_file_path = f"{output_path}/{output_filename}"
        
        # Create comprehensive report
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_transcripts": analysis_data["section_distribution"]["total_transcripts"],
                "total_paragraph_records": analysis_data.get("total_records", 0),
                "section_count_distribution": {
                    str(k): len(v) for k, v in analysis_data["section_distribution"]["section_count_distribution"].items()
                }
            },
            "section_distribution_analysis": analysis_data["section_distribution"],
            "section_names_analysis": analysis_data["section_names"],
            "content_length_analysis": analysis_data["content_lengths"],
            "outlier_analysis": analysis_data["outliers"],
            "detailed_transcript_breakdown": analysis_data["detailed_analysis"]
        }
        
        # Convert to JSON and upload
        report_json = json.dumps(report, indent=2)
        report_obj = io.BytesIO(report_json.encode("utf-8"))
        
        if nas_upload_file(nas_conn, report_obj, output_file_path):
            log_console(f"✅ Analysis report saved: {output_filename}")
            return True
        else:
            log_console("❌ Failed to save analysis report", "ERROR")
            return False
            
    except Exception as e:
        log_console(f"❌ Error saving analysis report: {e}", "ERROR")
        return False


def print_summary_results(analysis_data: Dict[str, Any]):
    """Print summary results to console."""
    log_console("\n" + "="*80)
    log_console("📊 TRANSCRIPT STRUCTURE ANALYSIS SUMMARY")
    log_console("="*80)
    
    # Section Distribution Summary
    section_dist = analysis_data["section_distribution"]
    log_console(f"\n📈 SECTION COUNT DISTRIBUTION:")
    log_console(f"Total Transcripts: {section_dist['total_transcripts']}")
    
    for count, files in section_dist["section_count_distribution"].items():
        log_console(f"  {count} sections: {len(files)} transcripts")
    
    stats = section_dist["statistics"]
    log_console(f"  Average sections per transcript: {stats['mean_sections']:.1f}")
    log_console(f"  Range: {stats['min_sections']} - {stats['max_sections']} sections")
    
    # Section Names Summary
    section_names = analysis_data["section_names"]
    log_console(f"\n📋 SECTION NAMES ANALYSIS:")
    
    log_console(f"First Section Names ({section_names['first_section_names']['total']} transcripts):")
    for name, count in section_names["first_section_names"]["name_counts"].most_common(5):
        log_console(f"  '{name}': {count} times")
    
    log_console(f"Second Section Names ({section_names['second_section_names']['total']} transcripts):")
    for name, count in section_names["second_section_names"]["name_counts"].most_common(5):
        log_console(f"  '{name}': {count} times")
    
    # Content Length Summary
    content_lengths = analysis_data["content_lengths"]
    log_console(f"\n📏 CONTENT LENGTH ANALYSIS:")
    
    log_console(f"Section Lengths (characters):")
    stats = content_lengths["section_character_lengths"]
    log_console(f"  Average: {stats['mean']:.0f} chars, Range: {stats['min']:.0f} - {stats['max']:.0f}")
    
    log_console(f"Speaker Block Lengths (characters):")
    stats = content_lengths["speaker_block_character_lengths"]
    log_console(f"  Average: {stats['mean']:.0f} chars, Range: {stats['min']:.0f} - {stats['max']:.0f}")
    
    log_console(f"Paragraph Lengths (characters):")
    stats = content_lengths["paragraph_character_lengths"]
    log_console(f"  Average: {stats['mean']:.0f} chars, Range: {stats['min']:.0f} - {stats['max']:.0f}")
    
    # Outlier Summary
    outliers = analysis_data["outliers"]
    log_console(f"\n🎯 OUTLIER DETECTION:")
    for metric, outlier_info in outliers.items():
        log_console(f"{metric}: {outlier_info['outlier_count']} outliers ({outlier_info['outlier_percentage']:.1f}%)")
    
    # Detailed Breakdown Summary
    detailed = analysis_data["detailed_analysis"]
    counts = detailed["counts"]
    log_console(f"\n🔍 TRANSCRIPT BREAKDOWN:")
    log_console(f"  1 section: {counts['one_section']} transcripts")
    log_console(f"  2 sections: {counts['two_section']} transcripts") 
    log_console(f"  3+ sections: {counts['multi_section']} transcripts")
    
    # Show specific examples for non-standard transcripts
    if counts["one_section"] > 0:
        log_console(f"\n📄 ONE SECTION EXAMPLES:")
        for transcript in detailed["one_section_transcripts"][:3]:
            log_console(f"  {transcript['filename']} ({transcript['ticker']}): '{transcript['sections'][0][1]}'")
    
    if counts["multi_section"] > 0:
        log_console(f"\n📄 MULTI-SECTION EXAMPLES:")
        for transcript in detailed["multi_section_transcripts"][:3]:
            section_names = [section[1] for section in transcript["sections"]]
            log_console(f"  {transcript['filename']} ({transcript['ticker']}): {len(transcript['sections'])} sections")
            for i, name in enumerate(section_names, 1):
                log_console(f"    Section {i}: '{name}'")
    
    log_console("\n" + "="*80)


def main():
    """Main analysis function."""
    global logger
    
    # Initialize logging
    logger = setup_logging()
    log_console("=== TRANSCRIPT STRUCTURE ANALYSIS ===")
    
    nas_conn = None
    
    try:
        # Connect to NAS
        log_console("Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            log_console("Failed to establish NAS connection", "ERROR")
            return
        
        # Load Stage 3 output
        log_console("Loading Stage 3 output data...")
        records = load_stage3_output(nas_conn)
        if not records:
            log_console("No data loaded from Stage 3 output", "ERROR")
            return
        
        log_console(f"Analyzing {len(records)} paragraph records...")
        
        # Perform all analyses
        analysis_data = {}
        
        # 1. Section distribution analysis
        analysis_data["section_distribution"] = analyze_section_distribution(records)
        
        # 2. Section names analysis
        analysis_data["section_names"] = analyze_section_names(records)
        
        # 3. Content length analysis
        analysis_data["content_lengths"] = analyze_content_lengths(records)
        
        # 4. Outlier detection
        analysis_data["outliers"] = identify_outliers(analysis_data["content_lengths"])
        
        # 5. Detailed transcript analysis
        analysis_data["detailed_analysis"] = generate_detailed_transcript_analysis(records)
        
        # Add total records count
        analysis_data["total_records"] = len(records)
        
        # Save comprehensive report
        log_console("Saving analysis report...")
        save_analysis_report(nas_conn, analysis_data)
        
        # Print summary to console
        print_summary_results(analysis_data)
        
        log_console("✅ Analysis complete!")
        
    except Exception as e:
        log_console(f"❌ Analysis failed: {e}", "ERROR")
        
    finally:
        if nas_conn:
            try:
                nas_conn.close()
                log_console("NAS connection closed")
            except Exception as e:
                log_console(f"Error closing NAS connection: {e}", "WARNING")


if __name__ == "__main__":
    main()