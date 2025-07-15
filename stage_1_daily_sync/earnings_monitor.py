#!/usr/bin/env python3
"""
Earnings Transcript Monitor
Runs Stage 1 daily sync every 5 minutes and provides real-time macOS notifications
for new earnings transcripts. Perfect for monitoring during earnings season!
"""

import subprocess
import time
import json
import os
import sys
import signal
import threading
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict

# Configuration
SCRIPT_PATH = Path(__file__).parent / "1_transcript_daily_sync.py"
RUN_FREQUENCY = 300  # 5 minutes in seconds
STATE_FILE = Path(__file__).parent / ".monitor_state.json"
HISTORY_FILE = Path(__file__).parent / ".transcript_history.json"

# Enhanced regex patterns for parsing
PATTERNS = {
    "processing": re.compile(r"Processing: (.+?) \((.+?)\) - (\d+) transcripts"),
    "new_download": re.compile(r"Downloaded and uploaded (.+?) \([\d,]+ bytes\)"),
    "updated_version": re.compile(r"Downloaded updated version: (.+)"),
    "completed": re.compile(r"Completed (.+?): Downloaded (\d+) new transcripts total"),
    "found_new": re.compile(r"Found (\d+) new and (\d+) updated (.+?) transcripts to download"),
    "stage_complete": re.compile(r"STAGE 1 DAILY SYNC COMPLETE"),
    "total_downloaded": re.compile(r"Total new transcripts downloaded: (\d+)"),
    "no_transcripts": re.compile(r"No transcripts found for (.+)")
}

# Global flag for graceful shutdown
shutdown_requested = False
monitor_thread = None

def send_notification(title, message, subtitle=None, sound=True):
    """Send enhanced macOS notification."""
    try:
        script_parts = [f'display notification "{message}"']
        script_parts.append(f'with title "{title}"')
        
        if subtitle:
            script_parts.append(f'subtitle "{subtitle}"')
        
        if sound:
            script_parts.append('sound name "Glass"')
        
        script = " ".join(script_parts)
        
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True
        )
    except Exception as e:
        print(f"Failed to send notification: {e}")

def load_state():
    """Load monitor state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
            print(f"Failed to load state: {e}")
            pass
    return {
        "last_run": None,
        "runs_since_start": 0,
        "total_transcripts_session": 0,
        "institutions_with_activity": set()
    }

def save_state(state):
    """Save monitor state."""
    try:
        # Convert sets to lists for JSON serialization
        state_copy = state.copy()
        if "institutions_with_activity" in state_copy:
            state_copy["institutions_with_activity"] = list(state_copy["institutions_with_activity"])
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state_copy, f, indent=2)
    except Exception as e:
        print(f"Failed to save state: {e}")

def load_history():
    """Load transcript history."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
            print(f"Failed to load history: {e}")
            pass
    return {"transcripts": {}, "last_updated": None}

def save_history(history):
    """Save transcript history."""
    try:
        history["last_updated"] = datetime.now().isoformat()
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Failed to save history: {e}")

def parse_transcript_filename(filename):
    """Extract information from transcript filename."""
    # Format: {ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml
    parts = filename.replace('.xml', '').split('_')
    if len(parts) >= 7:
        return {
            "ticker": parts[0],
            "date": parts[1],
            "event_type": parts[2],
            "transcript_type": parts[3],
            "filename": filename
        }
    return None

def monitor_output(process, state, history):
    """Monitor process output in real-time."""
    current_run_results = {
        "new_transcripts": [],
        "updated_transcripts": [],
        "institutions": defaultdict(lambda: {"new": 0, "updated": 0, "total": 0}),
        "total_downloaded": 0
    }
    
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        
        line = line.strip()
        print(line)  # Echo to terminal
        
        # Check for new transcript downloads
        match = PATTERNS["new_download"].search(line)
        if match:
            filename = match.group(1)
            current_run_results["new_transcripts"].append(filename)
            
            # Parse filename for details
            info = parse_transcript_filename(filename)
            if info:
                ticker = info["ticker"]
                current_run_results["institutions"][ticker]["new"] += 1
                
                # Send immediate notification for new transcript
                send_notification(
                    f"New Transcript: {ticker}",
                    f"{info['transcript_type']} transcript for {info['date']}",
                    f"Event: {info['event_type']}"
                )
        
        # Check for updated versions
        match = PATTERNS["updated_version"].search(line)
        if match:
            filename = match.group(1)
            current_run_results["updated_transcripts"].append(filename)
            
            info = parse_transcript_filename(filename)
            if info:
                ticker = info["ticker"]
                current_run_results["institutions"][ticker]["updated"] += 1
        
        # Check for summary information
        match = PATTERNS["found_new"].search(line)
        if match:
            new_count = int(match.group(1))
            updated_count = int(match.group(2))
            transcript_type = match.group(3)
            
            if new_count > 0 or updated_count > 0:
                # Extract ticker from previous lines (this is a bit fragile but works)
                for prev_line in line.split('\n')[-5:]:
                    if "Processing:" in prev_line:
                        ticker_match = PATTERNS["processing"].search(prev_line)
                        if ticker_match:
                            ticker = ticker_match.group(2)
                            send_notification(
                                f"Found Transcripts: {ticker}",
                                f"{new_count} new, {updated_count} updated",
                                f"{transcript_type} transcripts"
                            )
                            break
        
        # Check for completion
        match = PATTERNS["completed"].search(line)
        if match:
            ticker = match.group(1)
            downloaded = int(match.group(2))
            current_run_results["institutions"][ticker]["total"] = downloaded
            current_run_results["total_downloaded"] += downloaded
            
            if downloaded > 0:
                state["institutions_with_activity"].add(ticker)
        
        # Check for stage completion
        if PATTERNS["stage_complete"].search(line):
            # Always send summary notification after each run
            if current_run_results["total_downloaded"] > 0:
                institutions_with_new = [
                    f"{ticker} ({info['new']})"
                    for ticker, info in current_run_results["institutions"].items()
                    if info["new"] > 0
                ]
                
                message = "Banks with new transcripts: " + ", ".join(institutions_with_new[:3])
                if len(institutions_with_new) > 3:
                    message += f" and {len(institutions_with_new) - 3} more"
                
                send_notification(
                    "Sync Complete - New Transcripts!",
                    message,
                    f"Total: {current_run_results['total_downloaded']} transcripts downloaded"
                )
            else:
                # Send notification even when no new transcripts found
                session_total = state.get('total_transcripts_session', 0)
                send_notification(
                    "Sync Complete - No New Transcripts",
                    "No new transcripts found this run",
                    f"Session total: {session_total} transcripts today",
                    sound=False  # Quieter notification for no-activity runs
                )
    
    return current_run_results

def run_stage_1_with_monitoring(state, history):
    """Run Stage 1 with real-time monitoring."""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Stage 1 daily sync...")
    
    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT_PATH)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Monitor output in real-time
        results = monitor_output(process, state, history)
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code != 0:
            send_notification(
                "Stage 1 Sync Failed",
                f"Script failed with return code {return_code}",
                "Check terminal for details"
            )
            return None
        
        return results
        
    except Exception as e:
        print(f"Error running Stage 1: {e}")
        send_notification(
            "Stage 1 Sync Error",
            str(e),
            "Check terminal for details"
        )
        return None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Shutdown requested...")
    shutdown_requested = True

def format_status_notification(state):
    """Format periodic status notification."""
    runs = state["runs_since_start"]
    total = state["total_transcripts_session"]
    active_institutions = len(state.get("institutions_with_activity", []))
    
    if runs > 0 and runs % 12 == 0:  # Every hour (12 * 5 minutes)
        send_notification(
            "Stage 1 Monitor Status",
            f"Running for {runs * 5} minutes",
            f"{total} transcripts from {active_institutions} institutions today",
            sound=False
        )

def main():
    """Main monitor loop."""
    global shutdown_requested
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\nEarnings Transcript Monitor")
    print("=" * 60)
    print(f"\nChecking for new transcripts every {RUN_FREQUENCY // 60} minutes")
    print("\nFeatures:")
    print("  ✓ Real-time popup notifications for each new transcript")
    print("  ✓ Summary notifications after each run")
    print("  ✓ Tracks all banks with activity")
    print("  ✓ Automatic version management")
    print("\nRequirements:")
    print("  • config.json on NAS must have stage_1 settings")
    print("  • .env file with all credentials")
    print("  • macOS notification permissions enabled")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)
    
    # Send startup notification
    send_notification(
        "Earnings Transcript Monitor Started",
        "You'll be notified immediately when new transcripts are detected",
        f"Checking every {RUN_FREQUENCY // 60} minutes"
    )
    
    # Load state and history
    state = load_state()
    if "institutions_with_activity" in state:
        state["institutions_with_activity"] = set(state["institutions_with_activity"])
    else:
        state["institutions_with_activity"] = set()
    
    history = load_history()
    
    # Reset session counters
    state["runs_since_start"] = 0
    state["total_transcripts_session"] = 0
    
    # Main loop
    while not shutdown_requested:
        try:
            # Increment run counter
            state["runs_since_start"] += 1
            
            # Run Stage 1 with monitoring
            results = run_stage_1_with_monitoring(state, history)
            
            if results:
                # Update state
                state["last_run"] = datetime.now().isoformat()
                state["total_transcripts_session"] += results["total_downloaded"]
                save_state(state)
                
                # Update history
                for transcript in results["new_transcripts"] + results["updated_transcripts"]:
                    info = parse_transcript_filename(transcript)
                    if info:
                        key = f"{info['ticker']}_{info['date']}_{info['event_type']}"
                        history["transcripts"][key] = {
                            "filename": transcript,
                            "first_seen": history["transcripts"].get(key, {}).get("first_seen", datetime.now().isoformat()),
                            "last_updated": datetime.now().isoformat(),
                            "transcript_type": info["transcript_type"]
                        }
                save_history(history)
            
            # Send periodic status updates
            format_status_notification(state)
            
            # Wait for next run
            if not shutdown_requested:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Next run in {RUN_FREQUENCY // 60} minutes...")
                for _ in range(RUN_FREQUENCY):
                    if shutdown_requested:
                        break
                    time.sleep(1)
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitor error: {e}")
            send_notification(
                "Monitor Error",
                str(e),
                "Monitor will continue running"
            )
            if not shutdown_requested:
                time.sleep(RUN_FREQUENCY)
    
    # Cleanup and final notification
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Monitor stopped")
    
    # Send summary notification
    total = state.get("total_transcripts_session", 0)
    runs = state.get("runs_since_start", 0)
    institutions = len(state.get("institutions_with_activity", []))
    
    if total > 0:
        send_notification(
            "Monitor Session Complete",
            f"Downloaded {total} transcripts from {institutions} institutions",
            f"Ran {runs} times over {runs * 5} minutes"
        )
    else:
        send_notification(
            "Monitor Stopped",
            "No new transcripts were found during this session",
            f"Ran {runs} times"
        )

if __name__ == "__main__":
    main()