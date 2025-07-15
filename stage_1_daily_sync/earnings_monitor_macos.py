#!/usr/bin/env python3
"""
Earnings Transcript Monitor
Runs Stage 1 daily sync every 5 minutes and provides real-time GUI notifications
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
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import queue

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
qt_app = None
notification_queue = queue.Queue()

class NotificationWindow(QWidget):
    """PyQt5 notification window that auto-closes."""
    def __init__(self, title, message, subtitle=None, duration=5000):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        # Set up layout
        layout = QVBoxLayout()
        
        # Title label
        title_label = QLabel(title)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # Subtitle label (if provided)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_font = QFont()
            subtitle_font.setPointSize(11)
            subtitle_label.setFont(subtitle_font)
            subtitle_label.setStyleSheet("color: #666;")
            layout.addWidget(subtitle_label)
        
        # Message label
        message_label = QLabel(message)
        message_font = QFont()
        message_font.setPointSize(12)
        message_label.setFont(message_font)
        layout.addWidget(message_label)
        
        self.setLayout(layout)
        
        # Style the window
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        # Position in top-right corner
        self.adjustSize()
        screen = QApplication.desktop().screenGeometry()
        self.move(screen.width() - self.width() - 20, 40)
        
        # Auto-close timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.close)
        self.timer.start(duration)
        
        self.show()

def send_notification(title, message, subtitle=None, sound=True):
    """Queue notification for display in Qt thread."""
    notification_queue.put((title, message, subtitle))

def load_state():
    """Load monitor state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except:
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
        except:
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
        "total_downloaded": 0,
        "has_activity": False
    }
    
    # Track if we're in verbose mode (when transcripts are found)
    verbose_mode = False
    
    for line in iter(process.stdout.readline, ''):
        if not line:
            break
        
        line = line.strip()
        
        # Detect if we have transcript activity
        if any(pattern in line for pattern in [
            "Found", "Downloaded", "new transcript", "updated", 
            "Processing:", "Completed"
        ]) and not "No transcripts found" in line:
            verbose_mode = True
            current_run_results["has_activity"] = True
        
        # Only print detailed logs if we have activity
        if verbose_mode or "STAGE 1 DAILY SYNC COMPLETE" in line:
            print(line)
        elif "No new transcripts found" in line or "Discovery complete: No new transcripts" in line:
            # Simple one-liner for no activity
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] No new transcripts found - monitoring continues...")
        
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
            # Send summary notification
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
                # Silent when no new transcripts
                pass
    
    return current_run_results

def run_stage_1_with_monitoring(state, history):
    """Run Stage 1 with real-time monitoring."""
    # Only print starting message if we had activity in the last run
    if state.get("last_run_had_activity", False):
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

class NotificationHandler(QThread):
    """Handles notifications in Qt thread."""
    def __init__(self):
        super().__init__()
        self.active = True
        
    def run(self):
        """Process notification queue."""
        while self.active:
            try:
                # Check for notifications with timeout
                title, message, subtitle = notification_queue.get(timeout=0.1)
                
                # Create and show notification
                notification = NotificationWindow(title, message, subtitle)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Notification handler error: {e}")
                
    def stop(self):
        """Stop the notification handler."""
        self.active = False

def main():
    """Main monitor loop."""
    global shutdown_requested, qt_app
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\nEarnings Transcript Monitor")
    print("=" * 60)
    print(f"\nChecking for new transcripts every {RUN_FREQUENCY // 60} minutes")
    print("\nFeatures:")
    print("  ✓ Real-time GUI popup notifications for each new transcript")
    print("  ✓ Summary notifications after each run")
    print("  ✓ Tracks all banks with activity")
    print("  ✓ Automatic version management")
    print("  ✓ Streamlined logging (minimal when no activity)")
    print("\nRequirements:")
    print("  • config.json on NAS must have stage_1 settings")
    print("  • .env file with all credentials")
    print("  • PyQt5 installed (pip install PyQt5)")
    print("\nPress Ctrl+C to stop")
    print("-" * 60)
    
    # Initialize Qt application
    qt_app = QApplication(sys.argv)
    
    # Start notification handler thread
    notification_handler = NotificationHandler()
    notification_handler.start()
    
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
                state["last_run_had_activity"] = results.get("has_activity", False)
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
                # Only print "next run" message if we had activity
                if results and results.get("has_activity", False):
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
    
    # Stop notification handler
    notification_handler.stop()
    notification_handler.wait()
    
    # Clean up Qt application
    qt_app.quit()

if __name__ == "__main__":
    main()