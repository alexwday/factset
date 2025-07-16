"""
Standalone NearRealTime Transcript Cleanup Script

Purpose: One-time removal of all NearRealTime folders and transcripts from NAS.
This script safely removes existing NearRealTime data before updating the configuration 
to prevent future downloads.

Features:
- Comprehensive NAS scanning for NearRealTime folders
- Dry-run mode for safe preview before deletion
- Detailed logging of all operations
- Secure NTLM authentication
- Complete audit trail

Usage:
    python cleanup_nearrealtime_transcripts.py --dry-run    # Preview only
    python cleanup_nearrealtime_transcripts.py             # Execute deletion
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Set, Optional
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment
NAS_USERNAME = os.getenv('NAS_USERNAME')
NAS_PASSWORD = os.getenv('NAS_PASSWORD')
NAS_SERVER_IP = os.getenv('NAS_SERVER_IP')
NAS_SERVER_NAME = os.getenv('NAS_SERVER_NAME')
NAS_SHARE_NAME = os.getenv('NAS_SHARE_NAME')
NAS_BASE_PATH = os.getenv('NAS_BASE_PATH')
NAS_PORT = int(os.getenv('NAS_PORT', 445))
CLIENT_MACHINE_NAME = os.getenv('CLIENT_MACHINE_NAME')

# Validate required environment variables
required_env_vars = [
    'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
    'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CLIENT_MACHINE_NAME'
]

def validate_environment() -> None:
    """Validate all required environment variables are present."""
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")

def validate_nas_path(path: str) -> bool:
    """Validate NAS path for security - prevents directory traversal attacks."""
    if not path or not isinstance(path, str):
        return False
    if '..' in path or path.startswith('/'):
        return False
    return True

def sanitize_path_for_logging(path: str) -> str:
    """Remove sensitive information from paths before logging."""
    # Replace server info with placeholder for security
    if NAS_SERVER_NAME and NAS_SERVER_NAME in path:
        path = path.replace(NAS_SERVER_NAME, "[NAS_SERVER]")
    if NAS_SHARE_NAME and NAS_SHARE_NAME in path:
        path = path.replace(NAS_SHARE_NAME, "[SHARE]")
    return path

def setup_logging(dry_run: bool = False) -> logging.Logger:
    """Setup comprehensive logging for the cleanup operation."""
    # Create logs directory if it doesn't exist
    log_dir = "cleanup_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "dry_run" if dry_run else "execution"
    log_filename = f"{log_dir}/nearrealtime_cleanup_{mode_suffix}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting NearRealTime cleanup - Mode: {'DRY RUN' if dry_run else 'EXECUTION'}")
    logger.info(f"Log file: {log_filename}")
    
    return logger

class NASConnection:
    """Secure NAS connection management with proper cleanup."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.connection: Optional[SMBConnection] = None
    
    def __enter__(self):
        """Context manager entry - establish NAS connection."""
        try:
            self.logger.info("Establishing NAS connection...")
            self.connection = SMBConnection(
                username=NAS_USERNAME,
                password=NAS_PASSWORD,
                my_name=CLIENT_MACHINE_NAME,
                remote_name=NAS_SERVER_NAME,
                domain='',
                use_ntlm_v2=True,
                is_direct_tcp=True
            )
            
            if not self.connection.connect(NAS_SERVER_IP, NAS_PORT):
                raise ConnectionError("Failed to connect to NAS server")
            
            self.logger.info("NAS connection established successfully")
            return self
            
        except Exception as e:
            self.logger.error(f"Failed to establish NAS connection: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure connection cleanup."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("NAS connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing NAS connection: {e}")

class NearRealTimeCleanup:
    """Main cleanup class for removing NearRealTime transcripts."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = setup_logging(dry_run)
        self.nearrealtime_folders: List[Dict[str, str]] = []
        self.deletion_summary = {
            'folders_found': 0,
            'folders_deleted': 0,
            'files_deleted': 0,
            'total_size_bytes': 0,
            'errors': []
        }
    
    def discover_nearrealtime_folders(self, nas_conn: NASConnection) -> None:
        """Recursively discover all NearRealTime folders on NAS."""
        self.logger.info("Starting discovery of NearRealTime folders...")
        
        try:
            # Start scanning from the Data directory
            data_path = f"{NAS_BASE_PATH}/Outputs/Data"
            self._scan_directory_recursive(nas_conn.connection, data_path, "")
            
            self.deletion_summary['folders_found'] = len(self.nearrealtime_folders)
            self.logger.info(f"Discovery complete. Found {len(self.nearrealtime_folders)} NearRealTime folders")
            
            # Log all discovered folders
            for folder_info in self.nearrealtime_folders:
                safe_path = sanitize_path_for_logging(folder_info['full_path'])
                self.logger.info(f"Found NearRealTime folder: {safe_path}")
                
        except Exception as e:
            self.logger.error(f"Error during folder discovery: {e}")
            raise
    
    def _scan_directory_recursive(self, connection: SMBConnection, base_path: str, current_path: str) -> None:
        """Recursively scan directories for NearRealTime folders."""
        try:
            full_path = f"{base_path}/{current_path}".strip('/')
            if not validate_nas_path(full_path):
                self.logger.warning(f"Skipping invalid path: {sanitize_path_for_logging(full_path)}")
                return
            
            # List directory contents
            file_list = connection.listPath(NAS_SHARE_NAME, full_path)
            
            for file_info in file_list:
                # Skip . and .. entries
                if file_info.filename in ['.', '..']:
                    continue
                
                # Build path for this item
                item_path = f"{current_path}/{file_info.filename}".strip('/')
                full_item_path = f"{base_path}/{item_path}".strip('/')
                
                if file_info.isDirectory:
                    # Check if this is a NearRealTime folder
                    if file_info.filename == "NearRealTime":
                        self.nearrealtime_folders.append({
                            'folder_name': file_info.filename,
                            'parent_path': f"{base_path}/{current_path}".strip('/'),
                            'full_path': full_item_path,
                            'relative_path': item_path
                        })
                        self.logger.debug(f"Found NearRealTime folder: {sanitize_path_for_logging(full_item_path)}")
                    else:
                        # Recursively scan subdirectories
                        self._scan_directory_recursive(connection, base_path, item_path)
                        
        except Exception as e:
            safe_path = sanitize_path_for_logging(f"{base_path}/{current_path}")
            self.logger.warning(f"Error scanning directory {safe_path}: {e}")
    
    def delete_nearrealtime_folders(self, nas_conn: NASConnection) -> None:
        """Delete all discovered NearRealTime folders."""
        if not self.nearrealtime_folders:
            self.logger.info("No NearRealTime folders found to delete")
            return
        
        self.logger.info(f"{'[DRY RUN] Would delete' if self.dry_run else 'Deleting'} {len(self.nearrealtime_folders)} NearRealTime folders...")
        
        for folder_info in self.nearrealtime_folders:
            try:
                self._delete_folder_recursive(nas_conn.connection, folder_info)
                self.deletion_summary['folders_deleted'] += 1
                
            except Exception as e:
                error_msg = f"Error deleting folder {sanitize_path_for_logging(folder_info['full_path'])}: {e}"
                self.logger.error(error_msg)
                self.deletion_summary['errors'].append(error_msg)
    
    def _delete_folder_recursive(self, connection: SMBConnection, folder_info: Dict[str, str]) -> None:
        """Recursively delete a folder and all its contents."""
        folder_path = folder_info['full_path']
        safe_path = sanitize_path_for_logging(folder_path)
        
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would delete folder: {safe_path}")
            # In dry run, still count files that would be deleted
            self._count_files_recursive(connection, folder_path)
            return
        
        try:
            # First, delete all files and subdirectories
            self._delete_folder_contents(connection, folder_path)
            
            # Then delete the folder itself
            connection.deleteDirectory(NAS_SHARE_NAME, folder_path)
            self.logger.info(f"Successfully deleted folder: {safe_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete folder {safe_path}: {e}")
            raise
    
    def _delete_folder_contents(self, connection: SMBConnection, folder_path: str) -> None:
        """Delete all contents of a folder recursively."""
        try:
            file_list = connection.listPath(NAS_SHARE_NAME, folder_path)
            
            for file_info in file_list:
                if file_info.filename in ['.', '..']:
                    continue
                
                item_path = f"{folder_path}/{file_info.filename}"
                safe_item_path = sanitize_path_for_logging(item_path)
                
                if file_info.isDirectory:
                    # Recursively delete subdirectory
                    self._delete_folder_contents(connection, item_path)
                    connection.deleteDirectory(NAS_SHARE_NAME, item_path)
                    self.logger.debug(f"Deleted subdirectory: {safe_item_path}")
                else:
                    # Delete file
                    connection.deleteFiles(NAS_SHARE_NAME, item_path)
                    self.deletion_summary['files_deleted'] += 1
                    self.deletion_summary['total_size_bytes'] += file_info.file_size
                    self.logger.debug(f"Deleted file: {safe_item_path} ({file_info.file_size} bytes)")
                    
        except Exception as e:
            safe_path = sanitize_path_for_logging(folder_path)
            self.logger.error(f"Error deleting contents of {safe_path}: {e}")
            raise
    
    def _count_files_recursive(self, connection: SMBConnection, folder_path: str) -> None:
        """Count files that would be deleted (for dry run mode)."""
        try:
            file_list = connection.listPath(NAS_SHARE_NAME, folder_path)
            
            for file_info in file_list:
                if file_info.filename in ['.', '..']:
                    continue
                
                item_path = f"{folder_path}/{file_info.filename}"
                
                if file_info.isDirectory:
                    self._count_files_recursive(connection, item_path)
                else:
                    self.deletion_summary['files_deleted'] += 1
                    self.deletion_summary['total_size_bytes'] += file_info.file_size
                    
        except Exception as e:
            safe_path = sanitize_path_for_logging(folder_path)
            self.logger.warning(f"Error counting files in {safe_path}: {e}")
    
    def generate_summary_report(self) -> None:
        """Generate and log comprehensive summary report."""
        self.logger.info("=" * 60)
        self.logger.info("NEARREALTIME CLEANUP SUMMARY REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'EXECUTION'}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("")
        self.logger.info("RESULTS:")
        self.logger.info(f"  NearRealTime folders found: {self.deletion_summary['folders_found']}")
        
        if self.dry_run:
            self.logger.info(f"  NearRealTime folders that would be deleted: {self.deletion_summary['folders_deleted']}")
            self.logger.info(f"  Files that would be deleted: {self.deletion_summary['files_deleted']}")
            self.logger.info(f"  Total size that would be freed: {self.deletion_summary['total_size_bytes']:,} bytes ({self.deletion_summary['total_size_bytes'] / (1024*1024):.2f} MB)")
        else:
            self.logger.info(f"  NearRealTime folders deleted: {self.deletion_summary['folders_deleted']}")
            self.logger.info(f"  Files deleted: {self.deletion_summary['files_deleted']}")
            self.logger.info(f"  Total size freed: {self.deletion_summary['total_size_bytes']:,} bytes ({self.deletion_summary['total_size_bytes'] / (1024*1024):.2f} MB)")
        
        if self.deletion_summary['errors']:
            self.logger.info(f"  Errors encountered: {len(self.deletion_summary['errors'])}")
            for error in self.deletion_summary['errors']:
                self.logger.info(f"    - {error}")
        else:
            self.logger.info("  No errors encountered")
        
        self.logger.info("=" * 60)
        
        # Save summary to JSON file
        summary_file = f"cleanup_logs/summary_{'dry_run' if self.dry_run else 'execution'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(summary_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'dry_run' if self.dry_run else 'execution',
                    'summary': self.deletion_summary,
                    'folders_processed': self.nearrealtime_folders
                }, f, indent=2)
            self.logger.info(f"Summary report saved to: {summary_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save summary report: {e}")
    
    def run_cleanup(self) -> None:
        """Execute the complete cleanup process."""
        try:
            # Validate environment
            validate_environment()
            
            # Establish NAS connection and run cleanup
            with NASConnection(self.logger) as nas_conn:
                # Discover all NearRealTime folders
                self.discover_nearrealtime_folders(nas_conn)
                
                # Delete folders (or simulate in dry run)
                self.delete_nearrealtime_folders(nas_conn)
            
            # Generate summary report
            self.generate_summary_report()
            
            if self.dry_run:
                self.logger.info("DRY RUN COMPLETE - No actual deletions performed")
                self.logger.info("Run without --dry-run flag to execute actual deletion")
            else:
                self.logger.info("CLEANUP COMPLETE - All NearRealTime folders have been removed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Remove all NearRealTime transcript folders from NAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_nearrealtime_transcripts.py --dry-run    # Preview what would be deleted
  python cleanup_nearrealtime_transcripts.py             # Execute actual deletion

This script will:
1. Scan the entire NAS Data directory structure
2. Identify all folders named "NearRealTime"
3. Delete these folders and all their contents
4. Generate comprehensive audit logs

IMPORTANT: Run with --dry-run first to preview changes!
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview mode - show what would be deleted without actually deleting'
    )
    
    args = parser.parse_args()
    
    # Confirm execution if not dry run
    if not args.dry_run:
        print("WARNING: This will permanently delete all NearRealTime transcript folders!")
        print("Press Ctrl+C to cancel, or press Enter to continue...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return
    
    # Run cleanup
    cleanup = NearRealTimeCleanup(dry_run=args.dry_run)
    cleanup.run_cleanup()

if __name__ == "__main__":
    main()