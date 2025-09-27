"""
Stage 9: Master Database Consolidation & Archive
Consolidates processed records into master database and archives refresh folder.
"""

import os
import json
import csv
import io
import zipfile
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []
error_log = []


def setup_logging() -> logging.Logger:
    """Set up minimal console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def log_console(message: str, level: str = "INFO"):
    """Log minimal message to console."""
    global logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)


def log_execution(message: str, details: Dict[str, Any] = None):
    """Log detailed execution information for main log file."""
    global execution_log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "details": details or {},
    }
    execution_log.append(log_entry)


def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information for error log file."""
    global error_log
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    error_log.append(error_entry)


def validate_environment_variables() -> None:
    """Validate all required environment variables are present."""
    required_env_vars = [
        "NAS_USERNAME",
        "NAS_PASSWORD",
        "NAS_SERVER_IP",
        "NAS_SERVER_NAME",
        "NAS_SHARE_NAME",
        "NAS_PORT",
        "CONFIG_PATH",
        "CLIENT_MACHINE_NAME",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log_error(error_msg, "environment_validation", {"missing_variables": missing_vars})
        raise ValueError(error_msg)

    log_execution("Environment variables validated successfully",
                 {"total_variables": len(required_env_vars)})


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
            log_execution("NAS connection established successfully",
                         {"connection_type": "SMB/CIFS", "port": nas_port})
            return conn
        else:
            log_error("Failed to establish NAS connection", "nas_connection")
            return None

    except Exception as e:
        log_error(f"Error creating NAS connection: {e}", "nas_connection",
                 {"exception_type": type(e).__name__})
        return None


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""
    try:
        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            raise FileNotFoundError(f"Configuration file not found at {os.getenv('CONFIG_PATH')}")

        stage_config = yaml.safe_load(config_data.decode("utf-8"))
        log_execution("Configuration loaded successfully", {"sections": list(stage_config.keys())})

        # Validate Stage 9 configuration
        if "stage_09_master_consolidation" not in stage_config:
            # Create default configuration if not present
            stage_config["stage_09_master_consolidation"] = {
                "description": "Master database consolidation and archive",
                "stage_02_removal_queue": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_02_removal_queue.json",
                "stage_08_output_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_08_embeddings.csv",
                "master_database_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Database/master_embeddings.csv",
                "archive_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Archives",
                "refresh_folder_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh",
                "output_logs_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs",
                "chunk_size": 10000,  # Process CSVs in chunks for memory efficiency
                "archive_enabled": True,
                "delete_refresh_after_archive": False  # Safety: don't auto-delete
            }
            log_execution("Using default Stage 9 configuration")

        return stage_config

    except Exception as e:
        error_msg = f"Error loading configuration from NAS: {e}"
        log_error(error_msg, "config_load", {"exception_type": type(e).__name__})
        raise


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download",
                 {"path": nas_file_path, "error": str(e)})
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
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload",
                 {"path": nas_file_path, "error": str(e)})
        return False


def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file exists on the NAS."""
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), file_path)
        return True
    except Exception:
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
                log_error(f"Failed to create directory: {current_path}", "directory_creation",
                         {"path": current_path, "error": str(e)})
                return False

    return True


def nas_list_files(conn: SMBConnection, directory_path: str, pattern: str = None) -> List[str]:
    """List files in a NAS directory with optional pattern matching."""
    try:
        files = conn.listPath(os.getenv("NAS_SHARE_NAME"), directory_path)
        file_list = []
        for file_info in files:
            if not file_info.isDirectory and file_info.filename not in [".", ".."]:
                if pattern is None or pattern in file_info.filename:
                    file_list.append(file_info.filename)
        return file_list
    except Exception as e:
        log_error(f"Failed to list files in directory: {directory_path}", "nas_list_files",
                 {"directory": directory_path, "error": str(e)})
        return []


def load_removal_queue(nas_conn: SMBConnection) -> List[str]:
    """Load the removal queue from Stage 2 output."""
    removal_queue_path = config["stage_09_master_consolidation"]["stage_02_removal_queue"]

    log_console("Loading removal queue from Stage 2...")
    removal_data = nas_download_file(nas_conn, removal_queue_path)

    if not removal_data:
        log_console("No removal queue found - no files to remove")
        return []

    try:
        removal_records = json.loads(removal_data.decode("utf-8"))
        # Extract file_path from each record
        file_paths_to_remove = [record["file_path"] for record in removal_records]
        log_execution(f"Loaded {len(file_paths_to_remove)} files to remove",
                     {"count": len(file_paths_to_remove)})
        return file_paths_to_remove
    except Exception as e:
        log_error(f"Error parsing removal queue: {e}", "removal_queue_parse")
        return []


def process_master_csv_streaming(nas_conn: SMBConnection, removal_set: Set[str],
                                 new_records_path: str) -> Dict[str, Any]:
    """Process master CSV with streaming to handle large files efficiently."""
    master_path = config["stage_09_master_consolidation"]["master_database_path"]
    chunk_size = config["stage_09_master_consolidation"].get("chunk_size", 10000)

    stats = {
        "original_records": 0,
        "records_removed": 0,
        "new_records_added": 0,
        "final_records": 0,
        "file_size_mb": 0
    }

    # Create temporary output
    output = io.StringIO()
    writer = None
    fieldnames = None

    # Process existing master database if it exists
    if nas_file_exists(nas_conn, master_path):
        log_console("Processing existing master database...")
        master_data = nas_download_file(nas_conn, master_path)

        if master_data:
            # Read existing CSV
            master_csv = io.StringIO(master_data.decode("utf-8"))
            reader = csv.DictReader(master_csv)
            fieldnames = reader.fieldnames

            # Write header
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            # Process records, filtering out removals
            for row in reader:
                stats["original_records"] += 1

                # Check if this record should be removed
                # Stage 8 uses 'filename' field for the transcript filename (e.g., "AAPL_Q1_2024_Corrected_123_1.xml")
                # Stage 2 removal queue uses full 'file_path' from NAS
                # We need to check against the filename portion of the removal paths
                filename = row.get("filename", "")

                # Check if this filename matches any file in the removal set
                should_remove = False
                for removal_path in removal_set:
                    # Extract just the filename from the full path
                    removal_filename = removal_path.split('/')[-1] if '/' in removal_path else removal_path
                    if filename == removal_filename:
                        should_remove = True
                        break

                if not should_remove:
                    writer.writerow(row)
                    stats["final_records"] += 1
                else:
                    stats["records_removed"] += 1
                    log_execution(f"Removed record: {filename}")

            log_console(f"Processed {stats['original_records']} existing records, removed {stats['records_removed']}")

    # Load and append new records from Stage 8
    log_console("Loading new records from Stage 8...")
    new_data = nas_download_file(nas_conn, new_records_path)

    if new_data:
        new_csv = io.StringIO(new_data.decode("utf-8"))
        new_reader = csv.DictReader(new_csv)

        # If no existing master, use fieldnames from new data
        if fieldnames is None:
            fieldnames = new_reader.fieldnames
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

        # Append new records
        for row in new_reader:
            writer.writerow(row)
            stats["new_records_added"] += 1
            stats["final_records"] += 1

        log_console(f"Added {stats['new_records_added']} new records")

    # Save updated master database
    log_console("Saving updated master database...")
    output.seek(0)
    master_content = output.getvalue().encode("utf-8")
    stats["file_size_mb"] = len(master_content) / (1024 * 1024)

    master_file_obj = io.BytesIO(master_content)
    if nas_upload_file(nas_conn, master_file_obj, master_path):
        log_console(f"Master database updated successfully ({stats['file_size_mb']:.2f} MB)")
    else:
        raise RuntimeError("Failed to save master database")

    return stats


def create_refresh_archive(nas_conn: SMBConnection) -> Optional[str]:
    """Create an archive of the refresh folder."""
    if not config["stage_09_master_consolidation"].get("archive_enabled", True):
        log_console("Archiving is disabled in configuration")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_name = f"refresh_archive_{timestamp}.zip"
    archive_path = nas_path_join(
        config["stage_09_master_consolidation"]["archive_path"],
        archive_name
    )

    log_console(f"Creating archive: {archive_name}")

    try:
        # Create in-memory zip file
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            refresh_path = config["stage_09_master_consolidation"]["refresh_folder_path"]

            # List all files in refresh folder
            files_to_archive = []

            # Get all stage output files
            for stage_num in range(2, 9):  # Stages 2-8
                stage_files = nas_list_files(nas_conn, refresh_path, f"stage_{stage_num:02d}")
                for filename in stage_files:
                    file_path = nas_path_join(refresh_path, filename)
                    files_to_archive.append((filename, file_path))

            # Archive each file
            archived_count = 0
            for filename, file_path in files_to_archive:
                file_data = nas_download_file(nas_conn, file_path)
                if file_data:
                    zip_file.writestr(filename, file_data)
                    archived_count += 1

                    # Log progress for large archives
                    if archived_count % 10 == 0:
                        log_console(f"Archived {archived_count} files...")

            log_console(f"Added {archived_count} files to archive")

        # Upload archive to NAS
        zip_buffer.seek(0)
        if nas_upload_file(nas_conn, zip_buffer, archive_path):
            archive_size_mb = len(zip_buffer.getvalue()) / (1024 * 1024)
            log_console(f"Archive created successfully: {archive_name} ({archive_size_mb:.2f} MB)")
            log_execution("Archive created", {
                "archive_name": archive_name,
                "files_archived": archived_count,
                "size_mb": archive_size_mb
            })
            return archive_path
        else:
            log_error("Failed to upload archive to NAS", "archive_upload")
            return None

    except Exception as e:
        log_error(f"Error creating archive: {e}", "archive_creation",
                 {"exception_type": type(e).__name__})
        return None


def cleanup_refresh_folder(nas_conn: SMBConnection) -> bool:
    """Optionally clean up refresh folder after successful archive."""
    if not config["stage_09_master_consolidation"].get("delete_refresh_after_archive", False):
        log_console("Refresh folder cleanup is disabled (safety setting)")
        return True

    log_console("Cleaning up refresh folder...")
    refresh_path = config["stage_09_master_consolidation"]["refresh_folder_path"]

    try:
        # Delete stage output files
        files_deleted = 0
        for stage_num in range(2, 9):
            stage_files = nas_list_files(nas_conn, refresh_path, f"stage_{stage_num:02d}")
            for filename in stage_files:
                file_path = nas_path_join(refresh_path, filename)
                try:
                    conn.deleteFiles(os.getenv("NAS_SHARE_NAME"), file_path)
                    files_deleted += 1
                except Exception as e:
                    log_error(f"Failed to delete {filename}: {e}", "file_deletion")

        log_console(f"Deleted {files_deleted} files from refresh folder")
        return True

    except Exception as e:
        log_error(f"Error cleaning refresh folder: {e}", "cleanup",
                 {"exception_type": type(e).__name__})
        return False


def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any]):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["stage_09_master_consolidation"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "stage": "stage_09_master_consolidation",
        "execution_start": (
            execution_log[0]["timestamp"] if execution_log else datetime.now().isoformat()
        ),
        "execution_end": datetime.now().isoformat(),
        "summary": stage_summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"stage_09_master_consolidation_{timestamp}.json"
    main_log_path = nas_path_join(logs_path, main_log_filename)
    main_log_json = json.dumps(main_log_content, indent=2)
    main_log_obj = io.BytesIO(main_log_json.encode("utf-8"))

    if nas_upload_file(nas_conn, main_log_obj, main_log_path):
        log_console(f"Execution log saved: {main_log_filename}")

    # Save error log only if errors exist
    if error_log:
        errors_path = nas_path_join(logs_path, "Errors")
        nas_create_directory_recursive(nas_conn, errors_path)

        error_log_content = {
            "stage": "stage_09_master_consolidation",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "errors": error_log,
        }

        error_log_filename = f"stage_09_master_consolidation_errors_{timestamp}.json"
        error_log_path = nas_path_join(errors_path, error_log_filename)
        error_log_json = json.dumps(error_log_content, indent=2)
        error_log_obj = io.BytesIO(error_log_json.encode("utf-8"))

        if nas_upload_file(nas_conn, error_log_obj, error_log_path):
            log_console(f"Error log saved: {error_log_filename}", "WARNING")


def main() -> None:
    """Main function to orchestrate Stage 9 master consolidation."""
    global config, logger

    # Initialize logging
    logger = setup_logging()
    log_console("=== STAGE 9: MASTER DATABASE CONSOLIDATION & ARCHIVE ===")

    # Initialize stage summary
    stage_summary = {
        "status": "unknown",
        "execution_time_seconds": 0,
        "original_records": 0,
        "records_removed": 0,
        "new_records_added": 0,
        "final_records": 0,
        "master_size_mb": 0,
        "archive_created": None,
        "archive_size_mb": 0,
        "refresh_cleaned": False
    }

    start_time = datetime.now()
    nas_conn = None

    try:
        # Step 1: Environment validation
        log_console("Step 1: Validating environment variables...")
        validate_environment_variables()

        # Step 2: NAS connection
        log_console("Step 2: Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            stage_summary["status"] = "failed"
            log_console("Failed to establish NAS connection", "ERROR")
            return

        # Step 3: Configuration loading
        log_console("Step 3: Loading configuration...")
        config = load_config_from_nas(nas_conn)

        # Step 4: Load removal queue
        log_console("Step 4: Loading removal queue from Stage 2...")
        files_to_remove = load_removal_queue(nas_conn)
        removal_set = set(files_to_remove)
        log_console(f"Found {len(removal_set)} files to remove")

        # Step 5: Process master CSV
        log_console("Step 5: Processing master database...")
        new_records_path = config["stage_09_master_consolidation"]["stage_08_output_path"]

        # Check if Stage 8 output exists
        if not nas_file_exists(nas_conn, new_records_path):
            log_console("No Stage 8 output found - nothing to consolidate", "WARNING")
            stage_summary["status"] = "skipped_no_input"
            return

        # Process master database with streaming
        processing_stats = process_master_csv_streaming(nas_conn, removal_set, new_records_path)

        # Update summary with processing stats
        stage_summary.update(processing_stats)

        # Step 6: Create archive
        log_console("Step 6: Creating archive of refresh folder...")
        archive_path = create_refresh_archive(nas_conn)
        if archive_path:
            stage_summary["archive_created"] = archive_path

            # Step 7: Optional cleanup
            log_console("Step 7: Cleanup (if enabled)...")
            if cleanup_refresh_folder(nas_conn):
                stage_summary["refresh_cleaned"] = True

        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        stage_summary["execution_time_seconds"] = execution_time.total_seconds()
        stage_summary["status"] = "completed_successfully"

        # Final summary
        log_console("=== STAGE 9 CONSOLIDATION COMPLETE ===")
        log_console(f"Original records: {stage_summary['original_records']}")
        log_console(f"Records removed: {stage_summary['records_removed']}")
        log_console(f"New records added: {stage_summary['new_records_added']}")
        log_console(f"Final record count: {stage_summary['final_records']}")
        log_console(f"Master database size: {stage_summary.get('file_size_mb', 0):.2f} MB")
        if archive_path:
            log_console(f"Archive created: {os.path.basename(archive_path)}")
        log_console(f"Execution time: {execution_time}")

    except Exception as e:
        stage_summary["status"] = "failed"
        error_msg = f"Stage 9 consolidation failed: {e}"
        log_console(error_msg, "ERROR")
        log_error(error_msg, "main_execution", {"exception_type": type(e).__name__})

    finally:
        # Save logs to NAS
        if nas_conn:
            try:
                save_logs_to_nas(nas_conn, stage_summary)
            except Exception as e:
                log_console(f"Failed to save logs to NAS: {e}", "WARNING")

        # Cleanup
        if nas_conn:
            try:
                nas_conn.close()
                log_execution("NAS connection closed")
            except Exception as e:
                log_console(f"Error closing NAS connection: {e}", "WARNING")

        log_console(f"Stage 9 consolidation {stage_summary['status']}")


if __name__ == "__main__":
    main()