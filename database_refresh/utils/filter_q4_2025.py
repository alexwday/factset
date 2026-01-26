"""
Utility: Filter Q4 2025 Data from Master Embeddings
Removes Q4 2025 records from master_embeddings.csv for sharing.

Usage:
    # Dry run (default) - preview what will be filtered
    python filter_q4_2025.py

    # Execute - create filtered file (preserves original)
    python filter_q4_2025.py --execute
"""

import os
import json
import csv
import io
import logging
import argparse
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

        # Validate Stage 9 configuration (needed for master_database_path)
        if "stage_09_master_consolidation" not in stage_config:
            stage_config["stage_09_master_consolidation"] = {
                "master_database_path": "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Database/master_embeddings.csv",
            }
            log_execution("Using default Stage 9 configuration for master path")

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


def filter_q4_2025_records(nas_conn: SMBConnection, dry_run: bool = True) -> Dict[str, Any]:
    """Filter Q4 2025 records from master embeddings.

    Args:
        nas_conn: Active NAS connection
        dry_run: If True, only report statistics without saving changes

    Returns:
        Dictionary with filtering statistics
    """
    master_path = config["stage_09_master_consolidation"]["master_database_path"]

    # Stats tracking
    stats = {
        "total_records": 0,
        "q4_2025_records": 0,
        "remaining_records": 0,
        "affected_tickers": set(),
        "affected_companies": set(),
        "file_size_mb": 0,
        "original_file_size_mb": 0,
        "filtered_path": None
    }

    # Check if master exists
    if not nas_file_exists(nas_conn, master_path):
        log_console("Master embeddings file not found!", "ERROR")
        raise FileNotFoundError(f"Master file not found at: {master_path}")

    # Load master CSV
    log_console("Loading master embeddings file...")
    master_data = nas_download_file(nas_conn, master_path)
    if not master_data:
        raise RuntimeError("Failed to download master embeddings file")

    stats["original_file_size_mb"] = len(master_data) / (1024 * 1024)
    log_console(f"Original file size: {stats['original_file_size_mb']:.2f} MB")

    master_csv = io.StringIO(master_data.decode("utf-8"))
    reader = csv.DictReader(master_csv)

    if dry_run:
        # Dry run: just count and report
        log_console("Scanning records (dry run)...")

        for row in reader:
            stats["total_records"] += 1
            fiscal_year = row.get("fiscal_year", "")
            fiscal_quarter = row.get("fiscal_quarter", "")

            if fiscal_year == "2025" and fiscal_quarter == "Q4":
                stats["q4_2025_records"] += 1
                stats["affected_tickers"].add(row.get("ticker", "unknown"))
                stats["affected_companies"].add(row.get("company_name", "unknown"))
            else:
                stats["remaining_records"] += 1

            # Progress indicator
            if stats["total_records"] % 50000 == 0:
                log_console(f"  Scanned {stats['total_records']} records...")

        return stats

    else:
        # Execute mode: filter and save to new file
        log_console("Filtering records (execute mode)...")

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            stats["total_records"] += 1
            fiscal_year = row.get("fiscal_year", "")
            fiscal_quarter = row.get("fiscal_quarter", "")

            if fiscal_year == "2025" and fiscal_quarter == "Q4":
                stats["q4_2025_records"] += 1
                stats["affected_tickers"].add(row.get("ticker", "unknown"))
                stats["affected_companies"].add(row.get("company_name", "unknown"))
            else:
                writer.writerow(row)
                stats["remaining_records"] += 1

            # Progress indicator
            if stats["total_records"] % 50000 == 0:
                log_console(f"  Processed {stats['total_records']} records...")

        # Save to new filtered file (preserve original)
        output.seek(0)
        filtered_content = output.getvalue().encode("utf-8")
        stats["file_size_mb"] = len(filtered_content) / (1024 * 1024)

        # Save as master_embeddings_filtered.csv (same directory as original)
        filtered_path = master_path.replace(".csv", "_filtered.csv")
        stats["filtered_path"] = filtered_path

        log_console(f"Saving filtered file to: {filtered_path}")
        file_obj = io.BytesIO(filtered_content)

        if nas_upload_file(nas_conn, file_obj, filtered_path):
            log_console(f"Filtered file saved successfully ({stats['file_size_mb']:.2f} MB)")
            log_execution("Filtered file saved", {
                "path": filtered_path,
                "size_mb": stats["file_size_mb"]
            })
        else:
            raise RuntimeError("Failed to save filtered file to NAS")

        return stats


def main() -> None:
    """Main function to filter Q4 2025 data from master embeddings."""
    global config, logger

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Filter Q4 2025 data from master embeddings for sharing"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the filter and save (default is dry run)"
    )
    args = parser.parse_args()

    dry_run = not args.execute

    # Initialize logging
    logger = setup_logging()

    if dry_run:
        log_console("=== DRY RUN: Q4 2025 FILTER PREVIEW ===")
        log_console("(Run with --execute to create filtered file)")
    else:
        log_console("=== EXECUTING: Q4 2025 FILTER ===")

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
            log_console("Failed to establish NAS connection", "ERROR")
            return

        # Step 3: Configuration loading
        log_console("Step 3: Loading configuration...")
        config = load_config_from_nas(nas_conn)

        # Step 4: Filter Q4 2025 records
        log_console("Step 4: Processing master embeddings...")
        stats = filter_q4_2025_records(nas_conn, dry_run)

        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time

        # Display results
        log_console("")
        log_console("=" * 50)
        log_console("FILTER RESULTS")
        log_console("=" * 50)
        log_console(f"Total records scanned:     {stats['total_records']:,}")
        log_console(f"Q4 2025 records filtered:  {stats['q4_2025_records']:,}")
        log_console(f"Remaining records:         {stats['remaining_records']:,}")
        log_console(f"Affected tickers:          {len(stats['affected_tickers'])}")
        log_console("")
        log_console(f"Original file size:        {stats['original_file_size_mb']:.2f} MB")

        if not dry_run:
            log_console(f"Filtered file size:        {stats['file_size_mb']:.2f} MB")
            log_console(f"Filtered file path:        {stats['filtered_path']}")
        log_console("")

        # List affected tickers
        if stats['affected_tickers']:
            sorted_tickers = sorted(stats['affected_tickers'])
            log_console(f"Q4 2025 tickers ({len(sorted_tickers)}):")
            # Display in columns for readability
            ticker_line = ""
            for i, ticker in enumerate(sorted_tickers):
                ticker_line += f"{ticker:<8}"
                if (i + 1) % 10 == 0:
                    log_console(f"  {ticker_line}")
                    ticker_line = ""
            if ticker_line:
                log_console(f"  {ticker_line}")

        log_console("")
        log_console(f"Execution time: {execution_time}")
        log_console("")

        if dry_run:
            log_console("NO CHANGES MADE (dry run mode)")
            log_console("Run with --execute to create filtered file")
        else:
            log_console("FILTER COMPLETE")
            log_console("Original file preserved: master_embeddings.csv")
            log_console("Filtered file created:   master_embeddings_filtered.csv")

    except Exception as e:
        error_msg = f"Filter operation failed: {e}"
        log_console(error_msg, "ERROR")
        log_error(error_msg, "main_execution", {"exception_type": type(e).__name__})

    finally:
        # Cleanup
        if nas_conn:
            try:
                nas_conn.close()
                log_execution("NAS connection closed")
            except Exception as e:
                log_console(f"Error closing NAS connection: {e}", "WARNING")


if __name__ == "__main__":
    main()
