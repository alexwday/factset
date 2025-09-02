"""
Cleanup Script: Remove Invalid Category Folders from NAS Data Directory
This script removes category folders that don't match the current monitored_institutions.yaml configuration.
It uses the same NAS connection methods and configuration loading as the main pipeline.
"""

import os
import tempfile
import logging
import json
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Set
import io
import shutil

import yaml
from smb.SMBConnection import SMBConnection
from smb.base import SharedFile
from dotenv import load_dotenv

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
    """Log detailed execution information."""
    global execution_log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "details": details or {},
    }
    execution_log.append(log_entry)


def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information."""
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
        "NAS_BASE_PATH",
        "NAS_PORT",
        "CONFIG_PATH",
        "CLIENT_MACHINE_NAME",
    ]

    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log_error(error_msg, "environment_validation", {"missing_variables": missing_vars})
        raise ValueError(error_msg)

    log_execution("Environment variables validated successfully", {"total_variables": len(required_env_vars)})


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
            log_execution("NAS connection established successfully", {
                "connection_type": "SMB/CIFS",
                "port": nas_port,
                "share_name": os.getenv("NAS_SHARE_NAME"),
            })
            return conn
        else:
            log_error("Failed to establish NAS connection", "nas_connection",
                     {"server_ip": os.getenv("NAS_SERVER_IP"), "port": nas_port})
            return None

    except Exception as e:
        log_error(f"Error creating NAS connection: {e}", "nas_connection",
                 {"exception_type": type(e).__name__, "server_ip": os.getenv("NAS_SERVER_IP")})
        return None


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


def nas_list_directories(conn: SMBConnection, directory_path: str) -> List[str]:
    """List subdirectories in a NAS directory."""
    try:
        files = conn.listPath(os.getenv("NAS_SHARE_NAME"), directory_path)
        return [
            file_info.filename
            for file_info in files
            if file_info.isDirectory and file_info.filename not in [".", ".."]
        ]
    except Exception as e:
        log_error(f"Failed to list directories: {directory_path}", "nas_list_directories",
                 {"directory": directory_path, "error": str(e)})
        return []


def nas_delete_directory_recursive(conn: SMBConnection, dir_path: str, dry_run: bool = True) -> bool:
    """Recursively delete a directory and all its contents from NAS."""
    if dry_run:
        log_console(f"[DRY RUN] Would delete directory: {dir_path}")
        return True
    
    try:
        # First, list all contents
        items = conn.listPath(os.getenv("NAS_SHARE_NAME"), dir_path)
        
        # Delete all files and subdirectories
        for item in items:
            if item.filename in [".", ".."]:
                continue
                
            item_path = nas_path_join(dir_path, item.filename)
            
            if item.isDirectory:
                # Recursively delete subdirectory
                nas_delete_directory_recursive(conn, item_path, dry_run=False)
            else:
                # Delete file
                conn.deleteFiles(os.getenv("NAS_SHARE_NAME"), item_path)
                log_execution(f"Deleted file: {item_path}")
        
        # Finally, delete the empty directory
        conn.deleteDirectory(os.getenv("NAS_SHARE_NAME"), dir_path)
        log_console(f"✅ Deleted directory: {dir_path}")
        return True
        
    except Exception as e:
        log_error(f"Failed to delete directory: {dir_path}", "directory_deletion",
                 {"path": dir_path, "error": str(e)})
        return False


def nas_count_files_recursive(conn: SMBConnection, dir_path: str) -> int:
    """Count all files recursively in a directory."""
    count = 0
    try:
        items = conn.listPath(os.getenv("NAS_SHARE_NAME"), dir_path)
        for item in items:
            if item.filename in [".", ".."]:
                continue
            
            item_path = nas_path_join(dir_path, item.filename)
            if item.isDirectory:
                count += nas_count_files_recursive(conn, item_path)
            else:
                count += 1
    except Exception as e:
        log_error(f"Failed to count files in: {dir_path}", "file_count", {"error": str(e)})
    
    return count


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""
    try:
        log_execution("Loading configuration from NAS", {"config_path": os.getenv("CONFIG_PATH")})

        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            error_msg = f"Configuration file not found at {os.getenv('CONFIG_PATH')}"
            log_error(error_msg, "config_load", {"path": os.getenv("CONFIG_PATH")})
            raise FileNotFoundError(error_msg)

        # Parse YAML configuration
        stage_config = yaml.safe_load(config_data.decode("utf-8"))
        log_execution("Configuration loaded successfully", {"sections": list(stage_config.keys())})

        # Load monitored institutions from separate file
        config_path = os.getenv("CONFIG_PATH")
        config_dir = os.path.dirname(config_path) if config_path else ""
        institutions_path = os.path.join(config_dir, "monitored_institutions.yaml")
        
        institutions_data = nas_download_file(nas_conn, institutions_path)
        if institutions_data:
            try:
                stage_config["monitored_institutions"] = yaml.safe_load(institutions_data.decode("utf-8"))
                log_execution("Loaded monitored institutions successfully",
                            {"count": len(stage_config["monitored_institutions"])})
            except yaml.YAMLError as e:
                log_error("Failed to load monitored_institutions.yaml", "config_parse", {"yaml_error": str(e)})
        else:
            log_execution("monitored_institutions.yaml not found, using main config file")

        return stage_config

    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML in configuration file: {e}"
        log_error(error_msg, "config_parse", {"yaml_error": str(e)})
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Error loading configuration from NAS: {e}"
        log_error(error_msg, "config_load", {"exception_type": type(e).__name__})
        raise


def get_valid_categories(config: Dict[str, Any]) -> Set[str]:
    """Extract all valid category types from monitored institutions."""
    valid_categories = set()
    
    for ticker, info in config.get("monitored_institutions", {}).items():
        category = info.get("type")
        if category:
            valid_categories.add(category)
    
    return valid_categories


def cleanup_invalid_categories(nas_conn: SMBConnection, config: Dict[str, Any], dry_run: bool = True):
    """Main cleanup function to remove invalid category folders."""
    
    # Get valid categories from config
    valid_categories = get_valid_categories(config)
    log_console("=" * 60)
    log_console("VALID CATEGORIES FROM CONFIG:")
    for category in sorted(valid_categories):
        log_console(f"  ✅ {category}")
    log_console("=" * 60)
    
    # Get data path from config
    data_path = config.get("stage_02_database_sync", {}).get("input_data_path")
    if not data_path:
        data_path = "Outputs/Data"  # Default fallback
    
    log_console(f"\nScanning data directory: {data_path}")
    
    # Track statistics
    stats = {
        "years_scanned": 0,
        "quarters_scanned": 0,
        "valid_categories_found": 0,
        "invalid_categories_found": 0,
        "files_in_invalid_categories": 0,
        "directories_to_delete": []
    }
    
    # Scan all years
    years = nas_list_directories(nas_conn, data_path)
    log_console(f"Found {len(years)} year directories")
    
    for year in sorted(years):
        if year == "Unknown" or year == "InvalidTranscripts":
            continue
            
        stats["years_scanned"] += 1
        year_path = nas_path_join(data_path, year)
        quarters = nas_list_directories(nas_conn, year_path)
        
        for quarter in sorted(quarters):
            if quarter == "Unknown":
                continue
                
            stats["quarters_scanned"] += 1
            quarter_path = nas_path_join(year_path, quarter)
            categories = nas_list_directories(nas_conn, quarter_path)
            
            log_console(f"\nChecking {year}/{quarter}:")
            
            for category in categories:
                category_path = nas_path_join(quarter_path, category)
                
                if category in valid_categories:
                    stats["valid_categories_found"] += 1
                    log_console(f"  ✅ Valid category: {category}")
                else:
                    stats["invalid_categories_found"] += 1
                    file_count = nas_count_files_recursive(nas_conn, category_path)
                    stats["files_in_invalid_categories"] += file_count
                    stats["directories_to_delete"].append(category_path)
                    log_console(f"  ❌ INVALID category: {category} ({file_count} files)")
    
    # Summary
    log_console("\n" + "=" * 60)
    log_console("CLEANUP SUMMARY:")
    log_console(f"  Years scanned: {stats['years_scanned']}")
    log_console(f"  Quarters scanned: {stats['quarters_scanned']}")
    log_console(f"  Valid category folders: {stats['valid_categories_found']}")
    log_console(f"  Invalid category folders: {stats['invalid_categories_found']}")
    log_console(f"  Files in invalid categories: {stats['files_in_invalid_categories']}")
    log_console("=" * 60)
    
    if stats["directories_to_delete"]:
        log_console(f"\n{'DRY RUN - ' if dry_run else ''}Directories to delete:")
        for dir_path in stats["directories_to_delete"]:
            log_console(f"  - {dir_path}")
        
        if dry_run:
            log_console("\n⚠️  This is a DRY RUN - no files were deleted")
            log_console("To actually delete files, run with --execute flag")
        else:
            log_console("\n🗑️  Deleting invalid category directories...")
            deleted_count = 0
            for dir_path in stats["directories_to_delete"]:
                if nas_delete_directory_recursive(nas_conn, dir_path, dry_run=False):
                    deleted_count += 1
            
            log_console(f"\n✅ Successfully deleted {deleted_count}/{len(stats['directories_to_delete'])} directories")
    else:
        log_console("\n✅ No invalid category folders found - nothing to clean up!")
    
    return stats


def main():
    """Main function to run the cleanup."""
    global config, logger
    
    import sys
    
    # Check for execute flag
    dry_run = "--execute" not in sys.argv
    
    # Initialize logging
    logger = setup_logging()
    log_console("=" * 60)
    log_console("CATEGORY FOLDER CLEANUP SCRIPT")
    log_console("=" * 60)
    
    if dry_run:
        log_console("🔍 Running in DRY RUN mode - no files will be deleted")
    else:
        log_console("⚠️  Running in EXECUTE mode - files WILL be deleted!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() != "yes":
            log_console("Cleanup cancelled by user")
            return
    
    nas_conn = None
    
    try:
        # Validate environment
        log_console("\nStep 1: Validating environment variables...")
        validate_environment_variables()
        
        # Connect to NAS
        log_console("Step 2: Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            log_console("Failed to establish NAS connection", "ERROR")
            return
        
        # Load configuration
        log_console("Step 3: Loading configuration...")
        config = load_config_from_nas(nas_conn)
        log_console(f"Loaded {len(config.get('monitored_institutions', {}))} monitored institutions")
        
        # Run cleanup
        log_console("\nStep 4: Running cleanup...")
        stats = cleanup_invalid_categories(nas_conn, config, dry_run=dry_run)
        
        log_console("\n✅ Cleanup script completed successfully!")
        
    except Exception as e:
        log_console(f"Error during cleanup: {e}", "ERROR")
        log_error(f"Cleanup failed: {e}", "main_execution", {"exception_type": type(e).__name__})
        
    finally:
        if nas_conn:
            try:
                nas_conn.close()
                log_execution("NAS connection closed")
            except Exception as e:
                log_console(f"Error closing NAS connection: {e}", "WARNING")


if __name__ == "__main__":
    main()