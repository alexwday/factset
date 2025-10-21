#!/usr/bin/env python3
"""
Calendar Events Refresh
Single-script ETL to refresh calendar events data from FactSet API.
Replaces master CSV with fresh data each run (past 6 months + future 6 months).
"""

import os
import io
import csv
import json
import logging
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from urllib.parse import quote

import yaml
import pytz
from dateutil.parser import parse as dateutil_parser
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import calendar_events_api
from fds.sdk.EventsandTranscripts.models import (
    CompanyEventRequest,
    CompanyEventRequestData,
    CompanyEventRequestDataDateTime,
    CompanyEventRequestDataUniverse,
)

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
        "API_USERNAME",
        "API_PASSWORD",
        "PROXY_USER",
        "PROXY_PASSWORD",
        "PROXY_URL",
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

    log_execution("Environment variables validated successfully")


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
            log_execution("NAS connection established successfully")
            return conn
        else:
            log_error("Failed to establish NAS connection", "nas_connection")
            return None

    except Exception as e:
        log_error(f"Error creating NAS connection: {e}", "nas_connection")
        return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        log_error(f"Failed to download file from NAS {nas_file_path}: {e}", "nas_download")
        return None


def nas_upload_file(conn: SMBConnection, file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file to NAS."""
    try:
        file_obj.seek(0)
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS {nas_file_path}: {e}", "nas_upload")
        return False


def nas_create_directory_recursive(conn: SMBConnection, nas_path: str) -> bool:
    """Create directory and all parent directories on NAS if they don't exist."""
    try:
        parts = [p for p in nas_path.split("/") if p]
        current_path = ""

        for part in parts:
            current_path = f"{current_path}/{part}" if current_path else part
            try:
                conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
            except Exception:
                pass  # Directory might already exist

        return True
    except Exception as e:
        log_error(f"Failed to create directory {nas_path}: {e}", "nas_directory")
        return False


def nas_path_join(*parts: str) -> str:
    """Join path parts with forward slashes for NAS paths."""
    return "/".join(p.strip("/") for p in parts if p)


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""
    try:
        config_data = nas_download_file(nas_conn, os.getenv("CONFIG_PATH"))
        if not config_data:
            raise FileNotFoundError(f"Configuration file not found at {os.getenv('CONFIG_PATH')}")

        calendar_config = yaml.safe_load(config_data.decode("utf-8"))
        log_execution("Configuration loaded successfully")

        return calendar_config

    except Exception as e:
        log_error(f"Error loading configuration: {e}", "config_loading")
        raise


def setup_ssl_certificate(nas_conn: SMBConnection, ssl_cert_path: str) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
    try:
        log_console("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
        if cert_data:
            temp_cert = tempfile.NamedTemporaryFile(mode="wb", suffix=".cer", delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()

            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name

            log_execution("SSL certificate downloaded and configured")
            return temp_cert.name
        else:
            log_error("Failed to download SSL certificate", "ssl_setup")
            return None
    except Exception as e:
        log_error(f"Error downloading SSL certificate: {e}", "ssl_setup")
        return None


def calculate_date_range(past_months: int = 6, future_months: int = 6) -> tuple:
    """Calculate date range for API query."""
    today = datetime.now().date()
    start_date = today - timedelta(days=past_months * 30)  # Approximate
    end_date = today + timedelta(days=future_months * 30)  # Approximate

    log_execution(
        "Date range calculated",
        {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "past_months": past_months,
            "future_months": future_months,
        },
    )

    return start_date, end_date


def query_calendar_events(
    api_instance: calendar_events_api.CalendarEventsApi,
    monitored_tickers: List[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> List[Dict[str, Any]]:
    """Query FactSet Calendar Events API for all monitored institutions."""
    try:
        log_console(f"Querying events from {start_date} to {end_date}")

        # Convert dates to datetime objects for API
        start_datetime = dateutil_parser(f"{start_date}T00:00:00Z")
        end_datetime = dateutil_parser(f"{end_date}T23:59:59Z")

        # Build the request object for all monitored tickers
        company_event_request = CompanyEventRequest(
            data=CompanyEventRequestData(
                date_time=CompanyEventRequestDataDateTime(
                    start=start_datetime,
                    end=end_datetime,
                ),
                universe=CompanyEventRequestDataUniverse(
                    symbols=monitored_tickers,
                    type="Tickers",
                ),
                event_types=["Earnings"],
            ),
        )

        log_console(f"Checking events for {len(monitored_tickers)} institutions...")

        # Make the API call
        response = api_instance.get_company_event(company_event_request)

        events = []
        if response and hasattr(response, "data") and response.data:
            raw_events = [event.to_dict() for event in response.data]
            log_console(f"Found {len(raw_events)} total events")

            for event in raw_events:
                ticker = event.get("ticker", "Unknown")
                if ticker in monitored_tickers:
                    events.append(event)

            log_execution(
                "API query completed successfully",
                {"total_events": len(events), "institutions_queried": len(monitored_tickers)},
            )
        else:
            log_console("No events found for the specified date range", "WARNING")
            log_execution("API query returned no events")

        return events

    except Exception as e:
        log_error(f"Error querying calendar events: {e}", "api_query")
        return []


def enrich_event_with_institution_data(
    event: Dict[str, Any], institutions: Dict[str, Any]
) -> Dict[str, Any]:
    """Enrich event with institution metadata."""
    ticker = event.get("ticker", "")
    institution = institutions.get(ticker, {})

    enriched = {
        "event_id": event.get("event_id", ""),
        "ticker": ticker,
        "institution_name": institution.get("name", "Unknown"),
        "institution_id": institution.get("id", ""),
        "institution_type": institution.get("type", "Unknown"),
        "event_type": event.get("event_type", "Earnings"),
        "event_headline": event.get("event_headline", ""),
    }

    # Handle event datetime
    event_datetime = event.get("event_date_time")
    if event_datetime:
        # Convert to timezone-aware if needed
        if hasattr(event_datetime, "tzinfo") and event_datetime.tzinfo is None:
            event_datetime = pytz.UTC.localize(event_datetime)
        elif not hasattr(event_datetime, "tzinfo"):
            # It's a string, parse it
            event_datetime = dateutil_parser(str(event_datetime))
            if event_datetime.tzinfo is None:
                event_datetime = pytz.UTC.localize(event_datetime)

        # Store UTC datetime
        enriched["event_date_time_utc"] = event_datetime.isoformat()

        # Convert to Toronto time (EST/EDT) for local display
        toronto_tz = pytz.timezone("America/Toronto")
        event_datetime_local = event_datetime.astimezone(toronto_tz)
        enriched["event_date_time_local"] = event_datetime_local.isoformat()
        enriched["event_date"] = event_datetime_local.strftime("%Y-%m-%d")

        # Get timezone abbreviation (EST or EDT)
        tz_abbr = event_datetime_local.strftime("%Z")
        enriched["event_time_local"] = event_datetime_local.strftime(f"%H:%M {tz_abbr}")

    else:
        enriched["event_date_time_utc"] = ""
        enriched["event_date_time_local"] = ""
        enriched["event_date"] = ""
        enriched["event_time_local"] = ""

    # Webcast information
    enriched["webcast_status"] = event.get("webcast_status", "")
    enriched["webcast_url"] = event.get("webcast_url", "")
    enriched["dial_in_info"] = event.get("dial_in_info", "")

    # Try to parse fiscal year and quarter from headline
    headline = enriched["event_headline"]
    fiscal_year = ""
    fiscal_quarter = ""

    if headline:
        import re

        # Pattern: "Q1 2025 Earnings Call" or "Q4 2024 Earnings"
        quarter_match = re.search(r"Q([1-4])\s+(20\d{2})", headline)
        if quarter_match:
            fiscal_quarter = f"Q{quarter_match.group(1)}"
            fiscal_year = quarter_match.group(2)

    enriched["fiscal_year"] = fiscal_year
    enriched["fiscal_quarter"] = fiscal_quarter

    # Audit timestamp
    enriched["data_fetched_timestamp"] = datetime.now(pytz.UTC).isoformat()

    return enriched


def transform_events_to_csv_rows(
    events: List[Dict[str, Any]], institutions: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Transform API events into CSV-ready rows with enrichment."""
    csv_rows = []

    for event in events:
        enriched = enrich_event_with_institution_data(event, institutions)
        csv_rows.append(enriched)

    # Sort by event date
    csv_rows.sort(key=lambda x: x.get("event_date_time_utc", ""))

    log_execution("Events transformed to CSV format", {"total_rows": len(csv_rows)})

    return csv_rows


def save_master_csv(
    nas_conn: SMBConnection, csv_rows: List[Dict[str, Any]], output_path: str
) -> bool:
    """Save events as master CSV to NAS."""
    try:
        if not csv_rows:
            log_console("No events to save", "WARNING")
            return False

        # Define CSV field order (17 fields)
        fieldnames = [
            "event_id",
            "ticker",
            "institution_name",
            "institution_id",
            "institution_type",
            "event_type",
            "event_date_time_utc",
            "event_date_time_local",
            "event_date",
            "event_time_local",
            "event_headline",
            "webcast_status",
            "webcast_url",
            "dial_in_info",
            "fiscal_year",
            "fiscal_quarter",
            "data_fetched_timestamp",
        ]

        # Create CSV in memory
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

        # Convert to bytes for NAS upload
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode("utf-8"))

        # Create directory structure
        output_dir = "/".join(output_path.split("/")[:-1])
        nas_create_directory_recursive(nas_conn, output_dir)

        # Upload to NAS
        if nas_upload_file(nas_conn, csv_bytes, output_path):
            log_console(f"Master CSV saved: {output_path}")
            log_execution(
                "Master CSV saved successfully", {"output_path": output_path, "row_count": len(csv_rows)}
            )
            return True
        else:
            log_error("Failed to save master CSV", "csv_save", {"output_path": output_path})
            return False

    except Exception as e:
        log_error(f"Error saving master CSV: {e}", "csv_save")
        return False


def save_logs_to_nas(nas_conn: SMBConnection, summary: Dict[str, Any]):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logs_path = config["calendar_refresh"]["output_logs_path"]

    # Create logs directory
    nas_create_directory_recursive(nas_conn, logs_path)

    # Save main execution log
    main_log_content = {
        "script": "calendar_refresh",
        "execution_start": execution_log[0]["timestamp"] if execution_log else datetime.now().isoformat(),
        "execution_end": datetime.now().isoformat(),
        "summary": summary,
        "execution_log": execution_log,
    }

    main_log_filename = f"calendar_refresh_{timestamp}.json"
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
            "script": "calendar_refresh",
            "execution_time": datetime.now().isoformat(),
            "total_errors": len(error_log),
            "errors": error_log,
        }

        error_log_filename = f"calendar_refresh_errors_{timestamp}.json"
        error_log_path = nas_path_join(errors_path, error_log_filename)
        error_log_json = json.dumps(error_log_content, indent=2)
        error_log_obj = io.BytesIO(error_log_json.encode("utf-8"))

        if nas_upload_file(nas_conn, error_log_obj, error_log_path):
            log_console(f"Error log saved: {error_log_filename}", "WARNING")


def main():
    """Main execution function."""
    global config, logger

    # Setup
    logger = setup_logging()
    log_console("=" * 60)
    log_console("CALENDAR EVENTS REFRESH")
    log_console("=" * 60)

    start_time = datetime.now()
    summary = {
        "status": "started",
        "institutions_monitored": 0,
        "events_found": 0,
        "events_saved": 0,
        "errors": 0,
    }

    nas_conn = None
    temp_cert_path = None

    try:
        # Step 1: Validate environment
        log_console("Step 1: Validating environment...")
        validate_environment_variables()

        # Step 2: Connect to NAS
        log_console("Step 2: Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise Exception("Failed to connect to NAS")

        # Step 3: Load configuration
        log_console("Step 3: Loading configuration...")
        config = load_config_from_nas(nas_conn)

        # Step 4: Setup SSL certificate
        log_console("Step 4: Setting up SSL certificate...")
        temp_cert_path = setup_ssl_certificate(nas_conn, config["ssl_cert_path"])
        if not temp_cert_path:
            raise Exception("Failed to setup SSL certificate")

        # Step 5: Configure FactSet API
        log_console("Step 5: Configuring FactSet API client...")
        proxy_user = os.getenv("PROXY_USER")
        proxy_password = quote(os.getenv("PROXY_PASSWORD"))
        proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")
        proxy_url_base = os.getenv("PROXY_URL")

        escaped_domain = quote(proxy_domain + "\\" + proxy_user)
        proxy_url = f"http://{escaped_domain}:{proxy_password}@{proxy_url_base}"

        configuration = fds.sdk.EventsandTranscripts.Configuration(
            username=os.getenv("API_USERNAME"),
            password=os.getenv("API_PASSWORD"),
            proxy=proxy_url,
            ssl_ca_cert=temp_cert_path,
        )
        configuration.get_basic_auth_token()
        log_execution("FactSet API client configured")

        # Step 6: Get monitored institutions
        log_console("Step 6: Loading monitored institutions...")
        monitored_institutions = config["monitored_institutions"]
        monitored_tickers = list(monitored_institutions.keys())
        summary["institutions_monitored"] = len(monitored_tickers)
        log_console(f"Monitoring {len(monitored_tickers)} institutions")

        # Step 7: Calculate date range
        log_console("Step 7: Calculating date range...")
        calendar_config = config["calendar_refresh"]
        past_months = calendar_config["date_range"]["past_months"]
        future_months = calendar_config["date_range"]["future_months"]
        start_date, end_date = calculate_date_range(past_months, future_months)
        log_console(f"Date range: {start_date} to {end_date}")

        # Step 8: Query API for events
        log_console("Step 8: Querying Calendar Events API...")
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = calendar_events_api.CalendarEventsApi(api_client)
            events = query_calendar_events(api_instance, monitored_tickers, start_date, end_date)
            summary["events_found"] = len(events)

        # Step 9: Transform events to CSV format
        log_console("Step 9: Transforming events to CSV format...")
        csv_rows = transform_events_to_csv_rows(events, monitored_institutions)

        # Step 10: Save master CSV (replace existing)
        log_console("Step 10: Saving master CSV to NAS...")
        master_csv_path = calendar_config["master_database_path"]
        if save_master_csv(nas_conn, csv_rows, master_csv_path):
            summary["events_saved"] = len(csv_rows)
            summary["status"] = "success"
        else:
            summary["status"] = "failed"
            raise Exception("Failed to save master CSV")

        # Summary statistics
        log_console("=" * 60)
        log_console("REFRESH COMPLETE")
        log_console(f"Institutions Monitored: {summary['institutions_monitored']}")
        log_console(f"Events Found: {summary['events_found']}")
        log_console(f"Events Saved: {summary['events_saved']}")

        # Calculate upcoming vs past events
        now = datetime.now(pytz.UTC)
        upcoming_count = sum(
            1 for row in csv_rows
            if row.get("event_date_time_utc") and
            dateutil_parser(row["event_date_time_utc"]) > now
        )
        past_count = len(csv_rows) - upcoming_count
        log_console(f"Upcoming Events: {upcoming_count}")
        log_console(f"Past Events: {past_count}")
        log_console("=" * 60)

    except Exception as e:
        log_console(f"ERROR: {e}", "ERROR")
        log_error(f"Fatal error in main execution: {e}", "main_execution")
        summary["status"] = "failed"
        summary["errors"] = len(error_log)

    finally:
        # Step 11: Save execution logs
        if nas_conn:
            log_console("Saving execution logs...")
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            summary["execution_time_seconds"] = execution_time
            save_logs_to_nas(nas_conn, summary)

        # Cleanup
        if nas_conn:
            nas_conn.close()
            log_console("NAS connection closed")

        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
            except Exception:
                pass

        log_console(f"Total execution time: {summary.get('execution_time_seconds', 0):.2f} seconds")


if __name__ == "__main__":
    main()
