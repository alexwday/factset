#!/usr/bin/env python3
"""
Adhoc transcript availability checker for BNS Q1 2026.

This script reuses the same environment and setup approach as the stage refresh
scripts (env vars, NAS config load, proxy/SSL, FactSet SDK auth) and only
reports what is currently available from FactSet transcripts.

Read-only behavior:
- Reads config and SSL certificate from NAS
- Calls FactSet API / transcript links
- Does NOT write/upload to NAS
- Optionally saves matched Raw/Corrected transcript XMLs to local disk only
"""

from __future__ import annotations

import argparse
import collections
import io
import logging
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Set

import fds.sdk.EventsandTranscripts
import requests
import yaml
from dotenv import load_dotenv
from fds.sdk.EventsandTranscripts.api import transcripts_api
from smb.SMBConnection import SMBConnection

load_dotenv()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check FactSet transcript availability for BNS Q1 2026."
    )
    parser.add_argument("--ticker", default="BNS-CA", help="FactSet ticker ID")
    parser.add_argument(
        "--quarter",
        default="Q1",
        choices=["Q1", "Q2", "Q3", "Q4"],
        help="Target quarter",
    )
    parser.add_argument("--year", type=int, default=2026, help="Target year")
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=date(2026, 1, 1),
        help="API query start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=date.today(),
        help="API query end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fail-if-missing",
        action="store_true",
        help="Exit with code 1 when no matching transcript is found.",
    )
    parser.add_argument(
        "--all-primary-id-rows",
        action="store_true",
        help=(
            "List transcripts where ticker appears in primary_ids even when it is "
            "not the sole primary ID. Default remains sole-primary only."
        ),
    )
    parser.add_argument(
        "--types",
        default="all",
        help=(
            "Comma-separated transcript types to include (for example: "
            "Raw,Corrected). Default: all."
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling and warn when alert transcript types appear.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Polling interval in seconds when --watch is enabled (default: 300).",
    )
    parser.add_argument(
        "--alert-types",
        default="Raw,Corrected",
        help=(
            "Comma-separated transcript types that trigger a warning in watch mode "
            "(default: Raw,Corrected). Use 'all' to alert on any type."
        ),
    )
    parser.add_argument(
        "--exit-on-alert",
        action="store_true",
        help="Exit immediately after the first alert in watch mode.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Disable local download of matched Raw/Corrected transcript XML files "
            "(enabled by default)."
        ),
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help=(
            "Local folder for downloaded Raw/Corrected XML files "
            "(default: ./adhoc_transcript_downloads next to this script)."
        ),
    )
    return parser.parse_args()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def validate_environment_variables() -> None:
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
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )


def get_nas_connection() -> Optional[SMBConnection]:
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
        return conn
    return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception:
        return None


def sanitize_url_for_logging(url: str) -> str:
    if not url:
        return url
    sanitized = re.sub(
        r"(password|token|auth)=[^&]*", r"\1=***", url, flags=re.IGNORECASE
    )
    sanitized = re.sub(r"://[^@]*@", "://***:***@", sanitized)
    return sanitized


def validate_config_structure(config: Dict[str, Any]) -> None:
    required_sections = ["ssl_cert_path", "api_settings"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    required_api_settings = [
        "industry_categories",
        "sort_order",
        "pagination_limit",
        "pagination_offset",
    ]
    for setting in required_api_settings:
        if setting not in config["api_settings"]:
            raise ValueError(f"Missing required api_settings field: {setting}")


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    config_path = os.getenv("CONFIG_PATH")
    logger.info("Loading configuration from NAS: %s", sanitize_url_for_logging(config_path))

    config_data = nas_download_file(nas_conn, config_path)
    if not config_data:
        raise FileNotFoundError(
            f"Failed to download configuration file from NAS: {sanitize_url_for_logging(config_path)}"
        )

    config = yaml.safe_load(config_data.decode("utf-8")) or {}

    config_dir = os.path.dirname(config_path) if config_path else ""
    institutions_path = os.path.join(config_dir, "monitored_institutions.yaml")
    institutions_data = nas_download_file(nas_conn, institutions_path)
    if institutions_data:
        config["monitored_institutions"] = (
            yaml.safe_load(institutions_data.decode("utf-8")) or {}
        )
    else:
        config.setdefault("monitored_institutions", {})

    validate_config_structure(config)
    return config


def setup_ssl_certificate(nas_conn: SMBConnection, config: Dict[str, Any]) -> str:
    cert_path = config["ssl_cert_path"]
    cert_data = nas_download_file(nas_conn, cert_path)
    if not cert_data:
        raise FileNotFoundError(
            f"Failed to download SSL certificate from NAS: {sanitize_url_for_logging(cert_path)}"
        )

    temp_cert = tempfile.NamedTemporaryFile(mode="wb", suffix=".cer", delete=False)
    temp_cert.write(cert_data)
    temp_cert.close()

    os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
    os.environ["SSL_CERT_FILE"] = temp_cert.name
    return temp_cert.name


def setup_proxy_configuration() -> str:
    proxy_user = os.getenv("PROXY_USER")
    proxy_password = os.getenv("PROXY_PASSWORD")
    proxy_url = os.getenv("PROXY_URL")
    proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")

    escaped_domain = quote(proxy_domain + "\\" + proxy_user)
    quoted_password = quote(proxy_password)
    return f"http://{escaped_domain}:{quoted_password}@{proxy_url}"


def setup_factset_api_client(proxy_url: str, ssl_cert_path: str) -> Any:
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=os.getenv("API_USERNAME"),
        password=os.getenv("API_PASSWORD"),
        proxy=proxy_url,
        ssl_ca_cert=ssl_cert_path,
    )
    configuration.get_basic_auth_token()
    return configuration


def cleanup_temporary_files(ssl_cert_path: Optional[str]) -> None:
    if ssl_cert_path:
        try:
            os.unlink(ssl_cert_path)
        except Exception:
            pass


def parse_quarter_and_year_from_xml(xml_content: bytes) -> tuple[str, str, str]:
    try:
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return "Unknown", "Unknown", "No title found"

        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return "Unknown", "Unknown", "No title found"

        title = title_elem.text.strip()

        pattern = r"Q([1-4])\s+(20\d{2})"
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return f"Q{match.group(1)}", match.group(2), title

        quarter_patterns = [
            (r"First\s+Quarter\s+(20\d{2})", "Q1"),
            (r"Second\s+Quarter\s+(20\d{2})", "Q2"),
            (r"Third\s+Quarter\s+(20\d{2})", "Q3"),
            (r"Fourth\s+Quarter\s+(20\d{2})", "Q4"),
            (r"1Q(\d{2})", "Q1"),
            (r"2Q(\d{2})", "Q2"),
            (r"3Q(\d{2})", "Q3"),
            (r"4Q(\d{2})", "Q4"),
        ]
        for pattern, quarter_val in quarter_patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                year_val = match.group(1)
                if len(year_val) == 2:
                    year_val = "20" + year_val
                return quarter_val, year_val, title

        return "Unknown", "Unknown", title
    except Exception as exc:
        return "Unknown", "Unknown", f"Error parsing: {exc}"


def is_valid_earnings_call_title(title: str) -> bool:
    pattern = r"^Q([1-4])\s+(20\d{2})\s+Earnings\s+Call$"
    return bool(re.match(pattern, title, re.IGNORECASE))


def normalize_transcript_type(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def parse_type_filter(raw_value: str) -> Set[str]:
    value = (raw_value or "").strip()
    if not value or value.lower() == "all":
        return set()
    return {
        normalize_transcript_type(item.strip())
        for item in value.split(",")
        if item.strip()
    }


def is_type_selected(transcript_type: str, type_filter: Set[str], raw_filter: str) -> bool:
    if (raw_filter or "").strip().lower() in {"", "all"}:
        return True
    return normalize_transcript_type(transcript_type) in type_filter


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "Unknown"


def fetch_title_info(
    transcript: Dict[str, Any],
    api_configuration: Any,
    proxy_url: str,
) -> Dict[str, Any]:
    transcript_link = transcript.get("transcripts_link")
    if not transcript_link:
        return {
            "error": "missing_transcripts_link",
            "title": "",
            "parsed_quarter": "Unknown",
            "parsed_year": "Unknown",
        }

    headers = {
        "Accept": "application/xml,*/*",
        "Authorization": api_configuration.get_basic_auth_token(),
    }
    proxies = {"https": proxy_url, "http": proxy_url}

    response = requests.get(
        transcript_link,
        headers=headers,
        proxies=proxies,
        verify=api_configuration.ssl_ca_cert,
        timeout=30,
    )
    response.raise_for_status()

    parsed_quarter, parsed_year, title = parse_quarter_and_year_from_xml(response.content)
    strict_match = is_valid_earnings_call_title(title)

    return {
        "error": "",
        "title": title,
        "parsed_quarter": parsed_quarter,
        "parsed_year": parsed_year,
        "strict_title_match": strict_match,
        "xml_content": response.content,
    }


def run_single_check(
    args: argparse.Namespace,
    config: Dict[str, Any],
    api_configuration: Any,
    proxy_url: str,
    type_filter: Set[str],
) -> Dict[str, Any]:
    api_params = {
        "ids": [args.ticker],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "categories": config["api_settings"]["industry_categories"],
        "sort": config["api_settings"]["sort_order"],
        "pagination_limit": config["api_settings"]["pagination_limit"],
        "pagination_offset": config["api_settings"]["pagination_offset"],
    }

    with fds.sdk.EventsandTranscripts.ApiClient(api_configuration) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        response = api_instance.get_transcripts_ids(**api_params)

    raw_transcripts: List[Dict[str, Any]] = []
    if response and hasattr(response, "data") and response.data:
        raw_transcripts = [item.to_dict() for item in response.data]

    if args.all_primary_id_rows:
        scoped_transcripts = [
            t
            for t in raw_transcripts
            if isinstance(t.get("primary_ids"), list)
            and args.ticker in t.get("primary_ids")
            and is_type_selected(str(t.get("transcript_type", "")), type_filter, args.types)
        ]
    else:
        scoped_transcripts = [
            t
            for t in raw_transcripts
            if isinstance(t.get("primary_ids"), list)
            and t.get("primary_ids") == [args.ticker]
            and is_type_selected(str(t.get("transcript_type", "")), type_filter, args.types)
        ]

    inspected_rows: List[Dict[str, Any]] = []
    for transcript in scoped_transcripts:
        row: Dict[str, Any] = {
            "event_date": transcript.get("event_date", ""),
            "story_datetime": transcript.get("story_date_time", ""),
            "transcript_type": transcript.get("transcript_type", ""),
            "event_id": str(transcript.get("event_id", "")),
            "version_id": str(transcript.get("version_id", "")),
        }

        try:
            title_info = fetch_title_info(
                transcript=transcript,
                api_configuration=api_configuration,
                proxy_url=proxy_url,
            )
            row.update(title_info)
        except Exception as exc:
            row.update(
                {
                    "error": str(exc),
                    "title": "",
                    "parsed_quarter": "Unknown",
                    "parsed_year": "Unknown",
                    "strict_title_match": False,
                }
            )

        inspected_rows.append(row)

    target_year = str(args.year)
    matches = [
        row
        for row in inspected_rows
        if row.get("parsed_quarter", "").upper() == args.quarter
        and row.get("parsed_year") == target_year
    ]

    return {
        "raw_transcripts": raw_transcripts,
        "scoped_transcripts": scoped_transcripts,
        "inspected_rows": inspected_rows,
        "matches": matches,
    }


def print_snapshot(args: argparse.Namespace, result: Dict[str, Any], iteration: Optional[int]) -> None:
    raw_transcripts = result["raw_transcripts"]
    scoped_transcripts = result["scoped_transcripts"]
    inspected_rows = result["inspected_rows"]
    matches = result["matches"]
    target_year = str(args.year)

    print("")
    print("=== BNS Transcript Availability Check ===")
    if iteration is not None:
        print(f"Iteration: {iteration}")
    print(f"Run Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Ticker: {args.ticker}")
    print(f"Target: {args.quarter} {args.year}")
    print(f"Window: {args.start_date.isoformat()} to {args.end_date.isoformat()}")
    print("Mode: NAS READ-ONLY (local XML download enabled unless --no-download)")
    print(f"Type filter: {args.types}")
    print(f"Raw API rows: {len(raw_transcripts)}")
    if args.all_primary_id_rows:
        print(f"Scoped rows (ticker in primary_ids): {len(scoped_transcripts)}")
    else:
        print(f"Scoped rows (sole-primary only): {len(scoped_transcripts)}")
    print(f"Rows inspected by XML title parse: {len(inspected_rows)}")
    print(f"Matches for {args.quarter} {args.year}: {len(matches)}")

    type_counter = collections.Counter(
        str(row.get("transcript_type", "")).strip() for row in inspected_rows
    )
    print("")
    print("Transcript types present (scoped rows):")
    if type_counter:
        for transcript_type, count in sorted(type_counter.items()):
            print(f"- {transcript_type}: {count}")
    else:
        print("- none")

    if matches:
        print("STATUS: AVAILABLE")
        matches_sorted = sorted(
            matches, key=lambda row: (row.get("event_date", ""), row.get("event_id", ""))
        )
        for idx, match in enumerate(matches_sorted, start=1):
            print(
                f"{idx}. event_date={match.get('event_date')} "
                f"type={match.get('transcript_type')} "
                f"event_id={match.get('event_id')} "
                f"version_id={match.get('version_id')} "
                f"strict_title={match.get('strict_title_match')}"
            )
            print(f"   title={match.get('title')}")

        grouped_types: Dict[str, Set[str]] = {}
        for row in matches:
            event_id = str(row.get("event_id", ""))
            grouped_types.setdefault(event_id, set()).add(
                str(row.get("transcript_type", "")).strip()
            )

        print("")
        print("Target event type coverage:")
        required_types = ["Raw", "Corrected"]
        for event_id in sorted(grouped_types.keys()):
            available = sorted(grouped_types[event_id])
            missing = [t for t in required_types if t not in grouped_types[event_id]]
            print(
                f"- event_id={event_id} available={','.join(available) if available else 'none'} "
                f"missing_raw_corrected={','.join(missing) if missing else 'none'}"
            )
    else:
        print("STATUS: NOT AVAILABLE")
        if not inspected_rows:
            print("No scoped transcripts were returned for the query window.")

    print("")
    print("=== Full Transcript Listing (Scoped Rows) ===")
    if not inspected_rows:
        print("No rows to display.")
    else:
        all_rows_sorted = sorted(
            inspected_rows,
            key=lambda row: (row.get("event_date", ""), row.get("event_id", "")),
            reverse=True,
        )
        for idx, row in enumerate(all_rows_sorted, start=1):
            parsed_label = f"{row.get('parsed_quarter')} {row.get('parsed_year')}"
            is_target = (
                row.get("parsed_quarter", "").upper() == args.quarter
                and row.get("parsed_year") == target_year
            )
            target_flag = "TARGET_MATCH" if is_target else "non-target"
            print(
                f"{idx}. event_date={row.get('event_date')} "
                f"type={row.get('transcript_type')} "
                f"parsed={parsed_label} "
                f"strict_title={row.get('strict_title_match')} "
                f"{target_flag} "
                f"event_id={row.get('event_id')} "
                f"version_id={row.get('version_id')}"
            )
            title = row.get("title") or "<title unavailable>"
            print(f"   title={title}")
            if row.get("error"):
                print(f"   error={row.get('error')}")


def collect_alert_rows(
    matches: List[Dict[str, Any]], alert_type_filter: Set[str], alert_types_raw: str
) -> List[Dict[str, Any]]:
    return [
        row
        for row in matches
        if is_type_selected(
            str(row.get("transcript_type", "")), alert_type_filter, alert_types_raw
        )
    ]


def build_local_xml_path(download_root: Path, ticker: str, row: Dict[str, Any]) -> Path:
    ticker_clean = sanitize_path_component(ticker)
    quarter = sanitize_path_component(str(row.get("parsed_quarter", "Unknown")))
    year = sanitize_path_component(str(row.get("parsed_year", "Unknown")))
    transcript_type = sanitize_path_component(str(row.get("transcript_type", "Unknown")))
    event_id = sanitize_path_component(str(row.get("event_id", "Unknown")))
    version_id = sanitize_path_component(str(row.get("version_id", "Unknown")))

    filename = (
        f"{ticker_clean}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml"
    )
    target_dir = download_root / ticker_clean / f"{year}_{quarter}"
    return target_dir / filename


def download_raw_corrected_target_matches(
    matches: List[Dict[str, Any]],
    ticker: str,
    download_root: Path,
) -> Dict[str, Any]:
    download_types = {"raw", "corrected"}
    summary: Dict[str, Any] = {
        "eligible": 0,
        "downloaded": 0,
        "already_exists": 0,
        "failed": 0,
        "saved_paths": [],
    }

    for row in matches:
        if normalize_transcript_type(str(row.get("transcript_type", ""))) not in download_types:
            continue

        summary["eligible"] += 1
        xml_content = row.get("xml_content")
        if not isinstance(xml_content, (bytes, bytearray)) or not xml_content:
            summary["failed"] += 1
            continue

        target_path = build_local_xml_path(download_root, ticker, row)
        if target_path.exists():
            summary["already_exists"] += 1
            continue

        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(bytes(xml_content))
            summary["downloaded"] += 1
            summary["saved_paths"].append(str(target_path))
        except Exception:
            summary["failed"] += 1

    return summary


def main() -> int:
    global logger
    args = parse_args()
    logger = setup_logging()
    config: Dict[str, Any] = {}

    nas_conn = None
    ssl_cert_path: Optional[str] = None

    try:
        if args.watch and args.interval_seconds <= 0:
            raise ValueError("--interval-seconds must be a positive integer")

        validate_environment_variables()

        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to establish NAS connection")

        config = load_config_from_nas(nas_conn)
        ssl_cert_path = setup_ssl_certificate(nas_conn, config)
        if not ssl_cert_path:
            raise RuntimeError("Failed to configure SSL certificate")

        proxy_url = setup_proxy_configuration()
        api_configuration = setup_factset_api_client(proxy_url, ssl_cert_path)

        if args.ticker not in config.get("monitored_institutions", {}):
            print(
                f"WARNING: {args.ticker} is not in monitored_institutions. "
                "Proceeding with direct API lookup."
            )
        type_filter = parse_type_filter(args.types)
        alert_type_filter = parse_type_filter(args.alert_types)
        seen_alert_keys: Set[str] = set()
        iteration = 0
        download_root = Path(args.download_dir).expanduser() if args.download_dir else (
            Path(__file__).resolve().parent / "adhoc_transcript_downloads"
        )

        while True:
            iteration += 1
            result = run_single_check(
                args=args,
                config=config,
                api_configuration=api_configuration,
                proxy_url=proxy_url,
                type_filter=type_filter,
            )

            print_snapshot(args, result, iteration if args.watch else None)

            matches = result["matches"]

            if args.no_download:
                print("")
                print("Local download: disabled (--no-download)")
            else:
                download_summary = download_raw_corrected_target_matches(
                    matches=matches,
                    ticker=args.ticker,
                    download_root=download_root,
                )
                print("")
                print(
                    "Local Raw/Corrected XML downloads (target matches): "
                    f"eligible={download_summary['eligible']} "
                    f"downloaded={download_summary['downloaded']} "
                    f"already_exists={download_summary['already_exists']} "
                    f"failed={download_summary['failed']}"
                )
                print(f"Download folder: {download_root}")
                if download_summary["saved_paths"]:
                    print("New files:")
                    for path in download_summary["saved_paths"]:
                        print(f"- {path}")

            if args.watch:
                alert_rows = collect_alert_rows(
                    matches=matches,
                    alert_type_filter=alert_type_filter,
                    alert_types_raw=args.alert_types,
                )

                new_alert_rows: List[Dict[str, Any]] = []
                for row in alert_rows:
                    key = (
                        f"{row.get('event_id')}|{row.get('version_id')}|"
                        f"{normalize_transcript_type(str(row.get('transcript_type', '')))}"
                    )
                    if key not in seen_alert_keys:
                        seen_alert_keys.add(key)
                        new_alert_rows.append(row)

                if new_alert_rows:
                    print("")
                    print(
                        f"\aALERT [{datetime.now().isoformat(timespec='seconds')}]: "
                        f"New {args.alert_types} transcript type(s) detected for "
                        f"{args.ticker} {args.quarter} {args.year}"
                    )
                    for row in new_alert_rows:
                        print(
                            f"- event_date={row.get('event_date')} "
                            f"type={row.get('transcript_type')} "
                            f"event_id={row.get('event_id')} "
                            f"version_id={row.get('version_id')}"
                        )
                        print(f"  title={row.get('title')}")

                    if args.exit_on_alert:
                        return 0

                print("")
                print(
                    f"Watch mode active. Sleeping {args.interval_seconds}s "
                    f"(alert types: {args.alert_types})..."
                )
                time.sleep(args.interval_seconds)
                continue

            if args.fail_if_missing and not matches:
                return 1
            return 0

    except KeyboardInterrupt:
        print("\nWatch stopped by user.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if nas_conn:
            try:
                nas_conn.close()
            except Exception:
                pass
        cleanup_temporary_files(ssl_cert_path)


if __name__ == "__main__":
    sys.exit(main())
