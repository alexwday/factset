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
- Does NOT save transcript files
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import fds.sdk.EventsandTranscripts
import requests
from fds.sdk.EventsandTranscripts.api import transcripts_api


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
    return parser.parse_args()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Use YYYY-MM-DD."
        ) from exc


def load_stage1_module() -> Any:
    script_path = (
        Path(__file__).resolve().parent
        / "01_download_daily"
        / "main_daily_sync_with_ignore.py"
    )
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot locate stage script: {script_path}")

    spec = importlib.util.spec_from_file_location("stage1_daily_sync_module", script_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load import spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def fetch_title_info(
    transcript: Dict[str, Any],
    api_configuration: Any,
    proxy_url: str,
    stage1_module: Any,
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

    parsed_quarter, parsed_year, title = stage1_module.parse_quarter_and_year_from_xml(
        response.content
    )
    strict_match = stage1_module.is_valid_earnings_call_title(title)

    return {
        "error": "",
        "title": title,
        "parsed_quarter": parsed_quarter,
        "parsed_year": parsed_year,
        "strict_title_match": strict_match,
    }


def run_single_check(
    args: argparse.Namespace,
    stage1: Any,
    api_configuration: Any,
    proxy_url: str,
    type_filter: Set[str],
) -> Dict[str, Any]:
    api_params = {
        "ids": [args.ticker],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "categories": stage1.config["api_settings"]["industry_categories"],
        "sort": stage1.config["api_settings"]["sort_order"],
        "pagination_limit": stage1.config["api_settings"]["pagination_limit"],
        "pagination_offset": stage1.config["api_settings"]["pagination_offset"],
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
                stage1_module=stage1,
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
    print("Mode: READ-ONLY (no NAS writes, no transcript files saved)")
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


def main() -> int:
    args = parse_args()
    stage1 = load_stage1_module()

    stage1.logger = stage1.setup_logging()
    stage1.config = {}

    nas_conn = None
    ssl_cert_path: Optional[str] = None

    try:
        if args.watch and args.interval_seconds <= 0:
            raise ValueError("--interval-seconds must be a positive integer")

        stage1.validate_environment_variables()

        nas_conn = stage1.get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to establish NAS connection")

        stage1.config = stage1.load_config_from_nas(nas_conn)
        ssl_cert_path = stage1.setup_ssl_certificate(nas_conn)
        if not ssl_cert_path:
            raise RuntimeError("Failed to configure SSL certificate")

        proxy_url = stage1.setup_proxy_configuration()
        api_configuration = stage1.setup_factset_api_client(proxy_url, ssl_cert_path)

        if args.ticker not in stage1.config["monitored_institutions"]:
            print(
                f"WARNING: {args.ticker} is not in monitored_institutions. "
                "Proceeding with direct API lookup."
            )
        type_filter = parse_type_filter(args.types)
        alert_type_filter = parse_type_filter(args.alert_types)
        seen_alert_keys: Set[str] = set()
        iteration = 0

        while True:
            iteration += 1
            result = run_single_check(
                args=args,
                stage1=stage1,
                api_configuration=api_configuration,
                proxy_url=proxy_url,
                type_filter=type_filter,
            )

            print_snapshot(args, result, iteration if args.watch else None)

            matches = result["matches"]
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
        stage1.cleanup_temporary_files(ssl_cert_path)


if __name__ == "__main__":
    sys.exit(main())
