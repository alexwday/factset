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
import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def main() -> int:
    args = parse_args()
    stage1 = load_stage1_module()

    stage1.logger = stage1.setup_logging()
    stage1.config = {}

    nas_conn = None
    ssl_cert_path: Optional[str] = None

    try:
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

        raw_transcripts = []
        if response and hasattr(response, "data") and response.data:
            raw_transcripts = [item.to_dict() for item in response.data]

        allowed_types = set(stage1.config["api_settings"]["transcript_types"])
        scoped_transcripts = [
            t
            for t in raw_transcripts
            if isinstance(t.get("primary_ids"), list)
            and t.get("primary_ids") == [args.ticker]
            and t.get("transcript_type") in allowed_types
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

        print("")
        print("=== BNS Transcript Availability Check ===")
        print(f"Run Time: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Ticker: {args.ticker}")
        print(f"Target: {args.quarter} {args.year}")
        print(f"Window: {args.start_date.isoformat()} to {args.end_date.isoformat()}")
        print("Mode: READ-ONLY (no NAS writes, no transcript files saved)")
        print(f"Raw API rows: {len(raw_transcripts)}")
        print(f"Sole-primary + configured transcript types: {len(scoped_transcripts)}")
        print(f"Rows inspected by XML title parse: {len(inspected_rows)}")
        print(f"Matches for {args.quarter} {args.year}: {len(matches)}")

        if matches:
            print("STATUS: AVAILABLE")
            matches = sorted(
                matches, key=lambda row: (row.get("event_date", ""), row.get("event_id", ""))
            )
            for idx, match in enumerate(matches, start=1):
                print(
                    f"{idx}. event_date={match.get('event_date')} "
                    f"type={match.get('transcript_type')} "
                    f"event_id={match.get('event_id')} "
                    f"version_id={match.get('version_id')} "
                    f"strict_title={match.get('strict_title_match')}"
                )
                print(f"   title={match.get('title')}")
        else:
            print("STATUS: NOT AVAILABLE")
            recent_rows = sorted(
                inspected_rows,
                key=lambda row: (row.get("event_date", ""), row.get("event_id", "")),
                reverse=True,
            )[:5]
            if recent_rows:
                print("Most recent inspected titles:")
                for idx, row in enumerate(recent_rows, start=1):
                    title = row.get("title") or "<title unavailable>"
                    parsed_label = f"{row.get('parsed_quarter')} {row.get('parsed_year')}"
                    print(
                        f"{idx}. event_date={row.get('event_date')} "
                        f"type={row.get('transcript_type')} "
                        f"parsed={parsed_label} "
                        f"event_id={row.get('event_id')}"
                    )
                    print(f"   title={title}")
            else:
                print("No scoped transcripts were returned for the query window.")

        if args.fail_if_missing and not matches:
            return 1
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
