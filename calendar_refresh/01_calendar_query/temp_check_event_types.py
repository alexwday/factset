#!/usr/bin/env python3
"""
Temporary Test Script: Check Available Event Types
Tests various event types in the FactSet Calendar Events API to see what's available.
NO OUTPUT FILES - Console logging only.
"""

import os
import tempfile
from datetime import datetime, timedelta
from urllib.parse import quote

from dotenv import load_dotenv
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import calendar_events_api
from fds.sdk.EventsandTranscripts.models import (
    CompanyEventRequest,
    CompanyEventRequestData,
    CompanyEventRequestDataDateTime,
    CompanyEventRequestDataUniverse,
)
from dateutil.parser import parse as dateutil_parser
from smb.SMBConnection import SMBConnection
import yaml
import io

# Load environment variables
load_dotenv()

def get_nas_connection():
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
            print("✓ NAS connection established")
            return conn
        else:
            print("✗ Failed to establish NAS connection")
            return None
    except Exception as e:
        print(f"✗ Error creating NAS connection: {e}")
        return None

def nas_download_file(conn, nas_file_path):
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        print(f"✗ Failed to download file from NAS {nas_file_path}: {e}")
        return None

def load_config_from_nas(nas_conn):
    """Load and validate configuration from NAS."""
    try:
        config_path = os.getenv("CALENDAR_CONFIG_PATH") or os.getenv("CONFIG_PATH")
        if not config_path:
            raise ValueError("CALENDAR_CONFIG_PATH or CONFIG_PATH must be set")

        config_data = nas_download_file(nas_conn, config_path)
        if not config_data:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        calendar_config = yaml.safe_load(config_data.decode("utf-8"))
        print(f"✓ Configuration loaded from {config_path}")
        return calendar_config
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        raise

def load_monitored_institutions(nas_conn, config):
    """Load monitored institutions from separate file or fall back to config."""
    try:
        config_path = os.getenv("CONFIG_PATH")
        if not config_path:
            raise ValueError("CONFIG_PATH must be set to locate monitored_institutions.yaml")

        institutions_path = "/".join(config_path.split("/")[:-1]) + "/monitored_institutions.yaml"
        print(f"Loading institutions from {institutions_path}...")
        institutions_data = nas_download_file(nas_conn, institutions_path)

        if institutions_data:
            monitored_institutions = yaml.safe_load(institutions_data.decode("utf-8"))
            print(f"✓ Loaded {len(monitored_institutions)} institutions")
            return monitored_institutions
        else:
            if "monitored_institutions" in config:
                print(f"✓ Using institutions from config ({len(config['monitored_institutions'])} institutions)")
                return config["monitored_institutions"]
            else:
                raise ValueError("No monitored institutions found")
    except Exception as e:
        if "monitored_institutions" in config:
            print(f"⚠ Using institutions from config: {e}")
            return config["monitored_institutions"]
        else:
            raise

def setup_ssl_certificate(nas_conn, ssl_cert_path):
    """Download SSL certificate from NAS and set up for use."""
    try:
        print("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, ssl_cert_path)
        if cert_data:
            temp_cert = tempfile.NamedTemporaryFile(mode="wb", suffix=".cer", delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()

            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name

            print("✓ SSL certificate configured")
            return temp_cert.name
        else:
            print("✗ Failed to download SSL certificate")
            return None
    except Exception as e:
        print(f"✗ Error downloading SSL certificate: {e}")
        return None

def query_all_event_types(api_instance, monitored_tickers, start_date, end_date):
    """Query API WITHOUT event type filter to see what's available."""
    try:
        # FactSet API has a 90-day maximum date range limit
        # Split the overall date range into 90-day chunks
        MAX_DAYS_PER_QUERY = 89  # Use 89 to be safe

        date_ranges = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=MAX_DAYS_PER_QUERY), end_date)
            date_ranges.append((current_start, current_end))
            current_start = current_end + timedelta(days=1)

        print(f"  Date range split into {len(date_ranges)} chunks (max 90 days each)")

        all_events = []

        # Query all tickers for each date range chunk
        for range_num, (chunk_start, chunk_end) in enumerate(date_ranges, 1):
            print(f"  [{range_num}/{len(date_ranges)}] Querying {chunk_start} to {chunk_end}...", end=" ")

            # Convert dates to datetime objects for API
            start_datetime = dateutil_parser(f"{chunk_start}T00:00:00Z")
            end_datetime = dateutil_parser(f"{chunk_end}T23:59:59Z")

            # Build the request object WITHOUT event_types filter
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
                    # NOTE: No event_types parameter - get everything!
                ),
            )

            # Make the API call for this date range
            try:
                response = api_instance.get_company_event(company_event_request)

                if response and hasattr(response, "data") and response.data:
                    range_events = [event.to_dict() for event in response.data]
                    print(f"Found {len(range_events)} events")
                    all_events.extend(range_events)
                else:
                    print("No events")

            except Exception as e:
                print(f"ERROR: {str(e)[:80]}")
                continue

        return all_events

    except Exception as e:
        print(f"✗ Fatal error in query: {e}")
        return []

def main():
    """Main execution function."""
    print("=" * 70)
    print("FACTSET CALENDAR EVENTS API - EVENT TYPE CHECKER")
    print("=" * 70)

    nas_conn = None
    temp_cert_path = None

    try:
        # Connect to NAS
        print("\n1. Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise Exception("Failed to connect to NAS")

        # Load configuration
        print("\n2. Loading configuration...")
        config = load_config_from_nas(nas_conn)

        # Setup SSL certificate
        print("\n3. Setting up SSL certificate...")
        temp_cert_path = setup_ssl_certificate(nas_conn, config["ssl_cert_path"])
        if not temp_cert_path:
            raise Exception("Failed to setup SSL certificate")

        # Configure FactSet API
        print("\n4. Configuring FactSet API client...")
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
        print("✓ API client configured")

        # Load monitored institutions
        print("\n5. Loading monitored institutions...")
        monitored_institutions = load_monitored_institutions(nas_conn, config)
        monitored_tickers = list(monitored_institutions.keys())
        print(f"✓ Monitoring {len(monitored_tickers)} institutions")

        # Calculate date range (same as main script)
        print("\n6. Calculating date range...")
        calendar_config = config["calendar_refresh"]
        past_months = calendar_config["date_range"]["past_months"]
        future_months = calendar_config["date_range"]["future_months"]

        today = datetime.now().date()
        start_date = today - timedelta(days=past_months * 30)
        end_date = today + timedelta(days=future_months * 30)
        print(f"✓ Date range: {start_date} to {end_date}")

        # Query all events without event type filter
        print("\n7. Querying ALL Event Types (no filter)...")
        print("=" * 70)

        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = calendar_events_api.CalendarEventsApi(api_client)
            all_events = query_all_event_types(api_instance, monitored_tickers, start_date, end_date)

        print(f"\n✓ Total events retrieved: {len(all_events)}")

        # Group events by event_type
        if all_events:
            from collections import defaultdict
            event_type_counts = defaultdict(int)
            event_type_examples = defaultdict(list)

            for event in all_events:
                event_type = event.get("event_type", "Unknown")
                ticker = event.get("ticker", "Unknown")
                event_type_counts[event_type] += 1
                # Store first 3 examples
                if len(event_type_examples[event_type]) < 3:
                    event_type_examples[event_type].append({
                        "ticker": ticker,
                        "headline": event.get("description", "")[:60]
                    })

            # Summary
            print("\n" + "=" * 70)
            print("SUMMARY - Event Types Found:")
            print("=" * 70)

            for event_type, count in sorted(event_type_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"\n{event_type}: {count} events")
                # Show examples
                for example in event_type_examples[event_type]:
                    print(f"  - {example['ticker']}: {example['headline']}")

            print("\n" + "=" * 70)
            print(f"Total unique event types: {len(event_type_counts)}")
            print(f"Total events: {sum(event_type_counts.values())}")
            print("=" * 70)
        else:
            print("\n✗ No events returned!")

        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")

    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
            print("\n✓ NAS connection closed")

        if temp_cert_path:
            try:
                os.unlink(temp_cert_path)
            except Exception:
                pass

if __name__ == "__main__":
    main()
