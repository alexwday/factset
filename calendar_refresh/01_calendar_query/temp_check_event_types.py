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

def test_event_type(api_instance, event_type, monitored_tickers, start_date, end_date):
    """Test a specific event type and return count of events found."""
    try:
        # Convert dates to datetime objects for API
        start_datetime = dateutil_parser(f"{start_date}T00:00:00Z")
        end_datetime = dateutil_parser(f"{end_date}T23:59:59Z")

        # Build the request object
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
                event_types=[event_type],
            ),
        )

        # Make the API call
        response = api_instance.get_company_event(company_event_request)

        if response and hasattr(response, "data") and response.data:
            return len(response.data)
        else:
            return 0

    except Exception as e:
        error_msg = str(e)
        # Extract just the key error message
        if "Bad Request" in error_msg:
            return f"ERROR: Bad Request - {event_type} may not be a valid event type"
        elif "Invalid" in error_msg:
            return f"ERROR: Invalid - {event_type} not recognized"
        else:
            return f"ERROR: {error_msg[:100]}"

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

        # Test various event types
        print("\n7. Testing Event Types...")
        print("=" * 70)

        # Common event types to test
        event_types_to_test = [
            "Earnings",           # We know this works
            "Guidance",
            "Sales",
            "Shareholder/Analyst Call",
            "Analyst Meeting",
            "Conference Call",
            "Company Conference Presentations",
            "Shareholder Meeting",
            "Annual Meeting",
            "M&A",
            "Merger",
            "Acquisition",
            "Dividend",
            "Stock Split",
            "Special Situation",
            "Product Announcement",
            "Clinical Trial",
            "Board Meeting",
            "Investor Day",
            "Roadshow",
            "IPO",
        ]

        results = []

        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = calendar_events_api.CalendarEventsApi(api_client)

            for i, event_type in enumerate(event_types_to_test, 1):
                print(f"\n[{i}/{len(event_types_to_test)}] Testing: {event_type}")
                count = test_event_type(api_instance, event_type, monitored_tickers, start_date, end_date)

                if isinstance(count, int):
                    if count > 0:
                        print(f"  ✓ FOUND {count} events")
                        results.append((event_type, count))
                    else:
                        print(f"  - No events (valid type but no data)")
                else:
                    print(f"  ✗ {count}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY - Event Types with Data:")
        print("=" * 70)

        if results:
            for event_type, count in sorted(results, key=lambda x: x[1], reverse=True):
                print(f"  {event_type:40s} {count:>6d} events")
            print(f"\nTotal event types with data: {len(results)}")
        else:
            print("  No event types returned data!")

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
