"""
Ad-hoc script: Download all transcript types for European banks for March 2026.
Writes XML files to adhoc_downloads/output_european/<TICKER>/ for manual review.
Uses the same proxy, SSL, and FactSet SDK patterns as the main pipeline.
"""

import os
import sys
import io
import re
import tempfile
import time
import logging
from datetime import datetime, date
from urllib.parse import quote
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import yaml
import requests
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
import xml.etree.ElementTree as ET

# Load environment variables from project root .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ===== CONFIGURATION =====

TICKERS = {
    "UBS-US": "UBS Group AG",
    "BCS-US": "Barclays PLC",
    "DBK-DE": "Deutsche Bank AG",
    "GLE-FR": "Societe Generale",
    "BNP-FR": "BNP Paribas",
    "BBVA-ES": "Banco Bilbao Vizcaya Argentaria S.A.",
    "SAN-ES": "Banco Santander S.A.",
    "HSBA-GB": "HSBC Holdings plc",
    "LLOY-GB": "Lloyds Banking Group plc",
    "ING-US": "ING Groep N.V.",
    "STAN-GB": "Standard Chartered PLC",
    "RBS-GB": "NatWest Group plc",
    "UCG-IT": "UniCredit S.p.A.",
    "ISP-IT": "Intesa Sanpaolo",
}

START_DATE = date(2026, 3, 1)
END_DATE = min(date(2026, 3, 31), date.today())

OUTPUT_DIR = Path(__file__).resolve().parent / "output_european"

# ===== LOGGING =====

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ===== SETUP FUNCTIONS (mirror main pipeline) =====


def validate_environment_variables() -> None:
    """Validate required environment variables are present."""
    required = [
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
        "NAS_BASE_PATH",
        "NAS_PORT",
        "CONFIG_PATH",
        "CLIENT_MACHINE_NAME",
    ]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    logger.info("Environment variables validated")


def get_nas_connection() -> Optional[SMBConnection]:
    """Create SMB connection to NAS (needed for SSL cert and config)."""
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
            logger.info("NAS connection established")
            return conn
        else:
            logger.error("Failed to connect to NAS")
            return None
    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        logger.error(f"Failed to download from NAS {nas_file_path}: {e}")
        return None


def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load YAML configuration from NAS."""
    config_path = os.getenv("CONFIG_PATH")
    logger.info(f"Loading config from NAS: {config_path}")

    config_data = nas_download_file(nas_conn, config_path)
    if not config_data:
        raise FileNotFoundError(f"Failed to download config from NAS: {config_path}")

    config = yaml.safe_load(config_data.decode("utf-8"))
    logger.info("Configuration loaded successfully")
    return config


def setup_ssl_certificate(nas_conn: SMBConnection, config: Dict[str, Any]) -> Optional[str]:
    """Download SSL certificate from NAS and configure for API use."""
    try:
        cert_path = config["ssl_cert_path"]
        logger.info(f"Downloading SSL certificate from NAS")

        cert_data = nas_download_file(nas_conn, cert_path)
        if not cert_data:
            logger.error("Failed to download SSL certificate")
            return None

        temp_cert = tempfile.NamedTemporaryFile(mode="wb", suffix=".cer", delete=False)
        temp_cert.write(cert_data)
        temp_cert.close()

        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
        os.environ["SSL_CERT_FILE"] = temp_cert.name

        logger.info(f"SSL certificate configured: {temp_cert.name}")
        return temp_cert.name

    except Exception as e:
        logger.error(f"Error setting up SSL certificate: {e}")
        return None


def setup_proxy_configuration() -> str:
    """Configure proxy URL for API authentication."""
    proxy_user = os.getenv("PROXY_USER")
    proxy_password = os.getenv("PROXY_PASSWORD")
    proxy_url = os.getenv("PROXY_URL")
    proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")

    escaped_domain = quote(proxy_domain + "\\" + proxy_user)
    quoted_password = quote(proxy_password)
    proxy_url_formatted = f"http://{escaped_domain}:{quoted_password}@{proxy_url}"

    logger.info("Proxy configuration completed")
    return proxy_url_formatted


def setup_factset_api_client(proxy_url: str, ssl_cert_path: str):
    """Configure FactSet API client with proxy and SSL settings."""
    api_username = os.getenv("API_USERNAME")
    api_password = os.getenv("API_PASSWORD")

    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=api_username,
        password=api_password,
        proxy=proxy_url,
        ssl_ca_cert=ssl_cert_path,
    )
    configuration.get_basic_auth_token()

    logger.info("FactSet API client configured")
    return configuration


# ===== DOWNLOAD LOGIC =====


def get_transcripts_for_ticker(
    api_instance,
    ticker: str,
    start_date,
    end_date,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Query FactSet API for all transcripts for a ticker within date range."""
    max_retries = config["api_settings"]["max_retries"]

    for attempt in range(max_retries):
        try:
            logger.info(f"  Querying API for {ticker} (attempt {attempt + 1})")

            api_params = {
                "ids": [ticker],
                "start_date": start_date,
                "end_date": end_date,
                "categories": config["api_settings"]["industry_categories"],
                "sort": config["api_settings"]["sort_order"],
                "pagination_limit": config["api_settings"]["pagination_limit"],
                "pagination_offset": config["api_settings"]["pagination_offset"],
            }

            response = api_instance.get_transcripts_ids(**api_params)

            if not response or not hasattr(response, "data") or not response.data:
                logger.info(f"  No transcripts found for {ticker}")
                return []

            all_transcripts = [t.to_dict() for t in response.data]

            # Anti-contamination: only keep transcripts where ticker is sole primary ID
            filtered = []
            for t in all_transcripts:
                primary_ids = t.get("primary_ids", [])
                if isinstance(primary_ids, list) and primary_ids == [ticker]:
                    filtered.append(t)

            logger.info(
                f"  Found {len(all_transcripts)} total, {len(filtered)} after contamination filter"
            )
            return filtered

        except Exception as e:
            if attempt < max_retries - 1:
                base_delay = config["api_settings"]["retry_delay"]
                if config["api_settings"].get("use_exponential_backoff", False):
                    max_delay = config["api_settings"].get("max_backoff_delay", 120.0)
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    delay = base_delay
                logger.warning(f"  Attempt {attempt + 1} failed, retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"  All {max_retries} attempts failed for {ticker}: {e}")
                return []

    return []


def parse_title_from_xml(xml_content: bytes) -> str:
    """Extract the title from transcript XML content."""
    try:
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return "No title found"

        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return "No title found"

        return title_elem.text.strip()
    except Exception as e:
        return f"Error parsing: {e}"


def download_transcript(
    transcript: Dict[str, Any],
    ticker: str,
    api_configuration,
    config: Dict[str, Any],
) -> Optional[Tuple[bytes, str]]:
    """Download a single transcript XML. Returns (content, title) or None."""
    transcript_link = transcript.get("transcripts_link")
    if not transcript_link:
        logger.warning(f"  No download link for event_id={transcript.get('event_id')}")
        return None

    max_retries = config["api_settings"]["max_retries"]

    for attempt in range(max_retries):
        try:
            headers = {
                "Accept": "application/xml,*/*",
                "Authorization": api_configuration.get_basic_auth_token(),
            }

            proxy_user = os.getenv("PROXY_USER")
            proxy_password = os.getenv("PROXY_PASSWORD")
            proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")
            escaped_domain = quote(proxy_domain + "\\" + proxy_user)
            proxy_url = f"http://{escaped_domain}:{quote(proxy_password)}@{os.getenv('PROXY_URL')}"
            proxies = {"https": proxy_url, "http": proxy_url}

            response = requests.get(
                transcript_link,
                headers=headers,
                proxies=proxies,
                verify=api_configuration.ssl_ca_cert,
                timeout=30,
            )
            response.raise_for_status()

            title = parse_title_from_xml(response.content)
            return response.content, title

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                base_delay = config["api_settings"]["retry_delay"]
                if config["api_settings"].get("use_exponential_backoff", False):
                    max_delay = config["api_settings"].get("max_backoff_delay", 120.0)
                    delay = min(base_delay * (2 ** attempt), max_delay)
                else:
                    delay = base_delay
                logger.warning(f"  Download attempt {attempt + 1} failed, retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
            else:
                logger.error(f"  All download attempts failed: {e}")
                return None
        except Exception as e:
            logger.error(f"  Unexpected download error: {e}")
            return None

    return None


def build_filename(ticker: str, title: str, event_date) -> str:
    """Build filename as ticker_title_date.xml, sanitizing the title for filesystem safety."""
    safe_title = re.sub(r'[^\w\s\-]', '', title)  # remove special chars
    safe_title = re.sub(r'\s+', '_', safe_title.strip())  # spaces to underscores
    if not safe_title:
        safe_title = "untitled"

    # Format date
    if isinstance(event_date, (date, datetime)):
        date_str = event_date.strftime("%Y-%m-%d")
    elif isinstance(event_date, str):
        date_str = event_date
    else:
        date_str = "unknown-date"

    return f"{ticker}_{safe_title}_{date_str}.xml"


# ===== MAIN =====


def main():
    ssl_cert_path = None
    nas_conn = None

    try:
        # Setup
        validate_environment_variables()

        nas_conn = get_nas_connection()
        if not nas_conn:
            logger.error("Cannot proceed without NAS connection (needed for SSL cert)")
            sys.exit(1)

        config = load_config_from_nas(nas_conn)
        ssl_cert_path = setup_ssl_certificate(nas_conn, config)
        if not ssl_cert_path:
            logger.error("Cannot proceed without SSL certificate")
            sys.exit(1)

        # NAS connection no longer needed — close it
        nas_conn.close()
        nas_conn = None
        logger.info("NAS connection closed (only needed for config/cert)")

        proxy_url = setup_proxy_configuration()
        api_configuration = setup_factset_api_client(proxy_url, ssl_cert_path)

        logger.info(f"Date range: {START_DATE} to {END_DATE}")

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Create API instance
        api_client = fds.sdk.EventsandTranscripts.ApiClient(api_configuration)
        api_instance = transcripts_api.TranscriptsApi(api_client)

        # Process each bank
        total_downloaded = 0
        total_skipped = 0
        summary = {}
        event_map = defaultdict(set)

        for ticker, bank_name in TICKERS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {ticker} - {bank_name}")
            logger.info(f"{'='*60}")

            # Create per-bank output folder
            bank_dir = OUTPUT_DIR / ticker
            bank_dir.mkdir(parents=True, exist_ok=True)

            # Query API
            transcripts = get_transcripts_for_ticker(
                api_instance, ticker, START_DATE, END_DATE, config
            )

            if not transcripts:
                logger.info(f"  No transcripts found for {ticker}")
                summary[ticker] = {"found": 0, "downloaded": 0, "skipped": 0}
                continue

            bank_downloaded = 0
            bank_skipped = 0

            for t in transcripts:
                event_id = str(t.get("event_id", ""))
                event_date = t.get("event_date")

                # Download
                result = download_transcript(t, ticker, api_configuration, config)
                if result is None:
                    logger.warning(f"  Failed to download event_id={event_id}")
                    continue

                xml_content, title = result
                filename = build_filename(ticker, title, event_date)
                filepath = bank_dir / filename

                # Track event for summary
                event_map[title].add(ticker)

                # Skip if already downloaded
                if filepath.exists():
                    logger.info(f"  Already downloaded: {filename} — skipping")
                    bank_skipped += 1
                    continue

                filepath.write_bytes(xml_content)
                logger.info(f"  Saved: {ticker}/{filename}")
                bank_downloaded += 1

                # Rate limiting
                time.sleep(config["api_settings"]["request_delay"])

            summary[ticker] = {
                "found": len(transcripts),
                "downloaded": bank_downloaded,
                "skipped": bank_skipped,
            }
            total_downloaded += bank_downloaded
            total_skipped += bank_skipped

            # Delay between institutions
            time.sleep(config["api_settings"]["request_delay"])

        # Print download stats
        logger.info(f"\n{'='*60}")
        logger.info("DOWNLOAD STATS")
        logger.info(f"{'='*60}")
        for ticker, stats in summary.items():
            logger.info(
                f"  {ticker}: {stats['found']} found, "
                f"{stats['downloaded']} downloaded, "
                f"{stats['skipped']} skipped"
            )
        logger.info(f"  TOTAL: {total_downloaded} downloaded, {total_skipped} skipped")
        logger.info(f"  Output: {OUTPUT_DIR}")

        # Event summary
        shared_events = {t: tickers for t, tickers in event_map.items() if len(tickers) > 1}
        unique_events = {t: tickers for t, tickers in event_map.items() if len(tickers) == 1}

        print(f"\n{'='*60}")
        print("EVENT SUMMARY")
        print(f"{'='*60}")

        if shared_events:
            print(f"\nSHARED EVENTS ({len(shared_events)} events)")
            print("─" * 50)
            for title in sorted(shared_events, key=lambda t: len(shared_events[t]), reverse=True):
                tickers = sorted(shared_events[title])
                print(f"  {title}")
                print(f"    {', '.join(tickers)}")
                print()

        if unique_events:
            print(f"BANK-SPECIFIC EVENTS ({len(unique_events)} events)")
            print("─" * 50)
            # Group by bank
            bank_unique = defaultdict(list)
            for title, tickers in unique_events.items():
                ticker = next(iter(tickers))
                bank_unique[ticker].append(title)
            for ticker in sorted(bank_unique):
                print(f"  {ticker}")
                for title in sorted(bank_unique[ticker]):
                    print(f"    {title}")
                print()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Cleanup
        if ssl_cert_path:
            try:
                os.unlink(ssl_cert_path)
                logger.info("Temporary SSL certificate cleaned up")
            except (OSError, FileNotFoundError):
                pass
        if nas_conn:
            try:
                nas_conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
