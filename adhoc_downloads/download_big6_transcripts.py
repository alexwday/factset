"""
Ad-hoc script: Download all transcript types for the Big 6 Canadian banks.
Downloads transcripts from the last 4 fiscal quarters (fiscal year starts Oct 1),
organized by fiscal quarter date range, then by bank.
Writes XML files to a local folder (adhoc_downloads/output/) for manual review.
Uses the same proxy, SSL, and FactSet SDK patterns as the main pipeline.
"""

import os
import sys
import io
import re
import tempfile
import time
import logging
from datetime import datetime, date, timedelta
from urllib.parse import quote
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

BIG_6_TICKERS = {
    "RY-CA": "Royal Bank of Canada",
    "BMO-CA": "Bank of Montreal",
    "CM-CA": "Canadian Imperial Bank of Commerce",
    "NA-CA": "National Bank of Canada",
    "BNS-CA": "Bank of Nova Scotia",
    "TD-CA": "Toronto-Dominion Bank",
}

# Fiscal quarters (fiscal year starts Oct 1)
# FQ1: Oct 1 - Jan 31
# FQ2: Feb 1 - Apr 30
# FQ3: May 1 - Jul 31
# FQ4: Aug 1 - Sep 30
FISCAL_QUARTERS = [
    (10, 1, 1, 31),   # FQ1: Oct 1 - Jan 31 (crosses calendar year)
    (2, 1, 4, 30),    # FQ2: Feb 1 - Apr 30
    (5, 1, 7, 31),    # FQ3: May 1 - Jul 31
    (8, 1, 9, 30),    # FQ4: Aug 1 - Sep 30
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

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


def get_last_4_fiscal_quarters(today: date) -> List[Tuple[date, date, str]]:
    """Calculate the last 4 fiscal quarter date ranges ending at or before today.

    Fiscal year starts Oct 1:
      FQ1: Oct 1 - Jan 31
      FQ2: Feb 1 - Apr 30
      FQ3: May 1 - Jul 31
      FQ4: Aug 1 - Sep 30

    Returns list of (start_date, end_date, folder_label) sorted oldest first.
    The current (possibly partial) quarter is included.
    """
    # Determine which fiscal quarter today falls in
    month = today.month
    year = today.year

    if month >= 10:
        # FQ1 starts this calendar year
        current_fq_start = date(year, 10, 1)
        current_fq_end = date(year + 1, 1, 31)
    elif month <= 1:
        # FQ1 started last calendar year
        current_fq_start = date(year - 1, 10, 1)
        current_fq_end = date(year, 1, 31)
    elif month <= 4:
        current_fq_start = date(year, 2, 1)
        current_fq_end = date(year, 4, 30)
    elif month <= 7:
        current_fq_start = date(year, 5, 1)
        current_fq_end = date(year, 7, 31)
    else:
        current_fq_start = date(year, 8, 1)
        current_fq_end = date(year, 9, 30)

    quarters = [(current_fq_start, current_fq_end)]

    # Walk backwards 3 more quarters
    for _ in range(3):
        prev_end = quarters[-1][0] - timedelta(days=1)
        m = prev_end.month
        y = prev_end.year
        if m >= 10:
            prev_start = date(y, 10, 1)
        elif m >= 8:
            prev_start = date(y, 8, 1)
        elif m >= 5:
            prev_start = date(y, 5, 1)
        elif m >= 2:
            prev_start = date(y, 2, 1)
        else:
            prev_start = date(y - 1, 10, 1)
        quarters.append((prev_start, prev_end))

    # Reverse so oldest is first, and build labels
    quarters.reverse()
    result = []
    for start, end in quarters:
        label = f"{start.strftime('%Y-%m-%d')}_to_{end.strftime('%Y-%m-%d')}"
        result.append((start, end, label))

    return result


def get_fiscal_quarter_folder(event_date, quarters: List[Tuple[date, date, str]]) -> Optional[str]:
    """Determine which fiscal quarter folder an event_date falls into.
    Returns the folder label or None if outside all ranges."""
    if isinstance(event_date, str):
        try:
            event_date = datetime.strptime(event_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None
    elif isinstance(event_date, datetime):
        event_date = event_date.date()
    elif not isinstance(event_date, date):
        return None

    for start, end, label in quarters:
        if start <= event_date <= end:
            return label
    return None


def build_filename(ticker: str, transcript: Dict[str, Any], title: str) -> str:
    """Build filename from transcript metadata. Uses title to extract Q/year if possible."""
    event_id = str(transcript.get("event_id", "unknown"))
    version_id = str(transcript.get("version_id", "unknown"))
    transcript_type = transcript.get("transcript_type", "Unknown")

    # Try to extract quarter and year from title
    match = re.search(r"Q([1-4])\s+(20\d{2})", title, re.IGNORECASE)
    if match:
        quarter = f"Q{match.group(1)}"
        year = match.group(2)
    else:
        quarter = "Unknown"
        year = "Unknown"

    return f"{ticker}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml"


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

        # Calculate the last 4 fiscal quarters
        today = datetime.now().date()
        quarters = get_last_4_fiscal_quarters(today)
        api_start_date = quarters[0][0]
        api_end_date = today  # up to today, not the full quarter end

        logger.info(f"Fiscal quarters to download:")
        for start, end, label in quarters:
            logger.info(f"  {label}")
        logger.info(f"API date range: {api_start_date} to {api_end_date}")

        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Pre-create quarter/bank folder structure
        for _, _, label in quarters:
            for ticker in BIG_6_TICKERS:
                (OUTPUT_DIR / label / ticker).mkdir(parents=True, exist_ok=True)

        # Create API instance
        api_client = fds.sdk.EventsandTranscripts.ApiClient(api_configuration)
        api_instance = transcripts_api.TranscriptsApi(api_client)

        # Process each Big 6 bank
        total_downloaded = 0
        total_skipped = 0
        total_no_quarter = 0
        summary = {}

        for ticker, bank_name in BIG_6_TICKERS.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {ticker} - {bank_name}")
            logger.info(f"{'='*60}")

            # Query API for full date range
            transcripts = get_transcripts_for_ticker(
                api_instance, ticker, api_start_date, api_end_date, config
            )

            if not transcripts:
                logger.info(f"  No transcripts found for {ticker}")
                summary[ticker] = {"found": 0, "downloaded": 0, "skipped": 0, "no_quarter": 0}
                continue

            bank_downloaded = 0
            bank_skipped = 0
            bank_no_quarter = 0

            for t in transcripts:
                event_id = str(t.get("event_id", ""))
                version_id = str(t.get("version_id", ""))
                event_date = t.get("event_date")

                # Determine fiscal quarter folder from event_date
                quarter_label = get_fiscal_quarter_folder(event_date, quarters)
                if not quarter_label:
                    logger.warning(
                        f"  event_id={event_id} event_date={event_date} "
                        f"falls outside fiscal quarter ranges — skipping"
                    )
                    bank_no_quarter += 1
                    continue

                bank_dir = OUTPUT_DIR / quarter_label / ticker

                # Check if we already have this file locally (search across all quarter folders)
                existing = list(OUTPUT_DIR.glob(f"*/{ticker}/*_{event_id}_{version_id}.xml"))
                if existing:
                    logger.info(f"  Already downloaded: {existing[0].name} — skipping")
                    bank_skipped += 1
                    continue

                # Download
                result = download_transcript(t, ticker, api_configuration, config)
                if result is None:
                    logger.warning(f"  Failed to download event_id={event_id}")
                    continue

                xml_content, title = result
                filename = build_filename(ticker, t, title)
                filepath = bank_dir / filename

                filepath.write_bytes(xml_content)
                logger.info(f"  Saved: {quarter_label}/{ticker}/{filename}  (title: {title})")
                bank_downloaded += 1

                # Rate limiting
                time.sleep(config["api_settings"]["request_delay"])

            summary[ticker] = {
                "found": len(transcripts),
                "downloaded": bank_downloaded,
                "skipped": bank_skipped,
                "no_quarter": bank_no_quarter,
            }
            total_downloaded += bank_downloaded
            total_skipped += bank_skipped
            total_no_quarter += bank_no_quarter

            # Delay between institutions
            time.sleep(config["api_settings"]["request_delay"])

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("DOWNLOAD SUMMARY")
        logger.info(f"{'='*60}")
        for ticker, stats in summary.items():
            logger.info(
                f"  {ticker}: {stats['found']} found, "
                f"{stats['downloaded']} downloaded, "
                f"{stats['skipped']} skipped, "
                f"{stats['no_quarter']} outside range"
            )
        logger.info(
            f"  TOTAL: {total_downloaded} downloaded, "
            f"{total_skipped} skipped, {total_no_quarter} outside range"
        )
        logger.info(f"  Output directory: {OUTPUT_DIR}")

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
