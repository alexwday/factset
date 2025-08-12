#!/usr/bin/env python3
"""
Test script to investigate missing companies from FactSet API.
Uses the same NAS configuration, SSL certificates, and proxy setup as the main pipeline.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import yaml
from urllib.parse import quote

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
from fds.sdk.EventsandTranscripts.exceptions import ApiException
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables (matching main pipeline structure)
config = {}
ssl_cert_path = None

# Companies that aren't providing ANY transcripts (not even in ignore list)
# These need ticker verification
MISSING_COMPANIES = {
    "European Banks": {
        "BCS": "Barclays PLC",
        "ING": "ING Groep N.V.",
        "ISP-IT": "Intesa Sanpaolo",
        "UBS": "UBS Group AG"
    },
    "Nordic Banks": {
        "NDA-FI": "Nordea Bank Abp",
        "SEBA-SE": "Skandinaviska Enskilda Banken AB",
        "SWEDA-SE": "Swedbank AB"
    },
    "UK Wealth & Asset Managers": {
        "SJP-GB": "St. James's Place plc"
    },
    "US Banks": {
        "JEF": "Jefferies Financial Group Inc."
    },
    "US Wealth & Asset Managers": {
        "SCHW": "Charles Schwab Corporation"
    }
}

def validate_environment_variables() -> bool:
    """Validate all required environment variables are set."""
    required_vars = [
        "API_USERNAME", "API_PASSWORD",
        "NAS_USERNAME", "NAS_PASSWORD", 
        "NAS_SERVER_IP", "NAS_SERVER_NAME", "NAS_SHARE_NAME",
        "NAS_BASE_PATH", "CLIENT_MACHINE_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"ERROR: Missing required environment variables: {missing_vars}")
        print("\nRequired environment variables:")
        for var in required_vars:
            status = "✓" if os.getenv(var) else "✗"
            print(f"  {status} {var}")
        return False
    
    print("✓ All required environment variables are set")
    return True

def get_nas_connection() -> Optional[SMBConnection]:
    """Establish connection to NAS server."""
    try:
        nas_conn = SMBConnection(
            username=os.getenv("NAS_USERNAME"),
            password=os.getenv("NAS_PASSWORD"),
            my_name=os.getenv("CLIENT_MACHINE_NAME"),
            remote_name=os.getenv("NAS_SERVER_NAME"),
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if nas_conn.connect(os.getenv("NAS_SERVER_IP"), int(os.getenv("NAS_PORT", 445))):
            print("✓ Connected to NAS server")
            return nas_conn
        else:
            print("✗ Failed to connect to NAS server")
            return None
    except Exception as e:
        print(f"✗ Error connecting to NAS: {e}")
        return None

def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load configuration from NAS."""
    global config
    
    config_path = os.getenv("CONFIG_PATH", "database_refresh/config.yaml")
    print(f"Loading configuration from NAS: {config_path}")
    
    try:
        # Download config file from NAS
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.yaml') as temp_file:
            nas_conn.retrieveFile(
                os.getenv("NAS_SHARE_NAME"),
                config_path,
                temp_file
            )
            temp_path = temp_file.name
        
        # Load YAML configuration
        with open(temp_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Monitored institutions: {len(config.get('monitored_institutions', {}))}")
        print(f"  - API settings configured: {bool(config.get('api_settings'))}")
        
        return config
        
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        return {}

def setup_ssl_certificate(nas_conn: SMBConnection) -> str:
    """Download and set up SSL certificate from NAS."""
    global ssl_cert_path
    
    cert_nas_path = config.get("ssl_cert_path")
    if not cert_nas_path:
        print("⚠ No SSL certificate path in configuration")
        return None
    
    print(f"Setting up SSL certificate from: {cert_nas_path}")
    
    try:
        # Create temp file for SSL certificate
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cer') as temp_cert:
            nas_conn.retrieveFile(
                os.getenv("NAS_SHARE_NAME"),
                cert_nas_path,
                temp_cert
            )
            ssl_cert_path = temp_cert.name
        
        # Set environment variable for requests library
        os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
        os.environ['SSL_CERT_FILE'] = ssl_cert_path
        
        print(f"✓ SSL certificate configured: {ssl_cert_path}")
        return ssl_cert_path
        
    except Exception as e:
        print(f"✗ Error setting up SSL certificate: {e}")
        return None

def setup_proxy_configuration():
    """Configure proxy URL for API authentication (matching Stage 00 exactly)."""
    try:
        proxy_user = os.getenv("PROXY_USER")
        proxy_password = os.getenv("PROXY_PASSWORD")
        proxy_url = os.getenv("PROXY_URL")
        proxy_domain = os.getenv("PROXY_DOMAIN", "MAPLE")
        
        if not proxy_url:
            print("⚠ No proxy URL configured")
            return None
        
        # Escape domain and user for NTLM authentication
        escaped_domain = quote(proxy_domain + "\\" + proxy_user)
        quoted_password = quote(proxy_password)
        
        # Construct proxy URL (matching Stage 00 format exactly)
        proxy_url_formatted = f"http://{escaped_domain}:{quoted_password}@{proxy_url}"
        
        print(f"✓ Proxy configured with domain authentication: {proxy_domain}\\{proxy_user}")
        return proxy_url_formatted
        
    except Exception as e:
        print(f"✗ Error configuring proxy: {e}")
        return None

def setup_factset_api_client():
    """Set up FactSet API client with proxy and authentication (matching Stage 00)."""
    global config, ssl_cert_path
    
    try:
        # Get proxy configuration
        proxy_url = setup_proxy_configuration()
        
        # Configure FactSet API client
        configuration = fds.sdk.EventsandTranscripts.Configuration(
            username=os.getenv("API_USERNAME"),
            password=os.getenv("API_PASSWORD"),
            proxy=proxy_url,
            ssl_ca_cert=ssl_cert_path
        )
        
        # Generate authentication token (important!)
        configuration.get_basic_auth_token()
        
        print("✓ FactSet API client configured successfully")
        print(f"✓ SSL certificate: {ssl_cert_path}")
        
        # Return the configuration wrapped in API client
        return transcripts_api.TranscriptsApi(
            fds.sdk.EventsandTranscripts.ApiClient(configuration)
        )
        
    except Exception as e:
        print(f"✗ Error setting up FactSet API client: {e}")
        raise

def search_by_company_name(api_instance, company_name: str) -> List[Dict[str, Any]]:
    """Search for transcripts using company name keywords."""
    global config
    results = []
    
    # Extract key words from company name
    # Remove common suffixes
    clean_name = company_name.replace(" PLC", "").replace(" plc", "")
    clean_name = clean_name.replace(" AG", "").replace(" N.V.", "")
    clean_name = clean_name.replace(" Group", "").replace(" Corporation", "")
    clean_name = clean_name.replace(" Inc.", "").replace(" AB", "")
    clean_name = clean_name.replace(" Abp", "").replace(" S.p.A.", "")
    
    # Get main company name (first significant word)
    name_parts = clean_name.split()
    search_terms = []
    
    if name_parts:
        # Try full clean name
        search_terms.append(clean_name)
        # Try first word (often the main identifier)
        if name_parts[0] not in ["The", "St", "St."]:
            search_terms.append(name_parts[0])
        # For St. James's Place, try "James"
        if "James" in company_name:
            search_terms.append("James")
    
    print(f"\n  Searching by company name: {company_name}")
    print(f"  Search terms: {search_terms}")
    
    # Note: FactSet API doesn't support direct name search in transcripts endpoint
    # This is a placeholder for alternative search methods
    # In practice, you might need to use FactSet's entity search API first
    
    return results

def test_ticker_formats(api_instance, ticker: str, company_name: str) -> Dict[str, Any]:
    """Test various ticker format variations."""
    global config
    results = {}
    
    # Calculate date range (3 years)
    end_date = datetime.now().date()
    start_date = (datetime.now() - timedelta(days=365*3)).date()
    
    # Base ticker (remove any country code if present)
    base_ticker = ticker.split('-')[0] if '-' in ticker else ticker
    
    # Start with common variations
    ticker_variations = [
        ticker,                              # Original
        base_ticker,                         # Without country code
        ticker.upper(),                      # Uppercase
    ]
    
    # Add company-specific variations based on known exchange listings
    if base_ticker == "BCS":  # Barclays
        ticker_variations.extend([
            "BARC",         # Alternative ticker
            "BARC.L",       # London Stock Exchange
            "BARC-GB",
            "BCS-US",       # US ADR
            "BCS-N",        # NYSE
        ])
    elif base_ticker == "ING":  # ING Group
        ticker_variations.extend([
            "INGA",         # Amsterdam listing
            "INGA.AS",      # Euronext Amsterdam
            "ING-US",       # US ADR
            "ING-N",        # NYSE
        ])
    elif base_ticker == "ISP":  # Intesa Sanpaolo
        ticker_variations.extend([
            "ISP",
            "ISP.MI",       # Borsa Italiana Milan
            "ISP-IT",
            "ISNPY",        # US ADR
        ])
    elif base_ticker == "UBS":  # UBS Group
        ticker_variations.extend([
            "UBSG",         # SIX Swiss Exchange ticker
            "UBSG.SW",      
            "UBSG.S",
            "UBS-US",       # US listing
            "UBS-N",        # NYSE
        ])
    elif base_ticker == "NDA":  # Nordea Bank
        ticker_variations.extend([
            "NDA",
            "NDA-SE",       # Swedish listing
            "NDA.ST",       # Stockholm
            "NDA-FI",
            "NDA.HE",       # Helsinki
            "NRBAY",        # US ADR
        ])
    elif base_ticker == "SEBA":  # SEB
        ticker_variations.extend([
            "SEB-A",        # A shares
            "SEB-A.ST",     # Stockholm
            "SEBA",
            "SEBA.ST",
            "SEBA-SE",
        ])
    elif base_ticker == "SWEDA":  # Swedbank
        ticker_variations.extend([
            "SWED-A",       # A shares
            "SWEDA",
            "SWEDA.ST",     # Stockholm
            "SWED-A.ST",
            "SWDBY",        # US ADR
        ])
    elif base_ticker == "SJP":  # St. James's Place
        ticker_variations.extend([
            "STJ",          # Alternative ticker
            "STJ.L",        # London Stock Exchange
            "SJP.L",
            "SJP-GB",
        ])
    elif base_ticker == "JEF":  # Jefferies
        ticker_variations.extend([
            "JEF",
            "JEF-US",
            "JEF-N",        # NYSE
        ])
    elif base_ticker == "SCHW":  # Charles Schwab
        ticker_variations.extend([
            "SCHW",
            "SCHW-US",
            "SCHW-N",       # NYSE
        ])
    
    print(f"\nTesting {company_name} ({ticker})...")
    print(f"  Date range: {start_date} to {end_date}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in ticker_variations:
        if v not in seen:
            seen.add(v)
            unique_variations.append(v)
    
    for variant in unique_variations:
        try:
            print(f"  Trying ticker variant: {variant:20s} ... ", end="")
            
            api_params = {
                "ids": [variant],
                "start_date": start_date,
                "end_date": end_date,
                "categories": config["api_settings"]["industry_categories"],
                "sort": config["api_settings"]["sort_order"],
                "pagination_limit": 100,
                "pagination_offset": 0
            }
            
            response = api_instance.get_transcripts_ids(**api_params)
            
            if response and hasattr(response, 'data') and response.data:
                count = len(response.data)
                print(f"✓ Found {count} transcripts")
                
                # Get sample transcript details
                sample_transcripts = []
                for transcript in response.data[:3]:  # First 3 transcripts
                    t_dict = transcript.to_dict()
                    sample_transcripts.append({
                        "event_id": t_dict.get("event_id"),
                        "report_id": t_dict.get("report_id"),
                        "primary_ids": t_dict.get("primary_ids"),
                        "event_type": t_dict.get("event_type"),
                        "event_date": str(t_dict.get("event_date_time", ""))[:10] if t_dict.get("event_date_time") else None,
                        "all_regional_ids": t_dict.get("all_regional_ids", [])
                    })
                
                results[variant] = {
                    "status": "success",
                    "count": count,
                    "sample_transcripts": sample_transcripts
                }
            else:
                print("✗ No transcripts found")
                results[variant] = {
                    "status": "no_data",
                    "count": 0
                }
            
            # Add delay to avoid rate limiting (from config)
            time.sleep(config.get("api_settings", {}).get("request_delay", 1.0))
            
        except ApiException as e:
            print(f"✗ API Error: {e.status}")
            results[variant] = {
                "status": "api_error",
                "error": f"Status {e.status}: {e.reason}"
            }
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            results[variant] = {
                "status": "error",
                "error": str(e)[:200]
            }
    
    return results

def cleanup_ssl_certificate():
    """Clean up temporary SSL certificate file."""
    global ssl_cert_path
    if ssl_cert_path and os.path.exists(ssl_cert_path):
        try:
            os.unlink(ssl_cert_path)
            print(f"✓ Cleaned up SSL certificate: {ssl_cert_path}")
        except Exception as e:
            print(f"⚠ Could not clean up SSL certificate: {e}")

def test_known_working_ticker(api_instance):
    """Test with a known working ticker (RBC) to verify API connection."""
    print("\n" + "=" * 60)
    print("Testing API connection with known working ticker (RY-CA)")
    print("=" * 60)
    
    try:
        # Test with Royal Bank of Canada (known to work)
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=90)).date()  # Just last 90 days for quick test
        
        print(f"Testing RY-CA (Royal Bank of Canada)")
        print(f"Date range: {start_date} to {end_date}")
        
        api_params = {
            "ids": ["RY-CA"],
            "start_date": start_date,
            "end_date": end_date,
            "categories": ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"],
            "sort": ["-storyDateTime"],
            "pagination_limit": 10,
            "pagination_offset": 0
        }
        
        response = api_instance.get_transcripts_ids(**api_params)
        
        if response and hasattr(response, 'data') and response.data:
            count = len(response.data)
            print(f"✓ API CONNECTION SUCCESSFUL - Found {count} transcripts for RY-CA")
            
            # Show sample transcript
            if response.data:
                sample = response.data[0].to_dict()
                print(f"  Sample transcript:")
                print(f"    Event ID: {sample.get('event_id')}")
                print(f"    Primary IDs: {sample.get('primary_ids')}")
                print(f"    Event Type: {sample.get('event_type')}")
                print(f"    Date: {str(sample.get('event_date_time', ''))[:10]}")
            return True
        else:
            print("✗ WARNING: No transcripts found for RY-CA (but API connection worked)")
            return True  # Connection worked even if no data
            
    except ApiException as e:
        print(f"✗ API ERROR: Status {e.status} - {e.reason}")
        print(f"  Details: {e.body if hasattr(e, 'body') else 'No details'}")
        return False
    except Exception as e:
        print(f"✗ CONNECTION ERROR: {e}")
        print("\nPossible issues:")
        print("  1. Check API credentials (API_USERNAME, API_PASSWORD)")
        print("  2. Verify proxy settings (PROXY_URL, PROXY_USER, PROXY_PASSWORD, PROXY_DOMAIN)")
        print("  3. Ensure SSL certificate is valid")
        print("  4. Check network connectivity to api.factset.com")
        return False

def main():
    """Main test function."""
    print("=" * 80)
    print("FactSet API Missing Companies Investigation")
    print("Using NAS Configuration and SSL Certificates")
    print("=" * 80)
    
    # Step 1: Validate environment variables
    print("\nStep 1: Validating environment variables...")
    if not validate_environment_variables():
        sys.exit(1)
    
    # Step 2: Connect to NAS
    print("\nStep 2: Connecting to NAS...")
    nas_conn = get_nas_connection()
    if not nas_conn:
        print("ERROR: Could not connect to NAS")
        sys.exit(1)
    
    # Step 3: Load configuration
    print("\nStep 3: Loading configuration from NAS...")
    config = load_config_from_nas(nas_conn)
    if not config:
        print("ERROR: Could not load configuration")
        nas_conn.close()
        sys.exit(1)
    
    # Step 4: Set up SSL certificate
    print("\nStep 4: Setting up SSL certificate...")
    ssl_cert = setup_ssl_certificate(nas_conn)
    
    # Step 5: Set up API client
    print("\nStep 5: Setting up FactSet API client...")
    try:
        api_instance = setup_factset_api_client()
        print("✓ API client configured successfully")
    except Exception as e:
        print(f"ERROR: Failed to set up API client: {e}")
        cleanup_ssl_certificate()
        nas_conn.close()
        sys.exit(1)
    
    # Step 6: Test API connection with known working ticker
    print("\nStep 6: Testing API connection with known working ticker...")
    if not test_known_working_ticker(api_instance):
        print("\n✗ API connection test failed. Please fix connection issues before proceeding.")
        cleanup_ssl_certificate()
        nas_conn.close()
        sys.exit(1)
    
    print("\n✓ API connection verified successfully!")
    
    # Step 7: Test each missing company
    print("\nStep 7: Testing missing companies...")
    print("=" * 80)
    
    all_results = {}
    successful_variants = {}
    
    for category, companies in MISSING_COMPANIES.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {category}")
        print(f"{'=' * 60}")
        
        category_results = {}
        
        for ticker, company_name in companies.items():
            results = test_ticker_formats(api_instance, ticker, company_name)
            category_results[ticker] = results
            
            # Check if any variant was successful
            success_variants = [
                variant for variant, result in results.items() 
                if result.get("status") == "success" and result.get("count", 0) > 0
            ]
            
            if success_variants:
                successful_variants[ticker] = {
                    "company": company_name,
                    "working_tickers": success_variants,
                    "transcript_counts": {
                        v: results[v]["count"] for v in success_variants
                    },
                    "sample_data": results[success_variants[0]].get("sample_transcripts", [])
                }
            
            # Delay between companies
            time.sleep(1)
        
        all_results[category] = category_results
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"missing_companies_test_results_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Print successful variants
    if successful_variants:
        print("\n✓ COMPANIES WITH WORKING TICKER VARIANTS:")
        print("-" * 60)
        for ticker, info in successful_variants.items():
            print(f"\n{ticker}: {info['company']}")
            print(f"  Working tickers: {', '.join(info['working_tickers'])}")
            for variant, count in info['transcript_counts'].items():
                print(f"    {variant}: {count} transcripts")
            
            # Show sample primary IDs from actual transcripts
            if info.get('sample_data'):
                sample = info['sample_data'][0]
                if sample.get('primary_ids'):
                    print(f"  Actual primary IDs in API: {sample['primary_ids']}")
                if sample.get('all_regional_ids'):
                    print(f"  Regional IDs available: {sample['all_regional_ids']}")
    
    # Print companies with no results
    print("\n✗ COMPANIES WITH NO TRANSCRIPTS FOUND:")
    print("-" * 60)
    no_results = []
    for category, companies in MISSING_COMPANIES.items():
        for ticker, company_name in companies.items():
            if ticker not in successful_variants:
                no_results.append(f"{ticker}: {company_name} ({category})")
    
    for company in sorted(no_results):
        print(f"  {company}")
    
    # Recommendations based on results
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR CONFIG.YAML UPDATES:")
    print("-" * 80)
    
    if successful_variants:
        print("\n✓ Update config.yaml with working ticker variants:")
        print("\nmonitored_institutions:")
        for ticker, info in successful_variants.items():
            best_variant = info['working_tickers'][0]
            # Get the transcript count for best variant
            count = info['transcript_counts'][best_variant]
            
            # Show the YAML format for easy copy-paste
            print(f"  {best_variant}:")
            print(f"    name: \"{info['company']}\"")
            print(f"    type: # Update with correct type")
            print(f"    # Found {count} transcripts with this ticker")
            
            # Show what the actual API returns as primary IDs
            if info.get('sample_data') and info['sample_data'][0].get('primary_ids'):
                print(f"    # API returns primary_ids: {info['sample_data'][0]['primary_ids']}")
    
    print("\n✗ Companies with NO working tickers found:")
    print("\nThese companies may need alternative approaches:")
    for company in sorted(no_results):
        print(f"  - {company}")
    
    print("\n✓ SUGGESTED ACTIONS for companies with no results:")
    print("  1. These companies might not have earnings call coverage in FactSet")
    print("  2. Try using FactSet's entity search API to find correct identifiers")
    print("  3. Check FactSet Workstation or contact support for these specific companies")
    print("  4. Some might use SEDOL, CUSIP, or ISIN identifiers instead of tickers")
    print("  5. Verify if companies have been delisted, merged, or renamed recently")
    
    print("\n✓ Configuration insights from successful queries:")
    print(f"  - Industry categories used: {config['api_settings']['industry_categories']}")
    print(f"  - API date range: 3 years rolling window")
    print(f"  - Consider if missing companies are in different industry categories")
    
    # Cleanup
    print("\n" + "=" * 80)
    print("Cleaning up...")
    cleanup_ssl_certificate()
    nas_conn.close()
    print("✓ Test completed!")

if __name__ == "__main__":
    main()