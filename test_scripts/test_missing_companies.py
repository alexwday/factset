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

# Missing companies grouped by category
MISSING_COMPANIES = {
    "Canadian Monoline Lenders": {
        "MKP-CA": "MCAN Mortgage Corporation"
    },
    "European Banks": {
        "BCS-GB": "Barclays PLC",
        "ING-NL": "ING Groep N.V.",
        "ISP-IT": "Intesa Sanpaolo",
        "LLOY-GB": "Lloyds Banking Group plc",
        "STAN-GB": "Standard Chartered PLC",
        "UBS-CH": "UBS Group AG",
        "UCG-IT": "UniCredit S.p.A."
    },
    "Nordic Banks": {
        "DANSKE-DK": "Danske Bank A/S",
        "NDA-FI": "Nordea Bank Abp",
        "SEBA-SE": "Skandinaviska Enskilda Banken AB",
        "SWEDA-SE": "Swedbank AB"
    },
    "UK Wealth & Asset Managers": {
        "QLT-GB": "Quilter plc",
        "RAT-GB": "Rathbones Group Plc",
        "SJP-GB": "St. James's Place plc"
    },
    "US Banks": {
        "JEF-US": "Jefferies Financial Group Inc."
    },
    "US Wealth & Asset Managers": {
        "SCHW-US": "Charles Schwab Corporation",
        "TROW-US": "T. Rowe Price Group Inc."
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

def setup_factset_api_client():
    """Set up FactSet API client with proxy and authentication."""
    global config
    
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=os.getenv("API_USERNAME"),
        password=os.getenv("API_PASSWORD")
    )
    
    # Configure proxy if available
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url:
        proxy_user = os.getenv("PROXY_USER")
        proxy_pass = os.getenv("PROXY_PASSWORD")
        
        if proxy_user and proxy_pass:
            # URL encode credentials for special characters
            proxy_user_encoded = quote(proxy_user, safe='')
            proxy_pass_encoded = quote(proxy_pass, safe='')
            proxy_auth = f"http://{proxy_user_encoded}:{proxy_pass_encoded}@{proxy_url}"
        else:
            proxy_auth = f"http://{proxy_url}"
        
        configuration.proxy = proxy_auth
        print(f"✓ Proxy configured: {proxy_url}")
    
    # Set SSL certificate if available
    if ssl_cert_path and os.path.exists(ssl_cert_path):
        configuration.ssl_ca_cert = ssl_cert_path
        configuration.verify_ssl = True
        print(f"✓ SSL verification enabled with certificate")
    
    return transcripts_api.TranscriptsApi(
        fds.sdk.EventsandTranscripts.ApiClient(configuration)
    )

def test_ticker_formats(api_instance, ticker: str, company_name: str) -> Dict[str, Any]:
    """Test various ticker format variations."""
    global config
    results = {}
    
    # Calculate date range (3 years)
    end_date = datetime.now().date()
    start_date = (datetime.now() - timedelta(days=365*3)).date()
    
    # Different ticker formats to try
    ticker_variations = [
        ticker,                              # Original: MKP-CA
        ticker.split('-')[0] if '-' in ticker else ticker,  # Without country: MKP
        ticker.replace('-', '.'),            # Dot notation: MKP.CA
        ticker.replace('-', ':'),            # Colon notation: MKP:CA
        ticker.lower(),                      # Lowercase: mkp-ca
        ticker.upper(),                      # Uppercase: MKP-CA
    ]
    
    # For European tickers, try various exchange suffixes
    if '-' in ticker:
        base_ticker = ticker.split('-')[0]
        country_code = ticker.split('-')[1]
        
        # Add country-specific variations
        if country_code == 'GB':  # UK companies
            ticker_variations.extend([
                f"{base_ticker}",           # Just base
                f"{base_ticker}-GB",         # Original
                f"{base_ticker}.L",          # London Stock Exchange
                f"{base_ticker}-LON",        # London
                f"LON:{base_ticker}",        # Bloomberg style
            ])
        elif country_code in ['CH', 'SW']:  # Swiss companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-CH",
                f"{base_ticker}.SW",
                f"{base_ticker}.S",
                f"{base_ticker}-SWX",
            ])
        elif country_code == 'IT':  # Italian companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-IT",
                f"{base_ticker}.MI",         # Milan
                f"{base_ticker}-MIL",
            ])
        elif country_code == 'NL':  # Dutch companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-NL",
                f"{base_ticker}.AS",         # Amsterdam
                f"{base_ticker}-AMS",
            ])
        elif country_code == 'DK':  # Danish companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-DK",
                f"{base_ticker}.CO",         # Copenhagen
                f"{base_ticker}-CPH",
            ])
        elif country_code in ['SE', 'SW']:  # Swedish companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-SE",
                f"{base_ticker}.ST",         # Stockholm
                f"{base_ticker}-STO",
            ])
        elif country_code == 'FI':  # Finnish companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-FI",
                f"{base_ticker}.HE",         # Helsinki
                f"{base_ticker}-HEL",
            ])
        elif country_code == 'CA':  # Canadian companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-CA",
                f"{base_ticker}.TO",         # Toronto
                f"{base_ticker}-TSE",
                f"TSE:{base_ticker}",
            ])
        elif country_code == 'US':  # US companies
            ticker_variations.extend([
                f"{base_ticker}",
                f"{base_ticker}-US",
                f"{base_ticker}-NYSE",
                f"{base_ticker}-NASDAQ",
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
    
    # Step 6: Test each missing company
    print("\nStep 6: Testing missing companies...")
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
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    if successful_variants:
        print("\n✓ Update config.yaml with working ticker variants:")
        for ticker, info in successful_variants.items():
            best_variant = info['working_tickers'][0]
            if best_variant != ticker:
                print(f"  Change {ticker} to {best_variant}")
    
    print("\n✓ For companies with no results:")
    print("  1. Contact FactSet support to verify correct identifiers")
    print("  2. Check if companies are covered in FactSet's transcript database")
    print("  3. Some companies may use SEDOL, CUSIP, or ISIN identifiers instead")
    print("  4. Verify companies haven't been delisted, merged, or renamed")
    print("  5. Check if transcripts are available under parent company tickers")
    
    print("\n✓ Configuration insights from successful queries:")
    print(f"  - Industry categories used: {config['api_settings']['industry_categories']}")
    print(f"  - Consider if missing companies are in different categories")
    
    # Cleanup
    print("\n" + "=" * 80)
    print("Cleaning up...")
    cleanup_ssl_certificate()
    nas_conn.close()
    print("✓ Test completed!")

if __name__ == "__main__":
    main()