#!/usr/bin/env python3
"""
Test script to investigate missing companies from FactSet API.
Tests various ticker formats and company identifiers.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
from fds.sdk.EventsandTranscripts.model.company_event_request import CompanyEventRequest
from fds.sdk.EventsandTranscripts.exceptions import ApiException
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def setup_api_client():
    """Set up FactSet API client with authentication."""
    configuration = fds.sdk.EventsandTranscripts.Configuration(
        username=os.getenv("API_USERNAME"),
        password=os.getenv("API_PASSWORD")
    )
    
    # Set up proxy if needed
    proxy_url = os.getenv("PROXY_URL")
    if proxy_url:
        proxy_user = os.getenv("PROXY_USER")
        proxy_pass = os.getenv("PROXY_PASSWORD")
        if proxy_user and proxy_pass:
            configuration.proxy = f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
        else:
            configuration.proxy = f"http://{proxy_url}"
    
    return transcripts_api.TranscriptsApi(fds.sdk.EventsandTranscripts.ApiClient(configuration))

def test_ticker_formats(api_instance, ticker: str, company_name: str) -> Dict[str, Any]:
    """Test various ticker format variations."""
    results = {}
    
    # Calculate date range (3 years)
    end_date = datetime.now().date()
    start_date = (datetime.now() - timedelta(days=365*3)).date()
    
    # Different ticker formats to try
    ticker_variations = [
        ticker,                              # Original: MKP-CA
        ticker.split('-')[0],                # Without country: MKP
        ticker.replace('-', '.'),            # Dot notation: MKP.CA
        ticker.lower(),                      # Lowercase: mkp-ca
        ticker.upper(),                      # Uppercase: MKP-CA
        f"{ticker.split('-')[0]}-{ticker.split('-')[1].upper()}",  # Ensure uppercase country
    ]
    
    # For European tickers, try without country code
    if '-' in ticker:
        base_ticker = ticker.split('-')[0]
        ticker_variations.extend([
            base_ticker,                     # Just the base ticker
            f"{base_ticker}-EU",             # Try EU suffix
            f"{base_ticker}.L",              # London notation for UK stocks
            f"{base_ticker}.AS",             # Amsterdam notation
            f"{base_ticker}.MI",             # Milan notation
            f"{base_ticker}.CO",             # Copenhagen notation
            f"{base_ticker}.ST",             # Stockholm notation
            f"{base_ticker}.HE",             # Helsinki notation
            f"{base_ticker}.SW",             # Swiss notation
        ])
    
    print(f"\nTesting {company_name} ({ticker})...")
    print(f"Date range: {start_date} to {end_date}")
    
    for variant in set(ticker_variations):  # Use set to avoid duplicates
        try:
            print(f"  Trying ticker variant: {variant}...", end=" ")
            
            api_params = {
                "ids": [variant],
                "start_date": start_date,
                "end_date": end_date,
                "categories": ["IN:BANKS", "IN:FNLSVC", "IN:INS", "IN:SECS"],
                "sort": ["-storyDateTime"],
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
                        "event_date": str(t_dict.get("event_date_time", ""))[:10] if t_dict.get("event_date_time") else None
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
            
            # Add delay to avoid rate limiting
            time.sleep(0.5)
            
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

def search_by_company_name(api_instance, company_name: str) -> Optional[List[str]]:
    """Try to find company using company event search."""
    try:
        print(f"\n  Searching by company name: {company_name}...")
        
        # Create request for company events
        request_body = CompanyEventRequest(
            data={
                "universe": [company_name],  # Try company name directly
                "date_time": {
                    "start": (datetime.now() - timedelta(days=365*3)).isoformat(),
                    "end": datetime.now().isoformat()
                }
            }
        )
        
        # Note: This endpoint might not be available in all SDK versions
        # This is a placeholder - actual implementation depends on available endpoints
        print("    Note: Company name search requires specific API endpoint that may not be available")
        
    except Exception as e:
        print(f"    Error in company name search: {str(e)[:100]}")
    
    return None

def main():
    """Main test function."""
    print("=" * 80)
    print("FactSet API Missing Companies Investigation")
    print("=" * 80)
    
    # Verify environment variables
    required_vars = ["API_USERNAME", "API_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"ERROR: Missing environment variables: {missing_vars}")
        print("Please set API_USERNAME and API_PASSWORD")
        sys.exit(1)
    
    # Set up API client
    print("\nSetting up API client...")
    api_instance = setup_api_client()
    
    # Test each missing company
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
                    }
                }
            
            # Try company name search as fallback
            if not success_variants:
                search_by_company_name(api_instance, company_name)
            
            # Delay between companies
            time.sleep(1)
        
        all_results[category] = category_results
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    # Save detailed results to file
    output_file = "missing_companies_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print successful variants
    if successful_variants:
        print("\n✓ COMPANIES WITH WORKING TICKER VARIANTS:")
        print("-" * 60)
        for ticker, info in successful_variants.items():
            print(f"\n{ticker}: {info['company']}")
            print(f"  Working tickers: {', '.join(info['working_tickers'])}")
            for variant, count in info['transcript_counts'].items():
                print(f"    {variant}: {count} transcripts")
    
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
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    print("1. For companies with no results, verify the ticker symbols with FactSet")
    print("2. Some companies may use different identifiers (SEDOL, CUSIP, ISIN)")
    print("3. Check if companies have been delisted or merged")
    print("4. Consider using FactSet's entity search API to find correct identifiers")
    print("5. Update config.yaml with working ticker variants found above")
    
    print("\n" + "=" * 80)
    print("Test completed!")

if __name__ == "__main__":
    main()