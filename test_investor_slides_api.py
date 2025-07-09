"""
Test script for FactSet Investor Slides API
This script tests basic access to the investor slides endpoint
"""

import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
import os
from urllib.parse import quote
from datetime import datetime, timedelta
import pandas as pd

# =============================================================================
# CONFIGURATION VARIABLES (HARDCODED)
# =============================================================================

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Test Configuration
TEST_DAYS_BACK = 30  # Look for slides in the last 30 days

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication
user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Configure FactSet API client
configuration = fds.sdk.EventsandTranscripts.Configuration(
    username=API_USERNAME,
    password=API_PASSWORD,
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=SSL_CERT_PATH
)
configuration.get_basic_auth_token()

# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_investor_slides_basic():
    """
    Test basic access to investor slides API with minimal parameters
    """
    print("=" * 60)
    print("INVESTOR SLIDES API TEST")
    print("=" * 60)
    
    try:
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Calculate date range
            end_date = datetime.now().date()
            start_date = (datetime.now() - timedelta(days=TEST_DAYS_BACK)).date()
            
            print(f"\nTest 1: Basic API call")
            print(f"Date range: {start_date} to {end_date}")
            print("-" * 40)
            
            try:
                # Minimal API call - just date range
                response = api_instance.get_transcripts_investor_slides(
                    start_date=start_date,
                    end_date=end_date
                )
                
                if response and hasattr(response, 'data'):
                    print(f"✓ SUCCESS: Retrieved {len(response.data)} slides")
                    
                    if response.data:
                        # Show first slide info
                        first_slide = response.data[0]
                        print(f"\nFirst slide details:")
                        print(f"  Headline: {getattr(first_slide, 'headline', 'N/A')}")
                        print(f"  Date: {getattr(first_slide, 'story_date_time', 'N/A')}")
                        print(f"  Primary IDs: {getattr(first_slide, 'primary_ids', 'N/A')}")
                else:
                    print("✓ SUCCESS: API call succeeded but no slides found")
                    
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                if hasattr(e, 'status'):
                    print(f"HTTP Status: {e.status}")
                if hasattr(e, 'reason'):
                    print(f"Reason: {e.reason}")
                if hasattr(e, 'body'):
                    print(f"Response body: {e.body}")
            
            # Test 2: With specific company
            print(f"\n\nTest 2: API call with RBC")
            print("-" * 40)
            
            try:
                response = api_instance.get_transcripts_investor_slides(
                    start_date=start_date,
                    end_date=end_date,
                    ids=["RY-CA"]  # Royal Bank of Canada
                )
                
                if response and hasattr(response, 'data'):
                    print(f"✓ SUCCESS: Retrieved {len(response.data)} RBC slides")
                    
                    if response.data:
                        df = pd.DataFrame([s.to_dict() for s in response.data])
                        print(f"\nRBC slides summary:")
                        print(f"  Total slides: {len(df)}")
                        if 'event_id' in df.columns:
                            print(f"  Unique events: {df['event_id'].nunique()}")
                        if 'story_date_time' in df.columns:
                            print(f"  Date range: {df['story_date_time'].min()} to {df['story_date_time'].max()}")
                else:
                    print("✓ SUCCESS: API call succeeded but no RBC slides found")
                    
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                
            # Test 3: Very recent data (last 7 days)
            print(f"\n\nTest 3: Very recent data (last 7 days)")
            print("-" * 40)
            
            recent_start = (datetime.now() - timedelta(days=7)).date()
            
            try:
                response = api_instance.get_transcripts_investor_slides(
                    start_date=recent_start,
                    end_date=end_date,
                    pagination_limit=10
                )
                
                if response and hasattr(response, 'data'):
                    print(f"✓ SUCCESS: Retrieved {len(response.data)} recent slides")
                else:
                    print("✓ SUCCESS: API call succeeded but no recent slides found")
                    
            except Exception as e:
                print(f"✗ FAILED: {str(e)}")
                
    except Exception as e:
        print(f"\nFATAL ERROR: Could not initialize API client")
        print(f"Error: {str(e)}")

def main():
    """
    Run all tests
    """
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("\nStarting Investor Slides API tests...")
    print(f"API User: {API_USERNAME}")
    print(f"Testing date range: Last {TEST_DAYS_BACK} days")
    
    test_investor_slides_basic()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()