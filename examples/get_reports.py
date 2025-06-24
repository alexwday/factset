#!/usr/bin/env python3
"""
Example script to explore FactSet API and retrieve available reports
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_key_auth import FactSetAPIKeyClient
from src.oauth2_auth import FactSetOAuth2Client

def test_api_key_auth():
    """Test connection using API key authentication"""
    print("Testing API Key Authentication...")
    try:
        client = FactSetAPIKeyClient()
        
        # FactSet v2 API endpoints to try
        endpoints_to_try = [
            "/factset-funds/v2/reference",
            "/factset-prices/v2/reference",
            "/factset-fundamentals/v2/reference",
            "/factset-estimates/v2/reference",
            "/factset-analytics/v2/reference",
            "/factset-concordance/v2/reference"
        ]
        
        for endpoint in endpoints_to_try:
            print(f"\nTrying endpoint: {endpoint}")
            try:
                result = client.get(endpoint)
                print(f"Success! Found {len(result)} items")
                # Print first item as example
                if isinstance(result, list) and result:
                    print("Example item:", result[0])
                break
            except Exception as e:
                print(f"Failed: {e}")
                
    except Exception as e:
        print(f"API Key auth failed: {e}")

def test_oauth2_auth():
    """Test connection using OAuth2 authentication"""
    print("\n\nTesting OAuth2 Authentication...")
    try:
        client = FactSetOAuth2Client()
        
        # Try to get token first
        print("Attempting to get access token...")
        token = client._get_access_token()
        print("Successfully obtained access token!")
        
        # Same v2 endpoints as above
        endpoints_to_try = [
            "/factset-funds/v2/reference",
            "/factset-prices/v2/reference",
            "/factset-fundamentals/v2/reference",
            "/factset-estimates/v2/reference",
            "/factset-analytics/v2/reference",
            "/factset-concordance/v2/reference"
        ]
        
        for endpoint in endpoints_to_try:
            print(f"\nTrying endpoint: {endpoint}")
            try:
                result = client.get(endpoint)
                print(f"Success! Found data")
                break
            except Exception as e:
                print(f"Failed: {e}")
                
    except Exception as e:
        print(f"OAuth2 auth failed: {e}")

def explore_factset_api():
    """Main function to explore FactSet API"""
    print("FactSet API Explorer")
    print("=" * 50)
    
    # Check which credentials are available
    api_key_available = os.getenv('FACTSET_API_KEY') is not None
    oauth_available = (os.getenv('FACTSET_CLIENT_ID') is not None and 
                      os.getenv('FACTSET_CLIENT_SECRET') is not None)
    
    if api_key_available:
        test_api_key_auth()
    else:
        print("API Key not configured. Set FACTSET_API_KEY in .env file")
    
    if oauth_available:
        test_oauth2_auth()
    else:
        print("\nOAuth2 not configured. Set FACTSET_CLIENT_ID and FACTSET_CLIENT_SECRET in .env file")
    
    if not api_key_available and not oauth_available:
        print("\nNo authentication configured!")
        print("Please copy .env.example to .env and add your credentials")

if __name__ == "__main__":
    explore_factset_api()