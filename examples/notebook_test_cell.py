"""
Standalone notebook cell for testing FactSet API
Copy this entire code into a Jupyter/Dataiku notebook cell
"""

import requests
import json
from datetime import datetime, timedelta

# CONFIGURATION - UPDATE THESE VALUES
API_KEY = "your_api_key_here"  # <-- PASTE YOUR API KEY HERE
BASE_URL = "https://api.factset.com"

# Test function
def test_factset_api():
    """Test FactSet API connection and fetch some data"""
    
    # Create session with headers
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    })
    
    print("Testing FactSet API Connection...")
    print("=" * 50)
    
    # Test 1: Try to get income statement data
    endpoint = "/content/factset-fundamentals/v2/income-statement"
    url = f"{BASE_URL}{endpoint}"
    
    request_body = {
        "ids": ["AAPL-US"],  # Apple Inc.
        "periodicity": "ANN",  # Annual
        "fiscalPeriodStart": "2022-01-01",
        "fiscalPeriodEnd": "2023-12-31"
    }
    
    print(f"\nTest 1: Income Statement")
    print(f"URL: {url}")
    print(f"Request: {json.dumps(request_body, indent=2)}")
    
    try:
        response = session.post(url, json=request_body)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response preview:")
            data = response.json()
            print(json.dumps(data, indent=2)[:500] + "...")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    
    # Test 2: Try to get balance sheet data
    endpoint = "/content/factset-fundamentals/v2/balance-sheet"
    url = f"{BASE_URL}{endpoint}"
    
    request_body = {
        "ids": ["AAPL-US"],
        "periodicity": "ANN",
        "fiscalPeriodStart": "2022-01-01",
        "fiscalPeriodEnd": "2023-12-31"
    }
    
    print(f"\n\nTest 2: Balance Sheet")
    print(f"URL: {url}")
    
    try:
        response = session.post(url, json=request_body)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response preview:")
            data = response.json()
            print(json.dumps(data, indent=2)[:500] + "...")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
    
    # Test 3: Try to get cash flow data
    endpoint = "/content/factset-fundamentals/v2/cash-flow-statement"
    url = f"{BASE_URL}{endpoint}"
    
    request_body = {
        "ids": ["AAPL-US", "MSFT-US"],  # Multiple companies
        "periodicity": "ANN",
        "fiscalPeriodStart": "2023-01-01",
        "fiscalPeriodEnd": "2023-12-31"
    }
    
    print(f"\n\nTest 3: Cash Flow Statement (Multiple Companies)")
    print(f"URL: {url}")
    
    try:
        response = session.post(url, json=request_body)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response preview:")
            data = response.json()
            print(json.dumps(data, indent=2)[:500] + "...")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# Run the test
if __name__ == "__main__":
    if API_KEY == "your_api_key_here":
        print("ERROR: Please update the API_KEY variable with your actual API key!")
    else:
        test_factset_api()