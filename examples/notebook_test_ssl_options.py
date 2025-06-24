"""
FactSet API test with multiple SSL options
Paste this into a notebook cell and uncomment the option you need
"""

import requests
import json
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Configuration
API_KEY = "your_api_key_here"

# Test function
def test_factset_api(verify_option):
    """Test FactSet API with different SSL verification options"""
    
    url = "https://api.factset.com/content/factset-fundamentals/v2/income-statement"
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    body = {
        "ids": ["AAPL-US"],
        "periodicity": "ANN",
        "fiscalPeriodStart": "2023-01-01",
        "fiscalPeriodEnd": "2023-12-31"
    }
    
    print(f"Testing with verify={verify_option}")
    try:
        response = requests.post(url, json=body, headers=headers, verify=verify_option)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            data = response.json()
            print(f"Got {len(data.get('data', []))} records")
        else:
            print(f"Error: {response.text[:300]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)[:300]}")

# OPTION 1: Try with default SSL (this is what's failing for you)
print("Option 1: Default SSL verification")
test_factset_api(verify=True)

print("\n" + "="*50 + "\n")

# OPTION 2: Disable SSL verification (for testing only!)
print("Option 2: SSL verification disabled (NOT FOR PRODUCTION)")
warnings.filterwarnings('ignore', category=InsecureRequestWarning)
test_factset_api(verify=False)

print("\n" + "="*50 + "\n")

# OPTION 3: Use a local certificate file
print("Option 3: Using local certificate file")
# Update this path to your certificate location
cert_path = "/path/to/your/certificate.cer"
# test_factset_api(verify=cert_path)

print("\n" + "="*50 + "\n")

# OPTION 4: Download certificate from URL
print("Option 4: Download certificate from URL")
import tempfile
import urllib.request

cert_url = "https://your-server.com/path/to/certificate.cer"
try:
    with tempfile.NamedTemporaryFile(suffix='.cer', delete=False) as tmp_cert:
        # urllib.request.urlretrieve(cert_url, tmp_cert.name)
        # test_factset_api(verify=tmp_cert.name)
        print("Uncomment the lines above and update cert_url to use this option")
except Exception as e:
    print(f"Failed to download cert: {e}")