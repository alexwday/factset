"""
Final FactSet API Test - Complete notebook cell
Copy this entire code into a Dataiku/Jupyter notebook cell
"""

import requests
import json
import tempfile
import os
import time
from smb.SMBConnection import SMBConnection
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== CONFIGURATION - UPDATE ALL THESE VALUES =====
SMB_USER = "your_username"
SMB_PASS = "your_password" 
SMB_IP = "10.0.0.1"
SMB_SHARE = "shared"
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"

API_KEY = "your_api_key_here"

# Proxy settings
HTTP_PROXY = "http://proxy.company.com:8080"
HTTPS_PROXY = "http://proxy.company.com:8080"
# ===================================================

print("FactSet API Test - Final Version")
print("=" * 50)

# Step 1: Set proxy environment variables
os.environ['HTTP_PROXY'] = HTTP_PROXY
os.environ['HTTPS_PROXY'] = HTTPS_PROXY
print(f"✓ Proxy configured: {HTTP_PROXY}")

# Step 2: Connect to SMB and download SSL certificate
print("\n1. Downloading SSL certificate from SMB...")
try:
    conn = SMBConnection(SMB_USER, SMB_PASS, "python-client", "", use_ntlm_v2=True, is_direct_tcp=True)
    conn.connect(SMB_IP, 445)
    
    temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
    conn.retrieveFile(SMB_SHARE, SMB_CERT_PATH, temp_cert)
    temp_cert.close()
    conn.close()
    
    cert_size = os.path.getsize(temp_cert.name)
    print(f"✓ Certificate downloaded: {cert_size} bytes")
    
except Exception as e:
    print(f"✗ SMB Error: {e}")
    exit(1)

# Step 3: Set SSL environment variables
os.environ['REQUESTS_CA_BUNDLE'] = temp_cert.name
os.environ['SSL_CERT_FILE'] = temp_cert.name
os.environ['CURL_CA_BUNDLE'] = temp_cert.name
os.environ['PYTHONHTTPSVERIFY'] = '1'
print("✓ SSL environment variables set")

# Step 4: Create robust HTTP session
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

session.headers.update({
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
})

# Step 5: Test FactSet API
print("\n2. Testing FactSet API...")

url = "https://api.factset.com/content/factset-fundamentals/v2/income-statement"
request_body = {
    "ids": ["AAPL-US"],
    "periodicity": "ANN",
    "fiscalPeriodStart": "2023-01-01",
    "fiscalPeriodEnd": "2023-12-31"
}

print(f"URL: {url}")
print(f"Request: {json.dumps(request_body, indent=2)}")

try:
    response = session.post(
        url, 
        json=request_body, 
        verify=temp_cert.name,
        timeout=(30, 60)
    )
    
    print(f"\nResponse Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
    print(f"Response Size: {len(response.content)} bytes")
    
    if response.status_code == 200:
        content_type = response.headers.get('content-type', '').lower()
        if 'application/json' in content_type:
            try:
                data = response.json()
                print("\n3. SUCCESS! FactSet API Response:")
                print("=" * 50)
                
                # Pretty print the response
                if isinstance(data, dict):
                    if 'data' in data:
                        print(f"Records returned: {len(data['data'])}")
                        if data['data']:
                            print("\nFirst record:")
                            print(json.dumps(data['data'][0], indent=2))
                        else:
                            print("No data records returned")
                    else:
                        print("Response structure:")
                        for key in data.keys():
                            print(f"- {key}: {type(data[key])}")
                        print("\nFull response preview:")
                        print(json.dumps(data, indent=2)[:1000] + "...")
                else:
                    print(f"Response type: {type(data)}")
                    print(json.dumps(data, indent=2)[:1000])
                    
            except json.JSONDecodeError as e:
                print(f"\n✗ JSON Decode Error: {e}")
                print("Raw response:")
                print(response.text[:1000])
        else:
            print(f"\n✗ Unexpected content type: {content_type}")
            print("Raw response:")
            print(response.text[:1000])
    else:
        print(f"\n✗ API Error: {response.status_code}")
        print("Response:")
        print(response.text[:1000])
        
except requests.exceptions.ConnectionError as e:
    print(f"\n✗ Connection Error: {e}")
except requests.exceptions.Timeout as e:
    print(f"\n✗ Timeout Error: {e}")
except Exception as e:
    print(f"\n✗ Unexpected Error: {type(e).__name__}: {e}")

# Step 6: Cleanup
print("\n4. Cleaning up...")
try:
    os.unlink(temp_cert.name)
    print("✓ Temporary certificate deleted")
except:
    pass

# Clear environment variables
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'PYTHONHTTPSVERIFY']:
    os.environ.pop(var, None)
print("✓ SSL environment variables cleared")

print("\nTest completed!")