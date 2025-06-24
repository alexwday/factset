"""
Simple FactSet API test with SSL certificate from SMB share
Paste this entire code into a Dataiku/Jupyter notebook cell
"""

import requests
import json
import tempfile
import os
import time
from smb.SMBConnection import SMBConnection
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CONFIGURATION - UPDATE THESE VALUES
SMB_USER = "your_username"
SMB_PASS = "your_password"
SMB_IP = "10.0.0.1"
SMB_SHARE = "shared"
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"
API_KEY = "your_api_key_here"

# Connect to SMB and get certificate
print("Connecting to SMB share...")
conn = SMBConnection(SMB_USER, SMB_PASS, "python-client", "", use_ntlm_v2=True, is_direct_tcp=True)
conn.connect(SMB_IP, 445)

# Download certificate to temp file
temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
conn.retrieveFile(SMB_SHARE, SMB_CERT_PATH, temp_cert)
temp_cert.close()
conn.close()

print(f"Certificate downloaded to: {temp_cert.name}")

# Verify certificate file exists and has content
cert_size = os.path.getsize(temp_cert.name)
print(f"Certificate file size: {cert_size} bytes")

# Set SSL environment variables
os.environ['REQUESTS_CA_BUNDLE'] = temp_cert.name
os.environ['SSL_CERT_FILE'] = temp_cert.name
os.environ['CURL_CA_BUNDLE'] = temp_cert.name
os.environ['PYTHONHTTPSVERIFY'] = '1'
print("SSL environment variables set")

# Test FactSet API
url = "https://api.factset.com/content/factset-fundamentals/v2/income-statement"
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}
body = {
    "ids": ["AAPL-US"],
    "periodicity": "ANN",
    "fiscalPeriodStart": "2023-01-01",
    "fiscalPeriodEnd": "2023-12-31"
}

# Create session with retry strategy
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

# Set headers and timeout
session.headers.update(headers)

print(f"Making API request with SSL certificate: {temp_cert.name}")

try:
    response = session.post(
        url, 
        json=body, 
        verify=temp_cert.name,
        timeout=(30, 60),  # (connect timeout, read timeout)
        stream=False
    )
    
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
    print(f"Response size: {len(response.content)} bytes")
    
    if response.status_code == 200:
        print("Success!")
        # Check if response is actually JSON
        content_type = response.headers.get('content-type', '').lower()
        if 'application/json' in content_type:
            try:
                data = response.json()
                print(json.dumps(data, indent=2)[:1000])
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print("Raw response content:")
                print(response.text[:1000])
        else:
            print("Response is not JSON. Raw content:")
            print(response.text[:1000])
    else:
        print(f"Error response:")
        print(f"Raw content: {response.text[:500]}")
        
except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: {e}")
    print("This could be due to:")
    print("- Firewall blocking the connection")
    print("- Proxy settings needed")
    print("- Network connectivity issues")
    print("- Server rejecting the connection")
    
except requests.exceptions.Timeout as e:
    print(f"Timeout Error: {e}")
    print("The server took too long to respond")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Clean up
os.unlink(temp_cert.name)

# Clear SSL environment variables
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'PYTHONHTTPSVERIFY']:
    os.environ.pop(var, None)