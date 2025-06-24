"""
Simple FactSet API test with SSL certificate from SMB share
Paste this entire code into a Dataiku/Jupyter notebook cell
"""

import requests
import json
import tempfile
import os
from smb.SMBConnection import SMBConnection

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

print(f"Making API request with SSL certificate: {temp_cert.name}")
response = requests.post(url, json=body, headers=headers, verify=temp_cert.name)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2)[:1000])
else:
    print(f"Error: {response.text[:500]}")

# Clean up
os.unlink(temp_cert.name)

# Clear SSL environment variables
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'PYTHONHTTPSVERIFY']:
    os.environ.pop(var, None)