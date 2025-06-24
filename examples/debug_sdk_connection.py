"""
Debug FactSet SDK connection and configuration
"""

import os
import tempfile
from smb.SMBConnection import SMBConnection

# Configuration
SMB_USER = "your_username"
SMB_PASS = "your_password" 
SMB_IP = "10.0.0.1"
SMB_SHARE = "shared"
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"
USERNAME = "your_factset_username"
API_KEY = "your_api_key_here"
HTTP_PROXY = "http://proxy.company.com:8080"
HTTPS_PROXY = "http://proxy.company.com:8080"

# Setup SSL and proxy
os.environ['HTTP_PROXY'] = HTTP_PROXY
os.environ['HTTPS_PROXY'] = HTTPS_PROXY

conn = SMBConnection(SMB_USER, SMB_PASS, "python-client", "", use_ntlm_v2=True, is_direct_tcp=True)
conn.connect(SMB_IP, 445)
temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
conn.retrieveFile(SMB_SHARE, SMB_CERT_PATH, temp_cert)
temp_cert.close()
conn.close()

os.environ['REQUESTS_CA_BUNDLE'] = temp_cert.name
os.environ['SSL_CERT_FILE'] = temp_cert.name

print("Environment variables set:")
print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
print(f"REQUESTS_CA_BUNDLE: {os.environ.get('REQUESTS_CA_BUNDLE')}")

# Test 1: Direct requests with our settings
import requests
print("\n1. Testing direct requests...")
try:
    response = requests.get("https://api.factset.com", timeout=10, verify=temp_cert.name)
    print(f"✓ Direct requests work: {response.status_code}")
except Exception as e:
    print(f"✗ Direct requests fail: {e}")

# Test 2: Import and inspect SDK configuration
try:
    import fds.sdk.FactSetFundamentals
    print("\n2. Testing SDK configuration...")
    
    configuration = fds.sdk.FactSetFundamentals.Configuration(
        username=USERNAME,
        password=API_KEY
    )
    
    print("Available configuration attributes:")
    for attr in dir(configuration):
        if not attr.startswith('_'):
            print(f"- {attr}")
    
    # Try to set proxy and SSL
    if hasattr(configuration, 'proxy'):
        configuration.proxy = HTTP_PROXY
        print(f"✓ Proxy set: {configuration.proxy}")
    else:
        print("✗ No proxy attribute found")
        
    if hasattr(configuration, 'ssl_ca_cert'):
        configuration.ssl_ca_cert = temp_cert.name
        print(f"✓ SSL cert set: {configuration.ssl_ca_cert}")
    else:
        print("✗ No ssl_ca_cert attribute found")
        
    if hasattr(configuration, 'verify_ssl'):
        configuration.verify_ssl = True
        print(f"✓ SSL verify set: {configuration.verify_ssl}")
    else:
        print("✗ No verify_ssl attribute found")
        
    # Check host
    print(f"SDK host: {configuration.host}")
    
    # Try creating API client
    with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
        print("✓ API Client created successfully")
        
        # Check the underlying HTTP client configuration
        if hasattr(api_client, 'rest_client'):
            print("REST client attributes:")
            for attr in dir(api_client.rest_client):
                if 'proxy' in attr.lower() or 'ssl' in attr.lower():
                    print(f"- {attr}")
                    
except ImportError as e:
    print(f"SDK import failed: {e}")
except Exception as e:
    print(f"SDK configuration failed: {e}")

# Cleanup
os.unlink(temp_cert.name)