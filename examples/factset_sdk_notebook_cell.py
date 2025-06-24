"""
FactSet SDK Example with SMB SSL Certificate
Uses official FactSet SDK following their example pattern
Paste this entire code into a Dataiku/Jupyter notebook cell

Required pip installs:
pip install fds.sdk.utils
pip install fds.sdk.FactSetFundamentals
pip install pandas
pip install pysmb
"""

import os
import tempfile
import pandas as pd
from smb.SMBConnection import SMBConnection
from pprint import pprint

# ===== CONFIGURATION - UPDATE ALL THESE VALUES =====
SMB_USER = "your_username"
SMB_PASS = "your_password" 
SMB_IP = "10.0.0.1"
SMB_SHARE = "shared"
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"

USERNAME = "your_factset_username"
API_KEY = "your_api_key_here"

# Proxy settings
HTTP_PROXY = "http://proxy.company.com:8080"
HTTPS_PROXY = "http://proxy.company.com:8080"
# ===================================================

print("FactSet SDK Test with SMB SSL")
print("=" * 50)

# Step 1: Set proxy environment variables
os.environ['HTTP_PROXY'] = HTTP_PROXY
os.environ['HTTPS_PROXY'] = HTTPS_PROXY
print(f"✓ Proxy configured: {HTTP_PROXY}")

# Step 2: Download SSL certificate from SMB
print("\n1. Setting up SSL certificate...")
try:
    conn = SMBConnection(SMB_USER, SMB_PASS, "python-client", "", use_ntlm_v2=True, is_direct_tcp=True)
    conn.connect(SMB_IP, 445)
    
    temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
    conn.retrieveFile(SMB_SHARE, SMB_CERT_PATH, temp_cert)
    temp_cert.close()
    conn.close()
    
    # Set SSL environment variables
    os.environ['REQUESTS_CA_BUNDLE'] = temp_cert.name
    os.environ['SSL_CERT_FILE'] = temp_cert.name
    os.environ['CURL_CA_BUNDLE'] = temp_cert.name
    os.environ['PYTHONHTTPSVERIFY'] = '1'
    
    cert_size = os.path.getsize(temp_cert.name)
    print(f"✓ SSL Certificate ready: {cert_size} bytes")
    
except Exception as e:
    print(f"✗ SSL setup failed: {e}")
    exit(1)

# Step 3: Import FactSet SDK (after SSL is configured)
try:
    import fds.sdk.FactSetFundamentals
    from fds.sdk.FactSetFundamentals.api import company_reports_api
    from fds.sdk.FactSetFundamentals.models import *
    print("✓ FactSet SDK imported successfully")
except ImportError as e:
    print(f"✗ SDK import failed: {e}")
    print("Please install: pip install fds.sdk.FactSetFundamentals")
    exit(1)

# Step 4: Configure FactSet SDK (following their pattern)
print("\n2. Configuring FactSet SDK...")
configuration = fds.sdk.FactSetFundamentals.Configuration(
    username=USERNAME,
    password=API_KEY
)
print("✓ SDK configuration ready")

# Step 5: Test API calls (following their example pattern)
print("\n3. Testing FactSet Fundamentals API...")

try:
    # Create API client and instance
    with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
        api_instance = company_reports_api.CompanyReportsApi(api_client)
        
        # Test 1: Get company reports
        print("\nCompany Reports response:")
        
        # Define request parameters
        ids = ["AAPL-US", "MSFT-US"]  # Apple and Microsoft
        
        # Make API call
        api_response = api_instance.get_company_reports(ids=ids)
        
        # Convert to DataFrame (following their pattern)
        results = pd.DataFrame(api_response.to_dict().get('data', []))
        print(f"✓ Retrieved {len(results)} records")
        
        if not results.empty:
            print("\nResults preview:")
            print(results.head())
            
            # Show available columns
            print(f"\nAvailable columns: {list(results.columns)}")
        else:
            print("No data returned")
            
        # Print raw response structure
        print(f"\nAPI Response structure:")
        response_dict = api_response.to_dict()
        for key in response_dict.keys():
            print(f"- {key}: {type(response_dict[key])}")

except Exception as e:
    print(f"✗ API call failed: {type(e).__name__}: {e}")
    
    # Try alternative endpoint if available
    try:
        print("\nTrying alternative API endpoint...")
        # You can add other API endpoints here
        
    except Exception as e2:
        print(f"✗ Alternative API also failed: {e2}")

# Step 6: Cleanup
print("\n4. Cleaning up...")
try:
    os.unlink(temp_cert.name)
    print("✓ Temporary certificate deleted")
except:
    pass

# Clear SSL environment variables
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'PYTHONHTTPSVERIFY']:
    os.environ.pop(var, None)
print("✓ Environment variables cleared")

print("\nSDK test completed!")
print("If successful, you can now use FactSet SDK in your workflows.")