"""
FactSet SDK Complete Notebook Cell
SMB SSL + Proxy + Official SDK Configuration
Paste this entire code into a Dataiku/Jupyter notebook cell

Required pip installs:
pip install fds.sdk.utils  
pip install fds.sdk.FactSetFundamentals
pip install fds.sdk.EventsandTranscripts==0.21.8
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

# Proxy settings (with authentication if needed)
PROXY_USER = "proxy_username"  # Leave empty "" if no auth needed
PROXY_PASS = "proxy_password"  # Leave empty "" if no auth needed  
PROXY_HOST = "proxy.company.com:8080"

# Build proxy URLs
if PROXY_USER and PROXY_PASS:
    HTTP_PROXY = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}"
    HTTPS_PROXY = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}"
else:
    HTTP_PROXY = f"http://{PROXY_HOST}"
    HTTPS_PROXY = f"http://{PROXY_HOST}"
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
    from fds.sdk.FactSetFundamentals.api import income_statement_api
    from fds.sdk.FactSetFundamentals.models import *
    print("✓ FactSet SDK imported successfully")
except ImportError as e:
    print(f"✗ SDK import failed: {e}")
    print("Please install: pip install fds.sdk.FactSetFundamentals")
    exit(1)

# Step 4: Configure FactSet SDK with API Key Authentication (official docs)
print("\n2. Configuring FactSet SDK...")

# API Key authentication (official pattern)
configuration = fds.sdk.FactSetFundamentals.Configuration(
    username=USERNAME,  # USERNAME-SERIAL from Developer Portal
    password=API_KEY,   # API-KEY from Developer Portal
    # Proxy configuration
    proxy=HTTPS_PROXY,
    proxy_headers={} if PROXY_USER else None,
    # SSL configuration
    ssl_ca_cert=temp_cert.name
)

print("✓ FactSet SDK configured with API Key authentication, proxy and SSL")

# Step 5: Test FactSet Fundamentals API
print("\n3. Testing FactSet Fundamentals API...")

try:
    with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
        api_instance = income_statement_api.IncomeStatementApi(api_client)
        
        # Use correct method name
        ids = ["AAPL-US", "MSFT-US"]
        api_response = api_instance.get_income_statement(
            ids=ids,
            periodicity="ANN",
            fiscal_period_start="2023-01-01",
            fiscal_period_end="2023-12-31"
        )
        
        results = pd.DataFrame(api_response.to_dict().get('data', []))
        print(f"✓ Fundamentals: Retrieved {len(results)} records")
        
        if not results.empty:
            print("Sample data:")
            print(results.head())

except Exception as e:
    print(f"✗ Fundamentals API failed: {e}")
    print("Available methods:")
    try:
        with fds.sdk.FactSetFundamentals.ApiClient(configuration) as api_client:
            api_instance = income_statement_api.IncomeStatementApi(api_client)
            methods = [m for m in dir(api_instance) if not m.startswith('_')]
            print(f"Available methods: {methods}")
    except:
        pass

# Step 6: Test Events & Transcripts API
print("\n4. Testing Events & Transcripts API...")

try:
    import fds.sdk.EventsandTranscripts
    from fds.sdk.EventsandTranscripts.api import transcripts_api
    from dateutil.parser import parse as dateutil_parser
    
    # Configure Events SDK with same settings
    events_config = fds.sdk.EventsandTranscripts.Configuration(
        username=USERNAME,
        password=API_KEY,
        proxy=HTTPS_PROXY,
        proxy_headers={} if PROXY_USER else None,
        ssl_ca_cert=temp_cert.name
    )
    
    with fds.sdk.EventsandTranscripts.ApiClient(events_config) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        
        # Get transcripts dates (their example)
        start_date = dateutil_parser('2020-10-01').date() 
        end_date = dateutil_parser('2020-12-26').date() 
        
        api_response = api_instance.get_transcripts_dates(
            start_date=start_date, 
            end_date=end_date, 
            time_zone="America/New_York",
            sort=["-storyDateTime"], 
            pagination_limit=5
        )
        
        results = pd.DataFrame(api_response.to_dict()['data'])
        print(f"✓ Events: Retrieved {len(results)} transcript records")
        print(results.head())

except Exception as e:
    print(f"✗ Events API failed: {e}")

# Step 7: Cleanup
print("\n5. Cleaning up...")
try:
    os.unlink(temp_cert.name)
    print("✓ Temporary certificate deleted")
except:
    pass

# Clear SSL environment variables
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CA_BUNDLE', 'PYTHONHTTPSVERIFY']:
    os.environ.pop(var, None)
print("✓ Environment variables cleared")

print("\nFactSet SDK test completed!")
print("If successful, you can now use FactSet SDK with API Key authentication in your workflows.")