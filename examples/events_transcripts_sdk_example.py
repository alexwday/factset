"""
FactSet Events & Transcripts SDK Example with SMB SSL
Matches their exact example pattern but with SMB SSL setup
"""

import os
import tempfile
import pandas as pd
import time
import requests
from urllib.parse import urlsplit, urlparse
from smb.SMBConnection import SMBConnection
from dateutil.parser import parse as dateutil_parser
from pprint import pprint

# ===== CONFIGURATION =====
SMB_USER = "your_username"
SMB_PASS = "your_password" 
SMB_IP = "10.0.0.1"
SMB_SHARE = "shared"
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"

USERNAME = "your_factset_username"
API_KEY = "your_api_key_here"
HTTP_PROXY = "http://proxy.company.com:8080"
HTTPS_PROXY = "http://proxy.company.com:8080"

output_dir = '/tmp/factset_transcripts'  # Directory to save transcript files
# ========================

print("FactSet Events & Transcripts SDK Test")
print("=" * 50)

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
print("✓ SSL and proxy configured")

# Import SDK (following their exact pattern)
try:
    import fds.sdk.EventsandTranscripts
    from fds.sdk.EventsandTranscripts.api import transcripts_api
    from fds.sdk.EventsandTranscripts.models import *
    print("✓ EventsandTranscripts SDK imported")
except ImportError as e:
    print(f"Please install: pip install fds.sdk.EventsandTranscripts==0.21.8")
    print(f"Error: {e}")
    exit(1)

# API-KEY authorization with proxy and SSL configuration
configuration = fds.sdk.EventsandTranscripts.Configuration(
     username=USERNAME,
     password=API_KEY
)

# Configure proxy and SSL for the SDK
try:
    configuration.proxy = HTTPS_PROXY
    configuration.ssl_ca_cert = temp_cert.name
    configuration.verify_ssl = True
    print("✓ SDK proxy and SSL configured")
except AttributeError as e:
    print(f"Warning: Could not set SDK proxy/SSL: {e}")
    print("SDK will use environment variables")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

authorization = (USERNAME, API_KEY)  # For viewing the XML files returned by transcripts endpoints.

print("\ntranscripts/dates response: ")

# Enter a context with an instance of the API client (their exact code)
with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = transcripts_api.TranscriptsApi(api_client)

    start_date = dateutil_parser('2020-10-01').date() 
    end_date = dateutil_parser('2020-12-26').date() 
    time_zone = "America/New_York"
    sort = ["-storyDateTime"] 
    pagination_limit = 20 
    pagination_offset = 0 

    try:
        api_response = api_instance.get_transcripts_dates(
            start_date=start_date, 
            end_date=end_date, 
            time_zone=time_zone, 
            sort=sort, 
            pagination_limit=pagination_limit, 
            pagination_offset=pagination_offset
        )

        results = pd.DataFrame(api_response.to_dict()['data'])
        print(results)    
        
        print("\n File download: \n")

        url_series = results['transcripts_link']
        total_records = len(url_series)
        for i in range(min(total_records, 5)):  # Limit to 5 files for testing
            url = url_series.iloc[i]  # Get the URL at index 'i'

            # Get the content of the URL
            response = requests.get(url, auth=authorization)

            filename = f"transcript_dates_{i}.xml"

            # Create the full path to save the file
            path = os.path.join(output_dir, filename)

            # Write the content to the file
            with open(path, 'wb') as f:
                f.write(response.content)

            # Print progress
            print(f"{i + 1} / {min(total_records, 5)} files downloaded: {filename}")
            
    except Exception as e:
        print(f"Error getting transcripts dates: {e}")

print('\ntranscripts/time-zones: ')

# Enter a context with an instance of the API client
with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
    api_instance = transcripts_api.TranscriptsApi(api_client)

    try:
        api_response = api_instance.get_timezone()
        results = pd.DataFrame(api_response.to_dict()['data'])
        print(results)
    except Exception as e:
        print(f"Error getting timezones: {e}")
    
print('\ntranscripts/categories: ')

# Enter a context with an instance of the API client
with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
    api_instance = transcripts_api.TranscriptsApi(api_client)

    try:
        api_response = api_instance.getcategories()
        results = pd.DataFrame(api_response.to_dict()['data'])
        print(results)
    except Exception as e:
        print(f"Error getting categories: {e}")

# Cleanup
os.unlink(temp_cert.name)
for var in ['REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE']:
    os.environ.pop(var, None)

print(f"\nCompleted! Files saved to: {output_dir}")