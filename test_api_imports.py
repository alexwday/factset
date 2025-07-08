"""
Test what's available in the EventsandTranscripts API module
"""

import os
from urllib.parse import quote

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication
user = PROXY_USER
password = quote(PROXY_PASSWORD)

try:
    print("Testing EventsandTranscripts API imports...")
    
    # Test basic import
    import fds.sdk.EventsandTranscripts
    print("✓ Successfully imported fds.sdk.EventsandTranscripts")
    
    # Test api module
    from fds.sdk.EventsandTranscripts import api
    print("✓ Successfully imported api module")
    
    # List what's in the api module
    print("\nAvailable in api module:")
    api_contents = dir(api)
    for item in sorted(api_contents):
        if not item.startswith('_'):
            print(f"  - {item}")
    
    # Try specific imports
    print("\nTesting specific imports:")
    
    try:
        from fds.sdk.EventsandTranscripts.api import transcripts_api
        print("✓ Successfully imported transcripts_api")
    except ImportError as e:
        print(f"✗ Failed to import transcripts_api: {e}")
    
    try:
        from fds.sdk.EventsandTranscripts.api import calendar_events_api
        print("✓ Successfully imported calendar_events_api")
    except ImportError as e:
        print(f"✗ Failed to import calendar_events_api: {e}")
    
    try:
        from fds.sdk.EventsandTranscripts.api import calendar_api
        print("✓ Successfully imported calendar_api")
    except ImportError as e:
        print(f"✗ Failed to import calendar_api: {e}")
    
    # Check what's in calendar_events_api if it exists
    try:
        from fds.sdk.EventsandTranscripts.api import calendar_events_api
        print("\nAvailable in calendar_events_api module:")
        for item in sorted(dir(calendar_events_api)):
            if not item.startswith('_'):
                print(f"  - {item}")
    except:
        pass
    
except Exception as e:
    print(f"Error during import test: {e}")
    import traceback
    traceback.print_exc()