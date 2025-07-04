"""
Test script to debug Calendar Events API
"""

import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import calendar_events_api
from fds.sdk.EventsandTranscripts.model.company_event_request import CompanyEventRequest
from fds.sdk.EventsandTranscripts.model.company_event_request_data import CompanyEventRequestData
from fds.sdk.EventsandTranscripts.model.company_event_request_data_date_time import CompanyEventRequestDataDateTime
from fds.sdk.EventsandTranscripts.model.company_event_request_data_universe import CompanyEventRequestDataUniverse
from dateutil.parser import parse as dateutil_parser
import os
from urllib.parse import quote
from datetime import datetime, timedelta
from pprint import pprint

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication
user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Configure FactSet API client
configuration = fds.sdk.EventsandTranscripts.Configuration(
    username=API_USERNAME,
    password=API_PASSWORD,
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=SSL_CERT_PATH
)
configuration.get_basic_auth_token()

def test_minimal_request():
    """Test with minimal request matching documentation example"""
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = calendar_events_api.CalendarEventsApi(api_client)
        
        # Try exact format from documentation
        company_event_request = CompanyEventRequest(
            data=CompanyEventRequestData(
                date_time=CompanyEventRequestDataDateTime(
                    start=dateutil_parser('2024-12-01T00:00:00Z'),
                    end=dateutil_parser('2024-12-31T22:59:02Z'),
                ),
                universe=CompanyEventRequestDataUniverse(
                    symbols=["AAPL-US"],  # Just one symbol
                    type="Tickers",
                ),
                event_types=[
                    "Earnings",
                ]
            ),
        )
        
        try:
            print("Making API request...")
            api_response = api_instance.get_company_event(company_event_request)
            print("Success!")
            pprint(api_response)
            
        except fds.sdk.EventsandTranscripts.ApiException as e:
            print(f"API Exception: {e}")
            print(f"Status: {e.status}")
            print(f"Reason: {e.reason}")
            print(f"Body: {e.body}")
            print(f"Headers: {e.headers}")

def test_with_bank():
    """Test with one bank"""
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = calendar_events_api.CalendarEventsApi(api_client)
        
        # Current date range
        start_date = datetime.now()
        end_date = start_date + timedelta(days=90)
        
        company_event_request = CompanyEventRequest(
            data=CompanyEventRequestData(
                date_time=CompanyEventRequestDataDateTime(
                    start=dateutil_parser(start_date.strftime('%Y-%m-%dT00:00:00Z')),
                    end=dateutil_parser(end_date.strftime('%Y-%m-%dT23:59:59Z')),
                ),
                universe=CompanyEventRequestDataUniverse(
                    symbols=["RY-CA"],  # Just Royal Bank
                    type="Tickers",
                )
            ),
        )
        
        try:
            print("\nTesting with RY-CA...")
            api_response = api_instance.get_company_event(company_event_request)
            print("Success!")
            pprint(api_response)
            
        except fds.sdk.EventsandTranscripts.ApiException as e:
            print(f"API Exception: {e}")
            print(f"Status: {e.status}")
            print(f"Reason: {e.reason}")
            print(f"Body: {e.body}")

def test_with_last_modified():
    """Test using lastModifiedWithin instead of date range"""
    
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = calendar_events_api.CalendarEventsApi(api_client)
        
        company_event_request = CompanyEventRequest(
            data=CompanyEventRequestData(
                universe=CompanyEventRequestDataUniverse(
                    symbols=["RY-CA", "TD-CA"],
                    type="Tickers",
                ),
                last_modified_within="Today"
            ),
        )
        
        try:
            print("\nTesting with lastModifiedWithin...")
            api_response = api_instance.get_company_event(company_event_request)
            print("Success!")
            pprint(api_response)
            
        except fds.sdk.EventsandTranscripts.ApiException as e:
            print(f"API Exception: {e}")
            print(f"Status: {e.status}")
            print(f"Reason: {e.reason}")
            print(f"Body: {e.body}")

if __name__ == "__main__":
    print("Testing Calendar Events API...")
    print("=" * 60)
    
    # Test 1: Minimal request
    test_minimal_request()
    
    # Test 2: With bank
    test_with_bank()
    
    # Test 3: With lastModifiedWithin
    test_with_last_modified()