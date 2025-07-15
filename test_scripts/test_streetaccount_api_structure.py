#!/usr/bin/env python3
"""
Test script to verify StreetAccountNews API structure and method availability
"""

import fds.sdk.StreetAccountNews as streetaccount

print("Testing StreetAccountNews API structure...")

# Test that the main classes exist
try:
    from fds.sdk.StreetAccountNews.api import filters_api
    print("✅ filters_api module available")
    
    # Check FiltersApi methods
    api_methods = dir(filters_api.FiltersApi)
    expected_methods = [
        'get_street_account_filters_categories',
        'get_street_account_filters_sectors', 
        'get_street_account_filters_regions',
        'get_street_account_filters_topics'
    ]
    
    for method in expected_methods:
        if method in api_methods:
            print(f"✅ {method} - Available")
        else:
            print(f"❌ {method} - Not found")
    
except ImportError as e:
    print(f"❌ filters_api import failed: {e}")

try:
    from fds.sdk.StreetAccountNews.api import headlines_api
    print("✅ headlines_api module available")
    
    # Check HeadlinesApi methods
    api_methods = dir(headlines_api.HeadlinesApi)
    expected_methods = [
        'get_street_account_headlines',
        'get_street_account_headlines_by_view'
    ]
    
    for method in expected_methods:
        if method in api_methods:
            print(f"✅ {method} - Available")
        else:
            print(f"❌ {method} - Not found")
    
except ImportError as e:
    print(f"❌ headlines_api import failed: {e}")

# Test model imports
try:
    from fds.sdk.StreetAccountNews.model.headlines_request import HeadlinesRequest
    from fds.sdk.StreetAccountNews.model.headlines_request_data import HeadlinesRequestData
    from fds.sdk.StreetAccountNews.model.headlines_request_meta import HeadlinesRequestMeta
    print("✅ HeadlinesRequest models available")
except ImportError as e:
    print(f"❌ HeadlinesRequest models import failed: {e}")

print("\n✅ StreetAccountNews API structure test complete!")