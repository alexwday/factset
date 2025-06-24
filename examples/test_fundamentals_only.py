#!/usr/bin/env python3
"""
Test script specifically for factset-fundamentals/v2 API
Since only this server is authorized
"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_key_auth import FactSetAPIKeyClient

def test_fundamentals_endpoints():
    """Test various endpoints under factset-fundamentals/v2"""
    client = FactSetAPIKeyClient()
    
    # Test different fundamentals endpoints
    endpoints = [
        {
            "path": "/content/factset-fundamentals/v2/income-statement",
            "body": {
                "ids": ["AAPL-US"],
                "periodicity": "ANN",
                "fiscalPeriodStart": "2022-01-01",
                "fiscalPeriodEnd": "2023-12-31"
            }
        },
        {
            "path": "/content/factset-fundamentals/v2/balance-sheet",
            "body": {
                "ids": ["AAPL-US"],
                "periodicity": "ANN",
                "fiscalPeriodStart": "2022-01-01",
                "fiscalPeriodEnd": "2023-12-31"
            }
        },
        {
            "path": "/content/factset-fundamentals/v2/cash-flow-statement",
            "body": {
                "ids": ["AAPL-US"],
                "periodicity": "ANN",
                "fiscalPeriodStart": "2022-01-01",
                "fiscalPeriodEnd": "2023-12-31"
            }
        }
    ]
    
    for ep in endpoints:
        print(f"\nTesting: {ep['path']}")
        print(f"Request body: {json.dumps(ep['body'], indent=2)}")
        
        try:
            response = client.post(ep['path'], data=ep['body'])
            print("Success! Response:")
            print(json.dumps(response, indent=2)[:500] + "...")
        except Exception as e:
            print(f"Error: {e}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text[:500]}")

if __name__ == "__main__":
    print("Testing FactSet Fundamentals v2 API")
    print("=" * 50)
    test_fundamentals_endpoints()