#!/usr/bin/env python3
"""
FactSet v2 API Examples - Common use cases
"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_key_auth import FactSetAPIKeyClient
from datetime import datetime, timedelta

def get_price_history(ticker, days=30):
    """Get historical prices for a security"""
    client = FactSetAPIKeyClient()
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    endpoint = "/content/factset-prices/v2/prices"
    request_body = {
        "ids": [ticker],
        "startDate": start_date,
        "endDate": end_date,
        "frequency": "D",  # Daily
        "fields": ["price", "volume", "high", "low", "open"]
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        print(f"Price history for {ticker}:")
        print(json.dumps(response, indent=2)[:1000] + "...")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_company_overview(ticker):
    """Get company overview information"""
    client = FactSetAPIKeyClient()
    
    endpoint = "/content/factset-fundamentals/v2/company-overview"
    request_body = {
        "ids": [ticker],
        "fields": ["companyName", "businessDescription", "industryGroup", "sector", "employees", "marketCap"]
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        print(f"\nCompany overview for {ticker}:")
        print(json.dumps(response, indent=2))
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_balance_sheet(ticker):
    """Get balance sheet data"""
    client = FactSetAPIKeyClient()
    
    endpoint = "/content/factset-fundamentals/v2/balance-sheet"
    request_body = {
        "ids": [ticker],
        "periodicity": "ANN",
        "fiscalPeriodStart": "2022-01-01",
        "fiscalPeriodEnd": "2023-12-31",
        "fields": ["totalAssets", "totalLiabilities", "totalEquity", "cash", "debt"]
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        print(f"\nBalance sheet for {ticker}:")
        print(json.dumps(response, indent=2))
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def search_companies(search_term):
    """Search for companies by name"""
    client = FactSetAPIKeyClient()
    
    endpoint = "/content/factset-concordance/v2/search"
    request_body = {
        "searchText": search_term,
        "includePrivate": False,
        "limit": 10
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        print(f"\nSearch results for '{search_term}':")
        print(json.dumps(response, indent=2))
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("FactSet v2 API Examples")
    print("=" * 50)
    
    # Example usage - adjust ticker symbols as needed
    ticker = "AAPL-US"
    
    # Get price history
    get_price_history(ticker, days=30)
    
    # Get company overview
    get_company_overview(ticker)
    
    # Get balance sheet
    get_balance_sheet(ticker)
    
    # Search for companies
    search_companies("Apple")