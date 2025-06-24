#!/usr/bin/env python3
"""
Simple example to fetch a specific report type once you know the endpoint
"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api_key_auth import FactSetAPIKeyClient
import pandas as pd

def get_company_fundamentals(ticker):
    """Example: Get fundamental data for a company"""
    client = FactSetAPIKeyClient()
    
    # This is an example - actual endpoint may differ
    # Check FactSet API docs for exact endpoints
    endpoint = f"/v1/fundamentals/factset-fundamentals/metrics"
    
    # Example request body - adjust based on API docs
    request_body = {
        "ids": [ticker],  # e.g., ["AAPL-US", "MSFT-US"]
        "metrics": [
            "SALES",
            "TOTAL_REVENUE", 
            "NET_INCOME",
            "TOTAL_ASSETS",
            "TOTAL_EQUITY"
        ],
        "periodicity": "ANN",  # Annual
        "fiscalPeriodStart": "2020-01-01",
        "fiscalPeriodEnd": "2023-12-31"
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        return response
    except Exception as e:
        print(f"Error fetching fundamentals: {e}")
        return None

def get_estimates_data(ticker):
    """Example: Get consensus estimates for a company"""
    client = FactSetAPIKeyClient()
    
    endpoint = "/v1/estimates/consensus"
    
    request_body = {
        "ids": [ticker],
        "metrics": ["SALES_ESTIMATE", "EPS_ESTIMATE"],
        "periodicity": "ANN",
        "fiscalPeriod": 1  # Next fiscal year
    }
    
    try:
        response = client.post(endpoint, data=request_body)
        return response
    except Exception as e:
        print(f"Error fetching estimates: {e}")
        return None

def save_to_excel(data, filename):
    """Save data to Excel file"""
    if isinstance(data, dict):
        # Convert to DataFrame
        df = pd.DataFrame(data.get('data', []))
        df.to_excel(f"output/{filename}.xlsx", index=False)
        print(f"Data saved to output/{filename}.xlsx")
    else:
        print("Invalid data format")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Example tickers - use FactSet symbology
    tickers = ["AAPL-US", "MSFT-US"]
    
    print("Fetching company fundamentals...")
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Get fundamentals
        fundamentals = get_company_fundamentals(ticker)
        if fundamentals:
            print("Fundamentals retrieved successfully")
            # Pretty print JSON
            print(json.dumps(fundamentals, indent=2)[:500] + "...")  # First 500 chars
            
        # Get estimates
        estimates = get_estimates_data(ticker)
        if estimates:
            print("Estimates retrieved successfully")