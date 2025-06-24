"""
Minimal test for FactSet API - paste into notebook cell
"""

import requests
import json

# PASTE YOUR API KEY HERE
API_KEY = "your_api_key_here"

# Simple test
url = "https://api.factset.com/content/factset-fundamentals/v2/income-statement"
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}
body = {
    "ids": ["AAPL-US"],
    "periodicity": "ANN",
    "fiscalPeriodStart": "2023-01-01",
    "fiscalPeriodEnd": "2023-12-31"
}

response = requests.post(url, json=body, headers=headers)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2)[:1000])
else:
    print(f"Error: {response.text[:500]}")