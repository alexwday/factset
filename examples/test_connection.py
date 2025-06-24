#!/usr/bin/env python3
"""
Test connection with proxy and SSL certificate
"""
import sys
import os
import requests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

def test_basic_connection():
    """Test basic HTTPS connection with cert and proxy"""
    print("Testing basic connection to FactSet API...")
    print("=" * 50)
    
    # SSL Certificate
    cert_path = Path.home() / "Documents" / "cer" / "rbc-ca-bundle.cer"
    print(f"SSL Certificate: {cert_path}")
    print(f"Certificate exists: {cert_path.exists()}")
    
    # Proxy settings
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    
    print(f"\nProxy settings:")
    print(f"HTTP_PROXY: {http_proxy}")
    print(f"HTTPS_PROXY: {https_proxy}")
    
    # Test connection
    url = "https://api.factset.com/content/factset-fundamentals/v2"
    
    session = requests.Session()
    
    # Set proxy if available
    if http_proxy or https_proxy:
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy
        session.proxies.update(proxies)
    
    # Set SSL cert
    if cert_path.exists():
        session.verify = str(cert_path)
    
    # Add API key if available
    api_key = os.getenv('FACTSET_API_KEY')
    if api_key:
        session.headers['Authorization'] = f'Bearer {api_key}'
    
    print(f"\nTesting connection to: {url}")
    try:
        response = session.get(url)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        if response.status_code != 200:
            print(f"Response: {response.text[:500]}")
    except requests.exceptions.ProxyError as e:
        print(f"Proxy Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if you're connected to VPN")
        print("2. Verify proxy settings in .env file")
        print("3. Try setting both HTTP_PROXY and HTTPS_PROXY")
        print("4. Common proxy format: http://proxy.company.com:8080")
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("\nTroubleshooting:")
        print("1. Verify certificate path is correct")
        print("2. Check if certificate is valid")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    test_basic_connection()