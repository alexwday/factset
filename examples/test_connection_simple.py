"""
Simple connection test to diagnose network issues
"""

import requests
import socket

def test_basic_connectivity():
    """Test basic network connectivity to FactSet"""
    
    print("Testing basic connectivity...")
    
    # Test 1: DNS resolution
    try:
        ip = socket.gethostbyname('api.factset.com')
        print(f"✓ DNS resolution: api.factset.com -> {ip}")
    except Exception as e:
        print(f"✗ DNS resolution failed: {e}")
        return
    
    # Test 2: TCP connection
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex(('api.factset.com', 443))
        sock.close()
        if result == 0:
            print("✓ TCP connection to port 443 successful")
        else:
            print(f"✗ TCP connection failed: {result}")
            return
    except Exception as e:
        print(f"✗ TCP connection error: {e}")
        return
    
    # Test 3: HTTPS without SSL verification
    try:
        print("Testing HTTPS connection (no SSL verify)...")
        response = requests.get('https://api.factset.com', verify=False, timeout=10)
        print(f"✓ HTTPS connection successful: {response.status_code}")
    except Exception as e:
        print(f"✗ HTTPS connection failed: {e}")
        print("This suggests network/proxy issues")

if __name__ == "__main__":
    test_basic_connectivity()