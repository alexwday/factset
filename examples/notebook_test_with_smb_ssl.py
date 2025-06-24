"""
FactSet API test with SSL certificate from SMB share
"""

import requests
import json
import tempfile
import os
from smb.SMBConnection import SMBConnection

# SMB CONFIGURATION - UPDATE THESE
SMB_USER = "your_username"
SMB_PASS = "your_password"
SMB_SERVER_IP = "10.0.0.1"  # IP address of the file server
SMB_SERVER_NAME = "FILESERVER"  # NetBIOS name of the server
SMB_SHARE_NAME = "shared"  # Name of the share
SMB_CERT_PATH = "/path/to/rbc-ca-bundle.cer"  # Path within the share

# API CONFIGURATION
API_KEY = "your_api_key_here"

def download_cert_from_smb():
    """Download SSL certificate from SMB share"""
    print("Connecting to SMB share...")
    
    # Create SMB connection
    conn = SMBConnection(SMB_USER, SMB_PASS, "python-client", SMB_SERVER_NAME, use_ntlm_v2=True)
    
    try:
        # Connect to server
        conn.connect(SMB_SERVER_IP, 445)
        print("Connected to SMB server")
        
        # Create temp file for certificate
        temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
        temp_cert_path = temp_cert.name
        
        # Download certificate
        print(f"Downloading certificate from {SMB_SHARE_NAME}{SMB_CERT_PATH}")
        conn.retrieveFile(SMB_SHARE_NAME, SMB_CERT_PATH, temp_cert)
        temp_cert.close()
        
        print(f"Certificate downloaded to: {temp_cert_path}")
        return temp_cert_path
        
    except Exception as e:
        print(f"SMB Error: {e}")
        return None
    finally:
        conn.close()

def test_factset_with_ssl():
    """Test FactSet API with SSL certificate"""
    
    # Download certificate
    cert_path = download_cert_from_smb()
    if not cert_path:
        print("Failed to download certificate")
        return
    
    try:
        # Test API call
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
        
        print("\nTesting FactSet API with SSL certificate...")
        response = requests.post(url, json=body, headers=headers, verify=cert_path)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("Success!")
            print(json.dumps(response.json(), indent=2)[:1000])
        else:
            print(f"Error: {response.text[:500]}")
            
    except requests.exceptions.SSLError as e:
        print(f"SSL Error: {e}")
        print("\nTrying with SSL verification disabled (NOT RECOMMENDED for production)...")
        response = requests.post(url, json=body, headers=headers, verify=False)
        print(f"Status without SSL verify: {response.status_code}")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        
    finally:
        # Clean up temp certificate
        if cert_path and os.path.exists(cert_path):
            os.unlink(cert_path)
            print(f"\nCleaned up temporary certificate")

# Run the test
if __name__ == "__main__":
    if API_KEY == "your_api_key_here":
        print("ERROR: Please update the configuration variables!")
    else:
        test_factset_with_ssl()