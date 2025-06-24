import os
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class FactSetOAuth2Client:
    def __init__(self):
        self.client_id = os.getenv('FACTSET_CLIENT_ID')
        self.client_secret = os.getenv('FACTSET_CLIENT_SECRET')
        self.base_url = os.getenv('FACTSET_BASE_URL', 'https://api.factset.com')
        
        if not self.client_id or not self.client_secret:
            raise ValueError("FACTSET_CLIENT_ID and FACTSET_CLIENT_SECRET required")
        
        self.token_url = f"{self.base_url}/oauth/token"
        self.access_token = None
        self.token_expires_at = 0
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # SSL Certificate setup
        cert_path = Path.home() / "Documents" / "cer" / "rbc-ca-bundle.cer"
        if cert_path.exists():
            self.session.verify = str(cert_path)
            print(f"Using SSL certificate: {cert_path}")
        else:
            print(f"Warning: SSL certificate not found at {cert_path}")
            print("Using default SSL verification")
    
    def _get_access_token(self):
        """Get access token using client credentials flow"""
        if self.access_token and time.time() < self.token_expires_at:
            return self.access_token
        
        # Request new token
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'scope': 'read'  # Adjust scope as needed
        }
        
        # Use same SSL cert for token request
        cert_path = Path.home() / "Documents" / "cer" / "rbc-ca-bundle.cer"
        verify = str(cert_path) if cert_path.exists() else True
        response = requests.post(self.token_url, data=data, verify=verify)
        response.raise_for_status()
        
        token_data = response.json()
        self.access_token = token_data['access_token']
        # Set expiry with 5 minute buffer
        self.token_expires_at = time.time() + token_data.get('expires_in', 3600) - 300
        
        return self.access_token
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make authenticated request"""
        token = self._get_access_token()
        url = f"{self.base_url}{endpoint}"
        
        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {token}'
        
        response = self.session.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get(self, endpoint, params=None):
        """Make GET request"""
        return self._make_request('GET', endpoint, params=params)
    
    def post(self, endpoint, data=None):
        """Make POST request"""
        return self._make_request('POST', endpoint, json=data)