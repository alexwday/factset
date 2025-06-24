import os
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class FactSetAPIKeyClient:
    def __init__(self):
        self.api_key = os.getenv('FACTSET_API_KEY')
        self.base_url = os.getenv('FACTSET_BASE_URL', 'https://api.factset.com')
        
        if not self.api_key:
            raise ValueError("FACTSET_API_KEY not found in environment variables")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Proxy configuration
        http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
        https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
        no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy')
        
        if http_proxy or https_proxy:
            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
            self.session.proxies.update(proxies)
            print(f"Using proxy: {proxies}")
            
        # SSL Certificate setup
        cert_path = Path.home() / "Documents" / "cer" / "rbc-ca-bundle.cer"
        if cert_path.exists():
            self.session.verify = str(cert_path)
            print(f"Using SSL certificate: {cert_path}")
        else:
            print(f"Warning: SSL certificate not found at {cert_path}")
            print("Using default SSL verification")
        
        # Rate limiting: 10 requests per second
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def get(self, endpoint, params=None):
        """Make GET request to FactSet API"""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint, data=None):
        """Make POST request to FactSet API"""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_available_endpoints(self):
        """Get list of available API endpoints"""
        try:
            # This is a common pattern - adjust based on actual FactSet API
            return self.get("/v1/catalog")
        except Exception as e:
            print(f"Error getting endpoints: {e}")
            return None