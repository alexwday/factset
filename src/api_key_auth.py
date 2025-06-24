import os
import requests
from dotenv import load_dotenv

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
    
    def get(self, endpoint, params=None):
        """Make GET request to FactSet API"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint, data=None):
        """Make POST request to FactSet API"""
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