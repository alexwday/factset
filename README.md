# FactSet API Test Project

This project provides examples for connecting to FactSet API using both API Key and OAuth2 authentication methods.

## Setup

1. **Clone and create virtual environment**:
   ```bash
   git clone https://github.com/alexwday/factset.git
   cd factset
   
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it (Mac/Linux)
   source venv/bin/activate
   
   # Or on Windows
   # venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```
   
   Or use the setup script:
   ```bash
   ./setup.sh
   ```

2. **Configure credentials**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your FactSet credentials.

## Authentication Methods

### API Key (Simpler)
- Generate an API key from FactSet Developer Portal
- Add to `.env`: `FACTSET_API_KEY=your_key_here`
- Best for: Quick testing, development

### OAuth2 (More Secure)
- Register your application in FactSet Developer Portal
- Get Client ID and Client Secret
- Add to `.env`:
  ```
  FACTSET_CLIENT_ID=your_client_id
  FACTSET_CLIENT_SECRET=your_client_secret
  ```
- Best for: Production use, better security

## Usage

### Explore Available APIs
```bash
python examples/get_reports.py
```

### Fetch Specific Reports
```bash
python examples/simple_report_example.py
```

## Project Structure
```
factset-api-test/
├── src/
│   ├── api_key_auth.py      # API Key authentication
│   └── oauth2_auth.py        # OAuth2 authentication
├── examples/
│   ├── get_reports.py        # Discover available endpoints
│   └── simple_report_example.py  # Fetch specific data
├── config/                   # Configuration files
├── .env.example             # Environment template
└── requirements.txt         # Python dependencies
```

## Next Steps

1. **Get credentials**: Log into FactSet Developer Portal and create either an API key or OAuth2 app
2. **Check API documentation**: Visit https://developer.factset.com/api-catalog to see available endpoints
3. **Run examples**: Start with `get_reports.py` to discover what's available
4. **Customize**: Modify the examples for your specific use case

## Common Report Types

FactSet typically provides:
- Fundamentals (financial statements)
- Estimates (analyst consensus)
- Prices (historical/real-time)
- Analytics (ratios, calculations)
- ESG data
- Supply chain data
- News and events

Check the API catalog for specific endpoints and required parameters.