# FactSet Earnings Transcript Pipeline

A multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API.

## Project Structure

```
factset/
├── .env                                    # Shared authentication (create from .env.example)
├── .env.example                           # Template for environment variables
├── 0_transcript_bulk_sync_working.py      # Working version of bulk sync script
├── docs/                                  # Documentation & FactSet SDK docs
├── stage_0_bulk_refresh/                  # Historical bulk download (optional)
├── stage_1_daily_sync/                    # Daily incremental sync (scheduled)
├── stage_2_processing/                    # Transcript processing & analysis
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Quick Setup

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd factset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Authentication

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
- `API_USERNAME` - Your FactSet API username
- `API_PASSWORD` - Your FactSet API password  
- `PROXY_USER` - Corporate proxy username
- `PROXY_PASSWORD` - Corporate proxy password
- `NAS_USERNAME` - NAS server username
- `NAS_PASSWORD` - NAS server password
- `NAS_SERVER_IP` - NAS server IP address

### 3. Run Scripts

Each stage is a standalone Python script:

```bash
# Stage 0: Bulk historical sync (optional)
python stage_0_bulk_refresh/0_transcript_bulk_sync.py

# Stage 1: Daily incremental sync
python stage_1_daily_sync/1_transcript_daily_sync.py

# Stage 2: Processing & analysis
python stage_2_processing/2_transcript_processing.py
```

## Pipeline Stages

### Stage 0: Bulk Refresh (Optional)
- **Purpose**: Download ALL historical transcripts from 2023-present
- **When to use**: Initial setup or complete repository refresh
- **Output**: Complete transcript repository on NAS

### Stage 1: Daily Sync (Scheduled)
- **Purpose**: Check for new transcripts daily and download incrementally
- **When to use**: Regular scheduled operations
- **Output**: New transcripts added to existing repository

### Stage 2: Processing (Future)
- **Purpose**: Process, enrich, and analyze downloaded transcripts
- **When to use**: After transcripts are downloaded
- **Output**: Processed data and analysis results

## Configuration

### Authentication (.env file)
- Contains only credentials and connection details
- Shared across all stages
- Never committed to git (in .gitignore)

### Operational Settings (NAS config files)
- Stored on NAS in `Inputs/config/` folder
- Stage-specific: `stage_0_config.json`, `stage_1_config.json`, etc.
- Contains monitored institutions, API settings, processing parameters

## Monitored Institutions

### Canadian Banks
- Royal Bank of Canada (RY-CA)
- Toronto-Dominion Bank (TD-CA)
- Bank of Nova Scotia (BNS-CA)
- Bank of Montreal (BMO-CA)
- Canadian Imperial Bank of Commerce (CM-CA)
- National Bank of Canada (NA-CA)

### US Banks
- JPMorgan Chase & Co. (JPM-US)
- Bank of America Corporation (BAC-US)
- Wells Fargo & Company (WFC-US)
- Citigroup Inc. (C-US)
- Goldman Sachs Group Inc. (GS-US)
- Morgan Stanley (MS-US)

### Insurance Companies
- Manulife Financial Corporation (MFC-CA)
- Sun Life Financial Inc. (SLF-CA)
- UnitedHealth Group Incorporated (UNH-US)

## Troubleshooting

### Common Issues

1. **Environment Variables Not Found**
   - Ensure .env file exists and contains all required variables
   - Check file is in correct location (project root)

2. **NAS Connection Failed**
   - Verify NAS credentials and IP address
   - Check network connectivity
   - Ensure NAS is accessible from your location

3. **API Authentication Failed**
   - Verify FactSet API credentials
   - Check if API access is enabled for your account
   - Ensure proxy settings are correct if behind corporate firewall

4. **Import Errors**
   - Activate virtual environment: `source venv/bin/activate`
   - Install requirements: `pip install -r requirements.txt`

### Logs and Monitoring

- Scripts generate detailed logs during execution
- Failed downloads are tracked and reported
- Check console output for real-time status
- Log files stored on NAS for historical analysis

## Development

### Adding New Stages
1. Create new stage directory: `stage_X_<purpose>/`
2. Create standalone script: `X_<descriptive_name>.py`
3. Copy shared functions from existing scripts
4. Create stage-specific config file on NAS
5. Update this README

### Script Requirements
- Each script must be completely standalone
- No imports between stage scripts
- Shared functions are copy-pasted (not imported)
- Scripts work from terminal or any Python environment
- All scripts use same .env file for authentication

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify configuration files are correct
3. Test network connectivity to NAS and FactSet API
4. Review this README for common solutions