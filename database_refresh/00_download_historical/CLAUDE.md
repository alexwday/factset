# 🚀 Stage 00: Historical Transcript Sync - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Historical Financial Transcript Download and Synchronization System
- Stack: Python 3.11+, FactSet SDK, SMB/NAS Integration, XML Processing
- Architecture: API-driven download system with NAS storage and 3-year rolling window
- Focus: Earnings transcript acquisition, validation, and file management

## 🚨 CRITICAL RULES
- NEVER expose API credentials or proxy passwords in logs
- ALWAYS validate SSL certificates for API connections
- MUST handle API rate limits and implement exponential backoff
- Monitor NAS storage capacity and file organization
- Downloads ALL transcripts (title filtering moved to Stage 02)

## 🛠️ DEVELOPMENT WORKFLOW

### Environment Requirements
```bash
# Required environment variables
API_USERNAME=<factset_api_username>
API_PASSWORD=<factset_api_password>
PROXY_USER=<corporate_proxy_user>
PROXY_PASSWORD=<corporate_proxy_password>
PROXY_URL=<proxy_server_url>
NAS_USERNAME=<nas_server_username>
NAS_PASSWORD=<nas_server_password>
NAS_SERVER_IP=<nas_server_ip>
NAS_SERVER_NAME=<nas_server_name>
NAS_SHARE_NAME=<nas_share_name>
NAS_BASE_PATH=<nas_base_path>
NAS_PORT=445
CONFIG_PATH=<config_yaml_path_on_nas>
CLIENT_MACHINE_NAME=<local_machine_name>
```

### Git Integration
- Branch pattern: feature/stage-00-*, bugfix/stage-00-*, hotfix/stage-00-*
- Commit format: "Stage 00: [description]"
- Focus: Historical transcript download and synchronization features

### Code Quality Standards
```bash
# Pre-commit checks specific to Stage 00
python -m black database_refresh/00_download_historical/ --line-length 88
python -m isort database_refresh/00_download_historical/ --profile black
python -m pylint database_refresh/00_download_historical/ --min-score 8.0
```

## 🤖 STAGE 00 CONFIGURATION

### Core Functionality
- **Purpose**: Download ALL transcripts from FactSet API for monitored institutions
- **Scope**: 3-year rolling window from current date (or fixed start year if configured)
- **Storage**: Organized NAS directory structure by year/quarter/company_type/company
- **Validation**: No title filtering - all transcripts are downloaded for archival

### Key Components
- **Authentication**: FactSet API credentials with corporate proxy support
- **SSL/TLS**: Certificate-based authentication for secure API calls
- **Rate Limiting**: Configurable delays and exponential backoff retry logic
- **File Management**: NAS integration with SMB/CIFS protocol

## 📁 PROJECT STRUCTURE
```
database_refresh/00_download_historical/
  main_historical_sync.py    # Main execution script
config.yaml (on NAS)         # Configuration file with API settings
SSL certificates (on NAS)    # Required for API authentication

NAS Output Structure:
/Outputs/Data/
  2024/                      # Fiscal year
    Q1/                      # Quarter
      Financial_Services/    # Company type
        JPM_JPMorgan_Chase/  # Ticker_Company structure
          JPM_Q1_2024_E1_12345_67890.xml  # Transcript files
```

## 🔧 ESSENTIAL COMMANDS

### Stage 00 Operations
```bash
# Run historical transcript sync
cd database_refresh/00_download_historical
python main_historical_sync.py

# Environment setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt  # Install dependencies

# Testing configuration
python -c "from main_historical_sync import validate_environment_variables; validate_environment_variables()"
```

### Debugging Commands
```bash
# Test NAS connection
python -c "from main_historical_sync import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Validate SSL setup
python -c "from main_historical_sync import setup_ssl_certificate, get_nas_connection; setup_ssl_certificate(get_nas_connection())"

# Check configuration
python -c "from main_historical_sync import load_config_from_nas, get_nas_connection; config = load_config_from_nas(get_nas_connection()); print(f'Institutions: {len(config[\"monitored_institutions\"])}')"
```

## 🔗 INTEGRATIONS

### FactSet API Integration
- **SDK**: fds.sdk.EventsandTranscripts
- **Authentication**: HTTP Basic Auth with proxy support
- **Endpoints**: Transcripts API for earnings call data
- **Rate Limits**: Configurable delays between requests

### NAS Storage Integration
- **Protocol**: SMB/CIFS
- **Operations**: Upload, download, directory creation, file removal
- **Structure**: Hierarchical organization by fiscal year and quarter

### Required Dependencies
```python
import yaml                    # Configuration file parsing
import fds.sdk.EventsandTranscripts  # FactSet API SDK
from smb.SMBConnection import SMBConnection  # NAS connectivity
from dotenv import load_dotenv # Environment variable management
import requests               # HTTP requests for transcript downloads
import xml.etree.ElementTree  # XML parsing and validation
```

## 📋 CURRENT FOCUS
- Historical transcript synchronization with 3-year rolling window
- API authentication and proxy configuration
- NAS storage organization and file management
- XML transcript validation and title filtering

## ⚠️ KNOWN ISSUES
- **Proxy Authentication**: Complex NTLM domain authentication required
- **SSL Certificates**: Must be downloaded from NAS for each execution
- **Rate Limiting**: API may throttle requests during high-volume periods
- **Title Validation**: Strict format requirement may reject valid transcripts with slight variations

## 🚫 DO NOT MODIFY
- Environment variable names (hardcoded in multiple functions)
- File naming convention: ticker_quarter_year_transcripttype_eventid_versionid.xml
- NAS directory structure pattern
- XML title validation regex pattern

## 💡 DEVELOPMENT NOTES

### Core Functions (main_historical_sync.py)
- `main()` - Main execution flow with error handling
- `validate_environment_variables()` - Checks all required env vars
- `get_nas_connection()` - Establishes SMB connection
- `setup_ssl_certificate()` - Downloads and configures SSL cert
- `setup_factset_api_client()` - Configures API client with auth
- `scan_existing_transcripts()` - Inventories current NAS files
- `get_api_transcripts_for_company()` - Queries API for company transcripts
- `download_transcript()` - Downloads transcripts without title rejection
- `parse_quarter_and_year_from_xml()` - Extracts quarter/year from XML title for file organization
- `compare_transcripts()` - Determines sync actions needed

### Logging System
- **Execution Log**: Detailed operational logging for audit trails
- **Error Log**: Specific error tracking with context
- **Console Log**: Minimal user-facing status messages

### Data Flow
1. Environment validation and NAS connection
2. Configuration and SSL certificate setup
3. FactSet API client initialization
4. Existing transcript inventory scan
5. 3-year rolling window calculation
6. Per-institution API querying and comparison
7. Download new/updated transcripts
8. Remove out-of-window transcripts
9. Log generation and cleanup

## 🔍 CODE CONVENTIONS
- Use `global logger, config` for shared state
- Prefix NAS utility functions with `nas_`
- Implement retry logic with exponential backoff
- Sanitize URLs before logging to remove credentials
- Use descriptive variable names for financial concepts (ticker, quarter, fiscal_year)

## 📊 DATA HANDLING
- **Security**: Credentials never logged, URLs sanitized
- **Validation**: XML structure and title format verification
- **Error Handling**: Comprehensive logging with contextual details
- **File Organization**: Consistent directory structure for data retrieval
- **Audit Trail**: Complete execution logging for compliance

## 🔄 PROCESSING LOGIC
- **3-Year Window**: Dynamic calculation from current date (or fixed start year)
- **Contamination Prevention**: Filter transcripts to single primary ID
- **Earnings Focus**: Only download transcripts with "Earnings" event type
- **Version Management**: API version always considered authoritative
- **No Title Filtering**: ALL transcripts downloaded (filtering moved to Stage 02)