# 🚀 Stage 1: Daily Transcript Sync - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Daily Financial Transcript Synchronization System
- Stage: 01_download_daily - First stage in earnings transcript processing pipeline
- Stack: Python 3.11+, FactSet SDK, SMB/NAS integration, XML processing
- Architecture: Date-based API querying with NAS storage and title filtering
- Focus: Daily incremental sync of quarterly earnings call transcripts

## 🚨 CRITICAL RULES
- NEVER commit credentials (API keys, NAS passwords, proxy auth)
- ALWAYS validate transcript titles match "Qx 20xx Earnings Call" format exactly
- MUST use date-based queries for efficient daily sync operations
- Handle API rate limiting with exponential backoff
- Maintain audit trails via detailed execution logging

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/stage-01-*, bugfix/stage-01-*, hotfix/stage-01-*
- Commit format: "Stage 1: [description]" preferred
- Focus: Daily sync improvements, API optimization, error handling

### Code Quality Standards
```bash
# Pre-commit checks for Stage 1
python -m black database_refresh/01_download_daily/ --line-length 88
python -m isort database_refresh/01_download_daily/ --profile black
python -m pylint database_refresh/01_download_daily/ --min-score 8.0
python -m flake8 database_refresh/01_download_daily/ --max-line-length 88
```

### Testing Strategy
- Local: Mock FactSet API responses and NAS connections
- Integration: Test with dev environment credentials
- API testing: Validate date-based queries and title parsing
- NAS testing: Directory creation and file operations

## 🔧 ESSENTIAL COMMANDS

### Stage 1 Operations
```bash
# Run daily sync
cd /Users/alexwday/Projects/factset/database_refresh/01_download_daily
python main_daily_sync.py

# Test with different sync ranges (modify config.yaml)
# sync_date_range: 1   # Yesterday + today
# sync_date_range: 7   # Last week + today
# sync_date_range: 30  # Last month + today

# Debug environment setup
python -c "from main_daily_sync import validate_environment_variables; validate_environment_variables()"
```

### Development Tools
```bash
# Check existing transcript inventory
grep -r "transcript_inventory" main_daily_sync.py

# Monitor API calls and responses
grep -r "get_daily_transcripts_by_date" main_daily_sync.py

# Validate XML title parsing
python -c "from main_daily_sync import parse_quarter_and_year_from_xml; print('Q1 2024', parse_quarter_and_year_from_xml(b'<root><meta><title>Q1 2024 Earnings Call</title></meta></root>'))"
```

## 📁 PROJECT STRUCTURE
```
database_refresh/01_download_daily/
├── main_daily_sync.py          # Main execution script
├── CLAUDE.md                   # This file
└── [logs/]                     # Runtime logs (created dynamically)

Key Functions:
├── Authentication & Setup
│   ├── validate_environment_variables()
│   ├── get_nas_connection()
│   ├── setup_ssl_certificate()
│   └── setup_factset_api_client()
├── Core Business Logic
│   ├── calculate_daily_sync_dates()
│   ├── get_daily_transcripts_by_date()
│   ├── compare_transcripts()
│   └── download_transcript_with_title_filtering()
├── Data Management
│   ├── scan_existing_transcripts()
│   ├── create_data_directory_structure()
│   └── parse_quarter_and_year_from_xml()
└── Utility Functions
    ├── nas_path_join()
    ├── nas_create_directory_recursive()
    └── sanitize_url_for_logging()
```

## 🤖 STAGE 1 SPECIFIC CONFIGURATION

### API Integration
- FactSet Events and Transcripts API
- Date-based querying: `get_transcripts_dates(start_date, end_date)`
- Rate limiting: Configurable delays between requests
- Retry logic: Exponential backoff for failed requests
- Authentication: Basic auth with proxy support

### Title Filtering Rules
```python
# EXACT pattern required - no variations accepted
pattern = r"^Q([1-4])\s+(20\d{2})\s+Earnings\s+Call$"

# ✅ ACCEPTED formats:
# "Q1 2024 Earnings Call"
# "Q3 2023 Earnings Call"

# ❌ REJECTED formats:
# "Q1 2024 Earnings Call - Preliminary" 
# "First Quarter 2024"
# "Q1 2024 Conference Call"
```

### Data Processing Pipeline
1. **Date Range Calculation**: Current date minus `sync_date_range` days
2. **API Discovery**: Query each date for all transcripts
3. **Institution Filtering**: Filter to monitored tickers with single primary ID
4. **Earnings Filtering**: Only transcripts with "Earnings" in event_type
5. **Title Validation**: Strict "Qx 20xx Earnings Call" format
6. **Version Management**: Download new versions, remove old ones
7. **Directory Structure**: `{year}/{quarter}/{type}/{ticker_company}/`

## 🔗 INTEGRATIONS

### Required Environment Variables
```bash
# FactSet API Authentication
API_USERNAME=""
API_PASSWORD=""

# Proxy Configuration
PROXY_USER=""
PROXY_PASSWORD=""
PROXY_URL=""
PROXY_DOMAIN="MAPLE"

# NAS Connection
NAS_USERNAME=""
NAS_PASSWORD=""
NAS_SERVER_IP=""
NAS_SERVER_NAME=""
NAS_SHARE_NAME=""
NAS_BASE_PATH=""
NAS_PORT="445"

# Configuration
CONFIG_PATH=""
CLIENT_MACHINE_NAME=""
```

### File Dependencies
- `config.yaml`: Stage configuration (sync_date_range, monitored_institutions)
- SSL certificate: Downloaded from NAS for API authentication
- NAS storage: SMB/CIFS connection for file operations

## 📋 CURRENT FOCUS
- Daily sync optimization for configurable date ranges
- Enhanced error handling and retry logic
- Simplified download logic without title validation
- Performance optimization for institutions with transcripts

## ⚠️ KNOWN ISSUES
- Title parsing requires EXACT format matching
- Rate limiting essential for API stability
- NAS directory creation needs recursive implementation
- SSL certificate handling requires temporary file management

## 🔍 STAGE 1 CODE CONVENTIONS

### Naming Patterns
```python
# Configuration sections
config["stage_01_download_daily"]["sync_date_range"]
config["api_settings"]["retry_delay"]
config["monitored_institutions"][ticker]

# Logging functions
log_console()       # Minimal console output
log_execution()     # Detailed audit trail
log_error()         # Error tracking

# File operations
nas_path_join()     # NAS path construction
nas_create_directory_recursive()  # Directory creation
download_transcript_with_title_filtering()  # Main download logic
```

### Error Handling
- API failures: Retry with exponential backoff
- Title validation: Skip non-conforming transcripts
- NAS operations: Graceful degradation with logging
- Authentication: Clear error messages for credential issues

### Data Structures
```python
# Transcript inventory format
transcript_record = {
    "fiscal_year": "2024",
    "quarter": "Q1", 
    "company_type": "Bank",
    "company": "JPM_JPMorgan_Chase_Co",
    "ticker": "JPM",
    "event_id": "12345",
    "version_id": "1",
    "filename": "JPM_Q1_2024_Script_12345_1.xml"
}

# API transcript format  
api_transcript = {
    "event_id": "12345",
    "version_id": "1",
    "transcript_type": "Script",
    "event_date": "2024-01-15",
    "transcripts_link": "https://api.url/transcript"
}
```

## 📊 DATA HANDLING
- **Input**: Date ranges for API queries, existing NAS inventory
- **Processing**: Title validation, version comparison, directory management
- **Output**: Downloaded XML transcripts in organized directory structure
- **Logging**: Comprehensive execution logs and error tracking to NAS
- **Security**: Credential sanitization in logs, secure NAS operations

## 🚫 DO NOT MODIFY
- Environment variable validation logic
- SSL certificate download and setup procedures  
- API authentication token generation
- NAS connection establishment patterns
- Title parsing regex pattern (exact format required)

## 💡 DEVELOPMENT NOTES
- Use `sync_date_range` in config.yaml to control how many days back to sync
- Date-based API queries are more efficient than institution-based queries
- Title filtering prevents processing of non-earnings transcripts
- Only institutions with transcripts are processed (performance optimization)
- Version management ensures latest transcript versions are maintained
- Daily sync is additive - doesn't remove files outside date range