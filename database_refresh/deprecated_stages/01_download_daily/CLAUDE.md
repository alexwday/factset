# 🚀 Stage 1: Daily Transcript Sync with Invalid Tracking - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Daily Financial Transcript Synchronization with Title Validation
- Stage: 01_download_daily - Daily incremental update for earnings transcript pipeline
- Stack: Python 3.11+, FactSet SDK, SMB/NAS integration, XML processing, Pandas/Excel
- Architecture: Date-based API querying with invalid transcript tracking via Excel
- Focus: Daily incremental sync with strict "Qx 20xx Earnings Call" title validation
- Main Script: `main_daily_sync_with_ignore.py` (1802 lines)

## 🚨 CRITICAL RULES
- NEVER commit credentials (API keys, NAS passwords, proxy auth)
- ALWAYS validate transcript titles match "Qx 20xx Earnings Call" format exactly
- MUST use date-based queries for efficient daily sync operations
- Handle API rate limiting with exponential backoff
- Maintain audit trails via detailed execution logging
- Check invalid list BEFORE downloading to avoid re-downloading rejected transcripts
- Save invalid list incrementally after processing each date range

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
# Run daily sync with invalid tracking
cd /Users/alexwday/Projects/factset/database_refresh/01_download_daily
python main_daily_sync_with_ignore.py

# Test with different sync ranges (modify config.yaml)
# sync_date_range: 0   # Today only
# sync_date_range: 1   # Yesterday + today (2 days total)
# sync_date_range: 7   # Last 7 days + today (8 days total)
# sync_date_range: 30  # Last 30 days + today (31 days total)

# Debug environment setup
python -c "from main_daily_sync_with_ignore import validate_environment_variables; validate_environment_variables()"
```

### Development Tools
```bash
# Check existing transcript inventory (line 926-1043)
grep -r "scan_existing_transcripts" main_daily_sync_with_ignore.py

# Monitor API calls and responses (line 1081-1158)
grep -r "get_daily_transcripts_by_date" main_daily_sync_with_ignore.py

# Check invalid transcript list (line 803-834)
python -c "from main_daily_sync_with_ignore import load_invalid_transcript_list, get_nas_connection; df = load_invalid_transcript_list(get_nas_connection()); print(f'Invalid transcripts: {len(df)}')"

# Validate XML title parsing (line 716-779)
python -c "from main_daily_sync_with_ignore import parse_quarter_and_year_from_xml; print('Q1 2024', parse_quarter_and_year_from_xml(b'<root><meta><title>Q1 2024 Earnings Call</title></meta></root>'))"
```

## 📁 PROJECT STRUCTURE
```
database_refresh/01_download_daily/
├── main_daily_sync_with_ignore.py  # Main execution script with invalid tracking
├── CLAUDE.md                       # This file
└── [logs/]                         # Runtime logs (created dynamically)

NAS Output Structure:
/Outputs/Data/
  InvalidTranscripts/               # Invalid transcript tracking
    invalid_transcripts.xlsx        # Excel list of rejected transcripts

Key Functions:
├── Authentication & Setup
│   ├── validate_environment_variables() [line 137-177]
│   ├── get_nas_connection() [line 184-223]
│   ├── setup_ssl_certificate() [line 417-448]
│   └── setup_factset_api_client() [line 475-500]
├── Core Business Logic
│   ├── calculate_daily_sync_dates() [line 1045-1079]
│   ├── get_daily_transcripts_by_date() [line 1081-1158]
│   ├── compare_transcripts_with_invalid_list() [line 1184-1281]
│   └── download_transcript_with_validation() [line 1283-1479]
├── Invalid List Management
│   ├── load_invalid_transcript_list() [line 803-834]
│   ├── save_invalid_transcript_list() [line 837-874]
│   ├── add_to_invalid_list() [line 876-894]
│   └── is_transcript_in_invalid_list() [line 896-902]
├── Data Management
│   ├── scan_existing_transcripts() [line 926-1043]
│   ├── create_data_directory_structure() [line 907-924]
│   └── parse_quarter_and_year_from_xml() [line 716-779]
└── Utility Functions
    ├── nas_path_join() [line 517-519]
    ├── nas_create_directory_recursive() [line 566-624]
    └── sanitize_url_for_logging() [line 787-798]
```

## 🤖 STAGE 1 SPECIFIC CONFIGURATION

### API Integration
- FactSet Events and Transcripts API [line 1090-1095]
- Date-based querying: `get_transcripts_dates(start_date, end_date, sort, pagination_limit)` [line 1090-1095]
- Rate limiting: Configurable delays between requests via `config['api_settings']['request_delay']` [line 1614, 1679]
- Retry logic: Exponential backoff for failed requests with `use_exponential_backoff` config [line 1142-1150]
- Authentication: Basic auth with proxy support [line 484-492]

### Title Filtering Rules
```python
# EXACT pattern required - no variations accepted (line 781-784)
def is_valid_earnings_call_title(title: str) -> bool:
    pattern = r"^Q([1-4])\s+(20\d{2})\s+Earnings\s+Call$"
    return bool(re.match(pattern, title, re.IGNORECASE))

# ✅ ACCEPTED formats:
# "Q1 2024 Earnings Call"
# "Q3 2023 Earnings Call"

# ❌ REJECTED formats:
# "Q1 2024 Earnings Call - Preliminary" 
# "First Quarter 2024"
# "Q1 2024 Conference Call"
```

### Data Processing Pipeline
1. **Date Range Calculation** [line 1045-1079]: Current date minus `sync_date_range` days
2. **API Discovery** [line 1081-1158]: Query each date for all transcripts using `get_transcripts_dates`
3. **Invalid List Check** [line 1219-1223]: Skip transcripts already marked as invalid
4. **Institution Filtering** [line 1110-1124]: Filter to monitored tickers with single primary ID
5. **Title Validation** [line 1342]: Download and validate - strict "Qx 20xx Earnings Call" format
6. **Invalid Tracking** [line 1402-1410]: Add rejected titles to Excel list, save incrementally
7. **Version Management** [line 1241-1245]: Download new versions, remove old ones
8. **Directory Structure** [line 1353-1359]: `{year}/{quarter}/{type}/{ticker_company}/` for valid only

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
- `config.yaml`: Stage configuration loaded from NAS [line 361-414]
  - `stage_01_download_daily.sync_date_range`: Days to look back [line 1048]
  - `stage_01_download_daily.output_data_path`: Base data directory [line 1354]
  - `stage_01_download_daily.output_logs_path`: Logs directory [line 87]
  - `api_settings.request_delay`: Delay between API calls [line 1614, 1679]
  - `api_settings.max_retries`: Retry attempts for failed requests [line 1087]
  - `api_settings.use_exponential_backoff`: Enable exponential retry delay [line 1142]
- `monitored_institutions.yaml`: Separate list of monitored tickers (loaded at line 386-399)
- `invalid_transcripts.xlsx`: Excel list of rejected transcripts on NAS [line 808]
- SSL certificate: Downloaded from NAS for API authentication [line 417-448]
- NAS storage: SMB/CIFS connection for file operations [line 184-223]

## 📋 CURRENT FOCUS
- Daily sync optimization with date-based API queries [line 1090-1095]
- Invalid transcript tracking via Excel to prevent re-downloads [line 1219-1223]
- Strict title validation for "Qx 20xx Earnings Call" format [line 1342]
- Incremental saving of invalid list after each bank to prevent data loss [line 1688-1690]
- Performance optimization via date-based queries instead of per-institution [line 1601-1616]

## ⚠️ KNOWN ISSUES
- Title parsing requires EXACT format matching - any variation is rejected [line 781-784]
- Rate limiting essential for API stability [line 1614, 1679, 1734]
- NAS directory creation implemented with recursive support [line 566-624]
- SSL certificate handling requires temporary file management [line 434-436]
- Invalid list grows over time, may need periodic archival [line 808]
- Date-based queries return ALL transcripts, requiring post-filtering [line 1110-1124]

## 🔍 STAGE 1 CODE CONVENTIONS

### Logging System
- `log_console()`: User-facing minimal output [line 48-57]
- `log_execution()`: Detailed JSON audit entries [line 59-68]
- `log_error()`: Error tracking with categorization [line 70-79]
- `save_logs_to_nas()`: Final log persistence [line 82-135]

### Naming Patterns
```python
# Configuration sections
config["stage_01_download_daily"]["sync_date_range"]  # line 1048
config["api_settings"]["retry_delay"]  # line 1149
config["monitored_institutions"][ticker]  # line 1633

# Logging functions  
log_console()       # Minimal console output [line 48-57]
log_execution()     # Detailed audit trail [line 59-68]
log_error()         # Error tracking [line 70-79]

# File operations
nas_path_join()     # NAS path construction [line 517-519]
nas_create_directory_recursive()  # Directory creation [line 566-624]
download_transcript_with_validation()  # Main download logic with title check [line 1283-1479]

# Invalid list operations
load_invalid_transcript_list()  # Load Excel from NAS [line 803-834]
save_invalid_transcript_list()  # Save Excel to NAS [line 837-874]
is_transcript_in_invalid_list()  # Check before download [line 896-902]
```

### Error Handling
- API failures: Retry with exponential backoff [line 1142-1150, 1426-1441]
- Title validation: Add to invalid list, skip non-conforming transcripts [line 1400-1422]
- Invalid list: Check before download to avoid duplicates [line 1219-1223]
- NAS operations: Graceful degradation with logging [line 594-621]
- Authentication: Clear error messages for credential issues [line 158-168]

### Data Structures
```python
# Transcript inventory format [line 985-998]
transcript_record = {
    "fiscal_year": "2024",
    "quarter": "Q1", 
    "company_type": "Bank",
    "company": "JPM_JPMorgan_Chase_Co",
    "ticker": "JPM",
    "file_quarter": "Q1",
    "file_year": "2024",
    "transcript_type": "Script",
    "event_id": "12345",
    "version_id": "1",
    "filename": "JPM_Q1_2024_Script_12345_1.xml",
    "full_path": "nas_path/to/file.xml"
}

# API transcript format [line 1169-1179]
api_record = {
    "company_type": "Bank",
    "company": "JPM",
    "ticker": "JPM",
    "transcript_type": "Script",
    "event_id": "12345",
    "version_id": "1",
    "event_date": "2024-01-15",
    "transcripts_link": "https://api.url/transcript"
}

# Invalid transcript record [line 880-890]
invalid_entry = {
    "ticker": "JPM",
    "institution_name": "JPMorgan Chase & Co",
    "event_id": "12345",
    "version_id": "1",
    "title_found": "Q1 2024 Earnings Call - Preliminary",
    "event_date": "2024-01-15",
    "transcript_type": "Script",
    "reason": "Title format not 'Qx 20xx Earnings Call'",
    "date_added": "2024-01-20T10:30:00"
}
```

## 📊 DATA HANDLING
- **Input**: Date ranges for API queries (line 1045-1079), existing NAS inventory (line 926-1043)
- **Processing**: Title validation (line 781-784), version comparison (line 1241), directory management (line 566-624)
- **Output**: Downloaded XML transcripts in organized directory structure (line 1353-1373)
- **Logging**: Comprehensive execution logs (line 82-135) and error tracking to NAS
- **Security**: Credential sanitization in logs (line 787-798), secure NAS operations

## 🚫 DO NOT MODIFY
- Environment variable validation logic [line 137-177]
- SSL certificate download and setup procedures [line 417-448]
- API authentication token generation [line 492]
- NAS connection establishment patterns [line 184-223]
- Title parsing regex pattern (exact format required) [line 783]
- Invalid list Excel column structure [line 821-830]
- Filename format: `ticker_quarter_year_transcripttype_eventid_versionid.xml` [line 677]

## 💡 DEVELOPMENT NOTES

### Key Implementation Details
- **Date Range Calculation** [line 1051-1054]: Includes current date (range + 1 days total)
- **API Query Strategy** [line 1601-1616]: All dates queried first, then grouped by ticker
- **Primary ID Filtering** [line 1119-1122]: Only transcripts with single primary ID matching monitored tickers
- **Invalid List Integration** [line 1219-1223]: Checked during comparison before download attempt
- **Title Validation Process** [line 1339-1342]: XML parsed, title extracted, strict regex match applied
- **Incremental Saves** [line 1688-1690]: Invalid list saved after each institution to prevent data loss
- **Version Management** [line 1241-1245]: New versions replace old, maintaining latest only
- **Directory Creation** [line 1361-1368]: Recursive creation ensures full path exists
- **Excel Storage**: Invalid list at `/Outputs/Data/InvalidTranscripts/invalid_transcripts.xlsx` [line 808]

### Performance Optimizations
- Date-based queries more efficient than per-institution loops [line 1090-1095]
- Batch discovery phase before processing [line 1601-1625]
- Only process institutions with transcripts found [line 1627-1630]
- Rate limiting between API calls prevents throttling [line 1614, 1679, 1734]