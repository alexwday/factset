# 🚀 Stage 0: Historical Transcript Sync with Invalid Tracking - AI Development Configuration

## 🎯 PROJECT CONTEXT
- **Type**: Historical Financial Transcript Download with Title Validation
- **Script**: `main_historical_sync_with_ignore.py` (NOT main_historical_sync.py)
- **Stack**: Python 3.11+, FactSet SDK, SMB/NAS, XML parsing, Pandas/Excel (openpyxl)
- **Architecture**: API-driven download with Excel-based invalid transcript tracking
- **Focus**: Download earnings transcripts with strict "Qx 20xx Earnings Call" validation

## 🚨 CRITICAL RULES
- STRICT title validation: Only "Qx 20xx Earnings Call" format accepted (line 761)
- Invalid transcripts tracked in Excel file at `{NAS_BASE_PATH}/Outputs/Data/InvalidTranscripts/invalid_transcripts.xlsx`
- Check invalid list BEFORE downloading to avoid re-processing known invalid transcripts
- Incremental saving of invalid list after each institution to prevent data loss
- NEVER expose credentials in logs (sanitize_url_for_logging function)
- Event IDs and Version IDs stored as strings for consistent comparison

## 🛠️ DEVELOPMENT WORKFLOW

### Environment Requirements
```bash
# Required environment variables (validated at line 140)
API_USERNAME=<factset_api_username>
API_PASSWORD=<factset_api_password>
PROXY_USER=<corporate_proxy_user>
PROXY_PASSWORD=<corporate_proxy_password>
PROXY_URL=<proxy_server_url>
PROXY_DOMAIN=MAPLE  # Default value
NAS_USERNAME=<nas_server_username>
NAS_PASSWORD=<nas_server_password>
NAS_SERVER_IP=<nas_server_ip>
NAS_SERVER_NAME=<nas_server_name>
NAS_SHARE_NAME=<nas_share_name>
NAS_BASE_PATH=<nas_base_path>
NAS_PORT=445  # Default
CONFIG_PATH=<config_yaml_path_on_nas>
CLIENT_MACHINE_NAME=<local_machine_name>
```

### Git Integration
- Branch pattern: feature/stage-00-*, bugfix/stage-00-*, hotfix/stage-00-*
- Commit format: "Stage 0: [description]"
- Focus: Invalid tracking and title validation features

## 🤖 STAGE 0 CONFIGURATION

### Core Functionality
- Downloads transcripts from FactSet API for monitored institutions
- Date window: Configurable 3-year rolling OR fixed start year (calculate_rolling_window, line 1071)
- Title validation: Downloads file, parses XML, checks title format (line 1427-1431)
- Invalid tracking: Excel-based list with full metadata
- Contamination prevention: Only accepts single primary ID transcripts (line 1177)

### Invalid Transcript Management
- **Load**: `load_invalid_transcript_list()` (line 781) - Creates new if doesn't exist
- **Save**: `save_invalid_transcript_list()` (line 825) - Extensive DEBUG logging
- **Add**: `add_to_invalid_list()` (line 883) - Converts IDs to strings
- **Check**: `is_transcript_in_invalid_list()` (line 909) - String comparison with strip

### Excel Structure (Invalid List)
```
Columns:
- ticker
- institution_name  
- event_id (stored as string)
- version_id (stored as string)
- title_found
- event_date
- transcript_type
- reason
- date_added
```

## 📁 PROJECT STRUCTURE
```
database_refresh/00_download_historical/
  main_historical_sync_with_ignore.py  # Main execution script
  CLAUDE.md                            # This file

Configuration Files (loaded from NAS):
  config.yaml                          # Main configuration
  monitored_institutions.yaml          # Separate institution list (line 386)
  SSL certificate                      # Downloaded to temp file

NAS Output Structure:
/Outputs/Data/
  YYYY/                                # Fiscal year
    QX/                                # Quarter
      CompanyType/                     # Institution type
        TICKER_CompanyName/            # Company folder
          *.xml                        # Valid transcripts only
  InvalidTranscripts/
    invalid_transcripts.xlsx          # Excel tracking file
```

## 🔧 ESSENTIAL COMMANDS

### Stage 0 Operations
```bash
# Run historical transcript sync with invalid tracking
cd database_refresh/00_download_historical
python main_historical_sync_with_ignore.py

# Environment setup
python -m venv venv && source venv/bin/activate
pip install pandas openpyxl pyyaml pysmb fds.sdk.EventsandTranscripts python-dotenv

# Testing
python -c "from main_historical_sync_with_ignore import validate_environment_variables; validate_environment_variables()"
```

### Debugging Commands
```bash
# Test NAS connection
python -c "from main_historical_sync_with_ignore import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Load and check invalid list
python -c "
from main_historical_sync_with_ignore import load_invalid_transcript_list, get_nas_connection
conn = get_nas_connection()
df = load_invalid_transcript_list(conn)
print(f'Invalid transcripts: {len(df)}')
if not df.empty:
    print(df[['ticker', 'event_id', 'version_id', 'title_found']].head())
"

# Check configuration loading
python -c "
from main_historical_sync_with_ignore import load_config_from_nas, get_nas_connection
conn = get_nas_connection()
config = load_config_from_nas(conn)
print(f'Institutions: {len(config[\"monitored_institutions\"])}')
print(f'Start year: {config[\"stage_00_download_historical\"].get(\"start_year\", \"3-year rolling\")}')
"
```

## 🔗 INTEGRATIONS

### FactSet API Integration
- SDK: `fds.sdk.EventsandTranscripts` (line 23-24)
- Authentication: HTTP Basic Auth with proxy support
- API method: `get_transcripts_ids()` (line 1164)
- Rate limiting: Configurable delays with exponential backoff

### NAS Storage Integration  
- Protocol: SMB/CIFS via pysmb
- Key functions:
  - `nas_create_directory_recursive()` (line 565) - Extensive DEBUG logging
  - `nas_upload_file()` (line 241) - DEBUG logging for troubleshooting
  - `nas_download_file()` (line 220)
- Share name dynamically retrieved from config or environment

### Configuration Management
- Main config: `config.yaml` loaded from NAS (line 360)
- Institutions: `monitored_institutions.yaml` loaded separately (line 386)
- Falls back to main config if separate file not found (line 395)
- SSL certificate downloaded to temp file (line 433)

## 📋 PROCESSING FLOW

### Main Execution Flow (main function, line 1573)
1. **Setup Phase** (lines 1605-1634)
   - Validate environment variables
   - Connect to NAS
   - Load config and monitored_institutions.yaml
   - Setup SSL certificate (temp file)
   - Configure proxy (NTLM authentication)
   - Setup FactSet API client

2. **Invalid List Loading** (lines 1636-1653)
   - Load existing Excel file or create new
   - Show sample entries for debugging
   - Track initial count for summary

3. **Inventory Scan** (lines 1674-1690)
   - Scan existing transcripts on NAS
   - Track unparseable filenames
   - Filter out InvalidTranscripts folder

4. **Date Window Calculation** (lines 1692-1698)
   - Check for fixed start_year in config
   - Calculate 3-year rolling window if not fixed
   - Enhanced logging with coverage details

5. **Institution Processing** (lines 1700-1840)
   - For each monitored institution:
     - Query API for transcripts (with contamination filter)
     - Check against invalid list
     - Compare with existing NAS inventory
     - Download and validate new transcripts
     - Update invalid list for rejected titles
     - Save invalid list incrementally (line 1804)
     - Log detailed statistics per institution

6. **Completion** (lines 1842-1900)
   - Final save of invalid list
   - Save execution and error logs to NAS
   - Cleanup temp SSL certificate
   - Close NAS connection

## ⚠️ KNOWN ISSUES
- Extensive DEBUG logging (search for "DEBUG:") - may need cleanup for production
- Excel file operations can fail with large invalid lists
- Monitored institutions loading falls back silently if file missing
- Rate limiting between institutions may slow processing
- Invalid list grows indefinitely - no archival mechanism

## 🚫 DO NOT MODIFY
- Environment variable names (hardcoded throughout)
- File naming convention: `ticker_quarter_year_transcripttype_eventid_versionid.xml`
- Invalid list Excel structure and column names
- Title validation regex: `^Q([1-4])\s+(20\d{2})\s+Earnings\s+Call$` (line 761)
- Event/Version ID string conversion logic (consistency critical)

## 💡 DEVELOPMENT NOTES

### Key Functions
- `calculate_rolling_window()` (line 1071) - Handles both fixed and rolling windows
- `get_api_transcripts_for_company()` (line 1130) - Contamination filtering
- `compare_transcripts_with_invalid_list()` (line 1265) - Check invalid BEFORE download
- `download_transcript_with_validation()` (line 1372) - Title validation logic
- `parse_quarter_and_year_from_xml()` (line 715) - Returns quarter, year, AND title
- `is_valid_earnings_call_title()` (line 759) - Strict regex validation

### Logging System
- **Console**: Minimal output via `log_console()`
- **Execution Log**: Detailed JSON saved to NAS
- **Error Log**: Separate error tracking by type
- **DEBUG**: Extensive debug messages (production concern)

### Statistics Tracked (per institution)
- API transcripts found
- Existing valid earnings calls
- Downloads attempted
- Downloads successful (valid title)
- Downloads rejected (invalid title)  
- Skipped (already in invalid list)
- Total valid after processing
- Total invalid tracked

## 🔄 PROCESSING LOGIC
- **Date Window**: Configurable via `start_year` in config or 3-year rolling
- **Contamination Prevention**: Only single primary ID transcripts (line 1177)
- **Invalid Check First**: Skip if already in invalid list (line 1301)
- **Title Validation**: Download, parse XML, check exact format (line 1431)
- **Version Management**: API version always authoritative (line 1330)
- **Incremental Saving**: Invalid list saved after each institution
- **No Removals**: Archive mode - never deletes transcripts (line 1858)

## 📊 DATA HANDLING
- **Security**: URL sanitization in logs (line 765), no credential logging
- **XML Parsing**: ElementTree with namespace handling (line 726)
- **Excel Management**: Pandas with openpyxl engine (line 846)
- **String Handling**: Event/Version IDs consistently as strings
- **Error Categories**: Tracked by type for analysis
- **File Organization**: Year/Quarter/Type/Company structure