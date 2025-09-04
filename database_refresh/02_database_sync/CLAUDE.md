# 🚀 Stage 2: Database Synchronization with Title Validation - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Database Synchronization with Earnings Call Identification
- Stack: Python 3.11+, SMB/CIFS protocol, NAS integration, XML parsing
- Architecture: Self-contained file synchronization with title-based filtering
- Focus: Master database sync with earnings call detection via XML title parsing

## 📄 STAGE OVERVIEW
**Stage 2: Transcript Consolidation & Master Database Synchronization**

This stage performs comprehensive file synchronization between the NAS file system and master database, WITH title filtering for processing queues. Key responsibilities:
- Scans NAS for ALL transcript files across all years/quarters/companies
- Downloads and parses each XML file to extract title for validation
- Identifies earnings calls via strict "Qx 20xx Earnings Call" format validation
- Marks each file with `is_earnings_call` flag based on title parsing
- Extracts ticker and institution_id from filename for metadata enrichment
- Loads existing master database for comparison
- Detects changes (new, modified, deleted files) via delta analysis
- **Version selection**: Automatically selects best version when multiple exist (highest version_id, type priority)
- Generates processing queues for downstream stages (ONLY earnings calls)
- Maintains complete inventory but filters processing to earnings calls only
- Self-contained with runtime configuration loading from NAS

## 🚨 CRITICAL RULES
- ALWAYS validate NAS paths for security (no directory traversal)
- NEVER commit credentials or API keys
- MUST handle SMB connection failures gracefully
- Monitor file permissions and access rights
- Handle large file inventories efficiently

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/stage-02-*, bugfix/stage-02-*, hotfix/stage-02-*
- Commit format: conventional commits preferred
- Focus: Database sync and NAS operations
- Test: Mock SMB connections for development

### Code Quality Standards
```bash
# Pre-commit checks
python -m black . --line-length 88
python -m isort . --profile black
python -m pylint database_refresh/02_database_sync/ --min-score 8.0
python -m flake8 database_refresh/02_database_sync/ --max-line-length 88

# Manual quality check
make lint  # If available
```

### Testing Strategy
- Local: Mock SMB connections and NAS operations
- Integration: Test with actual NAS in staging environment
- Security: Validate path sanitization and input validation
- Performance: Test with large file inventories

## 🤖 STAGE 2 CONFIGURATION

### Core Dependencies
- `smb.SMBConnection`: SMB/CIFS protocol handling
- `yaml`: Configuration file parsing
- `xml.etree.ElementTree`: XML parsing for title validation
- `requests`: HTTP operations with SSL/proxy support
- `dotenv`: Environment variable management
- `json`: Database and queue serialization
- `io.BytesIO`: In-memory file operations

### Processing Pipeline
1. **Environment Validation**: Verify all required credentials (line 131-171)
2. **NAS Connection**: Establish SMB connection with authentication (line 173-212)
3. **Configuration Loading**: Load YAML config from NAS at runtime, including separate monitored_institutions.yaml (line 214-259)
4. **SSL Setup**: Download and configure SSL certificates (line 318-348)
5. **Proxy Configuration**: Setup proxy with domain authentication (line 350-377)
6. **File Scanning**: Comprehensive NAS inventory scan of all XML files (line 615-706)
7. **Title Validation**: Download each file and parse XML to identify earnings calls (line 564-613)
8. **Database Loading**: Load existing master database (JSON format) (line 708-732)
9. **Delta Detection**: Compare NAS vs database inventories with timestamps (line 747-791)
10. **Version Selection**: Select best version when duplicates exist (line 793-858)
11. **Queue Generation**: Create processing queues (earnings calls only) with version deduplication (line 860-998)

## 📁 PROJECT STRUCTURE
```
database_refresh/02_database_sync/
  main_sync_updates.py     # Main synchronization script
  CLAUDE.md               # This configuration file
```

## 🔧 ESSENTIAL COMMANDS

### Development
```bash
# Environment setup (from project root)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run Stage 2 synchronization
cd database_refresh/02_database_sync
python main_sync_updates.py

# Test NAS connectivity (development)
python -c "from main_sync_updates import get_nas_connection; print('Connected' if get_nas_connection() else 'Failed')"

# View processing summary (after execution)
cat stage_02_database_sync_consolidation_*.json | jq '.summary'
```

### Debugging and Monitoring
```bash
# Check environment variables (requires all 14 variables)
python -c "from main_sync_updates import validate_environment_variables; validate_environment_variables()"

# Validate configuration structure
python -c "import yaml; from main_sync_updates import validate_config_structure; validate_config_structure(yaml.safe_load(open('../../config.yaml')))"

# Test path validation
python -c "from main_sync_updates import validate_nas_path; print(validate_nas_path('Data/2023/Q1/Banks/JPM/transcript.xml'))"

# Check title validation
python -c "from main_sync_updates import parse_quarter_and_year_from_xml; import io; xml='<root><meta><title>Q1 2023 Earnings Call</title></meta></root>'; print(parse_quarter_and_year_from_xml(xml.encode()))"
```

## 🔗 INTEGRATIONS

### Data Sources
- **NAS File System**: Primary source via SMB/CIFS protocol
- **Master Database**: JSON-based file inventory at `master_database_path`
- **Configuration**: YAML config loaded from NAS at runtime
- **Monitored Institutions**: Separate `monitored_institutions.yaml` file (fallback to main config)

### External Dependencies
- **SMB Share**: Network-attached storage access
- **SSL Certificates**: Downloaded from NAS for secure connections
- **Proxy Configuration**: Corporate proxy with domain authentication

### Required Environment Variables
- `API_USERNAME`, `API_PASSWORD`: API credentials
- `PROXY_USER`, `PROXY_PASSWORD`, `PROXY_URL`: Proxy authentication
- `NAS_USERNAME`, `NAS_PASSWORD`: NAS credentials
- `NAS_SERVER_IP`, `NAS_SERVER_NAME`, `NAS_SHARE_NAME`: NAS connection details
- `NAS_BASE_PATH`, `NAS_PORT`: NAS configuration
- `CONFIG_PATH`: Path to configuration file on NAS
- `CLIENT_MACHINE_NAME`: SMB client identification

## 📋 CURRENT FOCUS
- Comprehensive file synchronization with title validation for ALL files
- XML parsing to identify earnings calls ("Qx 20xx Earnings Call" exact format)
- Each file marked with `is_earnings_call` boolean flag
- Institution ID and ticker extraction from filename
- Delta detection between NAS and database inventories
- **Version selection**: Automatically selects best version from duplicates using:
  - Highest version_id (primary factor)
  - Transcript type priority: Corrected > Final > Script > Raw
  - Most recent modification date (tiebreaker)
- Processing queue generation for earnings calls only (filtered by flag)
- Master database maintains complete inventory (all transcripts)
- Robust error handling and logging with detailed execution tracking

## ⚠️ KNOWN ISSUES
- Large file inventories may impact memory usage (downloads each file for title check)
- SMB connection timeouts in poor network conditions
- Path validation critical for security (directory traversal prevention)
- SSL certificate setup required for external API calls
- Performance impact from downloading every file to check title
- Title validation requires exact "Qx 20xx Earnings Call" format
- Version selection examples limited to 5 for readability in console output

## 🚫 DO NOT MODIFY
- Environment variable validation logic
- NAS path validation functions (security-critical)
- SMB connection authentication flow
- Configuration structure validation

## 💡 DEVELOPMENT NOTES

### Key Functions and Patterns
- `scan_nas_for_all_transcripts()`: Comprehensive file inventory with title validation (line 615-706)
- `parse_quarter_and_year_from_xml()`: XML title validation for earnings calls (line 564-613)
- `detect_changes()`: Delta detection logic comparing NAS vs database (line 747-791)
- `select_best_version()`: Version selection logic for duplicate transcripts (line 793-858)
- `save_processing_queues()`: Queue generation filtered by is_earnings_call flag with version deduplication (line 860-998)
- `validate_nas_path()`: Security validation (line 533-552)
- `validate_file_path()`: File path security validation (line 512-531)
- `load_master_database()`: Load existing JSON database from NAS (line 708-732)
- `database_to_comparison_format()`: Convert database to comparison format (line 734-745)

### Data Flow Patterns
```python
# Enhanced file record structure with earnings call flag
file_record = {
    "file_path": "/Data/YYYY/QX/Type/Company/file.xml",
    "date_last_modified": "2023-12-01T10:30:00",
    "is_earnings_call": True,  # Based on XML title validation
    "institution_id": "12345",  # From monitored_institutions config
    "ticker": "JPM"  # Extracted from filename
}

# Version selection for duplicates (line 793-858)
# Format: TICKER_QUARTER_YEAR_TYPE_EVENTID_VERSIONID.xml
# Groups files by base key (ticker_quarter_year)
# Selects best version based on version_id and type priority

# Processing queue output format (earnings calls only + version deduplication)
process_queue = [file_record, ...]  # Only files with is_earnings_call=True and best versions
removal_queue = [file_record, ...]  # Files to remove from database
```

### Error Handling
- All operations log to execution_log and error_log arrays
- Errors categorized by type for summary reporting:
  - `environment_validation`: Missing environment variables
  - `nas_connection`: SMB connection failures
  - `config_load`: Configuration loading errors
  - `ssl_setup`: SSL certificate issues
  - `nas_download`/`nas_upload`: File transfer errors
  - `database_load`: Database corruption or load failures
  - `queue_save`: Queue generation failures
  - `path_validation`: Security validation failures
- Graceful degradation with comprehensive cleanup
- Logs saved to NAS even on failure (if connection exists)

## 🔍 CODE CONVENTIONS
- Use `nas_path_join()` for NAS path construction (line 379-388)
- Validate all file paths with `validate_nas_path()` (line 533-552)
- Additional validation with `validate_file_path()` for general paths (line 512-531)
- Log both console messages (`log_console()`) and detailed execution logs (`log_execution()`)
- Handle SMB exceptions with specific error types
- Use `io.BytesIO` for in-memory file operations
- Sanitize URLs before logging with `sanitize_url_for_logging()` (line 554-562)
- Global variables for config and logger management

## 📊 DATA HANDLING
- **File Structure**: Data/YYYY/QX/Type/Company/*.xml
- **Filename Format**: TICKER_QUARTER_YEAR_TYPE_EVENTID_VERSIONID.xml
- **Inventory Format**: file_path -> {file_path, date_last_modified, is_earnings_call, institution_id, ticker}
- **Master Database**: JSON format with complete file inventory
- **Version Handling**: Groups files by base transcript (ticker_quarter_year), selects best version
- **Queue Output**: JSON arrays in refresh output folder:
  - `stage_02_process_queue.json`: Earnings calls only, best versions selected
  - `stage_02_removal_queue.json`: Files to remove from database
- **Title Validation**: Downloads each file to parse XML and check title (exact pattern match)
- **Security**: Path validation prevents directory traversal
- **Performance**: Stream-based file operations, but downloads all files for validation

## 🔒 SECURITY CONSIDERATIONS
- NAS credentials in environment variables only
- Path sanitization prevents directory traversal attacks
- SSL certificate validation for external connections
- Proxy authentication with credential escaping
- No credentials logged in execution traces

## 📈 MONITORING AND LOGGING
- **Execution Logs**: Saved to `output_logs_path` with timestamp
  - Main log: `stage_02_database_sync_consolidation_YYYY-MM-DD_HH-MM-SS.json`
  - Error log: `Errors/stage_02_database_sync_consolidation_errors_YYYY-MM-DD_HH-MM-SS.json` (if errors occur)
- **Stage Summary**: Includes:
  - `total_nas_files`: Files found on NAS
  - `total_database_files`: Files in existing database
  - `files_to_process`: New/modified files
  - `files_to_remove`: Deleted/outdated files
  - `execution_time_seconds`: Total runtime
  - Error counts by category
- **Console Logging**: Real-time progress updates with minimal output
- **Version Selection Logging**: Details about duplicate handling and best version selection