# 🚀 Stage 2: Database Synchronization - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Database Synchronization and File Management Pipeline
- Stack: Python 3.11+, SMB/CIFS protocol, NAS integration
- Architecture: Self-contained file synchronization with delta detection
- Focus: Transcript consolidation and master database synchronization

## 📄 STAGE OVERVIEW
**Stage 2: Transcript Consolidation & Master Database Synchronization**

This stage performs comprehensive file synchronization between the NAS file system and master database, WITH title filtering for processing queues. Key responsibilities:
- Scans NAS for ALL transcript files across all years/quarters/companies
- Validates each transcript's title to identify earnings calls ("Qx 20xx Earnings Call" format)
- Loads existing master database for comparison
- Detects changes (new, modified, deleted files) via delta analysis
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
- `requests`: HTTP operations with SSL/proxy support
- `dotenv`: Environment variable management

### Processing Pipeline
1. **Environment Validation**: Verify all required credentials
2. **NAS Connection**: Establish SMB connection with authentication
3. **Configuration Loading**: Load YAML config from NAS at runtime
4. **SSL Setup**: Download and configure SSL certificates
5. **File Scanning**: Comprehensive NAS inventory scan with title validation
6. **Title Validation**: Parse XML to identify earnings calls ("Qx 20xx Earnings Call")
7. **Database Loading**: Load existing master database
8. **Delta Detection**: Compare NAS vs database inventories
9. **Queue Generation**: Create processing queues (earnings calls only) for Stage 3+

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
```

### Debugging and Monitoring
```bash
# Check environment variables
python -c "from main_sync_updates import validate_environment_variables; validate_environment_variables()"

# Validate configuration structure
python -c "import yaml; from main_sync_updates import validate_config_structure; validate_config_structure(yaml.safe_load(open('../../config.yaml')))"

# Test path validation
python -c "from main_sync_updates import validate_nas_path; print(validate_nas_path('Data/2023/Q1/Banks/JPM/transcript.xml'))"
```

## 🔗 INTEGRATIONS

### Data Sources
- **NAS File System**: Primary source via SMB/CIFS protocol
- **Master Database**: JSON-based file inventory
- **Configuration**: YAML config loaded from NAS at runtime

### External Dependencies
- **SMB Share**: Network-attached storage access
- **SSL Certificates**: Downloaded from NAS for secure connections
- **Proxy Configuration**: Corporate proxy with domain authentication

## 📋 CURRENT FOCUS
- Comprehensive file synchronization with title validation
- XML parsing to identify earnings calls vs other transcripts
- Delta detection between NAS and database inventories
- Processing queue generation for earnings calls only
- Robust error handling and logging

## ⚠️ KNOWN ISSUES
- Large file inventories may impact memory usage
- SMB connection timeouts in poor network conditions
- Path validation critical for security (directory traversal prevention)
- SSL certificate setup required for external API calls

## 🚫 DO NOT MODIFY
- Environment variable validation logic
- NAS path validation functions (security-critical)
- SMB connection authentication flow
- Configuration structure validation

## 💡 DEVELOPMENT NOTES

### Key Functions and Patterns
- `scan_nas_for_all_transcripts()`: Comprehensive file inventory (main_sync_updates.py:547)
- `detect_changes()`: Delta detection logic (main_sync_updates.py:637)
- `save_processing_queues()`: Queue generation (main_sync_updates.py:683)
- `validate_nas_path()`: Security validation (main_sync_updates.py:516)

### Data Flow Patterns
```python
# Standard file record structure
file_record = {
    "file_path": "/Data/YYYY/QX/Type/Company/file.xml",
    "date_last_modified": "2023-12-01T10:30:00"
}

# Processing queue output format
process_queue = [file_record, ...]  # Files to process
removal_queue = [file_record, ...]  # Files to remove
```

### Error Handling
- All operations log to execution_log and error_log arrays
- Errors categorized by type for summary reporting
- Graceful degradation with comprehensive cleanup

## 🔍 CODE CONVENTIONS
- Use `nas_path_join()` for NAS path construction
- Validate all file paths with `validate_nas_path()`
- Log both console messages and detailed execution logs
- Handle SMB exceptions with specific error types
- Use io.BytesIO for in-memory file operations

## 📊 DATA HANDLING
- **File Structure**: Data/YYYY/QX/Type/Company/*.xml
- **Inventory Format**: file_path -> {file_path, date_last_modified}
- **Queue Output**: JSON arrays in refresh output folder
- **Security**: Path validation prevents directory traversal
- **Performance**: Stream-based file operations for large inventories

## 🔒 SECURITY CONSIDERATIONS
- NAS credentials in environment variables only
- Path sanitization prevents directory traversal attacks
- SSL certificate validation for external connections
- Proxy authentication with credential escaping
- No credentials logged in execution traces

## 📈 MONITORING AND LOGGING
- Execution logs saved to NAS with timestamps
- Error logs saved separately if errors occur
- Stage summary includes file counts and error statistics
- Console logging for real-time monitoring
- Detailed execution tracking for audit purposes