# Stage 3: Transcript Content Extraction & Paragraph-Level Breakdown

> **Version**: 3.0 | **Updated**: 2025-07-30  
> **Purpose**: XML content extraction and paragraph-level breakdown with enhanced field extraction and JSON validation

---

## Project Context

- **Stage**: Stage 3 - Transcript Content Extraction & Paragraph-Level Breakdown
- **Primary Purpose**: Process Stage 2 queue files, extract metadata from paths/XML, and create paragraph-level database records
- **Pipeline Position**: Between Stage 2 (Consolidation) and Stage 4 (LLM Classification)
- **Production Status**: PRODUCTION READY ✅

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: XML parsing with ElementTree, NAS file operations
- **Authentication**: Corporate proxy (MAPLE domain) + NAS NTLM v2
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for this stage
import os                           # File system operations
import tempfile                     # Temporary file handling
import logging                      # Logging system
import json                         # JSON processing
import xml.etree.ElementTree as ET  # XML parsing
import yaml                         # Configuration parsing
import requests                     # Network operations  
from smb.SMBConnection import SMBConnection  # NAS connectivity
from dotenv import load_dotenv      # Environment variables
from datetime import datetime       # Timestamp handling
from typing import Dict, Any, Optional, List, Tuple  # Type hints
```

### Environment Requirements
```bash
# Required .env variables (14 required)
API_USERNAME=                # FactSet API credentials (consistency)
API_PASSWORD=
PROXY_USER=                  # Corporate proxy (MAPLE domain)
PROXY_PASSWORD=
PROXY_URL=
PROXY_DOMAIN=               # Defaults to 'MAPLE' if not set
NAS_USERNAME=                # NAS authentication
NAS_PASSWORD=
NAS_SERVER_IP=
NAS_SERVER_NAME=
NAS_SHARE_NAME=
NAS_BASE_PATH=
NAS_PORT=                    # Defaults to 445 if not set
CONFIG_PATH=                 # NAS configuration path
CLIENT_MACHINE_NAME=
```

---

## Architecture & Design

### Core Design Patterns
- **Stage 2 Integration**: Processes simplified Stage 2 output (file_path + date_last_modified)
- **Enhanced Field Extraction**: Extracts all required metadata from file paths, XML content, and config lookups
- **XML Content Processing**: Comprehensive XML parsing with namespace handling and participant mapping
- **Paragraph-Level Breakdown**: Creates individual database records for each paragraph with full context

### File Structure
```
03_extract_content/
├── main_content_extraction.py      # Primary execution script
├── CLAUDE.md                      # This context file
└── old/                          # Backup of previous implementation
```

### Key Components
1. **Processing Queue Loader**: Loads Stage 2's files_to_process.json from NAS
2. **Metadata Extractor**: Extracts all required fields from file paths and config lookups
3. **XML Content Parser**: Comprehensive XML processing with namespace and participant handling
4. **Paragraph Record Creator**: Generates database records with full metadata and sequential IDs

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for Stage 3
stage_03_extract_content:
  description: "XML content extraction and paragraph-level breakdown"
  input_queue_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/files_to_process.json"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"
  output_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  dev_mode: true
  dev_max_files: 2
```

### Validation Requirements
- **Schema Validation**: Validates all required stage_03_extract_content parameters exist
- **Security Validation**: NAS path validation, file path security checks, XML parsing safety
- **Business Rule Validation**: Monitored institutions structure validation for company name lookups

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Environment Setup**: Validate environment variables and establish NAS connection
2. **Configuration Loading**: Load and validate YAML configuration from NAS
3. **SSL/Proxy Setup**: Configure SSL certificates and corporate proxy (for consistency)
4. **Processing Queue Loading**: Load Stage 2's files_to_process.json from NAS
5. **Development Mode Check**: Apply file limits if in development mode
6. **File Processing Loop**: For each file, extract metadata, download XML, parse content, create paragraph records
7. **Content Aggregation**: Combine all paragraph records across all processed files
8. **JSON Validation & Preview**: Validate JSON structure and preview first 3 records before saving
9. **Output Generation**: Save stage_03_extracted_content.json to NAS
10. **Log Cleanup**: Save execution logs and clean up temporary files

### Key Business Rules
- **Enhanced Field Extraction**: Extract all required fields that were simplified out of Stage 2's output
- **Filename Parsing**: Correctly extracts ticker, transcript_type, event_id, version_id from filenames
- **Path Metadata**: Extracts fiscal_year, fiscal_quarter, institution_type from NAS folder structure
- **Metadata Completion**: Combine Stage 2 data with extracted path metadata, XML titles, and config lookups  
- **Sequential Numbering**: Maintain paragraph IDs and speaker block IDs for document reconstruction
- **Speaker Attribution**: Format participants into readable speaker strings using name, title, affiliation
- **Q&A Detection**: Use XML speaker type attributes ("q" for question, "a" for answer) to flag Q&A content
- **JSON Validation**: Pre-save validation with record preview and comprehensive error detection

### Data Processing Logic
- **Input**: Stage 2's simplified files_to_process.json (file_path + date_last_modified)
- **Field Extraction**: 
  - Parse paths for fiscal_year/quarter/type/ticker from Data/YYYY/QX/Type/Company structure
  - Extract filename components: ticker_quarter_year_type_eventid_versionid.xml
  - Lookup company names from monitored_institutions configuration
- **XML Processing**: 
  - Download XML from NAS, parse with namespace handling
  - Extract title from meta section
  - Build participant mapping from meta/participants section
  - Process body/section/speaker/plist/p structure for content
- **Content Breakdown**: Convert XML paragraphs into individual database records with full metadata
- **Output**: stage_03_extracted_content.json with paragraph-level records ready for Stage 4

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions - implemented:
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
```

### Security Standards Checklist
- [x] All input validation implemented
- [x] No credential exposure in logs  
- [x] File paths validated against directory traversal
- [x] URLs sanitized before logging
- [x] Configuration schema validated
- [x] XML parsing with error handling to prevent XML bombs

### Compliance Requirements
- **Audit Trail**: Comprehensive logging of all file operations and extraction decisions
- **Data Retention**: No local data storage - all operations on NAS
- **Error Tracking**: Separate error logs by category for targeted investigation

---

## Commands & Operations

### Primary Commands
```bash
# Development and testing
python main_content_extraction.py           # Run Stage 3
python -m py_compile main_content_extraction.py  # Syntax check

# Configuration validation
python -c "import yaml; yaml.safe_load(open('config.yaml'))"  # Validate YAML
```

### Execution Modes
- **Terminal Mode**: `python main_content_extraction.py` - Standard command-line execution
- **Notebook Mode**: Import and run from Jupyter notebooks
- **Development Mode**: Configurable via `dev_mode: true` for limited file processing

### Testing Commands
```bash
# Syntax validation
python -m py_compile main_content_extraction.py

# Configuration testing
python -c "from main_content_extraction import validate_config_structure"
```

---

## Error Handling & Recovery

### Error Categories
- **Environment Validation**: Missing .env variables, invalid configurations
- **NAS Connection**: SMB/CIFS connectivity issues, authentication failures  
- **Config Load**: YAML parsing errors, missing configuration sections
- **SSL Setup**: Certificate download or configuration failures
- **Queue Load**: Stage 2 output file missing or corrupted
- **Metadata Extraction**: File path parsing failures, config lookup failures
- **XML Processing**: XML parsing errors, malformed content, namespace issues
- **Content Extraction**: Missing participants, empty sections, content structure failures

### Error Handling Standards (MANDATORY)
```python
# NEVER DO THIS - CAUSES PRODUCTION FAILURES:
except:
    pass

# ALWAYS DO THIS - SPECIFIC ERROR HANDLING:
except ET.ParseError as e:
    log_error(f"XML parsing error: {e}", "xml_parsing", {...})
    
except yaml.YAMLError as e:
    log_error(f"Invalid YAML in configuration file: {e}", "config_parse", {...})
```

### Recovery Mechanisms
- **Graceful Failures**: Continue processing other files if individual files fail
- **Error Logging**: Comprehensive error logs with actionable context
- **Development Mode**: Test with limited files to avoid processing issues at scale

---

## Integration Points

### Upstream Dependencies
- **Stage 2**: Must complete successfully and create files_to_process.json
- **Configuration Sources**: NAS-based YAML configuration with monitored_institutions
- **SSL Certificates**: Downloaded from NAS for consistent authentication setup

### Downstream Outputs  
- **stage_03_extracted_content.json**: Paragraph-level database records for Stage 4+ consumption
- **Enhanced Metadata**: All fields required by downstream stages now included in each record
- **Sequential IDs**: Paragraph and speaker block IDs for document reconstruction

### External System Integration
- **NAS File System**: SMB/CIFS operations for file downloads and output storage
- **Corporate Proxy**: NTLM authentication for SSL certificate downloads
- **Configuration System**: YAML-based configuration loaded from NAS

---

## Performance & Monitoring

### Performance Considerations
- **File Processing**: Sequential processing of transcript files with comprehensive error recovery
- **Memory Management**: Process files individually to avoid memory buildup from large XML files
- **Connection Management**: Single NAS connection reused throughout execution
- **Development Mode**: Use for iterative testing to avoid processing large volumes

### Monitoring Points
- **Processing Rate**: Track files processed and paragraphs extracted per execution
- **Error Rates**: Monitor XML parsing failures, metadata extraction failures, file download failures
- **Content Quality**: Validate paragraph extraction completeness and metadata accuracy
- **Resource Utilization**: Monitor memory usage during XML processing and JSON generation

---

## Code Style & Standards

### Code Organization
- **Function Naming**: Descriptive names following snake_case convention
- **Module Organization**: Single comprehensive script with clear functional sections
- **Documentation**: All public functions have comprehensive docstrings
- **Type Hints**: Optional type hints used for clarity in complex data structures

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: ALL file paths, XML content, and configuration validated
- **Credential Protection**: NEVER log credentials, server IPs, or sensitive data
- **Error Handling**: NO bare except clauses - specific exception handling only
- **Resource Management**: Proper cleanup of SSL certificates and connections in finally blocks

### Quality Standards
- **Three-Tier Logging**: Console, execution, and error logs with proper categorization
- **Comprehensive Error Context**: All errors include actionable details and context
- **Field Preservation**: All Stage 2 fields preserved plus newly extracted metadata
- **Sequential Consistency**: Maintain document order through paragraph and speaker block IDs

---

## Development Workflow

### Pre-Development Checklist
- [x] Environment variables configured in .env
- [x] NAS access confirmed and tested
- [x] Configuration file validated with stage_03_extract_content section
- [x] Stage 2 output files available for testing

### Development Process
1. **Setup**: Environment validation, NAS connection, configuration loading
2. **Development**: Field extraction logic, XML processing, paragraph record creation
3. **Testing**: Validate with development mode using sample files
4. **Validation**: Ensure all required fields are extracted and formatted correctly
5. **Production**: Full processing with comprehensive logging and error handling

### Pre-Production Checklist (MANDATORY)
- [x] **Security Review**: All input validation implemented
- [x] **Error Handling Review**: No bare except clauses
- [x] **Resource Management Review**: Proper cleanup implemented
- [x] **Configuration Validation**: Schema validation working
- [x] **Integration Testing**: Stage 2 interface tested, output format verified

---

## Function Reference

### Core Functions

#### Configuration & Setup
- `validate_environment_variables()`: Validates all required environment variables
- `get_nas_connection()`: Creates and returns SMB connection to NAS
- `load_config_from_nas()`: Loads and validates YAML configuration from NAS
- `validate_config_structure()`: Validates configuration schema and requirements
- `setup_ssl_certificate()`: Downloads SSL certificate from NAS to temporary file
- `setup_proxy_configuration()`: Configures proxy with proper credential escaping

#### File Operations
- `nas_path_join()`: Joins path parts for NAS paths using forward slashes
- `nas_download_file()`: Downloads file from NAS and returns as bytes
- `nas_upload_file()`: Uploads file object to NAS
- `nas_file_exists()`: Checks if file exists on NAS
- `nas_create_directory_recursive()`: Creates directory on NAS with safe iterative parent creation

#### Security Functions
- `validate_file_path()`: Validates file path for security (directory traversal prevention)
- `validate_nas_path()`: Validates NAS path structure for security
- `sanitize_url_for_logging()`: Removes auth tokens from URLs before logging

#### Content Processing
- `load_processing_queue()`: Loads Stage 2 processing queue from NAS
- `extract_metadata_from_file_path()`: Extracts metadata from file path structure and filename
- `parse_transcript_xml()`: Parses transcript XML and extracts structured data
- `format_speaker_string()`: Formats speaker information into clean readable string
- `determine_qa_flag()`: Determines Q&A flag from XML speaker type attribute
- `extract_transcript_paragraphs()`: Extracts paragraph-level content from XML transcript
- `process_transcript_file()`: Processes single transcript file and returns paragraph records
- `validate_and_preview_json()`: Validates JSON structure and previews first few records
- `save_extracted_content()`: Saves extracted paragraph records to NAS

#### Logging Functions
- `setup_logging()`: Sets up minimal console logging configuration
- `log_console()`: Logs minimal message to console
- `log_execution()`: Logs detailed execution information for main log file
- `log_error()`: Logs error information for error log file
- `save_logs_to_nas()`: Saves execution and error logs to NAS at completion

---

## Known Issues & Limitations

### Current Limitations
- **Sequential Processing**: Files processed sequentially (not parallelized)
- **Memory Usage**: Large XML files may use significant memory during processing
- **Development Mode Only**: Currently configured for development mode by default

### Known Issues - RESOLVED ✅
- **Configuration Key**: ✅ Uses `stage_03_extract_content` (not `stage_3`)
- **Output Filename**: ✅ Saves as `stage_03_extracted_content.json`
- **Filename Parsing**: ✅ Correctly extracts components from ticker_quarter_year_type_eventid_versionid.xml
- **Speaker Lookup**: ✅ Participant mapping correctly uses 'id' attribute and formats speaker strings
- **JSON Formatting**: ✅ Special character cleaning prevents JSON corruption

### Future Enhancements
- **Parallel Processing**: Multi-threaded file processing for better performance
- **Streaming XML Parser**: Handle very large XML files with streaming parsers
- **Content Validation**: Add paragraph content quality scoring
- **Caching**: Cache parsed participant data across similar transcripts

---

## Troubleshooting Guide

### Common Issues
**Issue**: No files found in processing queue
**Cause**: Stage 2 has not run successfully or files_to_process.json is empty
**Solution**: Verify Stage 2 completed successfully and generated processing queue

**Issue**: XML parsing failures
**Cause**: Malformed XML content or unexpected namespace changes
**Solution**: Check error logs for specific parsing errors, validate XML structure manually

**Issue**: Missing participant/speaker information
**Cause**: XML participant structure changes or missing speaker ID mappings
**Solution**: Review XML structure, check participant extraction logic uses 'id' attribute

**Issue**: Incorrect metadata extraction
**Cause**: File path structure changes or filename parsing issues
**Solution**: Verify file path format matches expected Data/YYYY/QX/Type/Company/filename.xml structure

**Issue**: JSON validation fails
**Cause**: Invalid JSON structure due to unescaped characters or formatting issues
**Solution**: Check validation output for specific error location and character issues

### Debugging Commands
```bash
# Debug configuration loading
python -c "from main_content_extraction import load_config_from_nas; print('Config loaded')"

# Test NAS connectivity
python -c "from main_content_extraction import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Validate environment
python -c "from main_content_extraction import validate_environment_variables; validate_environment_variables(); print('Valid')"
```

### Log Analysis
- **Execution Logs**: Located in Outputs/Logs/ with comprehensive operation tracking
- **Error Logs**: Separate error files in Outputs/Logs/Errors/ categorized by error type  
- **Debug Information**: Detailed file processing results and metadata extraction details

---

## Stage-Specific Context

### Unique Requirements
- **Enhanced Field Extraction**: Must extract all metadata that Stage 2 simplified out of its output
- **XML Content Processing**: Comprehensive parsing with participant mapping and content structure extraction
- **Development Mode Support**: Configurable file limits for testing and development

### Stage-Specific Patterns
- **Stage 2 Compatibility**: Processes simplified Stage 2 output format (file_path + date_last_modified)
- **Path-Based Metadata**: Extracts fiscal year/quarter, institution type, ticker from NAS folder structure
- **Filename Parsing**: Handles ticker_quarter_year_type_eventid_versionid.xml format
- **Config Integration**: Looks up company names from monitored_institutions configuration

### Integration Notes
- **Stage 2 Interface**: Consumes files_to_process.json with minimal field set
- **Stage 4 Interface**: Produces paragraph records with complete metadata for LLM classification
- **Field Bridging**: Acts as metadata bridge between Stage 2's simplicity and downstream requirements

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Project overview and context
- **Configuration Schema**: Stage 03 section in config.yaml
- **Script Location**: `database_refresh/03_extract_content/main_content_extraction.py`

### External References
- **XML Processing**: Python ElementTree documentation for XML parsing
- **NAS Operations**: Python pysmb library documentation
- **Security Standards**: Input validation and path security best practices

### Change Log
- **Version 1.0**: Original implementation with JSON config
- **Version 2.0**: Updated to follow Stage 0/1/2 patterns with YAML config and enhanced field extraction
- **Version 2.1**: Added JSON validation/preview, fixed filename parsing, resolved speaker lookup issues
- **Version 3.0**: Updated documentation to reflect actual implementation (main_content_extraction.py, stage_03_extract_content config key)

---

## Support & Maintenance

### Support Contacts
- **Primary Developer**: Claude Code Assistant
- **Technical Architecture**: Aligned with Stage 0/1/2 patterns
- **Operations**: Standard Stage 0/1/2 operational procedures

### Maintenance Schedule
- **Regular Updates**: Follow Stage 0/1/2 maintenance patterns
- **Security Reviews**: Same schedule as other pipeline stages
- **Performance Reviews**: Monitor processing time and content extraction quality

### Escalation Procedures
1. **Level 1**: Check error logs in Outputs/Logs/Errors/ for specific failure categories
2. **Level 2**: Verify Stage 2 outputs and NAS connectivity
3. **Level 3**: Review XML structure changes or filename format modifications

---

> **Implementation Notes:**
> 1. Stage 3 now bridges the gap between Stage 2's simplified output and downstream requirements
> 2. Extracts all metadata fields that were removed from Stage 2's output format
> 3. Maintains all Stage 0/1/2 architectural patterns and security standards
> 4. Output records include complete metadata ready for Stage 4 LLM classification
> 5. Development mode enables safe testing with limited file volumes
> 6. Script name is `main_content_extraction.py` (not `3_transcript_content_extraction.py`)
> 7. Configuration key is `stage_03_extract_content` (not `stage_3`)
> 8. Output file is `stage_03_extracted_content.json`