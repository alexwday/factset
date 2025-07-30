# Stage 0: Historical Transcript Sync - Context for Claude

> **Template Version**: 2.0 | **Updated**: 2025-01-30  
> **Purpose**: Complete context for Stage 0 Historical Transcript Sync system
> **Script**: `main_historical_sync.py`

---

## Project Context
- **Stage**: 0 - Historical Transcript Sync (Bulk Download)
- **Primary Purpose**: Download ALL historical earnings transcripts from FactSet API (3-year rolling window)
- **Pipeline Position**: Initial stage that creates the foundation transcript repository
- **Production Status**: PRODUCTION READY ✅ (All critical issues resolved)
- **Run Frequency**: Monthly/Quarterly for maintenance, Initial run for setup

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: FactSet SDK (fds.sdk.EventsandTranscripts)
- **Authentication**: Basic Auth (FactSet API) + NTLM v2 (NAS) + Domain Auth (Proxy)
- **Storage**: NAS (SMB/CIFS) with fiscal quarter organization
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for Stage 0
import os, tempfile, logging, json, time, io, re
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
import xml.etree.ElementTree as ET

# External packages
import yaml                                          # Configuration parsing
import fds.sdk.EventsandTranscripts                  # FactSet Events & Transcripts API
from fds.sdk.EventsandTranscripts.api import transcripts_api
import requests                                      # Network operations & proxy handling
from smb.SMBConnection import SMBConnection          # NAS connectivity
from dotenv import load_dotenv                       # Environment variables
```

### Environment Requirements
```bash
# Required .env variables (14 total)
API_USERNAME=                # FactSet API credentials
API_PASSWORD=
PROXY_USER=                  # Corporate proxy (MAPLE domain)
PROXY_PASSWORD=
PROXY_URL=
PROXY_DOMAIN=               # Defaults to MAPLE if not specified
NAS_USERNAME=                # NAS authentication (NTLM v2)
NAS_PASSWORD=
NAS_SERVER_IP=
NAS_SERVER_NAME=
NAS_SHARE_NAME=
NAS_BASE_PATH=
NAS_PORT=                   # Default: 445
CONFIG_PATH=                # NAS path to configuration YAML
CLIENT_MACHINE_NAME=        # Client identification for NAS
```

---

## Architecture & Design

### Core Design Patterns
- **Security-First Architecture**: Input validation and credential protection throughout
- **Resource Management Pattern**: Proper connection lifecycle with cleanup
- **Event-Driven Processing**: Institution-by-institution with comprehensive audit trail
- **Configuration-Driven Behavior**: External YAML configuration with schema validation

### File Structure
```
00_download_historical/
├── main_historical_sync.py    # Primary execution script (1,639 lines)
├── CLAUDE.md                  # This context file
└── old/                       # Previous version archive (for reference)
```

### Key Components
1. **Security & Authentication Manager**: Environment validation, NAS connection, SSL setup
2. **Configuration Loader & Validator**: YAML schema validation, institution verification  
3. **Directory Structure Manager**: Fiscal quarter organization with path validation
4. **Transcript Inventory Scanner**: Existing file analysis with version management
5. **API Query Engine**: FactSet API integration with filtering and rate limiting
6. **Download & Validation Engine**: Transcript download with title validation
7. **Audit & Logging System**: Comprehensive execution and error tracking

---

## Configuration Management

### Configuration Schema (config.yaml on NAS)
```yaml
# NAS configuration structure
ssl_cert_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/certificate/rbc-ca-bundle.cer"

api_settings:
  industry_categories: 
    - "IN:BANKS"      # Banking institutions
    - "IN:FNLSVC"     # Financial services  
    - "IN:INS"        # Insurance companies
    - "IN:SECS"       # Securities/Asset management
  transcript_types: ["Corrected", "Raw"]  # NearRealTime excluded
  sort_order: "-storyDateTime"            # Latest first
  pagination_limit: 1000
  pagination_offset: 0
  request_delay: 3.0                      # Seconds between API calls
  max_retries: 8                          # Maximum retry attempts
  retry_delay: 5.0
  use_exponential_backoff: true           # Enable backoff strategy
  max_backoff_delay: 120.0
  
stage_00_download_historical:
  description: "Historical bulk download of earnings transcripts"
  output_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Data"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"
  
monitored_institutions:
  # Canadian Financial (30+ institutions)
  RY-CA: {name: "Royal Bank of Canada", type: "Canadian", path_safe_name: "Royal_Bank_of_Canada"}
  TD-CA: {name: "Toronto-Dominion Bank", type: "Canadian", path_safe_name: "Toronto_Dominion_Bank"}
  BNS-CA: {name: "Bank of Nova Scotia", type: "Canadian", path_safe_name: "Bank_of_Nova_Scotia"}
  BMO-CA: {name: "Bank of Montreal", type: "Canadian", path_safe_name: "Bank_of_Montreal"}
  CM-CA: {name: "Canadian Imperial Bank of Commerce", type: "Canadian", path_safe_name: "CIBC"}
  NA-CA: {name: "National Bank of Canada", type: "Canadian", path_safe_name: "National_Bank_of_Canada"}
  
  # US Financial (35+ institutions)
  JPM-US: {name: "JPMorgan Chase", type: "US", path_safe_name: "JPMorgan_Chase"}
  BAC-US: {name: "Bank of America", type: "US", path_safe_name: "Bank_of_America"}
  WFC-US: {name: "Wells Fargo", type: "US", path_safe_name: "Wells_Fargo"}
  C-US: {name: "Citigroup", type: "US", path_safe_name: "Citigroup"}
  GS-US: {name: "Goldman Sachs", type: "US", path_safe_name: "Goldman_Sachs"}
  MS-US: {name: "Morgan Stanley", type: "US", path_safe_name: "Morgan_Stanley"}
  
  # European Financial (30+ institutions)  
  BCS-GB: {name: "Barclays", type: "European", path_safe_name: "Barclays"}
  LLOY-GB: {name: "Lloyds Banking Group", type: "European", path_safe_name: "Lloyds"}
  HSBC-GB: {name: "HSBC", type: "European", path_safe_name: "HSBC"}
  UBS-CH: {name: "UBS", type: "European", path_safe_name: "UBS"}
  CS-CH: {name: "Credit Suisse", type: "European", path_safe_name: "Credit_Suisse"}
  
  # Insurance (5+ institutions)
  MFC-CA: {name: "Manulife Financial", type: "Insurance", path_safe_name: "Manulife"}
  SLF-CA: {name: "Sun Life Financial", type: "Insurance", path_safe_name: "Sun_Life"}
  IFC-CA: {name: "Intact Financial", type: "Insurance", path_safe_name: "Intact"}
  FFH-CA: {name: "Fairfax Financial", type: "Insurance", path_safe_name: "Fairfax"}
```

### Validation Requirements
- **Schema Validation**: YAML structure, data types, required fields
- **Security Validation**: Path validation, NAS path structure, URL sanitization
- **Business Rule Validation**: Ticker format, institution types, date ranges

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Security & Authentication Setup** (validate_environment_variables, get_nas_connection)
2. **Configuration Loading & Validation** (load_config_from_nas, validate_config_structure)
3. **SSL & Proxy Setup** (setup_ssl_certificate, setup_proxy_configuration)
4. **FactSet API Client Setup** (setup_factset_api_client)
5. **Directory Structure Creation** (create_data_directory_structure)
6. **Existing Inventory Scanning** (scan_existing_transcripts)
7. **Rolling Window Calculation** (calculate_rolling_window - 3 years)
8. **Institution Processing Loop** (Process 100+ institutions with filtering)
9. **Transcript Comparison & Version Management** (compare_transcripts)
10. **Download & Title Validation** (download_transcript_with_title_filtering)
11. **Completion & Audit Trail** (save_logs_to_nas, cleanup_temporary_files)

### Key Business Rules
- **Anti-Contamination Rule**: Only process transcripts where target ticker is SOLE primary company ID
- **3-Year Rolling Window**: Download all transcripts from exactly 3 years ago to present
- **Title Validation Rule**: Only accept transcripts with exact format "Qx 20xx Earnings Call"
- **Version Management Rule**: API version is always authoritative, remove old versions automatically
- **Earnings Only Rule**: Filter to "Earnings" event types only (excludes M&A, special events)
- **Filename Standardization**: `{ticker}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml`

### Data Processing Logic
- **Input**: FactSet Events & Transcripts API (get_transcripts_ids endpoint)
- **Processing**: Anti-contamination filtering, title validation, version comparison
- **Output**: Standardized XML files in fiscal quarter-organized directory structure
- **Validation**: Title format validation, filename standardization, path length checking

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions - ALL IMPLEMENTED in script:

def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    if not path or not isinstance(path, str):
        return False
    if ".." in path or path.startswith("/"):
        return False
    # Check for invalid characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
    if any(char in path for char in invalid_chars):
        return False
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False
    return True
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    if not path or not isinstance(path, str):
        return False
    normalized = path.strip("/")
    if not normalized:
        return False
    parts = normalized.split("/")
    for part in parts:
        if not part or part in [".", ".."]:
            return False
        if not validate_file_path(part):
            return False
    return True
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    if not url:
        return url
    sanitized = re.sub(r"(password|token|auth)=[^&]*", r"\1=***", url, flags=re.IGNORECASE)
    sanitized = re.sub(r"://[^@]*@", "://***:***@", sanitized)
    return sanitized
```

### Security Standards Checklist
- [x] All input validation implemented (paths, URLs, API responses)
- [x] No credential exposure in logs (URLs sanitized, IPs not logged)
- [x] File paths validated against directory traversal
- [x] URLs sanitized before logging  
- [x] Configuration schema validated with comprehensive checks

### Compliance Requirements
- **Audit Trail**: Timestamped execution logs with complete operation history
- **Data Retention**: 3-year rolling window with automatic cleanup of out-of-scope files
- **Error Tracking**: Comprehensive error categorization and JSON export

---

## Commands & Operations

### Primary Commands
```bash
# Development and testing
python main_historical_sync.py                    # Run Stage 0 bulk sync
python -m py_compile main_historical_sync.py     # Syntax validation

# Environment validation
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('ENV vars loaded')"

# Configuration validation
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

### Execution Modes
- **Terminal Mode**: `python main_historical_sync.py` - Standard execution
- **Notebook Mode**: Import and run functions from Jupyter notebooks
- **Scheduled Mode**: Can be executed via cron or task scheduler

### Testing Commands
```bash
# NAS connectivity test  
python -c "from smb.SMBConnection import SMBConnection; print('SMB import successful')"

# Environment completeness check (all 14 required)
python -c "import os; from dotenv import load_dotenv; load_dotenv(); required=['API_USERNAME','API_PASSWORD','NAS_USERNAME','NAS_PASSWORD','PROXY_USER','PROXY_PASSWORD','PROXY_URL','NAS_SERVER_IP','NAS_SERVER_NAME','NAS_SHARE_NAME','NAS_BASE_PATH','NAS_PORT','CONFIG_PATH','CLIENT_MACHINE_NAME']; missing=[k for k in required if not os.getenv(k)]; print(f'Missing: {missing}' if missing else 'All required env vars present')"

# FactSet SDK test
python -c "import fds.sdk.EventsandTranscripts; print('FactSet SDK imported successfully')"
```

---

## Error Handling & Recovery

### Error Categories
- **environment_validation**: Missing environment variables
- **nas_connection**: NAS connectivity issues, authentication failures
- **config_validation**: Configuration schema errors, missing sections
- **api_query**: FactSet API failures, rate limiting
- **download**: Individual transcript download failures
- **directory_creation**: File system operation failures
- **unparseable_filename**: Non-conforming filename formats
- **xml_parsing**: Title extraction failures

### Error Handling Standards (MANDATORY)
```python
# NEVER DO THIS - CAUSES PRODUCTION FAILURES:
except:
    pass
    
except Exception:
    return False

# ALWAYS DO THIS - SPECIFIC ERROR HANDLING:
except (OSError, FileNotFoundError) as e:
    log_error(f"File operation failed: {e}", "file_operation", {
        "path": file_path,
        "operation": "download",
        "error_details": str(e)
    })
    
except (requests.ConnectionError, requests.Timeout) as e:
    log_error(f"Network error: {e}", "network", {
        "url": sanitize_url_for_logging(url),
        "retry_attempt": attempt,
        "error_details": str(e)
    })
```

### Recovery Mechanisms
- **Retry Logic**: Exponential backoff with maximum 8 retries and 120-second ceiling
- **Fallback Strategies**: Continue processing other institutions on individual failures
- **Error Reporting**: Detailed JSON error logs uploaded to NAS for analysis

---

## Integration Points

### Upstream Dependencies
- **Configuration Source**: NAS-based YAML configuration file
- **SSL Certificate**: Downloaded from NAS at runtime (temporary file)
- **Environment Variables**: 14 required credentials from .env file

### Downstream Outputs  
- **Next Stage**: Organized transcript files for Stage 1 (Daily Sync) and Stage 2 (Consolidation)
- **File Outputs**: Standardized XML files in fiscal quarter-organized structure
- **Log Outputs**: Execution logs and error logs saved to NAS

### External System Integration
- **FactSet API**: EventsandTranscripts SDK with Basic Authentication
  - Endpoint: `transcripts_api.TranscriptsApi.get_transcripts_ids()`
  - Rate Limiting: 3-second delays, exponential backoff on errors
  - Filtering: Primary ID filtering, earnings-only, 3-year window
- **NAS File System**: SMB/CIFS operations with NTLM v2 authentication
  - Operations: Upload, download, directory creation, file removal
  - Path Structure: `Data/YYYY/QX/Type/Company/filename.xml`
- **Corporate Proxy**: NTLM authentication with domain escaping
  - Domain escaping for special characters in credentials
  - SSL certificate validation via downloaded certificate

---

## Performance & Monitoring

### Performance Considerations
- **Rate Limiting**: 3-second delays between API requests (FactSet requirement)
- **Memory Management**: Stream processing for large files, immediate cleanup
- **Connection Management**: Single connection per session, proper lifecycle management
- **Resource Cleanup**: SSL certificate cleanup, connection closure in finally blocks

### Monitoring Points
- **Execution Metrics**: Total transcripts processed, download success rate, execution time
- **Error Rates**: API errors, download failures, NAS connectivity issues
- **Performance Metrics**: Average request time, retry frequency, memory usage
- **Resource Utilization**: Connection pool usage, temporary file management

---

## Code Style & Standards

### Code Organization
- **Function Naming**: `snake_case` with descriptive names (e.g., `download_transcript_with_title_filtering`)
- **Module Organization**: Single comprehensive script with clear section comments
- **Documentation**: Comprehensive inline comments and function docstrings
- **Type Hints**: Used throughout for better code clarity

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: ALL paths, URLs, and API responses validated
- **Credential Protection**: NEVER log server IPs, auth tokens, or credentials
- **Error Handling**: NO bare except clauses - all exceptions specifically handled
- **Resource Management**: Proper cleanup in finally blocks, no resource leaks

### Quality Standards
- **Line Length**: Reasonable lengths for readability
- **Function Length**: Most functions under 50 lines, complex business logic up to 100 lines
- **Complexity**: Clear separation of concerns with comprehensive error handling
- **Documentation**: All major functions have descriptive docstrings

---

## Development Workflow

### Pre-Development Checklist
- [ ] Environment variables configured in .env (14 required)
- [ ] NAS access confirmed and tested
- [ ] FactSet API credentials validated
- [ ] Corporate proxy authentication configured
- [ ] SSL certificate accessible from NAS

### Development Process
1. **Setup**: Load environment, validate credentials, establish connections
2. **Development**: Implement business logic with security validation
3. **Testing**: Test with small institution subset first
4. **Validation**: Configuration schema validation, security review
5. **Deployment**: Production deployment with comprehensive audit trail

### Pre-Production Checklist (MANDATORY)
- [x] **Security Review**: All input validation implemented
- [x] **Error Handling Review**: No bare except clauses
- [x] **Resource Management Review**: Proper cleanup in finally blocks
- [x] **Configuration Validation**: Schema validation working
- [x] **Integration Testing**: FactSet API, NAS connectivity tested

---

## Known Issues & Limitations

### Current Limitations
- **Single-threaded Processing**: Institution processing is sequential (designed for reliability)
- **Memory Usage**: Large XML files loaded entirely into memory during processing
- **API Rate Limiting**: 3-second delays required between requests (FactSet limitation)

### Known Issues
- **Network Dependency**: Requires stable internet connection for FactSet API and NAS access
- **Proxy Authentication**: Corporate proxy changes may require credential updates
- **SSL Certificate Management**: Certificate updates require manual NAS file replacement

### Future Enhancements
- **Parallel Processing**: Multi-threaded institution processing for improved performance
- **Incremental Processing**: More sophisticated comparison to reduce redundant operations
- **Enhanced Monitoring**: Real-time progress monitoring and alerting capabilities

---

## Troubleshooting Guide

### Common Issues

**Issue**: SSL certificate validation errors
- **Cause**: Expired or missing certificate on NAS
- **Solution**: Verify certificate exists at configured NAS path, check expiration, update if needed

**Issue**: Corporate proxy authentication failures  
- **Cause**: Changed proxy credentials or domain configuration
- **Solution**: Verify PROXY_USER, PROXY_PASSWORD, PROXY_DOMAIN in .env

**Issue**: NAS connection timeouts
- **Cause**: Network connectivity or authentication issues
- **Solution**: Verify NAS credentials, test network connectivity to NAS_SERVER_IP

**Issue**: Title validation failures for transcripts
- **Cause**: Transcript titles don't match expected format "Qx 20xx Earnings Call"
- **Solution**: Review error logs for specific title formats, verify earnings call filtering

### Debugging Commands
```bash
# Debug configuration loading
python -c "import yaml; config = yaml.safe_load(open('config.yaml')); print('Config sections:', list(config.keys()))"

# Test NAS connectivity
python -c "from smb.SMBConnection import SMBConnection; import os; from dotenv import load_dotenv; load_dotenv(); conn = SMBConnection(os.getenv('NAS_USERNAME'), os.getenv('NAS_PASSWORD'), os.getenv('CLIENT_MACHINE_NAME'), os.getenv('NAS_SERVER_NAME')); print('NAS connection test:', conn.connect(os.getenv('NAS_SERVER_IP'), int(os.getenv('NAS_PORT', 445))))"

# Validate specific environment variable
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API_USERNAME:', 'SET' if os.getenv('API_USERNAME') else 'NOT SET')"
```

### Log Analysis
- **Execution Logs**: `Outputs/Logs/stage_00_download_historical_transcript_sync_YYYY-MM-DD_HH-MM-SS.json`
- **Error Logs**: `Outputs/Logs/Errors/stage_00_download_historical_transcript_sync_errors_YYYY-MM-DD_HH-MM-SS.json`
- **Console Output**: Real-time progress updates during execution

---

## Stage-Specific Context

### Unique Requirements
- **3-Year Rolling Window**: Maintains exactly 3 years of historical data with automatic cleanup
- **Fiscal Quarter Organization**: Dynamic directory structure based on parsed transcript titles
- **Anti-Contamination Filtering**: Ensures ticker isolation by checking primary ID arrays
- **Version Management**: Automatically handles vendor version updates with old version cleanup
- **Title Format Validation**: Strict validation of "Qx 20xx Earnings Call" format

### Stage-Specific Patterns
- **Standardized Filename Format**: `{ticker}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml`
- **Institution Category Organization**: Separate folders for Canadian, US, European, and Insurance companies
- **Comprehensive Error Categorization**: Multiple distinct error types with specific handling
- **Inventory Management**: Pre-processing scan of existing files with version-aware duplicate detection

### Integration Notes
- **Previous Stage Interface**: None (initial stage)
- **Next Stage Interface**: Produces organized transcript repository for Stage 1 (Daily Sync) consumption
- **Parallel Processing**: Designed for safe re-execution without duplication

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Complete pipeline overview and standards
- **Legacy Documentation**: `old/CLAUDE.md` - Previous version context and lessons learned

### External References
- **FactSet SDK Documentation**: Events and Transcripts API reference
- **SMB/CIFS Protocol**: Python SMB library documentation
- **Corporate Security Standards**: Internal security validation requirements

### Change Log
- **Version 2.0** (2025-01-30): Updated filename to main_historical_sync.py, aligned with current codebase
- **Version 1.0**: Complete rewrite with security hardening and comprehensive error handling
- **Legacy Version**: Archived in `old/` folder with historical context

---

## Support & Maintenance

### Support Contacts
- **Primary Developer**: Stage 0 development team
- **Technical Lead**: Pipeline architecture team  
- **Operations Team**: Production support and monitoring

### Maintenance Schedule
- **Daily Monitoring**: Automated execution logs review
- **Weekly Reviews**: Performance metrics and error rate analysis
- **Monthly Updates**: Configuration updates and institution list maintenance
- **Quarterly Reviews**: Security validation and compliance review

### Escalation Procedures
1. **Level 1**: Check logs, verify connectivity, validate configuration
2. **Level 2**: Advanced debugging, credential validation, system integration testing
3. **Level 3**: Architecture review, external system coordination, security incident response

---

> **Stage 0 Production Notes:**
> This script represents a mature, production-ready implementation with comprehensive security,
> error handling, and business logic validation. It successfully addresses the complexities of
> financial data processing while maintaining high standards for security and reliability.
> All critical production issues have been resolved through rigorous development standards.