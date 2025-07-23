# Stage 0: Historical Transcript Sync - Context for Claude

> **Template Version**: 1.0 | **Created**: 2024-07-21  
> **Purpose**: Complete context for Stage 0 Historical Transcript Sync system

---

## Project Context
- **Stage**: 0 - Historical Transcript Sync
- **Primary Purpose**: Bulk download of historical earnings transcripts from FactSet API (3-year rolling window)
- **Pipeline Position**: Initial stage that creates the foundation transcript repository
- **Production Status**: PRODUCTION READY ✅ (All critical issues resolved)

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
import fds.sdk.EventsandTranscripts      # FactSet Events & Transcripts API
from fds.sdk.EventsandTranscripts.api import transcripts_api
import requests                          # Network operations & proxy handling
from smb.SMBConnection import SMBConnection  # NAS connectivity
import yaml                             # Configuration parsing
from dotenv import load_dotenv          # Environment variables
import xml.etree.ElementTree as ET      # XML parsing for title validation
import tempfile                         # SSL certificate management
import os, sys, re, time               # Core utilities
from datetime import datetime, timedelta  # Date calculations
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
CONFIG_PATH=                 # NAS path to configuration YAML
CLIENT_MACHINE_NAME=         # Client identification for NAS
SSL_CERT_PATH=              # NAS path to SSL certificate
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
stage_0_bulk_refresh/
├── 0_historical_transcript_sync.py    # Primary execution script (1,638 lines)
├── CLAUDE.md                          # This context file
├── old/                               # Previous version archive
│   ├── 0_transcript_bulk_sync.py      # Legacy script
│   └── CLAUDE.md                      # Legacy documentation
└── tests/                             # Future test implementations
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

### Configuration Schema
```yaml
# NAS configuration structure (config.yaml)
ssl_cert_path: "Inputs/certificate/rbc-ca-bundle.cer"
api_settings:
  industry_categories: 
    - "IN:BANKS"      # Banking institutions
    - "IN:FNLSVC"     # Financial services  
    - "IN:INS"        # Insurance companies
    - "IN:SECS"       # Securities/Asset management
  transcript_types: ["Corrected", "Raw"]  # Raw excluded by business rule
  request_delay: 3.0                      # Seconds between API calls
  max_retries: 8                          # Maximum retry attempts
  use_exponential_backoff: true           # Enable backoff strategy
  
stage_0:
  output_data_path: "Outputs/Data"        # NAS data output path
  output_logs_path: "Outputs/Logs"       # NAS logs output path
  
monitored_institutions:
  # Canadian Financial (6 institutions)
  RY-CA: {name: "Royal Bank of Canada", type: "Canadian"}
  TD-CA: {name: "Toronto-Dominion Bank", type: "Canadian"}
  BNS-CA: {name: "Bank of Nova Scotia", type: "Canadian"}
  BMO-CA: {name: "Bank of Montreal", type: "Canadian"}
  CM-CA: {name: "Canadian Imperial Bank of Commerce", type: "Canadian"}
  NA-CA: {name: "National Bank of Canada", type: "Canadian"}
  
  # US Financial (15 institutions)
  JPM-US: {name: "JPMorgan Chase & Co", type: "US"}
  BAC-US: {name: "Bank of America Corporation", type: "US"}
  WFC-US: {name: "Wells Fargo & Company", type: "US"}
  USB-US: {name: "U.S. Bancorp", type: "US"}
  PNC-US: {name: "PNC Financial Services Group", type: "US"}
  TFC-US: {name: "Truist Financial Corporation", type: "US"}
  BLK-US: {name: "BlackRock Inc", type: "US"}
  # ... [Additional US institutions]
  
  # European Financial (14 institutions)  
  BCS-GB: {name: "Barclays PLC", type: "European"}
  LLOY-GB: {name: "Lloyds Banking Group plc", type: "European"}
  UBS-CH: {name: "UBS Group AG", type: "European"}
  CS-CH: {name: "Credit Suisse Group AG", type: "European"}
  # ... [Additional European institutions]
  
  # Insurance (4 institutions)
  MFC-CA: {name: "Manulife Financial Corporation", type: "Insurance"}
  SLF-CA: {name: "Sun Life Financial Inc", type: "Insurance"}
  IFC-CA: {name: "Intact Financial Corporation", type: "Insurance"}
  FFH-CA: {name: "Fairfax Financial Holdings Limited", type: "Insurance"}
```

### Validation Requirements
- **Schema Validation**: YAML structure, data types, required fields
- **Security Validation**: Path validation, NAS path structure, URL sanitization
- **Business Rule Validation**: Ticker format, institution types, date ranges

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Security & Authentication Setup**: Validate 14 environment variables, establish NAS connection, download SSL certificate
2. **Configuration Loading & Validation**: Download YAML from NAS, validate comprehensive schema
3. **Directory Structure Creation**: Create fiscal quarter-organized folders with Windows compatibility
4. **Existing Inventory Scanning**: Scan all existing files, parse standardized filenames, identify unparseable files
5. **Rolling Window Calculation**: Calculate exact 3-year window from current date
6. **Institution Processing Loop**: Process 90+ institutions with anti-contamination filtering
7. **Transcript Comparison & Version Management**: Compare API vs NAS inventory, handle version updates
8. **Download Processing with Title Validation**: Download with strict title format validation
9. **File Management Operations**: Upload, remove out-of-scope files, handle version cleanup
10. **Completion & Audit**: Generate execution summary, save logs to NAS, cleanup resources

### Key Business Rules
- **Anti-Contamination Rule**: Only process transcripts where target ticker is SOLE primary company
- **Rolling Window Rule**: Download all transcripts from exactly 3 years ago to present
- **Title Validation Rule**: Only accept transcripts with exact format "Qx 20xx Earnings Call"
- **Version Management Rule**: API version is always authoritative, remove old versions automatically
- **Earnings Only Rule**: Filter to "Earnings" event types only (excludes M&A, special events)

### Data Processing Logic
- **Input**: FactSet Events & Transcripts API responses
- **Processing**: Anti-contamination filtering, title validation, version comparison
- **Output**: Standardized XML files in fiscal quarter-organized directory structure
- **Validation**: Title format validation, filename standardization, path length checking

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions implemented:
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    return not (".." in path or path.startswith("/") or ":" in path[1:])
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    allowed_prefixes = ["Inputs/", "Outputs/"]
    return any(path.startswith(prefix) for prefix in allowed_prefixes)
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    if "?" in url:
        return url.split("?")[0] + "?[PARAMETERS_REDACTED]"
    return url
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
python 0_historical_transcript_sync.py           # Run Stage 0 bulk sync
python -m py_compile 0_historical_transcript_sync.py  # Syntax validation
python -c "import yaml; print('Config valid')"   # YAML syntax check

# Environment validation
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Required vars:', [k for k in ['API_USERNAME', 'API_PASSWORD', 'NAS_USERNAME', 'NAS_PASSWORD'] if k in os.environ])"
```

### Execution Modes
- **Terminal Mode**: `python 0_historical_transcript_sync.py` - Standard execution
- **Notebook Mode**: Import and run functions from Jupyter notebooks
- **Scheduled Mode**: Can be executed via cron or task scheduler

### Testing Commands
```bash
# Configuration testing
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# NAS connectivity test  
python -c "from smb.SMBConnection import SMBConnection; print('SMB import successful')"

# Environment completeness check
python -c "import os; required=['API_USERNAME','API_PASSWORD','NAS_USERNAME','NAS_PASSWORD','PROXY_USER','PROXY_PASSWORD','PROXY_URL','NAS_SERVER_IP','NAS_SERVER_NAME','NAS_SHARE_NAME','CONFIG_PATH','CLIENT_MACHINE_NAME','SSL_CERT_PATH']; missing=[k for k in required if not os.getenv(k)]; print(f'Missing: {missing}' if missing else 'All required env vars present')"
```

---

## Error Handling & Recovery

### Error Categories
- **api_query**: FactSet API connection failures, authentication errors, rate limiting
- **download**: Individual transcript download failures, network timeouts
- **nas_connection**: NAS connectivity issues, authentication failures, path errors
- **directory_creation**: File system operation failures, permission errors
- **unparseable_filename**: Non-conforming filename formats requiring manual review
- **xml_parsing**: XML structure issues, title extraction failures

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
        "institution": current_institution,
        "operation": "file_upload",
        "path": target_path
    })
    
except (requests.ConnectionError, requests.Timeout) as e:
    log_error(f"Network error: {e}", "network", {
        "institution": current_institution,
        "retry_attempt": retry_count,
        "url": sanitize_url_for_logging(api_url)
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
  - Path Structure: `Data/YYYY/QX/Type/Company/TranscriptType/`
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
- **Function Naming**: `snake_case` with descriptive names (e.g., `download_ssl_certificate_from_nas`)
- **Class Structure**: Not applicable (functional design)
- **Module Organization**: Single monolithic script (1,638 lines) with clear section comments
- **Documentation**: Comprehensive inline comments and function docstrings

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: ALL paths, URLs, and API responses validated
- **Credential Protection**: NEVER log server IPs, auth tokens, or credentials
- **Error Handling**: NO bare except clauses - all exceptions specifically handled
- **Resource Management**: Proper cleanup in finally blocks, no resource leaks

### Quality Standards
- **Line Length**: Maximum 120 characters for readability
- **Function Length**: Most functions under 50 lines, complex business logic functions up to 100 lines
- **Complexity**: Clear separation of concerns with comprehensive error handling
- **Documentation**: All major functions have descriptive docstrings and inline comments

---

## Development Workflow

### Pre-Development Checklist
- [x] Environment variables configured in .env (14 required)
- [x] NAS access confirmed and tested
- [x] FactSet API credentials validated
- [x] Corporate proxy authentication configured
- [x] SSL certificate accessible from NAS

### Development Process
1. **Setup**: Load environment, validate credentials, establish connections
2. **Development**: Implement business logic with security validation
3. **Testing**: Unit testing for individual functions, integration testing with external systems
4. **Validation**: Configuration schema validation, security review, performance testing
5. **Deployment**: Production deployment with comprehensive audit trail

### Pre-Production Checklist (MANDATORY)
- [x] **Security Review**: All input validation implemented and tested
- [x] **Error Handling Review**: No bare except clauses, specific exception handling
- [x] **Resource Management Review**: Proper cleanup and connection management
- [x] **Configuration Validation**: Schema validation working with clear error messages
- [x] **Integration Testing**: FactSet API, NAS connectivity, proxy authentication tested

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
**Cause**: Expired or missing certificate on NAS
**Solution**: 
1. Verify certificate exists at configured NAS path
2. Check certificate expiration date
3. Update certificate file on NAS if needed
4. Restart script to download fresh certificate

**Issue**: Corporate proxy authentication failures  
**Cause**: Changed proxy credentials or domain configuration
**Solution**:
1. Verify PROXY_USER, PROXY_PASSWORD, PROXY_DOMAIN in .env
2. Test proxy connectivity with simple HTTP request
3. Check for special characters requiring URL encoding
4. Verify domain escaping (e.g., MAPLE\\username)

**Issue**: NAS connection timeouts
**Cause**: Network connectivity or authentication issues
**Solution**:
1. Verify NAS credentials in .env file
2. Test network connectivity to NAS_SERVER_IP
3. Check NTLM v2 authentication configuration
4. Verify CLIENT_MACHINE_NAME matches expected value

**Issue**: Title validation failures for transcripts
**Cause**: Transcript titles don't match expected format "Qx 20xx Earnings Call"
**Solution**:
1. Review error logs for specific title formats
2. Check if transcript type is correctly identified
3. Verify earnings call filtering is working
4. Consider expanding title validation patterns if needed

### Debugging Commands
```bash
# Debug configuration loading
python -c "import yaml; config = yaml.safe_load(open('/path/to/config.yaml')); print('Config loaded:', list(config.keys()))"

# Test NAS connectivity
python -c "from smb.SMBConnection import SMBConnection; conn = SMBConnection('user', 'pass', 'client', 'server'); print('NAS connection:', conn.connect('server_ip', 139))"

# Validate environment completeness
python -c "import os; required=['API_USERNAME','API_PASSWORD','NAS_USERNAME','NAS_PASSWORD']; print({k: 'SET' if os.getenv(k) else 'MISSING' for k in required})"

# Test FactSet API connectivity
python -c "import fds.sdk.EventsandTranscripts; print('FactSet SDK imported successfully')"
```

### Log Analysis
- **Execution Logs**: `Outputs/Logs/stage_0_historical_transcript_sync_YYYY-MM-DD_HH-MM-SS.json`
  - Contains detailed execution timeline, success/failure counts, performance metrics
- **Error Logs**: `Outputs/Logs/Errors/stage_0_historical_transcript_sync_errors_YYYY-MM-DD_HH-MM-SS.json`  
  - Categorized error entries with context, recovery suggestions, affected institutions
- **Debug Information**: Console output during execution with real-time progress updates

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
- **Comprehensive Error Categorization**: 6 distinct error types with specific handling and recovery
- **Inventory Management**: Pre-processing scan of existing files with version-aware duplicate detection

### Integration Notes
- **Previous Stage Interface**: None (initial stage)
- **Next Stage Interface**: Produces organized transcript repository for Stage 1 (Daily Sync) consumption
- **Parallel Processing**: Designed for safe re-execution without duplication

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Complete pipeline overview and standards
- **Template Reference**: `/CLAUDE_MD_TEMPLATE.md` - Universal template for all stages
- **Legacy Documentation**: `old/CLAUDE.md` - Previous version context and lessons learned

### External References
- **FactSet SDK Documentation**: Events and Transcripts API reference
- **Anthropic CLAUDE.md Best Practices**: Official Claude Code documentation
- **SMB/CIFS Protocol**: Python SMB library documentation
- **Corporate Security Standards**: Internal security validation requirements

### Change Log
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
> All 9 critical production issues have been resolved through rigorous development standards.