# FactSet Earnings Transcript Pipeline - Context for Claude

## Project Overview
This is a multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API. The pipeline is designed with standalone stage-based scripts that can run independently from terminal or notebook environments.

## Current Status
- **Stage 0**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - PRODUCTION READY ✅ (All 9 critical issues resolved)
- **Reference Implementation**: Production-ready code with comprehensive validation, security hardening, and proper error handling
- **Architecture**: Stage-based approach where each stage is completely independent and follows security standards
- **Storage**: All data stored on NAS (no local storage) using pysmb with NTLM v2 authentication
- **Authentication**: Corporate proxy authentication required for API access

### Stage 0 Key Business Rules
1. **Anti-Contamination**: Only downloads transcripts where target ticker is SOLE primary company
2. **Security-First**: All credentials from environment, paths validated, URLs sanitized in logs
3. **Audit Trail**: Every operation logged with timestamps and error context
4. **Incremental Safe**: Checks existing files, safe to re-run without re-downloading
5. **Version Management**: Automatically handles vendor version ID updates, keeps only latest version
6. **Institution Coverage**: 47 global financial institutions across 4 regions (Canadian, US, European, Insurance)
7. **File Organization**: Organized by institution type → company → transcript type
8. **Standardized Naming**: Consistent filename format for easy identification and processing

## Key Technical Requirements

### Authentication & Configuration
- **Environment Variables (.env)**: 13 required variables
  - API_USERNAME, API_PASSWORD (FactSet API)
  - PROXY_USER, PROXY_PASSWORD, PROXY_URL, PROXY_DOMAIN (Corporate proxy - PROXY_DOMAIN defaults to 'MAPLE')
  - NAS_USERNAME, NAS_PASSWORD, NAS_SERVER_IP, NAS_SERVER_NAME (NAS connection)
  - NAS_SHARE_NAME, NAS_BASE_PATH (NAS paths)
  - CONFIG_PATH, CLIENT_MACHINE_NAME (Configuration)
- **NAS Config Files**: Single `config.json` with stage-specific sections
- **SSL Certificate**: Downloaded from NAS at runtime for FactSet API connections
- **Proxy**: Corporate proxy with NTLM authentication required

### FactSet API Integration
- **API**: EventsandTranscripts API using fds.sdk.EventsandTranscripts
- **Endpoints**: transcripts_api.TranscriptsApi for transcript downloads
- **Filtering**: Primary ID filtering to prevent cross-contamination between related companies
- **Rate Limiting**: 2-second delays between requests, 3 retry attempts with 5-second delays

### NAS Integration Details
- **Protocol**: SMB/CIFS using pysmb.SMBConnection
- **Authentication**: NTLM v2 with domain credentials
- **Enhanced Folder Structure** (Fiscal Quarter Organization):
  - `Inputs/certificate/` - SSL certificates
  - `Inputs/config/` - Configuration files (single config.json)
  - `Outputs/Data/YYYY/QX/Type/Company/TranscriptType/` - Transcripts organized by fiscal year and quarter
  - `Outputs/Logs/` - Execution logs
  - `Outputs/Logs/Errors/` - Separate error logs (parsing, download, filesystem, validation)
  - `Outputs/listing/` - Inventory JSON files (future use)

### Monitored Institutions (47 Total)
**Canadian Banks (8)**: RY-CA, TD-CA, BNS-CA, BMO-CA, CM-CA, NA-CA, LB-CA, CWB-CA  
**US Banks (11)**: JPM-US, BAC-US, WFC-US, C-US, GS-US, MS-US, USB-US, PNC-US, TFC-US, COF-US, SCHW-US  
**European Banks (17)**: BCS-GB, LLOY-GB, RBS-GB, HSBA-GB, STAN-GB, DBK-DE, CBK-DE, BNP-FR, ACA-FR, GLE-FR, SAN-ES, BBVA-ES, ISP-IT, UCG-IT, ING-NL, UBS-CH, CSGN-CH  
**Insurance (10)**: MFC-CA, SLF-CA, GWO-CA, IFC-CA, FFH-CA, UNH-US, BRK.A-US, AIG-US, TRV-US, PGR-US  
**Other (2)**: PDO-CA, AKBM-NO

### File Naming Convention
`{primary_id}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`

**Example**: `RY-CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml`
- **RY-CA**: Royal Bank of Canada ticker
- **2024-01-25**: Earnings call date
- **Earnings**: Event type (filtered to earnings only)
- **Corrected**: Transcript type (Raw/Corrected/NearRealTime)
- **12345**: Unique event identifier
- **67890**: Report identifier
- **1**: Version identifier

**Storage Path**: `Data/2024/Q1/Canadian/RY-CA_Royal_Bank_of_Canada/Corrected/RY-CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml`

## Stage 0 Business Logic & Process Flow

### Complete Step-by-Step Process

#### **1. Security & Authentication Setup**
- Validates all required credentials from `.env` file (13 environment variables)
- Connects to NAS using secure NTLM v2 authentication
- Downloads SSL certificate from NAS for secure FactSet API connections
- Configures corporate proxy authentication for API access with domain escaping

#### **2. Configuration Loading & Validation**
- Downloads `config.json` from NAS at runtime (never stored locally)
- Validates comprehensive configuration schema:
  - Institution list validation (ticker format, institution types)
  - API parameter validation (dates, transcript types, rate limits)
  - Security path validation to prevent directory traversal attacks
  - Data type validation for all configuration parameters

#### **3. Enhanced Directory Structure Creation**
Creates fiscal quarter-organized folder structure on NAS:
```
Outputs/Data/
├── 2024/                    # Fiscal year folders
│   ├── Q1/                  # Fiscal quarter folders
│   │   ├── Canadian/        # Institution type
│   │   │   └── RY-CA_Royal_Bank_of_Canada/
│   │   │       ├── Raw/     # Transcript type subfolders
│   │   │       ├── Corrected/
│   │   │       └── NearRealTime/
│   │   ├── US/
│   │   │   └── JPM-US_JPMorgan_Chase/
│   │   └── Insurance/
│   │       └── MFC-CA_Manulife_Financial/
│   ├── Q2/ Q3/ Q4/         # Other quarters in same structure
├── 2023/                   # Previous years
└── Unknown/                # Fallback for unparseable transcripts
    └── Unknown/
```

#### **4. Institution Processing Loop (47 Total)**
For each monitored institution:

**a) API Query with Critical Filters**
- Queries FactSet Events & Transcripts API for ALL transcripts from 2023-present
- Applies earnings filter (only "Earnings" event types)
- **CRITICAL Anti-Contamination Filter**: Only processes transcripts where the target ticker is the SOLE primary company
  - Prevents cross-contamination (e.g., NA-CA pulling CWB-CA transcripts)
  - Ensures clean data for individual institution analysis

**b) Transcript Type Processing**
- **Raw**: Original transcript as received from live call
- **Corrected**: Manually edited for accuracy, completeness, and formatting
- **NearRealTime**: Real-time transcript captured during live earnings call

**c) Duplicate Prevention Logic**
- Scans existing files on NAS for each institution/transcript-type combination
- Creates standardized filename: `{ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`
- Only downloads transcripts that don't already exist (safe to re-run)

**d) Download & Storage Process**
- Downloads transcript XML via FactSet API with corporate proxy authentication
- Validates file integrity and logs file sizes for audit
- Uploads directly to NAS (no local storage for security)
- Implements retry logic: 3 attempts with 5-second delays
- Rate limiting: 2-second delays between requests (API protection)

#### **5. Progress Tracking & Audit Trail**
- Tracks downloads by institution and transcript type
- Logs detailed progress, successes, and failures
- Maintains counts for reporting and audit purposes
- Generates comprehensive execution summary

#### **6. Final Reporting & Cleanup**
- Provides detailed summary:
  - Total transcripts downloaded
  - Execution time
  - Institutions with/without transcripts found
  - Breakdown by transcript type and institution
- Uploads timestamped execution log to NAS for permanent audit trail
- Cleans up temporary SSL certificate files
- Properly closes all network connections

### Core Functionality Features
- Downloads all earnings transcripts from 2023-present for monitored institutions
- **Enhanced Fiscal Quarter Organization**: Automatically organizes transcripts by fiscal year and quarter parsed from XML titles
- **Robust Title Parsing**: 4 regex patterns handle various formats ("Q1 2024 Earnings Call", "First Quarter 2024", "2024 Q1")
- Filters transcripts where target ticker is the ONLY primary ID (prevents cross-contamination)
- Supports all transcript types: Corrected, Raw, NearRealTime
- **Comprehensive Error Logging**: Separate JSON files for parsing, download, filesystem, and validation errors
- **Smart Fallbacks**: Uses "Unknown/Unknown" folder for unparseable transcripts with operator guidance
- **Windows Path Compatibility**: Validates path lengths and automatically shortens when needed
- Implements intelligent version management to handle vendor version ID updates
- Automatically removes old versions and keeps only latest version of each transcript
- Uses version-agnostic keys for duplicate detection (prevents duplicate downloads)
- Rate limiting with retry logic for API protection
- Comprehensive error handling and security validation
- Uploads all data directly to NAS (no local storage)

### Technical Implementation
- **Import Pattern**: `from fds.sdk.EventsandTranscripts.api import transcripts_api`
- **Enhanced Folder Structure**: Dynamic creation based on parsed fiscal quarter/year from XML titles
- **Quarter/Year Parsing**: Key functions for extracting temporal information:
  - `parse_quarter_and_year_from_xml()`: Parses XML content to extract Q1-Q4 and 20XX from titles
  - `validate_path_length()`: Ensures Windows compatibility (260 char limit)
  - `get_fallback_quarter_year()`: Returns "Unknown/Unknown" for unparseable transcripts
  - `create_enhanced_directory_structure()`: Creates fiscal quarter-organized paths
- **Enhanced Error Logging**: Separate tracking and JSON export for different error types
  - `EnhancedErrorLogger`: Manages parsing, download, filesystem, and validation errors
  - Saves actionable error reports to `Outputs/Logs/Errors/` with recovery instructions
- **Proxy Setup**: Uses requests proxies with MAPLE domain authentication
- **SSL Handling**: Downloads certificate from NAS to temp file, sets environment variables
- **Version Management**: Key functions for handling vendor version updates:
  - `create_version_agnostic_key()`: Creates duplicate detection keys without version_id
  - `parse_version_from_filename()`: Extracts version numbers from filenames
  - `get_existing_files_with_version_management()`: Automatically manages versions, removes old ones
  - `nas_remove_file()`: Safely removes old versions from NAS
- **Error Handling**: Pandas warnings suppressed, comprehensive try/catch blocks
- **Logging**: Simplified logging without email notifications (removed in latest version)

## Known Issues & Fixes Applied

### Historical Problems Solved
1. **Import Error**: Fixed `calendar_api` vs `calendar_events_api` import confusion
2. **Sort Parameter**: Changed from `"-event_date"` to `"-storyDateTime"`
3. **Pandas Warnings**: Fixed SettingWithCopyWarning with explicit `.copy()`
4. **Cross-contamination**: NA-CA pulling CWB-CA transcripts fixed with primary ID filtering
5. **Proxy URL**: Fixed proxy construction for requests library format
6. **Environment Variables**: Attempted but reverted - broke functionality, restored from git
7. **Version ID Duplicates**: Fixed duplicate downloads when vendors update version IDs by implementing version-agnostic duplicate detection and automatic cleanup of old versions
8. **Enhanced Folder Structure**: Implemented fiscal quarter organization with robust title parsing and comprehensive error logging (2024-07-12)

### Git History Context
- **Last Working Commit**: 1d0ceba (before environment variable changes)
- **Email Removal**: Commit removed email functionality, added type annotations, simplified logging
- **Line Count**: Reduced from 855 to 551 lines in cleanup

## Dependencies (requirements.txt)
```
fds.sdk.EventsandTranscripts
pandas
requests
python-dotenv
pysmb
python-dateutil
openai  # Required for Stage 4 LLM classification
pytz    # Required for test scripts
# Enhanced folder structure parsing:
xml.etree.ElementTree  # Built-in Python library for XML parsing
```

## Stage Architecture Plan

### Stage 0: Bulk Refresh ✅ PRODUCTION READY
- **Purpose**: Download ALL historical earnings transcripts from 2023-present for 47 monitored financial institutions
- **Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py`
- **Config**: `Inputs/config/config.json` on NAS
- **When**: Initial setup or complete repository refresh
- **Status**: All 9 critical issues resolved, security validated, production ready
- **Key Features**:
  - **Enhanced Fiscal Quarter Organization**: Automatically organizes by fiscal year/quarter parsed from XML titles
  - **Robust Title Parsing**: 4 regex patterns handle "Q1 2024 Earnings Call", "First Quarter 2024", "2024 Q1" formats
  - **Comprehensive Error Logging**: Separate JSON files for parsing, download, filesystem, validation errors
  - **Smart Fallbacks**: Uses "Unknown/Unknown" folder for unparseable transcripts with operator guidance
  - Anti-contamination filter: Only downloads transcripts where target ticker is SOLE primary company
  - Version management: Automatically handles vendor version ID updates, prevents duplicate downloads
  - Duplicate prevention: Uses version-agnostic keys for intelligent duplicate detection
  - Automatic cleanup: Removes old versions and keeps only latest version of each transcript
  - **Windows Path Compatibility**: Validates path lengths and automatically shortens when needed
  - Comprehensive audit trail: Timestamped logs uploaded to NAS
  - Standardized naming: `{ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`

### Stage 1: Daily Sync ✅ PRODUCTION READY
- **Purpose**: Incremental daily downloads  
- **Script**: `stage_1_daily_sync/1_transcript_daily_sync.py`
- **Config**: `Inputs/config/config.json` on NAS (uses stage_1 section)
- **When**: Scheduled daily operations or earnings day monitoring
- **Key Features**:
  - **Enhanced Fiscal Quarter Organization**: Same dynamic folder structure as Stage 0 (YYYY/QX/Type/Company/TranscriptType/)
  - **Comprehensive Error Logging**: Inherits all error tracking and JSON export capabilities
  - Date-based queries using `get_transcripts_dates()` API
  - Single API call per date (efficient vs Stage 0's 15 calls)
  - Inherits ALL version management from Stage 0
  - **Smart Title Parsing**: Same 4 regex patterns and fallback logic as Stage 0
  - Optional `earnings_monitor.py` for real-time notifications
  - Configurable lookback period (sync_date_range)

### Stage 2: Transcript Consolidation ✅ PRODUCTION READY
- **Purpose**: Consolidate and organize downloaded transcripts, create master database  
- **Script**: `stage_2_processing/2_transcript_consolidation.py`
- **Config**: `Inputs/config/config.json` on NAS (uses stage_2 section)
- **When**: After bulk download or periodic consolidation
- **Key Features**:
  - Enhanced file selection algorithm with intelligent duplicate handling
  - Multi-call earnings transcript consolidation (e.g., preliminary + corrected calls)
  - Master database creation with comprehensive transcript metadata
  - Version-aware file processing with automatic latest version selection
  - Comprehensive error logging with separate JSON files for different error types
  - Read-only approach - validates and consolidates without modifying original files
  - Smart fallback handling for edge cases and parsing failures

### Stage 3: Content Processing ✅ PRODUCTION READY
- **Purpose**: Extract and structure content from XML transcripts
- **Script**: `stage_3_content_processing/3_transcript_content_extraction.py`
- **Config**: `Inputs/config/config.json` on NAS (uses stage_3 section)
- **When**: After transcript consolidation
- **Key Features**:
  - Paragraph-level content extraction from XML transcripts
  - Speaker attribution and role identification (Operator, Executive, Analyst)
  - Q&A session detection and separation
  - Structured content output with speaker metadata
  - Comprehensive error logging and validation
  - Preserves original XML structure while extracting meaningful content

### Stage 4: LLM Classification ✅ PRODUCTION READY
- **Purpose**: LLM-based transcript section classification and analysis
- **Script**: `stage_4_llm_classification/4_transcript_llm_classification.py`
- **Config**: `Inputs/config/config.json` on NAS (uses stage_4 section)
- **When**: After content extraction for detailed analysis
- **Key Features**:
  - 3-level progressive classification system (Management Discussion vs Investor Q&A)
  - OAuth 2.0 authentication for secure LLM API access
  - CO-STAR prompt framework for consistent, structured responses
  - Comprehensive cost tracking with configurable token rates and real-time budget monitoring
  - Full section context with paragraph-level character limits (750 chars per paragraph)
  - Comprehensive error handling with retry logic and rate limiting
  - JSON-structured output with confidence scoring
  - Production-ready with full security validation and cost tracking

## MANDATORY Development Standards

> **CRITICAL**: These standards prevent security vulnerabilities and production failures. They were developed from resolving 9 critical issues in Stage 0.

### 🚨 Security-First Requirements (NON-NEGOTIABLE)

#### Input Validation Framework
ALL scripts MUST include these validation functions:
```python
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    
def validate_api_response_structure(response) -> bool:
    """Validate API responses before processing."""
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
```

#### Credential Protection
- **NEVER** log server IPs, URLs with auth tokens, or credentials
- **ALWAYS** sanitize URLs before logging
- **ALWAYS** use configurable domains (no hardcoded values)
- **ALWAYS** validate environment variables for format/security

### 🛡️ Error Handling Standards (MANDATORY)

#### Forbidden Patterns
```python
# NEVER DO THIS - CAUSES PRODUCTION FAILURES:
except:
    pass
    
except Exception:
    return False
```

#### Required Patterns
```python
# ALWAYS DO THIS:
except (OSError, FileNotFoundError) as e:
    logger.error(f"File operation failed: {e}")
    
except (requests.ConnectionError, requests.Timeout) as e:
    logger.warning(f"Network error (retryable): {e}")
    raise RetryableError(e)
```

### 🔧 Resource Management Standards (MANDATORY)

- **NEVER** close and immediately reopen connections
- **NEVER** use recursive functions without depth limits  
- **NEVER** access files while they may be in use (race conditions)
- **ALWAYS** use proper cleanup sequences
- **ALWAYS** declare global variables at function start

### 📋 Pre-Deployment Checklist (MANDATORY)

Every script MUST pass ALL checks:

#### Security Review ✅
- [ ] All input validation implemented
- [ ] No credential exposure in logs
- [ ] File paths validated against directory traversal
- [ ] URLs sanitized before logging
- [ ] Configuration schema validated

#### Error Handling Review ✅
- [ ] No bare `except:` clauses anywhere
- [ ] Specific exception types for each operation
- [ ] Appropriate logging levels used
- [ ] Error context preserved in logs

#### Resource Management Review ✅
- [ ] No connection open/close/reopen patterns
- [ ] Proper cleanup in finally blocks
- [ ] No race conditions in file operations
- [ ] Global variables properly declared

### Script Architecture Requirements

#### Standalone Requirements
- Each script must be completely standalone (no inter-script imports)
- Must work from both terminal and notebook environments
- Use same .env file for authentication across all stages
- Load stage-specific config from NAS at runtime
- Copy-paste shared functions (don't import between stages)

#### Configuration Management
- **.env file**: Authentication only (API, proxy, NAS credentials)
- **NAS config files**: Operational settings (monitored institutions, API parameters, processing settings)
- **Schema validation**: ALL configuration must be validated against defined schemas

### Testing Commands
```bash
# Python testing commands
python -m py_compile stage_X/*.py  # Syntax check
pylint stage_X/*.py                 # Linting (if configured)
python -m pytest stage_X/tests/     # Unit tests (if available)
```

### Git Workflow
- Always commit and push changes
- User tests in different environment
- Use descriptive commit messages with Claude Code attribution

## Development History & Lessons

### Stage 0 Critical Issues Resolved
The stage 0 script underwent comprehensive security and reliability review, resolving 9 critical issues:

1. **Undefined Variables** (CRITICAL): Fixed variable scope issues causing runtime failures
2. **Global Variable Scope** (CRITICAL): Added proper global declarations preventing UnboundLocalError
3. **Security Credential Exposure** (HIGH): Removed IP logging, sanitized URLs, made domain configurable
4. **Inefficient Connection Management** (MEDIUM): Eliminated unnecessary connection close/reopen patterns
5. **Unsafe Directory Recursion** (MEDIUM): Replaced dangerous recursive function with safe iterative approach
6. **Log File Race Condition** (MEDIUM): Fixed concurrent access to log files during upload
7. **Configuration Validation** (MEDIUM): Added comprehensive schema validation with clear error messages
8. **Generic Error Handling** (MEDIUM): Replaced bare except clauses with specific exception handling
9. **Input Validation** (LOW): Added validation for API responses, file paths, and configuration

### Reference Implementation
- **Historical Reference**: Previous version removed during cleanup (commit 1b3ff6c)
- **Production-Ready Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - follows all security standards
- **Development Standards**: See `stage_0_bulk_refresh/CLAUDE.md` for detailed lessons learned

### Code Quality Transformation
- **Before**: Basic functionality with security vulnerabilities and reliability issues
- **After**: Production-ready code with comprehensive validation, security hardening, and proper error handling
- **Standards**: All future scripts must follow the patterns established in Stage 0