# FactSet Earnings Transcript Pipeline - Context for Claude

## Project Overview
This is a multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API. The pipeline is designed with standalone stage-based scripts that can run independently from terminal or notebook environments.

## Current Status
- **Stage 0**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - PRODUCTION READY ✅ (All 9 critical issues resolved)
- **Working Script**: `0_transcript_bulk_sync_working.py` contains the original functional bulk sync script (551 lines)
- **Architecture**: Stage-based approach where each stage is completely independent and follows security standards
- **Storage**: All data stored on NAS (no local storage) using pysmb with NTLM v2 authentication
- **Authentication**: Corporate proxy authentication required for API access

### Stage 0 Key Business Rules
1. **Anti-Contamination**: Only downloads transcripts where target ticker is SOLE primary company
2. **Security-First**: All credentials from environment, paths validated, URLs sanitized in logs
3. **Audit Trail**: Every operation logged with timestamps and error context
4. **Incremental Safe**: Checks existing files, safe to re-run without re-downloading
5. **Version Management**: Automatically handles vendor version ID updates, keeps only latest version
6. **Institution Coverage**: 6 Canadian banks, 6 US banks, 3 insurance companies
7. **File Organization**: Organized by institution type → company → transcript type
8. **Standardized Naming**: Consistent filename format for easy identification and processing

## Key Technical Requirements

### Authentication & Configuration
- **Environment Variables (.env)**: 12 required variables
  - API_USERNAME, API_PASSWORD (FactSet API)
  - PROXY_USER, PROXY_PASSWORD, PROXY_URL, PROXY_DOMAIN (Corporate proxy)
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
- **Folder Structure**:
  - `Inputs/certificate/` - SSL certificates
  - `Inputs/config/` - Configuration files (single config.json)
  - `Outputs/Data/` - Downloaded transcripts by type and institution
  - `Outputs/Logs/` - Execution logs
  - `Outputs/listing/` - Inventory JSON files (future use)

### Monitored Institutions
**Canadian Banks**: RY-CA, TD-CA, BNS-CA, BMO-CA, CM-CA, NA-CA  
**US Banks**: JPM-US, BAC-US, WFC-US, C-US, GS-US, MS-US  
**Insurance**: MFC-CA, SLF-CA, UNH-US

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

## Stage 0 Business Logic & Process Flow

### Complete Step-by-Step Process

#### **1. Security & Authentication Setup**
- Validates all required credentials from `.env` file (12 environment variables)
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

#### **3. Directory Structure Creation**
Creates organized folder structure on NAS:
```
Outputs/Data/
├── Canadian/          # Canadian bank transcripts
│   ├── RY-CA_Royal_Bank_of_Canada/
│   │   ├── Raw/       # Raw transcript files
│   │   ├── Corrected/ # Corrected transcript files
│   │   └── NearRealTime/ # Near real-time transcript files
│   └── [5 other Canadian banks...]
├── US/               # US bank transcripts
│   ├── JPM-US_JPMorgan_Chase/
│   └── [5 other US banks...]
└── Insurance/        # Insurance company transcripts
    ├── MFC-CA_Manulife_Financial/
    └── [2 other insurance companies...]
```

#### **4. Institution Processing Loop (15 Total)**
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
- Filters transcripts where target ticker is the ONLY primary ID (prevents cross-contamination)
- Supports all transcript types: Corrected, Raw, NearRealTime
- Implements intelligent version management to handle vendor version ID updates
- Automatically removes old versions and keeps only latest version of each transcript
- Uses version-agnostic keys for duplicate detection (prevents duplicate downloads)
- Implements file-based inventory tracking to avoid re-downloads
- Rate limiting with retry logic for API protection
- Comprehensive error handling and security validation
- Uploads all data directly to NAS (no local storage)

### Technical Implementation
- **Import Pattern**: `from fds.sdk.EventsandTranscripts.api import transcripts_api`
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
```

## Stage Architecture Plan

### Stage 0: Bulk Refresh ✅ PRODUCTION READY
- **Purpose**: Download ALL historical earnings transcripts from 2023-present for 15 monitored financial institutions
- **Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py`
- **Config**: `Inputs/config/config.json` on NAS
- **When**: Initial setup or complete repository refresh
- **Status**: All 9 critical issues resolved, security validated, production ready
- **Key Features**:
  - Anti-contamination filter: Only downloads transcripts where target ticker is SOLE primary company
  - Version management: Automatically handles vendor version ID updates, prevents duplicate downloads
  - File organization: Creates institution-type folders (Canadian/US/Insurance) with transcript-type subfolders
  - Duplicate prevention: Uses version-agnostic keys for intelligent duplicate detection
  - Automatic cleanup: Removes old versions and keeps only latest version of each transcript
  - Comprehensive audit trail: Timestamped logs uploaded to NAS
  - Standardized naming: `{ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`

### Stage 1: Daily Sync ✅ PRODUCTION READY
- **Purpose**: Incremental daily downloads  
- **Script**: `stage_1_daily_sync/1_transcript_daily_sync.py`
- **Config**: `Inputs/config/config.json` on NAS (uses stage_1 section)
- **When**: Scheduled daily operations or earnings day monitoring
- **Key Features**:
  - Date-based queries using `get_transcripts_dates()` API
  - Single API call per date (efficient vs Stage 0's 15 calls)
  - Inherits ALL version management from Stage 0
  - Optional `earnings_monitor.py` for real-time notifications
  - Configurable lookback period (sync_date_range)

### Stage 2: Processing (Future Development)
- **Purpose**: Process and analyze downloaded transcripts
- **Script**: `stage_2_processing/2_transcript_processing.py` (not yet implemented)
- **Config**: `Inputs/config/stage_2_config.json` on NAS

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
- **Original Working Script**: `0_transcript_bulk_sync_working.py` (551 lines) - functional baseline
- **Production-Ready Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - follows all security standards
- **Development Standards**: See `stage_0_bulk_refresh/CLAUDE.md` for detailed lessons learned

### Code Quality Transformation
- **Before**: Basic functionality with security vulnerabilities and reliability issues
- **After**: Production-ready code with comprehensive validation, security hardening, and proper error handling
- **Standards**: All future scripts must follow the patterns established in Stage 0