# FactSet Earnings Transcript Pipeline - Context for Claude

## Project Overview
This is a multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API. The pipeline is designed with standalone stage-based scripts that can run independently from terminal or notebook environments.

## Current Status
- **Stage 0**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - PRODUCTION READY ✅ (All 9 critical issues resolved)
- **Working Script**: `0_transcript_bulk_sync_working.py` contains the original functional bulk sync script (551 lines)
- **Architecture**: Stage-based approach where each stage is completely independent and follows security standards
- **Storage**: All data stored on NAS (no local storage) using pysmb with NTLM v2 authentication
- **Authentication**: Corporate proxy authentication required for API access

## Key Technical Requirements

### Authentication & Configuration
- **Environment Variables (.env)**: API credentials, proxy settings, NAS connection details
- **NAS Config Files**: Operational settings stored in `Inputs/config/` on NAS
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
  - `Inputs/config/` - Configuration files per stage
  - `Outputs/data/` - Downloaded transcripts by type and institution
  - `Outputs/logs/` - Execution logs
  - `Outputs/listing/` - Inventory JSON files

### Monitored Institutions
**Canadian Banks**: RY-CA, TD-CA, BNS-CA, BMO-CA, CM-CA, NA-CA  
**US Banks**: JPM-US, BAC-US, WFC-US, C-US, GS-US, MS-US  
**Insurance**: MFC-CA, SLF-CA, UNH-US

### File Naming Convention
`{primary_id}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`

## Current Working Script Features

### Core Functionality (0_transcript_bulk_sync_working.py)
- Downloads all earnings transcripts from 2023-present for monitored institutions
- Filters transcripts where target ticker is the ONLY primary ID (prevents cross-contamination)
- Supports all transcript types: Corrected, Raw, NearRealTime
- Implements file-based inventory tracking to avoid re-downloads
- Concurrent downloads with rate limiting (10 req/sec max)
- Comprehensive error handling and retry logic
- Uploads all data directly to NAS (no local storage)

### Technical Implementation
- **Import Pattern**: `from fds.sdk.EventsandTranscripts.api import transcripts_api`
- **Proxy Setup**: Uses requests proxies with MAPLE domain authentication
- **SSL Handling**: Downloads certificate from NAS to temp file, sets environment variables
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
- **Purpose**: Download ALL historical transcripts from 2023-present
- **Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py`
- **Config**: `Inputs/config/stage_0_config.json` on NAS
- **When**: Initial setup or complete repository refresh
- **Status**: All 9 critical issues resolved, security validated, production ready

### Stage 1: Daily Sync (Future)
- **Purpose**: Incremental daily downloads
- **Script**: `stage_1_daily_sync/1_transcript_daily_sync.py`
- **Config**: `Inputs/config/stage_1_config.json` on NAS
- **When**: Scheduled daily operations

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
# Lint and typecheck commands (user will specify these)
npm run lint
npm run typecheck
# Or equivalent Python commands
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