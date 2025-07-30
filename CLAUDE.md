# FactSet Earnings Transcript Pipeline - Context for Claude

## Project Overview
This is a multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API. The pipeline is designed with standalone stage-based scripts that can run independently from terminal or notebook environments.

## Current Status
- **Stage 0**: `database_refresh/00_download_historical/main_historical_sync.py` - PRODUCTION READY ✅ (All 9 critical issues resolved)
- **Stage 1**: `database_refresh/01_download_daily/main_daily_sync.py` - PRODUCTION READY ✅ (Daily incremental sync)
- **Stage 2**: `database_refresh/02_database_sync/main_sync_updates.py` - PRODUCTION READY ✅ (Master database synchronization)
- **Stage 3**: `database_refresh/03_extract_content/main_content_extraction.py` - PRODUCTION READY ✅ (Content extraction)
- **Stage 4**: `database_refresh/04_validate_structure/4_transcript_llm_classification.py` - PRODUCTION READY ✅ (Structure validation)
- **Stage 5**: `database_refresh/05_qa_pairing/main_qa_pairing.py` - PRODUCTION READY ✅ (Q&A boundary detection)
- **Reference Implementation**: Production-ready code with comprehensive validation, security hardening, and proper error handling
- **Architecture**: Stage-based approach where each stage is completely independent and follows security standards
- **Storage**: All data stored on NAS (no local storage) using pysmb with NTLM v2 authentication
- **Authentication**: Corporate proxy authentication required for API access, LLM OAuth for Stage 5

### Stage 0 Key Business Rules
1. **Anti-Contamination**: Only downloads transcripts where target ticker is SOLE primary company
2. **Security-First**: All credentials from environment, paths validated, URLs sanitized in logs
3. **Audit Trail**: Every operation logged with timestamps and error context
4. **Incremental Safe**: Checks existing files, safe to re-run without re-downloading
5. **Version Management**: Automatically handles vendor version ID updates, keeps only latest version
6. **Institution Coverage**: 100+ global financial institutions across multiple regions and types
7. **File Organization**: Organized by fiscal quarter → institution type → company → transcript type
8. **Standardized Naming**: Consistent filename format for easy identification and processing

## Key Technical Requirements

### Authentication & Configuration
- **Environment Variables (.env)**: 15 required variables
  - API_USERNAME, API_PASSWORD (FactSet API)
  - PROXY_USER, PROXY_PASSWORD, PROXY_URL, PROXY_DOMAIN (Corporate proxy - PROXY_DOMAIN defaults to 'MAPLE')
  - NAS_USERNAME, NAS_PASSWORD, NAS_SERVER_IP, NAS_SERVER_NAME (NAS connection)
  - NAS_SHARE_NAME, NAS_BASE_PATH, NAS_PORT (NAS paths - NAS_PORT defaults to 445)
  - CONFIG_PATH, CLIENT_MACHINE_NAME (Configuration)
  - LLM_CLIENT_ID, LLM_CLIENT_SECRET (LLM API for Stage 5)
- **NAS Config Files**: YAML configuration with stage-specific sections (stage_00_download_historical through stage_05_qa_pairing)
- **SSL Certificate**: Downloaded from NAS at runtime for FactSet API and LLM connections
- **Proxy**: Corporate proxy with NTLM authentication required

### FactSet API Integration
- **API**: EventsandTranscripts API using fds.sdk.EventsandTranscripts
- **Endpoints**: 
  - `transcripts_api.TranscriptsApi.get_transcripts_ids()` for institution-based queries (Stage 0)
  - `transcripts_api.TranscriptsApi.get_transcripts_dates()` for date-based queries (Stage 1)
- **Filtering**: Primary ID filtering to prevent cross-contamination between related companies
- **Rate Limiting**: 2-3 second delays between requests, 3-8 retry attempts with exponential backoff

### NAS Integration Details
- **Protocol**: SMB/CIFS using pysmb.SMBConnection
- **Authentication**: NTLM v2 with domain credentials
- **Enhanced Folder Structure** (Fiscal Quarter Organization):
  - `Inputs/certificate/` - SSL certificates
  - `Inputs/config/` - Configuration files (config.yaml)
  - `Outputs/Data/YYYY/QX/Type/Company/TranscriptType/` - Transcripts organized by fiscal year and quarter
  - `Outputs/Logs/` - Execution logs
  - `Outputs/Logs/Errors/` - Separate error logs (parsing, download, filesystem, validation)
  - `Outputs/Database/` - Master database JSON
  - `Outputs/Refresh/` - Stage processing queues and outputs

### Monitored Institutions (100+ Total)
**Key Institution Categories**:
- **Canadian Financial**: Banks, Asset Managers, Insurance, Monoline (30+ institutions)
- **US Financial**: Major Banks, Regional Banks, Asset Managers, Investment Banks, Trust Banks (35+ institutions)  
- **European/International**: UK, German, French, Spanish, Italian, Swiss, Nordic, Australian banks (30+ institutions)
- **Specialized**: Insurance companies, Asset managers, Boutique investment banks, Trust companies (5+ institutions)

*Examples*: RY-CA, TD-CA, JPM-US, BAC-US, BCS-GB, UBS-CH, MFC-CA, BLK-US, and many more

### File Naming Convention
**Stage 0 Format**: `{ticker}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml`

**Example**: `RY-CA_Q1_2024_Corrected_12345_1.xml`
- **RY-CA**: Royal Bank of Canada ticker
- **Q1**: Fiscal quarter
- **2024**: Fiscal year
- **Corrected**: Transcript type (Raw/Corrected)
- **12345**: Unique event identifier
- **1**: Version identifier

**Storage Path**: `Data/2024/Q1/Canadian/RY-CA_Royal_Bank_of_Canada/Corrected/RY-CA_Q1_2024_Corrected_12345_1.xml`

## Dependencies (requirements.txt)
```
# Core Dependencies
fds.sdk.EventsandTranscripts  # FactSet API SDK
requests                      # Network operations
python-dotenv                 # Environment variables
pysmb                        # NAS connectivity
yaml                         # Configuration parsing
pytz                         # Timezone handling

# Data Processing
pandas                       # Data manipulation (Stage 0)
python-dateutil             # Date parsing
xml.etree.ElementTree       # Built-in XML parsing

# LLM Integration (Stage 5)
openai                      # OpenAI API client
```

## Stage Architecture Plan

### Stage 0: Historical Transcript Sync ✅ PRODUCTION READY
- **Purpose**: Download ALL historical earnings transcripts (3-year rolling window) for 100+ monitored financial institutions
- **Script**: `database_refresh/00_download_historical/main_historical_sync.py`
- **Config**: `config.yaml` on NAS (uses stage_00_download_historical section)
- **When**: Monthly/Quarterly for maintenance, Initial run for setup
- **Status**: All 9 critical issues resolved, security validated, production ready
- **Key Features**:
  - **3-Year Rolling Window**: Maintains exactly 3 years of historical data with automatic cleanup
  - **Title Format Validation**: Strict validation of "Qx 20xx Earnings Call" format
  - **Anti-Contamination Filter**: Only downloads transcripts where target ticker is SOLE primary company
  - **Version Management**: Automatically handles vendor version ID updates, removes old versions
  - **Fiscal Quarter Organization**: Dynamic folder structure based on parsed transcript titles
  - **Comprehensive Error Logging**: Separate JSON files for parsing, download, filesystem, validation errors
  - **Smart Fallbacks**: Uses "Unknown/Unknown" folder for unparseable transcripts with operator guidance
  - **Windows Path Compatibility**: Validates path lengths and automatically shortens when needed
  - **Standardized Naming**: `{ticker}_{quarter}_{year}_{transcript_type}_{event_id}_{version_id}.xml`

### Stage 1: Daily Transcript Sync ✅ PRODUCTION READY
- **Purpose**: Incremental daily downloads of recent earnings transcripts  
- **Script**: `database_refresh/01_download_daily/main_daily_sync.py`
- **Config**: `config.yaml` on NAS (uses stage_01_download_daily section)
- **When**: Scheduled daily operations or earnings day monitoring
- **Key Features**:
  - **Date-Based Discovery**: Uses get_transcripts_dates() API for efficient recent transcript discovery
  - **Configurable Lookback**: sync_date_range parameter allows flexible date range configuration
  - **Two-Phase Processing**: Date discovery phase followed by institution processing phase
  - **Performance Optimized**: Only processes institutions with actual transcripts (not all 100+)
  - **Same Organization**: Uses identical fiscal quarter folder structure as Stage 0
  - **Version Management**: Inherits all version management capabilities from Stage 0
  - **Optional Monitoring**: earnings_monitor.py for real-time notifications

### Stage 2: Master Database Synchronization ✅ PRODUCTION READY
- **Purpose**: Pure file synchronization between NAS file system and master database
- **Script**: `database_refresh/02_database_sync/main_sync_updates.py`
- **Config**: `config.yaml` on NAS (uses stage_02_database_sync section)
- **When**: After Stage 0/1 completes, before Stage 3 processing
- **Key Features**:
  - **Pure File Sync**: No selection logic - simply mirrors NAS to database
  - **Delta Detection**: Identifies new, modified, and deleted files
  - **Queue Generation**: Creates stage_02_process_queue.json and stage_02_removal_queue.json
  - **Path-Based Comparison**: Uses file_path as primary key for synchronization
  - **Date-Modified Checking**: Detects file modifications via timestamps
  - **No Database Creation**: Returns None if master database doesn't exist
  - **Minimal Schema**: {file_path, date_last_modified} structure

### Stage 3: Content Extraction & Processing ✅ PRODUCTION READY
- **Purpose**: Extract and structure content from XML transcripts at paragraph level
- **Script**: `database_refresh/03_extract_content/main_content_extraction.py`
- **Config**: `config.yaml` on NAS (uses stage_03_extract_content section)
- **When**: After Stage 2 generates processing queue
- **Key Features**:
  - **Enhanced Field Extraction**: Extracts all metadata from paths, XML, and config lookups
  - **Paragraph-Level Breakdown**: Creates individual records for each paragraph
  - **Speaker Attribution**: Formats participants with name, title, and affiliation
  - **Q&A Detection**: Uses XML speaker type attributes ("q" for question, "a" for answer)
  - **Sequential Numbering**: Maintains paragraph and speaker block IDs
  - **JSON Validation**: Pre-save validation with record preview
  - **Development Mode**: Configurable file limits for testing

### Stage 4: Transcript Structure Validation ✅ PRODUCTION READY
- **Purpose**: Validate transcript content structure (exactly 2 sections with expected names)
- **Script**: `database_refresh/04_validate_structure/4_transcript_llm_classification.py`
- **Config**: `config.yaml` on NAS (uses stage_4 section)
- **When**: After Stage 3 content extraction
- **Key Features**:
  - **Structure Validation**: Validates exactly 2 sections per transcript
  - **Expected Sections**: "MANAGEMENT DISCUSSION SECTION" and "Q&A"
  - **Binary Classification**: Separates valid from invalid transcripts
  - **Review Queue**: Creates invalid_content_for_review.json for manual inspection
  - **Complete Validation**: All sections must match expected names
  - **Development Mode**: Process limited transcripts for testing

### Stage 5: Q&A Boundary Detection & Pairing ✅ PRODUCTION READY
- **Purpose**: Detect conversation boundaries and pair questions with answers using LLM
- **Script**: `database_refresh/05_qa_pairing/main_qa_pairing.py`
- **Config**: `config.yaml` on NAS (uses stage_05_qa_pairing section)
- **When**: After Stage 4 validation for Q&A relationship mapping
- **Key Features**:
  - **Speaker Block Analysis**: Groups paragraphs by speaker (not individual paragraphs)
  - **Sliding Window Processing**: Analyzes configurable windows of speaker blocks
  - **LLM Boundary Detection**: Uses GPT-4 with structured tool calls
  - **Per-Transcript OAuth**: Refreshes token for each transcript
  - **Progressive Fallbacks**: LLM detection → XML attributes → conservative grouping
  - **Minimal Schema Extension**: Only adds qa_group_id field
  - **Cost Tracking**: Real-time token usage and cost monitoring
  - **Visualization Tool**: visualize_qa_pairing.py for result inspection

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
- **.env file**: Authentication only (API, proxy, NAS, LLM credentials)
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
- **Production-Ready Scripts**: All scripts in database_refresh/ follow security standards
- **Development Standards**: See individual stage CLAUDE.md files for detailed implementation notes

### Code Quality Transformation
- **Before**: Basic functionality with security vulnerabilities and reliability issues
- **After**: Production-ready code with comprehensive validation, security hardening, and proper error handling
- **Standards**: All scripts follow the patterns established in Stage 0