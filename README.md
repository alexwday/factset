# FactSet Earnings Transcript Pipeline

A multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API with enhanced fiscal quarter organization.

## Project Structure

```
factset/
├── .env                                    # Shared authentication (create from .env.example)
├── .env.example                           # Template for environment variables
├── 0_transcript_bulk_sync_working.py      # Working version of bulk sync script
├── docs/                                  # Documentation & FactSet SDK docs
├── stage_0_bulk_refresh/                  # Historical bulk download (optional)
├── stage_1_daily_sync/                    # Daily incremental sync (scheduled)
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

## Quick Setup

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd factset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Authentication

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
- `API_USERNAME` - Your FactSet API username
- `API_PASSWORD` - Your FactSet API password  
- `PROXY_USER` - Corporate proxy username
- `PROXY_PASSWORD` - Corporate proxy password
- `PROXY_URL` - Corporate proxy URL (e.g., oproxy.fg.rbc.com:8080)
- `NAS_USERNAME` - NAS server username
- `NAS_PASSWORD` - NAS server password
- `NAS_SERVER_IP` - NAS server IP address
- `NAS_SERVER_NAME` - NAS server name for NTLM
- `NAS_SHARE_NAME` - NAS share name
- `NAS_BASE_PATH` - Base path within NAS share
- `CLIENT_MACHINE_NAME` - Client machine name for SMB

### 3. Setup NAS Configuration

**IMPORTANT**: Before running any scripts, you must place the shared configuration file on your NAS:

1. Upload `config.json` to your NAS at the path: `{NAS_BASE_PATH}/Inputs/config/config.json`
2. Ensure the SSL certificate is available at: `{NAS_BASE_PATH}/Inputs/certificate/certificate.cer`

**Note**: The configuration file contains monitored institutions, API settings, and transcript types. Stage 0 validates the entire configuration schema before processing.

### 4. Run Scripts

Each stage is a standalone Python script:

```bash
# Stage 0: Bulk historical sync (implemented)
python stage_0_bulk_refresh/0_transcript_bulk_sync.py

# Stage 1: Daily incremental sync (implemented)
python stage_1_daily_sync/1_transcript_daily_sync.py

# Stage 1: With earnings monitor (runs every 5 minutes with notifications)
python stage_1_daily_sync/earnings_monitor.py

# Stage 2: Processing & analysis (planned - future development)
# python stage_2_processing/2_transcript_processing.py
```

**Current Status**: 
- Stage 0 is production-ready with enhanced fiscal quarter organization ✅
- Stage 1 is production-ready with enhanced folder structure and comprehensive error logging ✅
- Both stages feature robust title parsing with 4 regex patterns and smart fallbacks ✅

## Testing from Work Environment

### Prerequisites

1. **Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   # Copy and configure environment file
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **NAS Setup**:
   - Ensure `config.json` is uploaded to `{NAS_BASE_PATH}/Inputs/config/config.json`
   - Verify SSL certificate exists at `{NAS_BASE_PATH}/Inputs/certificate/certificate.cer`
   - Test NAS connectivity from your work computer

### Environment Variables Required

Complete your `.env` file with these values:

```env
# FactSet API Configuration
API_USERNAME=your_factset_username
API_PASSWORD=your_factset_api_password

# Proxy Configuration (Corporate Network)
PROXY_USER=your_proxy_username
PROXY_PASSWORD=your_proxy_password
PROXY_URL=oproxy.fg.rbc.com:8080

# NAS Configuration
NAS_USERNAME=your_nas_username
NAS_PASSWORD=your_nas_password
NAS_SERVER_IP=192.168.1.100
NAS_SERVER_NAME=NAS-SERVER
NAS_SHARE_NAME=shared_folder
NAS_BASE_PATH=transcript_repository
NAS_PORT=445
CONFIG_PATH=Inputs/config/config.json
CLIENT_MACHINE_NAME=SYNC-CLIENT
```

### Testing Stage 0

1. **Validate Configuration**:
   ```bash
   # Test script syntax
   python -m py_compile stage_0_bulk_refresh/0_transcript_bulk_sync.py
   ```

2. **Dry Run Check**:
   - Script will connect to NAS and load config before starting
   - Monitor initial log output for connection issues
   - Script will fail fast if config file not found on NAS

3. **Execute Full Sync**:
   ```bash
   python stage_0_bulk_refresh/0_transcript_bulk_sync.py
   ```

4. **Monitor Progress**:
   - Watch console output for real-time progress
   - Check NAS directory structure creation
   - Verify transcript downloads in `{NAS_BASE_PATH}/Outputs/data/`
   - Review logs in `{NAS_BASE_PATH}/Outputs/logs/`

### Expected Behavior

- **Connection**: Script connects to NAS and downloads `config.json`
- **Authentication**: Configures FactSet API with proxy settings
- **Directory Setup**: Creates full directory structure on NAS
- **Processing**: Downloads transcripts for all 15 monitored institutions
- **Output**: Generates inventory files and execution logs
- **Error Handling**: Retries failed downloads, logs all errors

### Troubleshooting

1. **"Config file not found"**: Upload `config.json` to NAS at correct path
2. **SSL Certificate Error**: Verify certificate file exists on NAS
3. **Proxy Authentication**: Check proxy credentials and URL format
4. **NAS Connection Failed**: Verify NAS credentials and network access
5. **API Authentication**: Confirm FactSet API credentials are valid

## Stage 0: Detailed Business Logic

### What Stage 0 Actually Does

Stage 0 is a comprehensive bulk download system that establishes your complete historical transcript baseline. Here's the step-by-step business process:

#### **1. Security & Authentication Setup**
- Validates all required credentials from `.env` file
- Connects to NAS using secure NTLM v2 authentication
- Downloads SSL certificate from NAS for secure FactSet API connections
- Configures corporate proxy authentication for API access

#### **2. Configuration & Validation**
- Downloads `config.json` from NAS (never stored locally)
- Validates configuration schema with comprehensive checks:
  - Institution list validation
  - API parameter validation (dates, transcript types, rate limits)
  - Security path validation to prevent directory traversal
  - Ticker format validation

#### **3. Directory Structure Creation**
Creates organized folder structure on NAS:
```
Outputs/Data/
├── Canadian/          # Canadian bank transcripts
│   ├── RY-CA_Royal_Bank_of_Canada/
│   │   ├── Raw/       # Raw transcript files
│   │   ├── Corrected/ # Corrected transcript files
│   │   └── NearRealTime/ # Near real-time transcript files
│   └── [other Canadian banks...]
├── US/               # US bank transcripts
│   ├── JPM-US_JPMorgan_Chase/
│   └── [other US banks...]
└── Insurance/        # Insurance company transcripts
    ├── MFC-CA_Manulife_Financial/
    └── [other insurance companies...]
```

#### **4. Institution Processing (15 Total)**
For each monitored institution:

**a) API Query with Filters**
- Queries FactSet API for ALL transcripts from 2023-present
- Applies earnings filter (only "Earnings" event types)
- **Critical Anti-Contamination Filter**: Only processes transcripts where the target ticker is the SOLE primary company (prevents downloading joint earnings calls)

**b) Transcript Type Processing**
- **Raw**: Original transcript as received
- **Corrected**: Edited for accuracy and completeness
- **NearRealTime**: Real-time transcript during live call

**c) Duplicate Prevention**
- Scans existing files on NAS for each institution/type
- Creates standardized filename: `{ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`
- Only downloads new transcripts not already present

**d) Download & Storage**
- Downloads transcript XML via FactSet API with corporate proxy
- Validates file integrity and logs file sizes
- Uploads directly to NAS (no local storage for security)
- Implements retry logic: 3 attempts with 5-second delays
- Rate limiting: 2-second delays between requests

#### **5. Progress Tracking & Audit**
- Tracks downloads by institution and transcript type
- Logs detailed progress and any failures
- Generates comprehensive execution summary
- Uploads timestamped execution log to NAS for audit trail

#### **6. Final Reporting**
Provides detailed summary including:
- Total transcripts downloaded
- Execution time
- Institutions with/without transcripts found
- Breakdown by transcript type and institution
- Any failures or issues encountered

### **Key Business Rules**

1. **Enhanced Fiscal Quarter Organization**: Automatically organizes all transcripts by fiscal year and quarter parsed from XML titles
2. **Robust Title Parsing**: 4 regex patterns handle various formats with smart fallbacks to "Unknown/Unknown" folder
3. **Comprehensive Error Logging**: Separate JSON files track parsing, download, filesystem, and validation errors
4. **Anti-Contamination**: Only downloads transcripts where target ticker is SOLE primary company
5. **Security-First**: All credentials from environment, paths validated, URLs sanitized in logs
6. **Audit Trail**: Every operation logged with timestamps and error context
7. **Incremental Safe**: Checks existing files across all folders, safe to re-run without re-downloading
8. **Version Management**: Automatically handles vendor version ID updates, prevents duplicate downloads
9. **Windows Compatibility**: Path length validation with automatic shortening
10. **Institution Coverage**: 6 Canadian banks, 6 US banks, 3 insurance companies

### **Expected Outcomes**

- **Enhanced Data Structure**: Organized by fiscal year → quarter → institution type → company → transcript type
- **Fiscal Quarter Organization**: Easy quarterly analysis and time-series studies
- **Error Management**: Comprehensive error logs with actionable recovery instructions
- **File Naming**: Standardized format for easy identification and processing
- **Audit Trail**: Complete log of all operations for compliance
- **Repository**: Complete historical baseline with fiscal quarter organization ready for Stage 1 daily updates

## Pipeline Stages

### Stage 0: Bulk Refresh ✅ PRODUCTION READY
- **Purpose**: Download ALL historical earnings transcripts from 2023-present for 15 monitored financial institutions
- **When to use**: Initial setup or complete repository refresh
- **Output**: Complete transcript repository on NAS with organized folder structure
- **Configuration**: Loads operational settings from `Inputs/config/config.json` on NAS
- **Features**: Self-contained script with environment variable authentication
- **Security**: All 9 critical security and reliability issues resolved
- **Anti-contamination**: Only downloads transcripts where target ticker is SOLE primary company
- **Version Management**: Automatically handles vendor version ID updates, prevents duplicate downloads
- **File Organization**: Creates institution-type folders (Canadian/US/Insurance) with transcript-type subfolders
- **Duplicate Prevention**: Uses version-agnostic keys for intelligent duplicate detection, safe to re-run
- **Automatic Cleanup**: Removes old versions and keeps only latest version of each transcript
- **Audit Trail**: Comprehensive logging with timestamped execution logs uploaded to NAS

### Stage 1: Daily Sync ✅ PRODUCTION READY
- **Purpose**: Check for new transcripts daily and download incrementally using efficient date-based queries
- **When to use**: Regular scheduled operations or earnings day monitoring
- **Output**: New transcripts added to existing repository
- **Features**: 
  - Date-based API queries (single call per date vs 15 company calls)
  - Configurable lookback period (sync_date_range)
  - Same security and version management as Stage 0
  - Optional earnings monitor with real-time macOS notifications
- **Usage**: 
  - Manual: `python stage_1_daily_sync/1_transcript_daily_sync.py`
  - Monitor: `python stage_1_daily_sync/earnings_monitor.py` (runs every 5 minutes with popups)

### Stage 2: Processing (Future Development)
- **Purpose**: Process, enrich, and analyze downloaded transcripts
- **When to use**: After transcripts are downloaded
- **Output**: Processed data and analysis results
- **Status**: Not yet implemented

## Configuration

### Authentication (.env file)
- Contains only credentials and connection details
- Shared across all stages
- Never committed to git (in .gitignore)
- Variables: API credentials, proxy settings, NAS connection details

### Operational Settings (NAS config files)
- Stored on NAS in `Inputs/config/config.json` (single file with stage-specific sections)
- Contains monitored institutions, API settings, processing parameters
- Downloaded by each script at runtime from NAS
- Stage 1 requires `stage_1` section with `sync_date_range` parameter

## Monitored Institutions

### Canadian Banks
- Royal Bank of Canada (RY-CA)
- Toronto-Dominion Bank (TD-CA)
- Bank of Nova Scotia (BNS-CA)
- Bank of Montreal (BMO-CA)
- Canadian Imperial Bank of Commerce (CM-CA)
- National Bank of Canada (NA-CA)

### US Banks
- JPMorgan Chase & Co. (JPM-US)
- Bank of America Corporation (BAC-US)
- Wells Fargo & Company (WFC-US)
- Citigroup Inc. (C-US)
- Goldman Sachs Group Inc. (GS-US)
- Morgan Stanley (MS-US)

### Insurance Companies
- Manulife Financial Corporation (MFC-CA)
- Sun Life Financial Inc. (SLF-CA)
- UnitedHealth Group Incorporated (UNH-US)

## Earnings Monitor (Stage 1)

### Overview
The earnings monitor runs Stage 1 automatically every 5 minutes and provides real-time macOS notifications when new transcripts are detected. Perfect for earnings season monitoring!

### Usage
```bash
# Start the monitor
python stage_1_daily_sync/earnings_monitor.py
```

### Features
- ✅ Runs Stage 1 every 5 minutes automatically
- ✅ Popup notification for EACH new transcript found
- ✅ Shows bank name, date, and transcript type in notifications
- ✅ Summary notifications after each sync run
- ✅ Tracks session statistics and history
- ✅ Graceful shutdown with Ctrl+C

### Notification Examples
- **Individual Transcript**: "New Transcript: RY-CA" → "Corrected transcript for 2024-01-25"
- **Multiple Found**: "Found Transcripts: TD-CA" → "3 new, 1 updated Raw transcripts"
- **Sync Summary**: "Sync Complete - New Transcripts!" → "Total: 5 transcripts downloaded"

### Requirements
- macOS (for notifications)
- NAS config.json with `stage_1` section
- All Stage 1 authentication configured

### Best Practices for Earnings Days
1. Start monitor in the morning before market open
2. Keep terminal visible but minimized
3. Monitor typically finds most activity 5-9 PM (after-hours earnings calls)
4. Press Ctrl+C to stop at end of day

## Troubleshooting

### Common Issues

1. **Environment Variables Not Found**
   - Ensure .env file exists and contains all required variables
   - Check file is in correct location (project root)

2. **NAS Connection Failed**
   - Verify NAS credentials and IP address
   - Check network connectivity
   - Ensure NAS is accessible from your location

3. **API Authentication Failed**
   - Verify FactSet API credentials
   - Check if API access is enabled for your account
   - Ensure proxy settings are correct if behind corporate firewall

4. **Import Errors**
   - Activate virtual environment: `source venv/bin/activate`
   - Install requirements: `pip install -r requirements.txt`

### Logs and Monitoring

- Scripts generate detailed logs during execution
- Failed downloads are tracked and reported
- Check console output for real-time status
- Log files stored on NAS for historical analysis

## Development Standards & Security

> **CRITICAL**: These standards are mandatory for all future development. They prevent security vulnerabilities and production failures discovered in Stage 0.

### 🚨 Security-First Development (NON-NEGOTIABLE)

ALL new scripts MUST include:

#### Input Validation Framework
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

#### Error Handling Standards
```python
# FORBIDDEN - CAUSES PRODUCTION FAILURES:
except:
    pass

# REQUIRED - SPECIFIC ERROR HANDLING:
except (OSError, FileNotFoundError) as e:
    logger.error(f"File operation failed: {e}")
```

### 📋 Pre-Deployment Checklist (MANDATORY)

Every script MUST pass ALL checks before deployment:

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

### Adding New Stages

1. **Create Directory Structure**:
   ```
   stage_X_<purpose>/
   ├── CLAUDE.md              # Stage-specific lessons and standards
   ├── X_<descriptive_name>.py # Main script following all standards
   └── tests/                  # Comprehensive test suite
   ```

2. **Copy Security Framework**:
   - Copy ALL validation functions from Stage 0
   - Implement ALL error handling patterns
   - Add ALL input validation

3. **Follow Architecture Requirements**:
   - Each script completely standalone
   - No imports between stage scripts
   - Shared functions copy-pasted (not imported)
   - Works from terminal or any Python environment
   - Uses same .env file for authentication

4. **Validate Against Standards**:
   - Pass ALL security checks
   - Pass ALL error handling checks
   - Pass ALL resource management checks
   - Complete comprehensive testing

### Development History

#### Stage 0 Critical Issues Resolved
Stage 0 underwent comprehensive security review, resolving 9 critical issues:

1. **Undefined Variables** (CRITICAL): Variable scope issues causing runtime failures
2. **Global Variable Scope** (CRITICAL): Missing global declarations causing UnboundLocalError
3. **Security Credential Exposure** (HIGH): IP logging, URL tokens, hardcoded domains
4. **Inefficient Connection Management** (MEDIUM): Unnecessary connection patterns
5. **Unsafe Directory Recursion** (MEDIUM): Stack overflow vulnerability
6. **Log File Race Condition** (MEDIUM): Concurrent file access issues
7. **Configuration Validation** (MEDIUM): Missing schema validation
8. **Generic Error Handling** (MEDIUM): Bare except clauses masking failures
9. **Input Validation** (LOW): Missing validation for external data

#### Code Quality Standards
- **Security-First**: Input validation, credential protection, path validation
- **Error Handling**: Specific exceptions, proper logging, error context preservation
- **Resource Management**: Proper cleanup, no race conditions, efficient patterns
- **Configuration**: Schema validation, environment variable validation
- **Testing**: Comprehensive test coverage for all failure scenarios

### Reference Implementation
- **Stage 0**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py` - Production-ready reference
- **Stage 0 Documentation**: `stage_0_bulk_refresh/CLAUDE.md` - Detailed lessons learned
- **Development Standards**: This section - Mandatory requirements for all future development

## Support

For issues or questions:
1. Check logs for detailed error messages
2. Verify configuration files are correct
3. Test network connectivity to NAS and FactSet API
4. Review this README for common solutions