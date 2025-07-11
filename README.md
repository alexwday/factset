# FactSet Earnings Transcript Pipeline

A multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API.

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

### 4. Run Scripts

Each stage is a standalone Python script:

```bash
# Stage 0: Bulk historical sync (implemented)
python stage_0_bulk_refresh/0_transcript_bulk_sync.py

# Stage 1: Daily incremental sync (planned)
python stage_1_daily_sync/1_transcript_daily_sync.py

# Stage 2: Processing & analysis (planned - future development)
# python stage_2_processing/2_transcript_processing.py
```

**Current Status**: Stage 0 is production-ready with all security issues resolved ✅. Stage 1 is planned for future development.

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

## Pipeline Stages

### Stage 0: Bulk Refresh ✅ PRODUCTION READY
- **Purpose**: Download ALL historical transcripts from 2023-present
- **When to use**: Initial setup or complete repository refresh
- **Output**: Complete transcript repository on NAS
- **Configuration**: Loads operational settings from `Inputs/config/stage_0_config.json` on NAS
- **Features**: Self-contained script with environment variable authentication
- **Security**: All 9 critical security and reliability issues resolved

### Stage 1: Daily Sync (Scheduled)
- **Purpose**: Check for new transcripts daily and download incrementally
- **When to use**: Regular scheduled operations
- **Output**: New transcripts added to existing repository

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
- Stored on NAS in `Inputs/config/` folder
- Stage-specific: `stage_0_config.json`, `stage_1_config.json`, etc.
- Contains monitored institutions, API settings, processing parameters
- Downloaded by each script at runtime from NAS

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