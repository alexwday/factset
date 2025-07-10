# FactSet Earnings Transcript Pipeline - Context for Claude

## Project Overview
This is a multi-stage pipeline for downloading, processing, and analyzing earnings transcripts from FactSet's Events & Transcripts API. The pipeline is designed with standalone stage-based scripts that can run independently from terminal or notebook environments.

## Current Status
- **Working Script**: `0_transcript_bulk_sync_working.py` contains the fully functional bulk sync script (551 lines)
- **Architecture**: Moving to stage-based approach where each stage is completely independent
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

### Stage 0: Bulk Refresh (In Development)
- **Purpose**: Download ALL historical transcripts from 2023-present
- **Script**: `stage_0_bulk_refresh/0_transcript_bulk_sync.py`
- **Config**: `Inputs/config/stage_0_config.json` on NAS
- **When**: Initial setup or complete repository refresh

### Stage 1: Daily Sync (Future)
- **Purpose**: Incremental daily downloads
- **Script**: `stage_1_daily_sync/1_transcript_daily_sync.py`
- **Config**: `Inputs/config/stage_1_config.json` on NAS
- **When**: Scheduled daily operations

### Stage 2: Processing (Future)
- **Purpose**: Process and analyze downloaded transcripts
- **Script**: `stage_2_processing/2_transcript_processing.py`
- **Config**: `Inputs/config/stage_2_config.json` on NAS

## Important Notes for Development

### Script Requirements
- Each script must be completely standalone (no inter-script imports)
- Must work from both terminal and notebook environments
- Use same .env file for authentication across all stages
- Load stage-specific config from NAS at runtime
- Copy-paste shared functions (don't import between stages)

### Configuration Split
- **.env file**: Authentication only (API, proxy, NAS credentials)
- **NAS config files**: Operational settings (monitored institutions, API parameters, processing settings)

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

## Current Working Code Location
The complete, tested, working implementation is in `0_transcript_bulk_sync_working.py` (551 lines). This should be used as the foundation for creating the stage-based architecture.