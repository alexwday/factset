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
├── stage_2_processing/                    # Transcript consolidation & change detection
├── stage_3_content_processing/            # XML content extraction & paragraph breakdown
├── stage_4_llm_classification/            # LLM-based section type classification
├── stage_5_qa_pairing/                    # Q&A conversation boundary detection
├── stage_6_detailed_classification/       # Detailed financial category classification
├── stage_7_content_enhancement/           # Paragraph-level content enhancement
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
- `LLM_CLIENT_ID` - LLM API client ID (for Stage 4 & 5)
- `LLM_CLIENT_SECRET` - LLM API client secret (for Stage 4 & 5)

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

# Stage 2: Transcript consolidation and change detection (implemented)
python stage_2_processing/2_transcript_consolidation.py

# Stage 3: XML content extraction and paragraph-level breakdown (implemented)
python stage_3_content_processing/3_transcript_content_extraction.py

# Stage 4: LLM-based transcript section classification (implemented)
python stage_4_llm_classification/4_transcript_llm_classification.py

# Stage 5: Q&A conversation boundary detection and pairing (implemented)
python stage_5_qa_pairing/5_transcript_qa_pairing.py

# Stage 6: Detailed financial category classification (implemented)
python stage_6_detailed_classification/6_transcript_detailed_classification.py

# Stage 7: Content enhancement system (implemented)
python stage_7_content_enhancement/7_transcript_content_enhancement.py

# Test Scripts: Analysis and visualization tools
python test_scripts/stage_4_analysis_visualizer.py
```

**Current Status**: 
- Stage 0 is production-ready with enhanced fiscal quarter organization ✅
- Stage 1 is production-ready with enhanced folder structure and comprehensive error logging ✅
- Stage 2 is production-ready with transcript consolidation and change detection ✅
- Stage 3 is production-ready with XML content extraction and paragraph-level breakdown ✅
- Stage 4 is production-ready with LLM-based section type classification and enhanced execution metrics ✅
- Stage 5 is production-ready with Q&A conversation boundary detection, fixed group counting, and comprehensive timing ✅
- Stage 6 is production-ready with detailed financial category classification using dual processing approaches ✅
- Stage 7 is production-ready with paragraph-level content enhancement using sliding window approach ✅
- All stages feature robust title parsing with 4 regex patterns and smart fallbacks ✅

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

# LLM Configuration (Stage 4 & 5)
LLM_CLIENT_ID=your_llm_client_id
LLM_CLIENT_SECRET=your_llm_client_secret
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

### Testing Stage 2

1. **Prerequisites**:
   - Stage 0 or Stage 1 must have run successfully (transcripts exist on NAS)
   - NAS config.json must include `stage_2` section

2. **Execute Consolidation**:
   ```bash
   python stage_2_processing/2_transcript_consolidation.py
   ```

3. **Expected Outputs**:
   - `files_to_process.json`: New/changed transcripts for future processing
   - `files_to_remove.json`: Database records to clean up (if database exists)
   - Comprehensive execution logs with selection decisions
   - Note: No master database is created (that's handled by Stage 3)

### Testing Stage 3

1. **Prerequisites**:
   - Stage 2 must have run successfully (creates `files_to_process.json`)
   - NAS config.json must include `stage_3` section with `dev_mode: true`

2. **Execute Content Extraction**:
   ```bash
   python stage_3_content_processing/3_transcript_content_extraction.py
   ```

3. **Expected Outputs**:
   - `extracted_transcript_sections.json`: Paragraph-level database records
   - Each paragraph becomes a separate record with speaker attribution
   - All Stage 2 fields preserved plus new content fields
   - Development mode processes only 1-2 files for testing

4. **Development Mode Settings**:
   ```json
   "stage_3": {
     "dev_mode": true,
     "dev_max_files": 2
   }
   ```

### Testing Stage 4

1. **Prerequisites**:
   - Stage 3 must have run successfully (creates `extracted_transcript_sections.json`)
   - NAS config.json must include `stage_4` section with LLM configuration
   - LLM API credentials configured in .env file

2. **Execute LLM Classification**:
   ```bash
   python stage_4_llm_classification/4_transcript_llm_classification.py
   ```

3. **Expected Outputs**:
   - `classified_transcript_sections.json`: Complete records with section type classifications
   - Each paragraph gets 3 new fields: section_type, section_type_confidence, section_type_method
   - 3-level progressive classification: Direct → Breakpoint → Individual analysis
   - Development mode processes only 2 transcripts for testing

4. **Development Mode Settings**:
   ```json
   "stage_4": {
     "dev_mode": true,
     "dev_max_transcripts": 2,
     "llm_config": {
       "model": "gpt-4-turbo",
       "temperature": 0.1,
       "base_url": "https://your-llm-api.com/v1",
       "token_endpoint": "https://oauth.your-api.com/token",
       "cost_per_1k_prompt_tokens": 0.03,
       "cost_per_1k_completion_tokens": 0.06
     },
     "classification_thresholds": {
       "min_confidence": 0.7
     },
     "content_limits": {
       "max_paragraph_chars": 750
     }
   }
   ```

5. **Classification Results**:
   - **section_type**: "Management Discussion" or "Investor Q&A"
   - **section_type_confidence**: 0.0-1.0 confidence score
   - **section_type_method**: "section_uniform", "breakpoint_detection", or "contextual_individual"

### Testing Stage 5

1. **Prerequisites**:
   - Stage 4 must have run successfully (creates `classified_transcript_sections.json`)
   - NAS config.json must include `stage_5` section with LLM configuration
   - LLM API credentials configured in .env file (same as Stage 4)

2. **Execute Q&A Pairing**:
   ```bash
   python stage_5_qa_pairing/5_transcript_qa_pairing.py
   ```

3. **Expected Outputs**:
   - `qa_paired_transcript_sections.json`: Complete records with Q&A group assignments
   - Each paragraph gets 3 new fields: qa_group_id, qa_group_confidence, qa_group_method
   - Speaker block-based analysis with dynamic context windows
   - Development mode processes only 2 transcripts for testing

4. **Development Mode Settings**:
   ```json
   "stage_5": {
     "dev_mode": true,
     "dev_max_transcripts": 2,
     "window_config": {
       "context_blocks_before": 3,
       "context_blocks_after": 2,
       "dynamic_extension": true,
       "full_paragraphs": true
     },
     "llm_config": {
       "model": "gpt-4-turbo",
       "temperature": 0.1,
       "max_tokens": 1500,
       "base_url": "https://your-llm-api.com/v1",
       "token_endpoint": "https://oauth.your-api.com/token",
       "cost_per_1k_prompt_tokens": 0.03,
       "cost_per_1k_completion_tokens": 0.06
     }
   }
   ```

5. **Q&A Pairing Results**:
   - **qa_group_id**: Integer grouping related question-answer exchanges
   - **qa_group_confidence**: 0.0-1.0 aggregated confidence across speaker blocks
   - **qa_group_method**: "llm_detection", "llm_detection_medium_confidence", or "xml_fallback"

6. **Key Features**:
   - **Speaker Block Analysis**: Processes complete speaker blocks (not individual paragraphs)
   - **Dynamic Context Windows**: Extends back to question start when analyzing continuations
   - **Enhanced Speaker Formatting**: Clear role indicators ([ANALYST], [EXECUTIVE], [OPERATOR])
   - **Per-Transcript OAuth Refresh**: Fresh token for each transcript (eliminates expiration issues)
   - **Comprehensive Fallbacks**: LLM → XML type attributes → conservative grouping
   - **Cost Tracking**: Real-time token usage and final cost summary

### Testing Stage 6

1. **Prerequisites**:
   - Stage 5 must have run successfully (creates `qa_paired_transcript_sections.json`)
   - NAS config.json must include `stage_6` section with LLM configuration
   - LLM API credentials configured in .env file (same as Stage 4/5)

2. **Execute Detailed Classification**:
   ```bash
   python stage_6_detailed_classification/6_transcript_detailed_classification.py
   ```

3. **Expected Outputs**:
   - `detailed_classified_sections.json`: Complete records with detailed financial category classifications
   - Each paragraph gets 1 new field: detailed_classification (categories, confidence, method)
   - Management Discussion: Speaker block windowing with 750-char prior context
   - Q&A Groups: Complete conversation analysis using Stage 5 boundaries
   - Development mode processes only 2 transcripts for testing

4. **Development Mode Settings**:
   ```json
   "stage_6": {
     "dev_mode": true,
     "dev_max_transcripts": 2,
     "processing_config": {
       "md_paragraph_window_size": 5,
       "prior_block_preview_chars": 750,
       "max_speaker_blocks_context": 2
     },
     "llm_config": {
       "model": "gpt-4-turbo",
       "temperature": 0.1,
       "max_tokens": 1500,
       "base_url": "https://your-llm-api.com/v1",
       "token_endpoint": "https://oauth.your-api.com/token",
       "cost_per_1k_prompt_tokens": 0.03,
       "cost_per_1k_completion_tokens": 0.06
     }
   }
   ```

5. **Classification Results**:
   - **Categories**: Array of applicable financial categories (10 core categories)
   - **Confidence**: 0.0-1.0 LLM confidence in classification decision
   - **Method**: "speaker_block_windowing" (Management Discussion) or "complete_conversation" (Q&A)

6. **Key Features**:
   - **Dual Processing Approach**: Different strategies for Management Discussion vs Q&A sections
   - **10 Financial Categories**: Comprehensive classification covering all major earnings call topics
   - **Flexible Classification**: No minimum/maximum category requirements - applies all relevant
   - **Enhanced Context**: 750-char prior speaker block previews with previous classifications
   - **Per-Transcript OAuth Refresh**: Fresh token for each transcript (eliminates expiration issues)
   - **LLM-Only Approach**: No fallback classifications - comprehensive error handling instead
   - **Cost Tracking**: Real-time token usage and comprehensive cost monitoring

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

### Stage 2: Transcript Consolidation ✅ PRODUCTION READY
- **Purpose**: Select optimal single transcript per company per fiscal quarter/year and detect changes
- **When to use**: After Stage 0/1 populate transcripts repository
- **Output**: Processing queues for downstream analysis (no database creation)
- **Features**:
  - 4-tier enhanced file selection (call type → transcript type → version → date)
  - Handles multiple earnings calls per quarter (prioritizes primary over secondary)
  - Read-only comparison with existing master database (if present)
  - Delta detection for changed/new/removed files
  - Processing queues (files_to_process.json, files_to_remove.json)
- **Usage**: `python stage_2_processing/2_transcript_consolidation.py`

### Stage 3: XML Content Extraction ✅ PRODUCTION READY
- **Purpose**: Process XML transcripts and extract paragraph-level content with speaker attribution
- **When to use**: After Stage 2 creates processing queues
- **Input**: files_to_process.json from Stage 2 output
- **Output**: extracted_transcript_sections.json with paragraph-level database records
- **Features**:
  - Development mode (process only 1-2 files for testing)
  - Complete field preservation from Stage 2 input
  - Paragraph-level breakdown with sequential ordering
  - Speaker attribution with formatted strings (Name, Title, Affiliation)
  - Q&A detection (question/answer flags from XML)
  - Section tracking (Presentation, Q&A Session, etc.)
  - Enhanced error logging with separate JSON files
- **Usage**: `python stage_3_content_processing/3_transcript_content_extraction.py`
- **Development**: Set `"dev_mode": true` in config.json to process limited files during testing

### Stage 4: LLM-Based Classification ✅ PRODUCTION READY
- **Purpose**: Add section type classification using 3-level progressive LLM analysis
- **When to use**: After Stage 3 creates extracted content records
- **Input**: extracted_transcript_sections.json from Stage 3 output
- **Output**: classified_transcript_sections.json with section type classifications
- **Features**:
  - 3-level progressive classification system (Direct → Breakpoint → Individual)
  - OAuth 2.0 authentication with SSL certificate management
  - CO-STAR prompt framework with structured function calling
  - Comprehensive cost tracking with configurable token rates and real-time budget monitoring
  - Full section context with paragraph-level character limits (750 chars per paragraph)
  - Development mode (process only 2 transcripts for testing)
  - Comprehensive error handling and recovery mechanisms
  - Progressive API cost optimization (avg 1.5 calls per section)
  - Adds exactly 3 fields: section_type, section_type_confidence, section_type_method
- **Usage**: `python stage_4_llm_classification/4_transcript_llm_classification.py`
- **Development**: Set `"dev_mode": true` in config.json to process limited transcripts during testing
- **Authentication**: Requires LLM_CLIENT_ID and LLM_CLIENT_SECRET environment variables

### Stage 5: Q&A Conversation Boundary Detection ✅ PRODUCTION READY
- **Purpose**: Identify and group question-answer conversation boundaries using state-driven speaker block analysis
- **When to use**: After Stage 4 creates section type classifications
- **Input**: classified_transcript_sections.json from Stage 4 output
- **Output**: qa_paired_transcript_sections.json with Q&A group relationship mappings
- **Features**:
  - **State-Driven Analysis**: Dynamic tool definitions based on current group state (active vs none)
  - **Automatic Group ID Assignment**: Sequential numbering (1,2,3...) prevents invalid sequences
  - **Enhanced Operator Detection**: Content pattern matching for "thank you, next question" statements
  - **Real-Time Validation**: Built-in state machine with auto-correction and retry logic
  - **Speaker Pattern Focus**: Simple analyst→executive exchange detection vs complex thematic analysis
  - **Complete Exchange Capture**: Ensures full question→answer→follow-up→closing sequences
  - **Speaker Block Analysis**: Processes complete speaker blocks (not individual paragraphs)
  - **Dynamic Context Windows**: Extends back to question starts when analyzing continuations
  - **Enhanced Speaker Formatting**: Clear role indicators ([ANALYST], [EXECUTIVE], [OPERATOR])
  - **Per-Transcript OAuth Refresh**: Fresh token for each transcript (eliminates expiration issues)
  - **Comprehensive Fallback Strategies**: LLM detection → XML type attributes → conservative grouping
  - **Minimal Schema Extension**: Only 3 new fields (qa_group_id, qa_group_confidence, qa_group_method)
  - **Cost Tracking**: Real-time token usage and final cost summary (no budget limits)
  - **Development Mode**: Process only 2 transcripts for testing
  - **Validation & Retry**: Invalid LLM responses caught and retried with feedback
- **Usage**: `python stage_5_qa_pairing/5_transcript_qa_pairing.py`
- **Development**: Set `"dev_mode": true` in config.json to process limited transcripts during testing
- **Authentication**: Uses same LLM credentials as Stage 4 (LLM_CLIENT_ID and LLM_CLIENT_SECRET)
- **Key Improvements** (2024-07-13):
  - Eliminated duplicate group ending issues with state machine enforcement
  - Automatic group ID sequencing prevents invalid sequences (1,3,2,3 → 1,1,2,2,3,3)
  - Enhanced operator detection excludes "thank you, next question" from Q&A groups
  - Real-time validation eliminates false warnings about decision inconsistencies
  - Simplified prompting focuses on speaker patterns vs complex thematic analysis

### Stage 6: Detailed Financial Category Classification ✅ PRODUCTION READY
- **Purpose**: Classify transcript content with detailed financial categories using dual processing approaches
- **When to use**: After Stage 5 creates Q&A group boundaries for comprehensive content analysis
- **Input**: qa_paired_transcript_sections.json from Stage 5 output
- **Output**: detailed_classified_sections.json with comprehensive financial category classifications
- **Features**:
  - **Dual Processing Approach**: Management Discussion (speaker block windowing) vs Q&A Groups (complete conversation analysis)
  - **10 Financial Categories**: Comprehensive classification covering all major earnings call topics (Financial Performance, Credit Quality, Capital Management, etc.)
  - **Management Discussion Processing**: 5-paragraph windows within speaker blocks with 750-char prior context
  - **Q&A Group Processing**: Complete conversation analysis using Stage 5 boundaries (single LLM call per group)
  - **Flexible Classification**: No minimum/maximum category requirements - applies all relevant categories
  - **Enhanced Context Formatting**: Full current speaker block + 750-char prior speaker block previews with previous classifications
  - **Progressive Context**: Each window sees previous classifications within current speaker block
  - **LLM-Only Approach**: No fallback classifications - comprehensive error handling instead
  - **Per-Transcript OAuth Refresh**: Fresh token for each transcript (eliminates expiration issues)
  - **CO-STAR Prompt Framework**: Structured prompts with embedded category descriptions
  - **Function Calling Schemas**: Separate schemas for paragraph-level and conversation-level classification
  - **Minimal Schema Extension**: Only 1 new field (detailed_classification with categories, confidence, method)
  - **Cost Tracking**: Real-time token usage and comprehensive cost monitoring with detailed summaries
  - **Development Mode**: Process only 2 transcripts for testing
  - **Enhanced Error Logging**: Separate categories for LLM, authentication, classification, and processing errors
- **Usage**: `python stage_6_detailed_classification/6_transcript_detailed_classification.py`
- **Development**: Set `"dev_mode": true` in config.json to process limited transcripts during testing
- **Authentication**: Uses same LLM credentials as Stage 4/5 (LLM_CLIENT_ID and LLM_CLIENT_SECRET)
- **Classification Categories**:
  - Financial Performance & Results, Credit Quality & Risk Management, Capital & Regulatory Management
  - Strategic Initiatives & Transformation, Market Environment & Outlook, Operating Efficiency & Expenses
  - Asset & Liability Management, Non-Interest Revenue & Segments, ESG & Sustainability, Insurance Operations

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

4. **LLM Authentication Failed (Stage 4)**
   - Verify LLM_CLIENT_ID and LLM_CLIENT_SECRET environment variables
   - Check OAuth token endpoint URL in configuration
   - Ensure SSL certificate is available on NAS

5. **Import Errors**
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