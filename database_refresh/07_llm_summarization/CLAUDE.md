# 🚀 FactSet Project - Stage 7: LLM Summarization

## 🎯 PROJECT CONTEXT
- Type: LLM-based content summarization for financial earnings call transcripts
- Stack: Python 3.11+, OpenAI API, SMB/NAS integration
- Architecture: Self-contained summarization stage with OAuth authentication
- Focus: Creating block-level summaries for Q&A groups and Management Discussion speaker blocks
- Main Script: `main_llm_summarization.py` (Lines 1-1454)
- Creates condensed summaries at the block level:
  - Q&A Groups: One summary per complete Q&A conversation (all paragraphs in group get same summary)
  - Management Discussion: One summary per speaker block (all paragraphs in block get same summary)
  - Focus on speaker attribution, key points, and retrieval relevance for reranking models

## 🚨 CRITICAL RULES
- ALWAYS validate LLM API responses before processing
- NEVER commit LLM API keys or OAuth credentials
- MUST handle OAuth token refresh for each transcript
- Monitor token usage and costs continuously (accumulates costs)
- Handle financial transcript data with appropriate security measures
- MUST validate all NAS paths to prevent directory traversal attacks

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/*, bugfix/*, hotfix/*
- Commit format: conventional commits preferred
- Push workflow: commit → push → CI checks → PR
- Merge strategy: squash and merge for features

### Code Quality Standards
```bash
# Pre-commit checks
python -m black . --line-length 88
python -m isort . --profile black
python -m pylint database_refresh/07_llm_summarization/ --min-score 8.0
python -m flake8 database_refresh/07_llm_summarization/ --max-line-length 88
```

### Testing Strategy
- Local: Unit tests with mocked LLM responses
- Integration: Full summarization pipeline tests
- Coverage: Minimum 80% for new code
- AI-specific: Validate summary quality and relevance

## 🤖 AI/ML CONFIGURATION

### Model Setup
- Primary: OpenAI-compatible API (configurable model via `llm_config.model`)
- OAuth authentication with per-transcript token refresh (Line 527)
- Temperature: Configurable via `llm_config.temperature` (Lines 1002, 1170)
- Max tokens: Configurable via `llm_config.max_tokens` (Lines 1003, 1171)
- Timeout: Configurable via `llm_config.timeout` (Lines 515, 1004, 1172)
- SSL certificate handling for enterprise environments (Lines 438-466)

### Processing Pipeline
- Input: Stage 6 classified content from `input_data_path` (JSON array format)
- Processing: Two main summarization types:
  - Q&A Groups: Complete conversation summaries (Lines 942-1060)
  - Management Discussion: Speaker block-level summaries (Lines 1062-1228)
- Output: Enhanced records with `block_summary` field to `output_data_path`
- Incremental saving: Append results after each transcript using Stage 6 pattern

## 📁 PROJECT STRUCTURE
```
database_refresh/07_llm_summarization/
  main_llm_summarization.py    # Main summarization script (Lines 1-1411)
  CLAUDE.md                    # This file
```

## 🏗️ CORE COMPONENTS

### Environment Variables Required
- `API_USERNAME`, `API_PASSWORD`: FactSet API credentials
- `PROXY_USER`, `PROXY_PASSWORD`, `PROXY_URL`: Corporate proxy settings
- `NAS_USERNAME`, `NAS_PASSWORD`: NAS authentication
- `NAS_SERVER_IP`, `NAS_SERVER_NAME`, `NAS_SHARE_NAME`: NAS connection details
- `NAS_BASE_PATH`, `NAS_PORT`: NAS path and port configuration
- `CONFIG_PATH`: Path to shared configuration file on NAS
- `CLIENT_MACHINE_NAME`: SMB client identification
- `LLM_CLIENT_ID`, `LLM_CLIENT_SECRET`: OAuth credentials for LLM API

### Global State Management (Lines 36-45)
- `config`: Global configuration loaded from NAS (Line 37)
- `logger`: Console logging handler (Line 38)
- `execution_log`: Detailed execution tracking (Line 39)
- `error_log`: Error tracking list (Line 40)
- `llm_client`: OpenAI client instance (Line 43)
- `oauth_token`: Current OAuth access token (Line 44)
- `ssl_cert_path`: Path to temporary SSL certificate (Line 45)

### Logging Functions (Lines 48-90)
- `setup_logging()`: Configure console logging (Lines 48-56)
- `log_console()`: Minimal console output with level support (Lines 58-67)
- `log_execution()`: Detailed execution logging with details dict (Lines 69-78)
- `log_error()`: Error tracking with error_type categorization (Lines 80-90)

### Save Logs Function (Lines 183-237)
- `save_logs_to_nas()`: Export all logs and metrics (Lines 183-237)
- Saves both execution and error logs to NAS
- Creates separate directories for different log types
- Includes stage summary and metrics
- Calls enhanced error logger's save_error_logs() method

### Environment Validation (Lines 239-281)
- `validate_environment_variables()`: Check all required env vars (Lines 239-281)
- Validates all required environment variables at startup
- Comprehensive error reporting for missing variables
- Lists all required variables explicitly
- Required variables:
  - API credentials: API_USERNAME, API_PASSWORD
  - Proxy settings: PROXY_USER, PROXY_PASSWORD, PROXY_URL
  - NAS connection: NAS_USERNAME, NAS_PASSWORD, NAS_SERVER_IP, etc.
  - LLM OAuth: LLM_CLIENT_ID, LLM_CLIENT_SECRET

### Enhanced Error Logger Class (Lines 92-181)
- Categorizes errors by type: summarization, authentication, processing, validation
- Accumulates costs and token usage (Lines 139-146)
- Saves categorized error logs to NAS (Lines 147-181)
- Methods:
  - `log_summarization_error()`: Track LLM summarization failures (Lines 103-111)
  - `log_authentication_error()`: Track OAuth/SSL issues (Lines 113-119)
  - `log_processing_error()`: Track general processing errors (Lines 121-128)
  - `log_validation_error()`: Track validation issues (Lines 130-137)
  - `save_error_logs()`: Export categorized errors to NAS (Lines 147-181)

### NAS Operations (Lines 283-400)
- `get_nas_connection()`: SMB connection establishment (Lines 283-328)
- `nas_path_join()`: Path joining utility (Lines 330-333)
- `nas_file_exists()`: Check file existence (Lines 336-342)
- `nas_download_file()`: Download from NAS (Lines 345-359)
- `nas_create_directory_recursive()`: Create directories (Lines 361-379)
- `nas_upload_file()`: Upload to NAS (Lines 381-400)

### Configuration & Setup (Lines 402-543)
- `load_stage_config()`: Load YAML config from NAS (Lines 402-436)
- `setup_ssl_certificate()`: SSL cert setup (Lines 438-466)
- `get_oauth_token()`: OAuth token acquisition (Lines 469-507)
- `setup_llm_client()`: OpenAI client initialization (Lines 509-525)
- `refresh_oauth_token_for_transcript()`: Per-transcript refresh (Lines 527-543)

### Cost Calculation (Lines 545-564)
- `calculate_token_cost()`: Token usage to cost conversion (Lines 545-564)
- Uses configurable rates from config file

### Incremental Saving (Lines 566-648)
- `save_results_incrementally()`: Save after each transcript (Lines 566-598)
- `append_records_to_json_array()`: Append to JSON array (Lines 600-627)
- `close_json_array()`: Finalize JSON output (Lines 629-648)

### Security Validation (Lines 651-689)
- `validate_file_path()`: Directory traversal prevention (Lines 651-657)
- `validate_nas_path()`: NAS path security (Lines 660-677)
- `sanitize_url_for_logging()`: Remove credentials from logs (Lines 679-689)

### Data Loading & Grouping (Lines 691-774)
- `load_stage6_data()`: Load classified content (Lines 691-753)
  - Handles unclosed JSON arrays from Stage 6 (Lines 710-723)
  - Attempts to fix trailing commas and missing closing brackets
  - Provides detailed JSON parsing error context (Lines 729-744)
- `group_records_by_transcript()`: Group by transcript key (Lines 755-774)
  - Uses filename as primary grouping key
  - Fallback to ticker_event_id format if filename missing

### LLM Tools & Prompts (Lines 777-939)
- `create_qa_group_summary_tools()`: Q&A function schema (Lines 777-795)
- `create_management_summary_tools()`: Management function schema (Lines 798-816)
- `create_qa_group_summary_prompt()`: Q&A prompt generation (Lines 819-873)
- `create_management_speaker_block_prompt()`: Management prompt (Lines 876-939)

### Q&A Processing (Lines 942-1060)
- `process_transcript_qa_groups()`: Main Q&A processing (Lines 942-1060)
- Groups by qa_group_id (Lines 947-954)
- Creates conversation context with speaker attribution (Lines 963-984)
- Applies same summary to all paragraphs in group (Lines 1014-1016)
- Handles LLM errors gracefully (Lines 1033-1044)

### Management Discussion Processing (Lines 1062-1228)
- `process_transcript_management()`: Main Management processing (Lines 1062-1228)
- Groups by speaker_block_id (Lines 1072-1076)
- Uses sliding window context (previous 2 blocks) (Lines 1110-1140)
- Determines block position (opening/middle/closing) (Lines 1088-1094)
- Creates block-specific summaries (Lines 1183-1184)

### Other Records Processing (Lines 1230-1252)
- `process_other_records()`: Handle non-standard sections (Lines 1230-1252)
- Creates basic summaries for other content types (Lines 1239-1249)

### Transcript Processing (Lines 1254-1278)
- `process_transcript()`: Orchestrate all section processing (Lines 1254-1278)
- Combines Management, Q&A, and other sections (Lines 1260-1265)
- Returns enhanced records with summaries (Line 1268)

### Failed Transcript Handling (Lines 1280-1307)
- `save_failed_transcripts()`: Save failed transcript list (Lines 1280-1307)
- Creates separate JSON file with failure details (Lines 1286-1291)

### Main Execution (Lines 1309-1454)
- `main()`: Orchestrate complete Stage 7 execution (Lines 1309-1454)
- Environment validation and NAS connection (Lines 1319-1325, using validate_environment_variables from Lines 239-281)
- Configuration loading and SSL setup (Lines 1327-1333)
- Per-transcript processing with OAuth refresh (Lines 1362-1396)
- Incremental saving and error tracking (Lines 1376-1378)
- Final metrics and log saving (Lines 1407-1439)

## 🔧 ESSENTIAL COMMANDS

### Development
```bash
# Environment setup (from project root)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Load environment variables
source .env  # or use python-dotenv

# Run Stage 7 summarization
cd database_refresh/07_llm_summarization
python main_llm_summarization.py
```

### Stage-Specific Operations
```bash
# Run full Stage 7 summarization
python database_refresh/07_llm_summarization/main_llm_summarization.py

# Development mode (limited transcripts)
# Set dev_mode: true in config.yaml

# Check logs after execution
# Logs saved to NAS output_logs_path directory
```

### Debugging Commands
```bash
# Test OAuth token acquisition (Line 469)
python -c "from main_llm_summarization import get_oauth_token; print(get_oauth_token())"

# Test NAS connection (Line 283)
python -c "from main_llm_summarization import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Validate environment variables (Line 239)
python -c "from main_llm_summarization import validate_environment_variables; validate_environment_variables()"
```

## 🔗 INTEGRATIONS

### Data Sources
- Input: Stage 6 classified content from NAS
- Configuration: Shared YAML config file from NAS
- SSL Certificate: Downloaded from NAS for LLM API access

### Processing Tools
- OpenAI Python client for LLM interactions (Line 509)
- SMBConnection for NAS file operations (Lines 283-328)
- OAuth 2.0 client credentials flow for authentication (Lines 469-507)
- JSON incremental file writing using Stage 6 pattern (Lines 566-648)
- Function calling for structured responses (Lines 777-816)

## 📋 CURRENT FOCUS
- LLM-based summarization for earnings call transcripts
- Q&A group conversation summaries (all paragraphs get same summary) - Lines 942-1060
- Management Discussion speaker block summaries (with sliding window context) - Lines 1062-1228
- Per-transcript OAuth token refresh to prevent expiration - Line 527
- Enhanced error logging with separate error type categorization - Lines 92-181

## ⚠️ KNOWN ISSUES
- OAuth tokens can expire during long processing runs (mitigated by per-transcript refresh)
- Large transcripts may hit LLM token limits (handled with context windowing)
- SSL certificate setup required for enterprise LLM endpoints
- Network timeouts during NAS operations (retry logic needed)

## 🚫 DO NOT MODIFY
- OAuth credential environment variables
- SSL certificate validation logic
- NAS path security validation functions
- Cost calculation formulas (aligned with billing)

## 💡 DEVELOPMENT NOTES
- Use per-transcript OAuth refresh to handle long processing sessions (Line 1369)
- Q&A groups get single summary applied to all paragraphs in the conversation (Lines 1014-1016)
- Management Discussion uses speaker blocks with sliding window context (Lines 1110-1140)
- Enhanced error logger categorizes different failure types (Lines 92-181)
- Incremental saving prevents data loss during processing (Line 1376)
- Function calling used for structured LLM responses (Lines 1000-1001, 1168-1169)
- Stage 6 JSON format handling includes fixing unclosed arrays (Lines 710-723)

## 🔍 CODE CONVENTIONS
- Follow existing Stage 5/6 patterns for NAS operations
- Use `EnhancedErrorLogger` class for categorized error tracking (Lines 92-181)
- Implement per-transcript processing with OAuth refresh (Line 1369)
- Use sliding window context for Management Discussion blocks (Lines 1110-1140)
- Apply security validation to all file paths and URLs (Lines 651-689)
- Log execution details for audit and debugging

## 📊 DATA HANDLING
- Input: Stage 6 classified content records (JSON array, may be unclosed)
- Processing: Group by transcript using filename as key (Lines 755-774)
- Q&A Processing: Group by qa_group_id, summarize conversations (Lines 947-954)
- Management Processing: Group by speaker_block_id, use context windows (Lines 1072-1076)
- Output: Enhanced records with block_summary field (not paragraph_summary)
- Incremental saving: Stage 6 JSON array append pattern (Lines 600-627)
- Error handling: Null summary values for failed processing (Lines 1041-1043, 1208-1210)

## 🎨 SUMMARIZATION FEATURES

### Q&A Group Summarization (Lines 942-1060)
- Groups records by `qa_group_id` within transcript (Lines 947-954)
- Creates single summary for entire Q&A conversation
- Applies same summary to all paragraphs in the group (Lines 1014-1016)
- Optimized for post-retrieval reranking
- Includes speaker attribution and financial context (Lines 966-974)
- Uses `create_qa_group_summary_prompt()` (Lines 819-873)
- Function calling with `create_qa_group_summary_tools()` (Lines 777-795)

### Management Discussion Summarization (Lines 1062-1228)
- Groups records by `speaker_block_id` within transcript (Lines 1072-1076)
- Creates individual summaries for each speaker block
- Uses sliding window context (previous 2 speaker block summaries) (Lines 1110-1140)
- Distinguishes between opening, middle, and closing blocks (Lines 1088-1094)
- Preserves speaker names and block positioning context
- Uses `create_management_speaker_block_prompt()` (Lines 876-939)
- Function calling with `create_management_summary_tools()` (Lines 798-816)

### Summary Optimization
- Function calling for structured LLM responses
- Reranking-optimized prompts with financial terminology
- Speaker attribution preservation for relevance filtering
- Quantitative data and forward-looking statement emphasis
- Company and fiscal period context integration
- Three-part summary structure:
  1. SPEAKERS: Names and roles
  2. CONTENT: 2-3 sentence summary
  3. RELEVANCE: Query matching guidance

## 🔒 SECURITY CONSIDERATIONS
- OAuth client credentials for LLM API access
- SSL certificate validation for enterprise environments
- NAS path traversal attack prevention
- Sensitive data sanitization in logs
- Temporary SSL certificate file cleanup
- Environment variable validation for required credentials

## 💰 COST TRACKING
- Per-request token usage tracking (Lines 1021-1029, 1189-1197)
- Cost calculation based on prompt/completion tokens (Lines 545-564)
- Total cost accumulation across all transcripts (Lines 139-146)
- Enhanced error logs include cost breakdown
- Development mode for cost-controlled testing (Lines 1341-1346)

## 🚀 PERFORMANCE OPTIMIZATIONS
- Per-transcript OAuth token refresh (Lines 1368-1370)
- Incremental result saving after each transcript (Line 1376)
- Rate limiting between transcript processing (Line 1384)
- Sliding window context to reduce token usage (Lines 1110-1140)
- Function calling for efficient structured responses (Lines 777-816)
- Handles unclosed JSON arrays from Stage 6 (Lines 710-723)