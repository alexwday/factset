# 🚀 Stage 6: LLM-Based Financial Content Classification

## 🎯 PROJECT CONTEXT
- Stage: 06_llm_classification - Single-pass comprehensive financial categorization using LLM
- Purpose: Classify Management Discussion and Q&A sections with all applicable categories
- Input: Stage 5 output (structured speaker blocks and Q&A groups)
- Output: Classified content with multiple category assignments using ID-based system
- Architecture: OpenAI client with OAuth authentication, single-pass classification system
- Resume Capability: Supports resuming from interrupted processing via existing output detection

## 🚨 CRITICAL RULES
- ALWAYS use category IDs (0-22) from CATEGORY_REGISTRY, not category names
- NEVER expose OAuth tokens or LLM credentials in logs
- MUST handle token costs and usage tracking for all LLM calls
- Single-pass system: Comprehensive classification returning all applicable categories
- Handle authentication token refresh per transcript to prevent expiration
- Validate all NAS paths to prevent directory traversal attacks
- Category 0 (Non-Relevant) with secondary categories triggers automatic promotion
- Enable resume capability by default to handle processing interruptions

## 🛠️ DEVELOPMENT WORKFLOW

### Classification Process
1. Load Stage 5 output with structured speaker blocks and Q&A groups
2. Process Management Discussion by speaker blocks with single-pass classification
3. Process Q&A conversations as complete units (entire Q&A group together)
4. Single Pass: Assign ALL applicable categories per speaker block/conversation
5. Apply same classifications to all paragraphs within a speaker block
6. Apply same classifications to all records in a Q&A conversation
7. Save results incrementally using JSON array pattern

### Quality Standards
```bash
# Validate classification results
python -c "import json; data=json.load(open('output.json')); print(f'Records: {len(data)}')"

# Check category registry
python -c "from main_llm_classification import CATEGORY_REGISTRY; print(f'Categories: {len(CATEGORY_REGISTRY)}')"

# Test OAuth token acquisition
python -c "from main_llm_classification import get_oauth_token; print('Token acquired' if get_oauth_token() else 'Failed')"
```

## 🤖 AI/ML CONFIGURATION

### LLM Setup
- Client: OpenAI-compatible with OAuth client credentials flow
- Model: Configured in stage_06_llm_classification.llm_config.model
- Temperature: Low (0.1-0.3) for consistent financial categorization
- Token refresh: Per transcript to prevent expiration during long processes
- Cost tracking: Automatic accumulation with detailed breakdown

### Classification Methodology
- **Single-Pass System**: Comprehensive classification returning all applicable categories
- **Management Discussion**: Process each speaker block independently with context from previous 2 blocks
- **Q&A Conversations**: Process entire conversation as one unit (all Q&A group records together)
- **Function Calling**: Structured outputs using classify_speaker_block and classify_qa_conversation functions
- **Category System**: ID-based (0-22) with registry lookup for names
- **Fallback Strategy**: Category 0 (Non-Relevant) for failed classifications
- **No Auto-Promotion**: Single-pass system assigns all categories directly

### Financial Categories
- Loaded from financial_categories.yaml in NAS Config folder (lines 357-375)
- Falls back to config.yaml if separate file not found
- ID-based system (0-22) with CATEGORY_REGISTRY global dict (lines 40, 428-437)
- Each category has: id, name, description
- Primary: Single category ID per paragraph
- Secondary: Multiple category IDs for comprehensive retrieval coverage

## 📁 PROJECT STRUCTURE
```
06_llm_classification/
  main_llm_classification.py    # Complete self-contained script
  CLAUDE.md                     # This file
```

## 🔧 ESSENTIAL COMMANDS

### Development & Testing
```bash
# Run Stage 6 classification
python database_refresh/06_llm_classification/main_llm_classification.py

# Development mode (limited transcripts)
# Set dev_mode: true and dev_max_transcripts in config.yaml

# Test OAuth authentication
python -c "from main_llm_classification import get_oauth_token; print(get_oauth_token())"

# Check category registry
python -c "from main_llm_classification import CATEGORY_REGISTRY; print(f'Categories: {len(CATEGORY_REGISTRY)}')" 

# List category IDs and names
python -c "from main_llm_classification import CATEGORY_REGISTRY; [print(f'{k}: {v["name"]}') for k,v in CATEGORY_REGISTRY.items()]"

# Check for existing output (enables resume)
ls -la <output_path>/stage_06_classified_content.json

# Check failed transcripts log
ls -la <output_path>/stage_06_failed_transcripts.json
```

### Environment Setup
```bash
# Required environment variables
export API_USERNAME="your_api_username"
export API_PASSWORD="your_api_password"
export PROXY_USER="your_proxy_user"
export PROXY_PASSWORD="your_proxy_password"
export PROXY_URL="proxy.company.com:8080"
export NAS_USERNAME="nas_user"
export NAS_PASSWORD="nas_password"
export NAS_SERVER_IP="192.168.1.100"
export NAS_SERVER_NAME="NAS_SERVER"
export NAS_SHARE_NAME="shared_folder"
export NAS_BASE_PATH="/base/path"
export NAS_PORT="445"
export CONFIG_PATH="config/config.yaml"
export CLIENT_MACHINE_NAME="CLIENT_PC"
export LLM_CLIENT_ID="your_llm_client_id"
export LLM_CLIENT_SECRET="your_llm_client_secret"
```

### Configuration Validation
```bash
# Check config structure
python -c "from main_llm_classification import validate_config_structure; import yaml; validate_config_structure(yaml.safe_load(open('config.yaml')))"

# Verify SSL certificate path
python -c "from main_llm_classification import setup_ssl_certificate; print('SSL setup successful')"
```

## 🔗 INTEGRATIONS

### Stage Dependencies
- **Input**: Stage 5 output (configured in `input_data_path`)
- **Output**: `stage_06_classified_content.json` (incremental JSON array)
- **Logs**: Execution logs, error logs by type (classification, authentication, validation, processing)
- **Failed Records**: `stage_06_failed_transcripts.json`
- **Category Config**: `financial_categories.yaml` and `monitored_institutions.yaml` from NAS Config folder

### External Services
- **LLM Service**: OpenAI-compatible API with OAuth authentication
- **NAS Storage**: SMB/CIFS connection for config, input/output data
- **SSL Certificates**: Corporate SSL certificate for secure connections
- **Proxy**: Corporate proxy with domain authentication

## 📋 CURRENT FOCUS
- **Single-Pass Classification**: Comprehensive classification returning all applicable categories
- **ID-Based System**: Categories referenced by numeric IDs (0-22) not names
- **Processing Efficiency**: Speaker blocks for MD, entire conversations for Q&A
- **Enhanced Error Handling**: EnhancedErrorLogger class tracks errors by type (lines 88-177)
- **Cost Management**: Token tracking and cost calculation (lines 135-141, 587-605)
- **OAuth Management**: Per-transcript token refresh (lines 569-584, called at line 1351)
- **No Validation Logic**: Single-pass system doesn't need auto-promotion

## ⚠️ KNOWN ISSUES
- **Authentication**: OAuth tokens may expire during processing - refresh per transcript implemented
- **Category Validation**: Invalid category IDs from LLM - validation with fallback to 0
- **Field Name Compatibility**: Stage 3/5 use 'speaker' while Stage 6 expects 'speaker_name' (normalized at lines 1496-1508)
- **Large Speaker Blocks**: May exceed token limits - consider chunking
- **Network Timeouts**: LLM API calls may timeout - retry logic in get_oauth_token (lines 509-552)
- **SSL Certificate**: Corporate certificates require proper path setup (lines 448-477)

## 🚫 DO NOT MODIFY
- OAuth token refresh mechanism (lines 569-584) - critical for long-running processes
- Category validation logic in validate_config_structure (lines 385-440) - ensures data integrity
- CATEGORY_REGISTRY building (lines 422-432) - core ID-to-category mapping
- Incremental saving pattern (lines 608-689) - maintains data integrity
- Security validation functions (lines 780-817) - prevent directory traversal attacks
- No auto-promotion logic - single-pass system doesn't need it

## 💡 DEVELOPMENT NOTES

### Classification Patterns
- **Single-Pass Architecture**: One comprehensive classification per speaker block/conversation
- **Management Discussion**: Process each speaker block with `process_speaker_block_single_pass` (lines 1139-1305)
- **Q&A Conversations**: Process with `process_qa_conversation_single_pass` (lines 969-1136)
- **Category IDs**: Use numeric IDs (0-22) internally, map to names for output
- **No Validation**: Single-pass assigns all categories directly
- **Error Fallback**: Assign category 0 (Non-Relevant) if classification fails

### Single-Pass Prompt Structure

**Management Discussion Classification**:
- Prompt built inline in `process_speaker_block_single_pass` (lines 1164-1224)
- Focus: ALL applicable categories for the entire speaker block
- Context: Previous 2 speaker blocks with their classifications
- Retrieval: Comprehensive coverage for all topics mentioned
- Guidelines: Include main topic AND all secondary references
- All paragraphs in block receive the same classifications

**Q&A Conversation Classification**:
- Prompt built inline in `process_qa_conversation_single_pass` (lines 1011-1055)
- Focus: ALL applicable categories for the entire Q&A conversation
- Context: No previous context (each Q&A is independent)
- Retrieval: Complete conversation retrieved for any topic mentioned
- Guidelines: Be exhaustive, include all topics in questions AND answers
- All records in conversation receive the same classifications

### Performance Optimization
- OAuth token refresh per transcript via refresh_oauth_token_for_transcript (lines 569-584)
- Incremental saving after each transcript (lines 1614-1616)
- Speaker block processing maintains context from previous 2 blocks (line 1373)
- Cost tracking with token usage calculation (lines 587-605)
- Enhanced error logging by failure type using EnhancedErrorLogger class
- Retry logic for OAuth token acquisition (lines 504-547)

## 🔍 CODE CONVENTIONS
- **Error Handling**: EnhancedErrorLogger class (lines 88-177) for categorized error tracking
- **Logging**: log_console (lines 54-62), log_execution (lines 65-73), log_error (lines 76-85)
- **Security**: validate_file_path (lines 780-786), validate_nas_path (lines 789-805), sanitize_url_for_logging (lines 808-817)
- **NAS Operations**: nas_path_join (821-829), nas_download_file (832-845), nas_upload_file (848-865)
- **Key Functions**: 
  - process_speaker_block_single_pass (lines 1139-1305)
  - process_qa_conversation_single_pass (lines 969-1136)
  - process_transcript_v2 (lines 1308-1441)

## 📊 DATA HANDLING
- **Input Format**: Stage 5 JSON array with speaker blocks and Q&A groups
- **Output Format**: JSON array with added fields:
  - `classification_ids`: Array of numeric IDs (0-22) for all applicable categories
  - `classification_names`: Array of category names from registry
- **Category Validation**: IDs validated against CATEGORY_REGISTRY
- **Incremental Saving**: save_results_incrementally (lines 608-639), append_records_to_json_array (642-668), close_json_array (671-689)
- **Failed Records**: Separate tracking in stage_06_failed_transcripts.json
- **Cost Tracking**: calculate_token_cost (lines 587-605) for token usage and cost calculation

## 🏗️ ARCHITECTURE DETAILS

### Processing Flow (main function lines 1444-1721)
1. **Initialization**: validate_environment_variables (lines 241-281)
2. **NAS Connection**: get_nas_connection (lines 284-322)
3. **Config Loading**: load_config_from_nas (lines 325-388) with financial_categories.yaml
4. **SSL Setup**: setup_ssl_certificate (lines 448-477)
5. **OAuth Setup**: get_oauth_token (lines 509-552), setup_llm_client (lines 555-572)
6. **Data Loading**: Parse Stage 5 JSON array, normalize field names (lines 1902-1914)
7. **Transcript Grouping**: Group by MD, Q&A groups, unpaired Q&A (lines 1936-1960)
8. **Single-Pass Classification**: 
   - MD: process_speaker_block_two_pass per speaker block
   - Q&A: process_qa_conversation_two_pass per conversation
9. **Incremental Saving**: After each transcript (lines 1978-1982)
10. **Cleanup**: Close JSON array, save logs, cleanup SSL cert

### Key Functions

**Setup Functions**:
- validate_environment_variables (235-276)
- get_nas_connection (279-317)
- load_config_from_nas (320-382)
- setup_ssl_certificate (443-472)
- setup_proxy_configuration (475-501)
- get_oauth_token (504-547)
- setup_llm_client (550-566)

**Resume Functions**:
- load_existing_output (lines 714-776) - Check for and load completed transcripts
- nas_file_exists (lines 692-711) - Check file existence on NAS

**Classification Functions**:
- process_speaker_block_single_pass (1139-1305) - MD speaker blocks
- process_qa_conversation_single_pass (969-1136) - Q&A conversations
- process_transcript_v2 (1308-1441) - Main transcript processor

**Prompt Creation**:
- Prompts are built inline within classification functions
- MD prompt in process_speaker_block_single_pass (lines 1164-1224)
- Q&A prompt in process_qa_conversation_single_pass (lines 1011-1055)

**Helper Functions**:
- build_numbered_category_list (891-900)
- format_previous_speaker_blocks_with_classifications (903-942)
- format_current_speaker_block (945-964)
- calculate_token_cost (587-605)
- refresh_oauth_token_for_transcript (569-584)

## 🎛️ CONFIGURATION PARAMETERS

### Stage 6 Specific Config
```yaml
stage_06_llm_classification:
  description: "LLM-based financial content classification"
  input_data_path: "path/to/stage_05_qa_pairs.json"
  output_data_path: "path/to/output/directory"
  output_logs_path: "path/to/logs/directory"
  dev_mode: false
  dev_max_transcripts: 2
  enable_resume: true  # Resume from interruption (default: true)
  llm_config:
    base_url: "https://llm.api.endpoint"
    model: "gpt-4-turbo"
    temperature: 0.1
    max_tokens: 4096
    timeout: 120
    max_retries: 3
    token_endpoint: "https://auth.endpoint/token"
    cost_per_1k_prompt_tokens: 0.01
    cost_per_1k_completion_tokens: 0.03
  processing_config:
    md_paragraph_window_size: 5
  financial_categories:
    - name: "Revenue"
      description: "Revenue recognition, sales figures, and top-line growth"
      key_indicators: "sales, revenue, top-line, booking"
      # ... additional category details
```