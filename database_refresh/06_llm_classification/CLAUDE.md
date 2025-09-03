# 🚀 Stage 6: LLM-Based Financial Content Classification

## 🎯 PROJECT CONTEXT
- Stage: 06_llm_classification - Two-pass financial content categorization using LLM
- Purpose: Classify Management Discussion and Q&A sections with primary and secondary categories
- Input: Stage 5 output (structured speaker blocks and Q&A groups)
- Output: Classified content with primary/secondary category assignments using ID-based system
- Architecture: OpenAI client with OAuth authentication, two-pass classification system
- Resume Capability: Supports resuming from interrupted processing via existing output detection

## 🚨 CRITICAL RULES
- ALWAYS use category IDs (0-22) from CATEGORY_REGISTRY, not category names
- NEVER expose OAuth tokens or LLM credentials in logs
- MUST handle token costs and usage tracking for all LLM calls
- Two-pass system: Primary classification (single category) then Secondary (multiple categories)
- Handle authentication token refresh per transcript to prevent expiration
- Validate all NAS paths to prevent directory traversal attacks
- Category 0 (Non-Relevant) with secondary categories triggers automatic promotion
- Enable resume capability by default to handle processing interruptions

## 🛠️ DEVELOPMENT WORKFLOW

### Classification Process
1. Load Stage 5 output with structured speaker blocks and Q&A groups
2. Process Management Discussion by speaker blocks with two-pass classification
3. Process Q&A groups by speaker blocks within conversations (not whole conversations)
4. Pass 1: Assign one primary category per paragraph/speaker block
5. Pass 2: Assign comprehensive secondary categories for retrieval coverage
6. Validation: Fix Non-Relevant primary with secondary categories (auto-promotion)
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
- **Two-Pass System**: Primary (single category) then Secondary (multiple categories) classification
- **Management Discussion**: Process each speaker block independently with context from previous blocks
- **Q&A Conversations**: Process by speaker blocks within conversation (maintains question-answer context)
- **Function Calling**: Structured outputs using classify_speaker_block_primary/secondary functions
- **Category System**: ID-based (0-22) with registry lookup for names
- **Fallback Strategy**: Category 0 (Non-Relevant) for failed classifications
- **Auto-Promotion**: If primary is 0 but secondary exists, promote first secondary to primary

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
- **Two-Pass Classification**: Primary (priority) and Secondary (comprehensive coverage) categories
- **ID-Based System**: Categories referenced by numeric IDs (0-22) not names
- **Processing Efficiency**: Speaker blocks for both MD and Q&A (maintains context)
- **Enhanced Error Handling**: EnhancedErrorLogger class tracks errors by type (lines 87-183)
- **Cost Management**: Separate tracking for primary vs secondary pass tokens (lines 143-146, 1294-1297, 1455-1460)
- **OAuth Management**: Per-transcript token refresh (lines 574-589, called at line 1751)
- **Validation Logic**: Auto-promotion when primary is 0 but secondary exists (lines 1470-1483, 1674-1688)

## ⚠️ KNOWN ISSUES
- **Authentication**: OAuth tokens may expire during processing - refresh per transcript implemented
- **Category Validation**: Invalid category IDs from LLM - validation with fallback to 0
- **Field Name Compatibility**: Stage 3/5 use 'speaker' while Stage 6 expects 'speaker_name' (normalized at lines 1902-1914)
- **Large Speaker Blocks**: May exceed token limits - consider chunking
- **Network Timeouts**: LLM API calls may timeout - retry logic in get_oauth_token (lines 509-552)
- **SSL Certificate**: Corporate certificates require proper path setup (lines 448-477)

## 🚫 DO NOT MODIFY
- OAuth token refresh mechanism (lines 574-589) - critical for long-running processes
- Category validation logic in validate_config_structure (lines 390-445) - ensures data integrity
- CATEGORY_REGISTRY building (lines 428-437) - core ID-to-category mapping
- Incremental saving pattern (lines 613-694) - maintains data integrity
- Security validation functions (lines 698-735) - prevent directory traversal attacks
- Auto-promotion logic (lines 1470-1483, 1674-1688) - fixes Non-Relevant with secondary categories

## 💡 DEVELOPMENT NOTES

### Classification Patterns
- **Two-Pass Architecture**: Separate prompts and tools for primary vs secondary classification
- **Management Discussion**: Process each speaker block with `process_speaker_block_two_pass` (lines 1503-1706)
- **Q&A Conversations**: Process with `process_qa_conversation_two_pass` (lines 1126-1501)
- **Category IDs**: Use numeric IDs (0-22) internally, map to names for output
- **Validation**: Auto-promotion logic when primary=0 but secondary exists
- **Error Fallback**: Assign category 0 (Non-Relevant) if classification fails

### Two-Pass Prompt Structure

**Primary Classification Prompts**:
- create_primary_classification_prompt (lines 885-956): MD speaker blocks
- Q&A primary prompt inline (lines 1183-1224): Q&A speaker blocks  
- Focus: Single MOST relevant category per paragraph (MD) or per speaker block (Q&A)
- Context: Previous speaker blocks with classifications (up to 5 blocks)
- Retrieval: Primary category determines priority in search results
- Guidelines: Choose dominant financial theme, not minor mentions
- Important: Category 0 (Non-Relevant) only for pure procedural content

**Secondary Classification Prompts**:
- create_secondary_classification_prompt (lines 958-1044): MD speaker blocks  
- Q&A secondary prompt inline (lines 1332-1374): Q&A speaker blocks
- Focus: ALL other relevant categories (comprehensive coverage)
- Context: Current block with primary classifications already assigned
- Retrieval: Safety net ensuring content found through multiple search paths
- Guidelines: Be exhaustive, include all mentioned/referenced topics
- Validation: Never include primary category in secondary list

### Performance Optimization
- OAuth token refresh per transcript via refresh_oauth_token_for_transcript (lines 574-589)
- Incremental saving after each transcript (lines 1978-1982)
- Speaker block processing maintains context from previous 5 blocks (line 1773)
- Cost tracking with primary/secondary pass separation (lines 143-146)
- Enhanced error logging by failure type using EnhancedErrorLogger class
- Retry logic for OAuth token acquisition (lines 509-552)

## 🔍 CODE CONVENTIONS
- **Error Handling**: EnhancedErrorLogger class (lines 87-183) for categorized error tracking
- **Logging**: log_console (lines 53-62), log_execution (lines 64-73), log_error (lines 75-84)
- **Security**: validate_file_path (lines 698-704), validate_nas_path (lines 707-723), sanitize_url_for_logging (lines 726-735)
- **NAS Operations**: nas_path_join (739-747), nas_download_file (750-763), nas_upload_file (766-783)
- **Key Functions**: 
  - process_speaker_block_two_pass (lines 1503-1706)
  - process_qa_conversation_two_pass (lines 1126-1501)
  - process_transcript_v2 (lines 1708-1847)

## 📊 DATA HANDLING
- **Input Format**: Stage 5 JSON array with speaker blocks and Q&A groups
- **Output Format**: JSON array with added fields:
  - `primary_category_id`: Numeric ID (0-22)
  - `primary_category_name`: Category name from registry
  - `secondary_category_ids`: Array of numeric IDs
  - `secondary_category_names`: Array of category names
- **Category Validation**: IDs validated against CATEGORY_REGISTRY
- **Incremental Saving**: save_results_incrementally (lines 613-644), append_records_to_json_array (647-673), close_json_array (676-694)
- **Failed Records**: Separate tracking in stage_06_failed_transcripts.json
- **Cost Tracking**: calculate_token_cost (lines 592-610) with separate primary/secondary tracking

## 🏗️ ARCHITECTURE DETAILS

### Processing Flow (main function lines 1849-2089)
1. **Initialization**: validate_environment_variables (lines 241-281)
2. **NAS Connection**: get_nas_connection (lines 284-322)
3. **Config Loading**: load_config_from_nas (lines 325-388) with financial_categories.yaml
4. **SSL Setup**: setup_ssl_certificate (lines 448-477)
5. **OAuth Setup**: get_oauth_token (lines 509-552), setup_llm_client (lines 555-572)
6. **Data Loading**: Parse Stage 5 JSON array, normalize field names (lines 1902-1914)
7. **Transcript Grouping**: Group by MD, Q&A groups, unpaired Q&A (lines 1936-1960)
8. **Two-Pass Classification**: 
   - MD: process_speaker_block_two_pass per speaker block
   - Q&A: process_qa_conversation_two_pass per conversation
9. **Incremental Saving**: After each transcript (lines 1978-1982)
10. **Cleanup**: Close JSON array, save logs, cleanup SSL cert

### Key Functions

**Setup Functions**:
- validate_environment_variables (241-281)
- get_nas_connection (284-322)
- load_config_from_nas (325-388)
- setup_ssl_certificate (448-477)
- setup_proxy_configuration (480-507)
- get_oauth_token (509-552)
- setup_llm_client (555-572)

**Resume Functions**:
- load_existing_output (lines 734-793) - Check for and load completed transcripts
- nas_file_exists (lines 714-731) - Check file existence on NAS

**Classification Functions**:
- process_speaker_block_two_pass (1503-1706) - MD speaker blocks
- process_qa_conversation_two_pass (1126-1501) - Q&A conversations
- process_transcript_v2 (1708-1847) - Main transcript processor

**Prompt Creation**:
- create_primary_classification_prompt (885-956)
- create_secondary_classification_prompt (958-1044)
- create_primary_classification_tools (1047-1082)
- create_secondary_classification_tools (1085-1123)

**Helper Functions**:
- index_speaker_block_paragraphs (809-816)
- build_numbered_category_list (819-828)
- format_previous_speaker_blocks_with_classifications (831-868)
- format_current_speaker_block (871-882)
- calculate_token_cost (592-610)
- refresh_oauth_token_for_transcript (574-589)

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