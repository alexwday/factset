# 🚀 Stage 6: LLM-Based Financial Content Classification

## 🎯 PROJECT CONTEXT
- Stage: 06_llm_classification - Financial content categorization using LLM
- Purpose: Classify Management Discussion and Q&A sections into financial categories
- Input: Stage 5 output (structured speaker blocks and Q&A groups)
- Output: Classified content with category assignments and confidence scores
- Architecture: OpenAI client with OAuth authentication, speaker-block windowing, conversation-level classification

## 🚨 CRITICAL RULES
- ALWAYS validate financial categories against predefined set in config
- NEVER expose OAuth tokens or LLM credentials in logs
- MUST handle token costs and usage tracking for all LLM calls
- Monitor classification confidence scores and fallback to "Other" for errors
- Handle authentication token refresh per transcript to prevent expiration
- Validate all NAS paths to prevent directory traversal attacks

## 🛠️ DEVELOPMENT WORKFLOW

### Classification Process
1. Load Stage 5 output with structured speaker blocks and Q&A groups
2. Group Management Discussion by speaker blocks, process in configurable windows
3. Process Q&A groups as complete conversations (single LLM call per group)
4. Apply CO-STAR prompt methodology with comprehensive category descriptions
5. Save results incrementally using Stage 5's JSON array pattern

### Quality Standards
```bash
# Validate classification results
python -c "import json; data=json.load(open('output.json')); print(f'Records: {len(data)}')"

# Check category validation
python -c "from main_llm_classification import validate_categories; print(validate_categories(['Revenue', 'Other']))"

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
- **Management Discussion**: Speaker-block windowing with configurable window size
- **Q&A Conversations**: Complete conversation analysis (all paragraphs together)
- **CO-STAR Prompts**: Context, Objective, Style, Tone, Audience, Response format
- **Function Calling**: Structured outputs with category validation
- **Fallback Strategy**: "Other" category for failed classifications

### Financial Categories
- Loaded from config.yaml `financial_categories` section
- Each category includes: name, description, key_indicators, use_when, do_not_use_when, example_phrases
- Validation ensures only valid categories are assigned
- Support for multiple categories per content section

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

# Validate categories from config
python -c "from main_llm_classification import validate_categories, VALID_CATEGORIES; print(f'Categories: {len(VALID_CATEGORIES)}')"
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
- **Input**: Stage 5 output (`stage_05_qa_pairs.json`)
- **Output**: `stage_06_classified_content.json` (incremental JSON array)
- **Logs**: Execution logs, error logs by type (classification, authentication, validation, processing)
- **Failed Records**: `stage_06_failed_transcripts.json`

### External Services
- **LLM Service**: OpenAI-compatible API with OAuth authentication
- **NAS Storage**: SMB/CIFS connection for config, input/output data
- **SSL Certificates**: Corporate SSL certificate for secure connections
- **Proxy**: Corporate proxy with domain authentication

## 📋 CURRENT FOCUS
- **Classification Accuracy**: CO-STAR prompt methodology with comprehensive category descriptions
- **Processing Efficiency**: Speaker-block windowing for Management Discussion, conversation-level for Q&A
- **Error Handling**: Enhanced error logging by type with detailed failure analysis
- **Cost Management**: Token usage tracking and cost calculation per transcript
- **OAuth Management**: Per-transcript token refresh to prevent expiration

## ⚠️ KNOWN ISSUES
- **Authentication**: OAuth tokens may expire during processing - refresh per transcript implemented
- **Category Validation**: Invalid categories from LLM responses - validation and fallback to "Other"
- **Large Conversations**: Q&A groups with many paragraphs may exceed token limits
- **Network Timeouts**: LLM API calls may timeout - retry logic implemented
- **SSL Certificate**: Corporate certificates require proper path setup

## 🚫 DO NOT MODIFY
- OAuth token refresh mechanism - critical for long-running processes
- Category validation logic - ensures data integrity
- Incremental saving pattern - follows Stage 5 JSON array format
- Security validation functions - prevent directory traversal attacks

## 💡 DEVELOPMENT NOTES

### Classification Patterns
- **Management Discussion**: Process in windows (configurable size) with full speaker block context
- **Q&A Conversations**: Single classification call per complete conversation
- **Category Assignment**: Multiple categories allowed per content section
- **Confidence Scores**: Required for all classifications (0.0-1.0 range)
- **Error Fallback**: Always assign "Other" category if classification fails

### CO-STAR Prompt Structure
```
Context: Institution, fiscal period, speaker, section type
Objective: Classify content with all applicable financial categories
Style: Analyze against category descriptions, apply all relevant categories
Tone: Professional financial analysis
Audience: Financial analysts requiring content categorization
Response: Use function calling for structured output
```

### Performance Optimization
- OAuth token refresh per transcript (not per API call)
- Incremental saving to prevent data loss
- Configurable window sizes for processing efficiency
- Cost tracking with detailed breakdown
- Enhanced error logging by failure type

## 🔍 CODE CONVENTIONS
- **Error Handling**: Use EnhancedErrorLogger for categorized error tracking
- **Logging**: Separate execution logs and error logs with timestamps
- **Security**: Validate all file paths and sanitize URLs before logging
- **NAS Operations**: Use utility functions with path validation
- **Function Naming**: Descriptive names reflecting financial domain (e.g., `classify_management_discussion_paragraphs`)

## 📊 DATA HANDLING
- **Input Format**: Stage 5 JSON array with speaker blocks and Q&A groups
- **Output Format**: JSON array with added `category_type`, `category_type_confidence`, `category_type_method` fields
- **Category Validation**: All categories must exist in config `financial_categories`
- **Incremental Saving**: JSON array pattern with proper opening/closing brackets
- **Failed Records**: Separate tracking of failed transcripts with failure reasons
- **Cost Tracking**: Detailed token usage and cost calculation per transcript

## 🏗️ ARCHITECTURE DETAILS

### Processing Flow
1. **Initialization**: Environment validation, NAS connection, config loading
2. **SSL Setup**: Download and configure corporate SSL certificate
3. **LLM Authentication**: OAuth client credentials flow with token management
4. **Data Loading**: Stage 5 output parsing and transcript grouping
5. **Classification Processing**: Speaker-block windowing (MD) and conversation-level (Q&A)
6. **Result Saving**: Incremental JSON array writing with proper formatting
7. **Logging**: Comprehensive execution and error logging to NAS

### Function Categories
- **Setup Functions**: Environment, NAS, SSL, OAuth, proxy configuration
- **Classification Functions**: Management Discussion and Q&A processing
- **Utility Functions**: NAS operations, security validation, cost calculation
- **Error Handling**: Enhanced logging by error type with structured output
- **Data Processing**: Incremental saving, JSON array management

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