# 🚀 Stage 5: Q&A Boundary Detection & Conversation Pairing

## 🎯 PROJECT CONTEXT
- **Stage Purpose**: Q&A Boundary Detection & Conversation Pairing using LLM
- **Input**: Stage 4 validated content (`stage_04_validated_transcripts.json`)
- **Output**: Q&A conversation groups with proper boundaries (`stage_05_qa_paired_content.json`)
- **Processing Method**: Sliding window analysis with LLM-based boundary detection (two-phase approach)
- **Core Function**: Identifies question-answer conversation boundaries and groups speaker blocks into coherent Q&A pairs
- **Resume Capability**: Supports resuming from interrupted processing via existing output detection

## 🚨 CRITICAL RULES
- ALWAYS validate LLM boundary detection outputs before proceeding with assignment validation
- NEVER exceed maximum held blocks limit (default: 50) to prevent memory issues
- MUST handle consecutive skip limits (default: 5) to prevent infinite loops
- Monitor token usage and costs continuously for LLM API calls (tracked per transcript)
- Handle financial transcript data with appropriate security measures
- Validate all speaker block IDs exist before processing
- MUST refresh OAuth tokens per transcript to prevent expiration during long runs
- Enable resume capability by default to handle processing interruptions

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/stage-05-*, bugfix/stage-05-*, hotfix/stage-05-*
- Commit format: conventional commits preferred
- Focus: Q&A boundary detection improvements and conversation pairing logic

### Code Quality Standards
```bash
# Pre-commit checks
python -m black database_refresh/05_qa_pairing/ --line-length 88
python -m isort database_refresh/05_qa_pairing/ --profile black
python -m pylint database_refresh/05_qa_pairing/ --min-score 8.0
python -m flake8 database_refresh/05_qa_pairing/ --max-line-length 88
```

### Testing Strategy
- Local: Mock LLM responses for boundary detection testing
- Integration: Full Stage 4 → Stage 5 pipeline validation
- Coverage: Focus on sliding window logic and boundary detection accuracy
- AI-specific: Validate Q&A group formation quality and completeness

## 🤖 AI/ML CONFIGURATION

### Model Setup
- **Primary**: LLM model configured in `stage_05_qa_pairing.llm_config.model`
- **Temperature**: Configurable via `stage_05_qa_pairing.llm_config.temperature` (default: 0.1)
- **Max Tokens**: Configurable via `stage_05_qa_pairing.llm_config.max_tokens` 
- **Token Endpoint**: OAuth token endpoint for client credentials flow
- **Cost Tracking**: Per-token costs tracked via `cost_per_1k_prompt_tokens` and `cost_per_1k_completion_tokens`
- **Purpose**: Boundary detection (Phase 1) and assignment validation (Phase 2)

### Processing Pipeline
- **Input**: Stage 4 validated speaker block content from JSON file
- **Method**: Sliding window analysis with configurable window size (default: 10 blocks)
- **LLM Integration**: Two-phase approach:
  - Phase 1: `detect_analyst_breakpoint()` - Identifies last block of current analyst's turn
  - Phase 2: `validate_analyst_assignment()` - Validates proposed Q&A group assignment
- **Output**: Q&A conversation groups with assigned speaker blocks and metadata
- **Fallback Logic**: Force breakpoint on memory limits or excessive skips

## 📁 PROJECT STRUCTURE
```
database_refresh/05_qa_pairing/
├── main_qa_pairing.py          # Main stage 5 processing script (2455 lines)
└── CLAUDE.md                   # This file - stage-specific documentation
```

## 🔧 ESSENTIAL COMMANDS

### Stage 5 Execution
```bash
# Run Stage 5 Q&A Pairing
cd database_refresh/05_qa_pairing
python main_qa_pairing.py

# Check Stage 5 configuration
python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing'])"

# Monitor progress in real-time (single-line status updates)
# Output format: [progress] ticker | QA groups | Gaps | Cost | Time
```

### Development & Debugging
```bash
# Enable development mode (limits transcripts)
# Set dev_mode: true and dev_max_transcripts in config.yaml

# Validate Stage 4 input data exists
ls -la "$(python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing']['input_data_path'])")"

# Check output directory and resume capability
ls -la "$(python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing']['output_data_path'])")"

# Check for existing output (enables resume)
ls -la <output_path>/stage_05_qa_paired_content.json
ls -la <output_path>/stage_05_qa_pairing_failed_transcripts.json
```

### Configuration Management
```bash
# View stage-specific config parameters
python -c "
import yaml
config = yaml.safe_load(open('../../config.yaml'))
stage_config = config['stage_05_qa_pairing']
print('Window Size:', stage_config['window_size'])
print('Max Held Blocks:', stage_config.get('max_held_blocks', 50))
print('Max Consecutive Skips:', stage_config.get('max_consecutive_skips', 5))
print('Max Validation Retries:', stage_config.get('max_validation_retries_per_position', 2))
print('Enable Resume:', stage_config.get('enable_resume', True))
print('LLM Model:', stage_config['llm_config']['model'])
print('Temperature:', stage_config['llm_config']['temperature'])
print('Max API Retries:', stage_config.get('max_api_retries', 5))
"
```

## 🔗 INTEGRATIONS

### Data Sources
- **Input**: Stage 4 validated content (`stage_04_validated_transcripts.json`)
- **Configuration**: NAS-hosted `config.yaml` and `monitored_institutions.yaml`
- **Authentication**: OAuth client credentials flow with automatic token refresh
- **SSL Certificate**: Downloaded from NAS at runtime for secure API communications

### Processing Dependencies
- **SMB/NAS Connection**: For loading config and input/output data (SMBConnection)
- **OpenAI Client**: For LLM-based boundary detection (OpenAI Python SDK)
- **SSL Certificates**: Downloaded from NAS path specified in `ssl_cert_path`
- **Proxy Configuration**: Corporate proxy with domain authentication
- **OAuth Token**: Refreshed per transcript to prevent expiration

## 📋 CURRENT FOCUS
- **Core Function**: Q&A boundary detection using sliding window approach (line 1402)
- **Key Algorithm**: Two-phase LLM processing:
  - Phase 1: `detect_analyst_breakpoint()` (line 1166) - Tool-based LLM call
  - Phase 2: `validate_analyst_assignment()` (line 1301) - Validation with retries
- **Optimization**: Memory-efficient processing with configurable limits and infinite loop protection
- **Error Handling**: `EnhancedErrorLogger` class (line 90) with categorized error tracking
- **Resume Capability**: Automatic detection and skipping of completed transcripts

## ⚠️ KNOWN ISSUES
- **Memory Management**: Must respect `max_held_blocks` limit (default: 50) to prevent excessive memory usage
- **Infinite Loops**: Multiple protections:
  - `max_consecutive_skips` (default: 5) prevents endless skip decisions
  - `max_iterations` = 3x speaker blocks prevents runaway loops
  - `stuck_position_count` forces advancement after 3 iterations at same position
- **Validation Retries**: Limited retries per position via `max_validation_retries_per_position` (default: 2)
- **Token Costs**: Monitor LLM API usage costs with detailed token tracking and accumulation
- **OAuth Expiration**: Tokens refreshed per transcript to handle long-running processes

## 🚫 DO NOT MODIFY
- NAS configuration paths and connection logic
- OAuth authentication flow and token management
- SSL certificate handling for secure communications
- Core sliding window algorithm without thorough testing

## 💡 DEVELOPMENT NOTES

### Key Functions

#### Core Processing Functions
- `main()` (line 2107): Main orchestration with 12 processing steps
- `process_transcript_qa_pairing()` (line 2025): Per-transcript processing wrapper with metrics
- `process_qa_boundaries_sliding_window()` (line 1402): Core sliding window orchestration with state management

#### LLM Integration Functions
- `detect_analyst_breakpoint()` (line 1166): Main LLM boundary detection with tool calling
- `validate_analyst_assignment()` (line 1301): Secondary validation of proposed boundaries
- `create_breakpoint_detection_prompt()` (line 870): Generates structured prompt for boundary detection
- `create_validation_prompt()` (line 963): Generates validation prompt for assignment checking
- `create_breakpoint_detection_tool()` (line 1055): OpenAI function tool specification
- `create_validation_tool()` (line 1085): Validation tool specification

#### Setup and Configuration Functions
- `setup_llm_client()` (line 557): Initialize OpenAI client with OAuth authentication
- `get_oauth_token()` (line 493): OAuth client credentials flow with retry logic
- `setup_ssl_certificate()` (line 432): Download and configure SSL cert from NAS
- `setup_proxy_configuration()` (line 464): Configure corporate proxy with authentication

#### Data I/O Functions
- `load_stage_4_content()` (line 789): Load validated content from Stage 4
- `group_records_by_transcript()` (line 828): Organize records by transcript_id
- `load_existing_output()` (line 1884): Check for and load completed transcripts (resume)
- `append_records_to_output()` (line 1949): Streaming JSON append with retries
- `close_json_array()` (line 2003): Properly close JSON array in output file

#### Utility Functions
- `calculate_token_cost()` (line 1110): Calculate API costs based on token usage
- `create_indexed_speaker_blocks_from_ids()` (line 1141): Create indexed blocks for LLM
- `get_blocks_for_speaker_block_ids()` (line 847): Retrieve blocks by ID list

#### NAS Operations
- `nas_path_join()` (line 619): Safe path joining for NAS operations
- `nas_download_file()` (line 630): Download file from NAS
- `nas_upload_file()` (line 649): Upload file to NAS with retries
- `nas_file_exists()` (line 697): Check file existence on NAS
- `nas_create_directory_recursive()` (line 706): Create directory structure on NAS

### Configuration Parameters
```yaml
stage_05_qa_pairing:
  description: "Q&A boundary detection and conversation pairing"
  input_data_path: "/path/to/stage_04_output"       # Stage 4 validated content
  output_data_path: "/path/to/stage_05_output"      # Q&A paired content
  output_logs_path: "/path/to/stage_05_logs"        # Execution and error logs
  dev_mode: false                                    # Development mode flag
  dev_max_transcripts: 2                             # Limit for dev mode
  enable_resume: true                                # Resume from interruption
  window_size: 10                                    # Speaker blocks per analysis window
  max_held_blocks: 50                                # Memory limit for accumulated blocks
  max_consecutive_skips: 5                           # Prevent infinite skip loops
  max_validation_retries_per_position: 2             # Retries for boundary validation
  max_api_retries: 5                                 # API call retry attempts
  api_retry_delay: 1.0                               # Initial retry delay (exponential backoff)
  llm_config:
    base_url: "https://api.endpoint.com"            # LLM API endpoint
    model: "model-name"                             # Model identifier
    temperature: 0.1                                 # Low for factual analysis
    max_tokens: 500                                  # Response token limit
    timeout: 30                                      # API timeout seconds
    max_retries: 3                                   # Client-level retries
    token_endpoint: "https://oauth.endpoint.com"    # OAuth token endpoint
    cost_per_1k_prompt_tokens: 0.001                # Cost tracking
    cost_per_1k_completion_tokens: 0.002            # Cost tracking
```

### Error Handling Categories (EnhancedErrorLogger)
- **boundary_detection_errors**: LLM boundary detection failures (logged per speaker block)
- **authentication_errors**: OAuth and API access issues (includes token refresh failures)
- **validation_errors**: Q&A group validation failures (logged per transcript)
- **processing_errors**: General processing and data handling errors
- **Error Files**: Separate JSON files saved for each error category in `Errors/` subdirectory
- **Cost Tracking**: Total cost and tokens accumulated across all transcripts

## 🔍 CODE CONVENTIONS
- **Sliding Window State**: `sliding_state` dict tracks:
  - `current_qa_id`: Current Q&A group being formed
  - `current_block_id_index`: Position in speaker_block_id_sequence
  - `current_window_end_index`: End of current window (exclusive)
  - `processing_complete`: Completion flag
  - `all_qa_groups`: Accumulated Q&A groups
  - `speaker_block_id_sequence`: Ordered list of unique block IDs
  - `consecutive_skips`: Track skip decisions for limit enforcement
  - `stuck_position_count`: Detect stuck iterations
- **Speaker Block IDs**: Process using sequential speaker_block_id values for consistent ordering
- **LLM Prompts**: Structured prompts with company context and indexed speaker blocks (1-based)
- **Tool-based LLM**: OpenAI function calling with `tool_choice="required"`
- **Error Logging**: Separate error types using `EnhancedErrorLogger` class
- **Logging Levels**: Console (minimal) vs Execution Log (detailed)

## 📊 DATA HANDLING

### Input Format
JSON array from `stage_04_validated_transcripts.json` with:
- `speaker_block_id`: Unique identifier for speaker block
- `speaker_name`: Speaker identification
- `paragraph_content`: Text content (list of paragraphs)
- `section_name`: "Management Discussion" or "Q&A"
- `transcript_id`: Unique transcript identifier
- `company_name`: Company name for context
- `transcript_title`: Full transcript title

### Processing Unit
- Individual speaker blocks grouped by transcript_id
- Filtered to Q&A sections only (Management Discussion ignored)
- Sorted by speaker_block_id for sequential processing

### Output Format
- **Main Output**: `stage_05_qa_paired_content.json`
  - Enhanced records with `qa_group_id` field added to Q&A blocks
  - All original fields preserved
  - Non-Q&A records passed through unchanged
- **Failed Transcripts**: `stage_05_qa_pairing_failed_transcripts.json`
  - Original records for transcripts that failed processing

### Memory Efficiency
- Process transcripts sequentially (one at a time)
- Streaming JSON output with incremental appending
- Resume capability to handle interruptions
- Window-based processing limits memory usage per transcript

### Validation
- Ensure speaker_block_id sequences are sorted and tracked
- Skip blocks without proper IDs
- Handle gaps in speaker_block_id sequences
- Validate Q&A section exists before processing

## 🔧 DEBUGGING TIPS

### Common Issues and Solutions
- **Window State**: Monitor `sliding_state` for position tracking and block accumulation
- **LLM Responses**: Check tool call responses for proper JSON structure and required fields:
  - Breakpoint: `{"action": "breakpoint", "index": N}`
  - Skip: `{"action": "skip"}`
  - Validation: `{"is_valid": true/false, "confidence": 0.0-1.0}`
- **Token Usage**: Track costs using `calculate_token_cost()` function (line 820)
- **Block Sequences**: Validate speaker_block_id ordering within transcripts
- **Boundary Logic**: Test edge cases:
  - Transcript start/end boundaries
  - Single-speaker segments
  - Consecutive skip limits
  - Memory limit enforcement
- **Resume Testing**: Delete partial output to test resume capability
- **Progress Monitoring**: Watch single-line status updates during processing

### Output Status Symbols
- `✓` Success: Q&A groups created and saved
- `✗` Error: Processing or save failure
- `⚠` Warning: Save failed but transcript recorded
- `○` No Q&A: No Q&A sections found

### Log Files
- **Main Log**: `stage_05_qa_pairing_TIMESTAMP.json` - Complete execution log
- **Error Logs**: `Errors/stage_05_qa_pairing_TYPE_errors_TIMESTAMP.json`
  - Separate files for boundary_detection, authentication, validation, processing errors

## 📈 PERFORMANCE CONSIDERATIONS

### Processing Metrics
- **Window Size**: Larger windows provide more context but increase token costs (default: 10)
- **Validation Retries**: Balance accuracy vs processing time with retry limits (default: 2)
- **Memory Management**: `max_held_blocks` prevents memory exhaustion on long transcripts (default: 50)
- **API Efficiency**: 
  - Exponential backoff for retries (1s, 2s, 4s, 8s, 16s)
  - Token refresh per transcript prevents expiration
  - Tool-based calls reduce response parsing overhead
- **Resume Capability**: Automatic detection of completed transcripts reduces reprocessing
- **Progress Display**: Single-line updates minimize console output while maintaining visibility
- **Cost Optimization**: 
  - Track per-transcript costs
  - Accumulate total costs across all processing
  - Report cost/token metrics in final summary

### Typical Performance
- **Processing Time**: ~5-30 seconds per transcript (depends on size)
- **Token Usage**: ~2000-10000 tokens per transcript
- **Cost**: ~$0.01-0.05 per transcript (varies by model)
- **Q&A Groups**: Typically 10-30 per transcript
- **Memory Usage**: Limited by max_held_blocks setting

### Optimization Tips
- Enable development mode for testing (limits transcripts)
- Use resume capability for large batches
- Monitor console output for real-time progress
- Check error logs for specific failure patterns
- Adjust window_size for context vs cost tradeoff