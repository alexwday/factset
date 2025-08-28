# 🚀 Stage 5: Q&A Boundary Detection & Conversation Pairing

## 🎯 PROJECT CONTEXT
- **Stage Purpose**: Q&A Boundary Detection & Conversation Pairing
- **Input**: Stage 4 validated content (structured speaker blocks)
- **Output**: Q&A conversation groups with proper boundaries
- **Processing Method**: Sliding window analysis with LLM-based boundary detection
- **Core Function**: Identifies question-answer conversation boundaries and groups speaker blocks into coherent Q&A pairs

## 🚨 CRITICAL RULES
- ALWAYS validate LLM boundary detection outputs before proceeding
- NEVER exceed maximum held blocks limit (default: 50) to prevent memory issues
- MUST handle consecutive skip limits (default: 5) to prevent infinite loops
- Monitor token usage and costs continuously for LLM API calls
- Handle financial transcript data with appropriate security measures
- Validate all speaker block IDs exist before processing

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
- **Temperature**: Configurable via `stage_05_qa_pairing.llm_config.temperature`
- **Max Tokens**: Configurable via `stage_05_qa_pairing.llm_config.max_tokens`
- **Purpose**: Boundary detection and conversation validation

### Processing Pipeline
- **Input**: Stage 4 validated speaker block content
- **Method**: Sliding window analysis with configurable window size
- **LLM Integration**: Two-phase approach (detection + validation)
- **Output**: Q&A conversation groups with assigned speaker blocks

## 📁 PROJECT STRUCTURE
```
database_refresh/05_qa_pairing/
  main_qa_pairing.py          # Main stage 5 processing script
  CLAUDE.md                   # This file - stage-specific documentation
```

## 🔧 ESSENTIAL COMMANDS

### Stage 5 Execution
```bash
# Run Stage 5 Q&A Pairing
cd database_refresh/05_qa_pairing
python main_qa_pairing.py

# Check Stage 5 configuration
python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing'])"
```

### Development & Debugging
```bash
# Test with specific transcript
python main_qa_pairing.py --transcript-id SPECIFIC_ID  # If supported

# Validate Stage 4 input data exists
ls -la "$(python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing']['input_data_path'])")"

# Check output directory
ls -la "$(python -c "import yaml; print(yaml.safe_load(open('../../config.yaml'))['stage_05_qa_pairing']['output_data_path'])")"
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
print('LLM Model:', stage_config['llm_config']['model'])
"
```

## 🔗 INTEGRATIONS

### Data Sources
- **Input**: Stage 4 validated content (`stage_05_qa_pairing.input_data_path`)
- **Configuration**: NAS-hosted config.yaml with stage-specific parameters
- **Authentication**: OAuth token-based LLM API access

### Processing Dependencies
- **SMB/NAS Connection**: For loading config and input/output data
- **OpenAI Client**: For LLM-based boundary detection
- **SSL Certificates**: For secure API communications

## 📋 CURRENT FOCUS
- **Core Function**: Q&A boundary detection using sliding window approach
- **Key Algorithm**: Two-phase LLM processing (detection → validation)
- **Optimization**: Memory-efficient processing with configurable limits
- **Error Handling**: Enhanced error logging with separate error type categorization

## ⚠️ KNOWN ISSUES
- **Memory Management**: Must respect `max_held_blocks` limit to prevent excessive memory usage
- **Infinite Loops**: `max_consecutive_skips` prevents endless skip decisions
- **Validation Retries**: Limited retries per position via `max_validation_retries_per_position`
- **Token Costs**: Monitor LLM API usage costs with detailed token tracking

## 🚫 DO NOT MODIFY
- NAS configuration paths and connection logic
- OAuth authentication flow and token management
- SSL certificate handling for secure communications
- Core sliding window algorithm without thorough testing

## 💡 DEVELOPMENT NOTES

### Key Functions
- `detect_analyst_breakpoint()`: Main LLM boundary detection logic
- `validate_analyst_assignment()`: Secondary validation of proposed boundaries
- `process_qa_boundaries_sliding_window()`: Core sliding window orchestration
- `process_transcript_qa_pairing()`: Per-transcript processing wrapper

### Configuration Parameters
```yaml
stage_05_qa_pairing:
  window_size: 10                           # Speaker blocks per analysis window
  max_held_blocks: 50                       # Memory limit for accumulated blocks
  max_consecutive_skips: 5                  # Prevent infinite skip loops
  max_validation_retries_per_position: 2    # Retries for boundary validation
```

### Error Handling Categories
- **boundary_detection_errors**: LLM boundary detection failures
- **authentication_errors**: OAuth and API access issues
- **validation_errors**: Q&A group validation failures
- **processing_errors**: General processing and data handling errors

## 🔍 CODE CONVENTIONS
- **Sliding Window State**: Use `sliding_state` dict to track window position and accumulated blocks
- **Speaker Block IDs**: Process using sequential speaker_block_id values for consistent ordering
- **LLM Prompts**: Use structured prompts with company context and transcript metadata
- **Tool-based LLM**: Utilize OpenAI function calling for structured boundary detection responses
- **Error Logging**: Separate error types using `EnhancedErrorLogger` class

## 📊 DATA HANDLING
- **Input Format**: JSON records with speaker_block_id, speaker_name, and paragraph content
- **Processing Unit**: Individual speaker blocks grouped by transcript_id
- **Output Format**: Q&A groups with assigned speaker_block_ids and metadata
- **Memory Efficiency**: Process transcripts sequentially, not loading all into memory
- **Validation**: Ensure speaker_block_id sequences are continuous and complete

## 🔧 DEBUGGING TIPS
- **Window State**: Monitor `sliding_state` for position tracking and block accumulation
- **LLM Responses**: Check tool call responses for proper JSON structure and required fields
- **Token Usage**: Track costs using `calculate_token_cost()` function
- **Block Sequences**: Validate speaker_block_id ordering within transcripts
- **Boundary Logic**: Test edge cases around transcript start/end and single-speaker segments

## 📈 PERFORMANCE CONSIDERATIONS
- **Window Size**: Larger windows provide more context but increase token costs
- **Validation Retries**: Balance accuracy vs processing time with retry limits
- **Memory Management**: `max_held_blocks` prevents memory exhaustion on long transcripts
- **API Efficiency**: Batch processing within windows to minimize API calls