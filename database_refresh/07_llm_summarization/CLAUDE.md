# 🚀 FactSet Project - Stage 7: LLM Summarization

## 🎯 PROJECT CONTEXT
- Type: LLM-based content summarization for financial earnings call transcripts
- Stack: Python 3.11+, OpenAI API, SMB/NAS integration
- Architecture: Self-contained summarization stage with OAuth authentication
- Focus: Creating paragraph-level summaries for Q&A groups and Management Discussion speaker blocks

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
- Primary: OpenAI-compatible API (configurable model)
- OAuth authentication with per-transcript token refresh
- Temperature: Configurable via config (default for summarization tasks)
- SSL certificate handling for enterprise environments

### Processing Pipeline
- Input: Stage 6 classified content (JSON array format)
- Processing: Two main summarization types:
  - Q&A Groups: Complete conversation summaries
  - Management Discussion: Speaker block-level summaries
- Output: Enhanced records with paragraph_summary field
- Incremental saving: Append results after each transcript

## 📁 PROJECT STRUCTURE
```
database_refresh/07_llm_summarization/
  main_llm_summarization.py    # Main summarization script
  CLAUDE.md                    # This file
```

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
# Test OAuth token acquisition
python -c "from main_llm_summarization import get_oauth_token; print(get_oauth_token())"

# Test NAS connection
python -c "from main_llm_summarization import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Validate environment variables
python -c "from main_llm_summarization import validate_environment_variables; validate_environment_variables()"
```

## 🔗 INTEGRATIONS

### Data Sources
- Input: Stage 6 classified content from NAS
- Configuration: Shared YAML config file from NAS
- SSL Certificate: Downloaded from NAS for LLM API access

### Processing Tools
- OpenAI Python client for LLM interactions
- SMBConnection for NAS file operations
- OAuth 2.0 client credentials flow for authentication
- JSON incremental file writing (Stage 6 pattern)

## 📋 CURRENT FOCUS
- LLM-based summarization for earnings call transcripts
- Q&A group conversation summaries (all paragraphs get same summary)
- Management Discussion speaker block summaries (with sliding window context)
- Per-transcript OAuth token refresh to prevent expiration
- Enhanced error logging with separate error type categorization

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
- Use per-transcript OAuth refresh to handle long processing sessions
- Q&A groups get single summary applied to all paragraphs in the conversation
- Management Discussion uses speaker blocks with sliding window context
- Enhanced error logger categorizes different failure types
- Incremental saving prevents data loss during processing
- Function calling used for structured LLM responses

## 🔍 CODE CONVENTIONS
- Follow existing Stage 5/6 patterns for NAS operations
- Use EnhancedErrorLogger for categorized error tracking
- Implement per-transcript processing with OAuth refresh
- Use sliding window context for Management Discussion blocks
- Apply security validation to all file paths and URLs
- Log execution details for audit and debugging

## 📊 DATA HANDLING
- Input: Stage 6 classified content records (JSON array)
- Processing: Group by transcript, then by section type
- Q&A Processing: Group by qa_group_id, summarize conversations
- Management Processing: Group by speaker_block_id, use context windows
- Output: Enhanced records with paragraph_summary field
- Incremental saving: Stage 6 JSON array append pattern
- Error handling: Null summary values for failed processing

## 🎨 SUMMARIZATION FEATURES

### Q&A Group Summarization
- Groups records by qa_group_id within transcript
- Creates single summary for entire Q&A conversation
- Applies same summary to all paragraphs in the group
- Optimized for post-retrieval reranking
- Includes speaker attribution and financial context

### Management Discussion Summarization
- Groups records by speaker_block_id within transcript
- Creates individual summaries for each speaker block
- Uses sliding window context (previous 2 speaker block summaries)
- Distinguishes between opening, middle, and closing blocks
- Preserves speaker names and block positioning context

### Summary Optimization
- Function calling for structured LLM responses
- Reranking-optimized prompts with financial terminology
- Speaker attribution preservation for relevance filtering
- Quantitative data and forward-looking statement emphasis
- Company and fiscal period context integration

## 🔒 SECURITY CONSIDERATIONS
- OAuth client credentials for LLM API access
- SSL certificate validation for enterprise environments
- NAS path traversal attack prevention
- Sensitive data sanitization in logs
- Temporary SSL certificate file cleanup
- Environment variable validation for required credentials

## 💰 COST TRACKING
- Per-request token usage tracking
- Cost calculation based on prompt/completion tokens
- Total cost accumulation across all transcripts
- Enhanced error logs include cost breakdown
- Development mode for cost-controlled testing

## 🚀 PERFORMANCE OPTIMIZATIONS
- Per-transcript OAuth token refresh
- Incremental result saving after each transcript
- Rate limiting between transcript processing
- Sliding window context to reduce token usage
- Function calling for efficient structured responses