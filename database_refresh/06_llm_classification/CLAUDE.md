# Stage 6: LLM-Based Financial Content Classification

> **Version**: 2.0 | **Created**: 2025-07-30  
> **Purpose**: Apply detailed financial category classification to Management Discussion sections and Q&A conversations using LLM
> **Script**: `database_refresh/06_llm_classification/main_llm_classification.py`

---

## Project Context

- **Stage**: Stage 6 - LLM-Based Financial Content Classification
- **Primary Purpose**: Process Stage 5 output to add detailed financial category classifications using speaker blocks (MD) and Q&A groups
- **Pipeline Position**: Between Stage 5 (Q&A Pairing) and Stage 7 (Summarization)
- **Production Status**: PRODUCTION READY ✅ (v2.0 - rebuilt with Stage 5 architecture)

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: OpenAI API for LLM-based classification
- **Authentication**: OAuth 2.0 for LLM API + Corporate proxy (MAPLE domain) + NAS NTLM v2
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for this stage
import os                           # File system operations
import tempfile                     # Temporary file handling for SSL cert
import logging                      # Logging system
import json                         # JSON processing
import time                         # Rate limiting and retries
from datetime import datetime       # Timestamp handling
from urllib.parse import quote      # URL encoding for OAuth
from typing import Dict, Any, Optional, List, Tuple  # Type hints
import io                          # BytesIO for NAS operations
import re                          # Pattern matching
import requests                    # OAuth token requests

# Third-party dependencies
import yaml                        # Configuration parsing
from smb.SMBConnection import SMBConnection  # NAS connectivity
from dotenv import load_dotenv     # Environment variables
from openai import OpenAI          # LLM client
from collections import defaultdict  # Data structures
```

### Environment Requirements
```bash
# Required .env variables (16 required)
API_USERNAME=                # FactSet API credentials (consistency)
API_PASSWORD=
PROXY_USER=                  # Corporate proxy (MAPLE domain)
PROXY_PASSWORD=
PROXY_URL=
PROXY_DOMAIN=               # Default: MAPLE
NAS_USERNAME=               # NAS authentication
NAS_PASSWORD=
NAS_SERVER_IP=
NAS_SERVER_NAME=
NAS_SHARE_NAME=
NAS_BASE_PATH=
NAS_PORT=                   # Default: 445
CONFIG_PATH=                # NAS configuration path
CLIENT_MACHINE_NAME=
LLM_CLIENT_ID=              # OAuth client ID for LLM API
LLM_CLIENT_SECRET=          # OAuth client secret for LLM API
```

---

## Architecture & Design

### Core Design Patterns (Based on Stage 5)
- **Per-Transcript OAuth Refresh**: New token for each transcript to prevent expiration
- **Incremental Saving**: Results saved after each transcript completion
- **Failed Transcript Tracking**: Separate file for failed transcripts
- **Enhanced Error Logging**: Categorized error tracking with separate files
- **Cost Tracking**: Real-time token usage and cost monitoring
- **Development Mode**: Configurable transcript limits for testing

### Classification Approach
- **Management Discussion**: Speaker block windowing (5 paragraphs at a time)
- **Q&A Section**: Complete conversation classification (entire Q&A group)
- **Category Application**: Multiple categories per paragraph/conversation allowed
- **Validation**: Category validation with automatic correction
- **Fallback Strategy**: "Other" category for failed classifications

### File Structure
```
database_refresh/06_llm_classification/
├── main_llm_classification.py     # Primary execution script (v2.0)
├── main_classification.py         # Legacy version (deprecated)
└── CLAUDE.md                      # This context file
```

### Key Components
1. **OAuth Manager**: Per-transcript token refresh system
2. **Classification Engine**: MD windowing and Q&A conversation processors
3. **Category Validator**: Ensures valid category assignments
4. **Incremental Saver**: Saves results after each transcript
5. **Error Logger**: Enhanced error tracking with categorization
6. **Cost Tracker**: Monitors LLM usage and costs

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for Stage 6
stage_06_llm_classification:
  description: "LLM-based financial content classification"
  input_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_05_qa_paired_content.json"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"
  output_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  dev_mode: true
  dev_max_transcripts: 2
  
  # Processing configuration
  processing_config:
    md_paragraph_window_size: 5          # Paragraphs per MD classification call
    max_speaker_blocks_context: 2        # Prior speaker blocks for context
    prior_block_preview_chars: 750       # Characters to show from prior blocks
  
  # LLM configuration
  llm_config:
    base_url: "https://api.llm.internal.com/v1"
    model: "gpt-4o-2024-08-06"
    temperature: 0.0
    max_tokens: 500
    cost_per_1k_prompt_tokens: 0.0025
    cost_per_1k_completion_tokens: 0.01
    timeout: 30
    max_retries: 3
    token_endpoint: "https://auth.llm.internal.com/oauth/token"
  
  # Financial categories
  financial_categories:
    - name: "Revenue & Growth"
      description: "Discussions about revenue performance, growth rates, and sales trends"
      key_indicators: "revenue, sales, growth, top-line"
    - name: "Profitability & Margins"
      description: "Content about profit margins, profitability metrics, and margin expansion/compression"
      key_indicators: "margin, profit, profitability, bottom-line"
    # ... more categories ...
```

### Validation Requirements
- **Schema Validation**: All required parameters must exist
- **Category Validation**: Financial categories must be defined
- **Security Validation**: NAS paths and file paths validated
- **Configuration Completeness**: All sections properly structured

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Environment Setup**: Validate environment variables and establish NAS connection
2. **Configuration Loading**: Load and validate configuration from NAS
3. **SSL/Proxy Setup**: Configure SSL certificates and corporate proxy
4. **Stage 5 Data Loading**: Load qa_paired_content.json
5. **Development Mode**: Apply transcript limits if enabled
6. **Transcript Grouping**: Organize records by transcript and section type
7. **Per-Transcript Processing**:
   - Refresh OAuth token
   - Process Management Discussion speaker blocks
   - Process Q&A conversation groups
   - Save results incrementally
   - Track any failures
8. **Failed Transcript Handling**: Save failed transcripts to separate file
9. **Summary Generation**: Calculate costs and processing metrics
10. **Log Cleanup**: Save execution and error logs

### Key Business Rules
- **Speaker Block Integrity**: MD classification maintains speaker context
- **Conversation Unity**: Q&A groups classified as complete conversations
- **Multi-Category Support**: Paragraphs/conversations can have multiple categories
- **Validation & Correction**: Invalid categories automatically corrected
- **Complete Coverage**: All records receive classification (including "Other")
- **Cost Transparency**: Real-time tracking of LLM costs

### Classification Methods
```
Management Discussion:
1. Group paragraphs by speaker_block_id
2. Process in 5-paragraph windows
3. Provide speaker block context
4. Apply categories to individual paragraphs

Q&A Section:
1. Group paragraphs by qa_group_id
2. Process entire conversation at once
3. Apply same categories to all paragraphs in group
4. Maintain conversation coherence
```

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions - ALL IMPLEMENTED ✅
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    
def validate_environment_variables() -> None:
    """Validate all required environment variables."""
```

### OAuth Security
- **Token Storage**: OAuth tokens stored in memory only
- **Per-Transcript Refresh**: New token for each transcript
- **Credential Protection**: Client ID/secret from environment only
- **URL Sanitization**: Auth tokens removed from logs

---

## Core Functions Reference

### OAuth & Authentication Functions
- `get_oauth_token()`: Acquires OAuth token with retry logic
- `refresh_oauth_token_for_transcript()`: Per-transcript token refresh
- `setup_llm_client()`: Initializes OpenAI client with OAuth

### Classification Functions
- `process_management_discussion_section()`: Classifies MD speaker blocks
- `process_qa_group()`: Classifies complete Q&A conversations
- `process_transcript()`: Orchestrates single transcript processing

### Category Management Functions
- `validate_categories()`: Validates categories against allowed set
- `build_categories_description()`: Builds prompt descriptions
- `create_management_discussion_tools()`: MD function calling schema
- `create_qa_conversation_tools()`: Q&A function calling schema

### Prompt Generation Functions
- `create_management_discussion_costar_prompt()`: CO-STAR prompt for MD
- `create_qa_conversation_costar_prompt()`: CO-STAR prompt for Q&A
- `format_management_discussion_context()`: Formats MD context
- `format_qa_group_context()`: Formats Q&A conversation context

### Utility Functions
- `save_results_incrementally()`: Saves results after each transcript
- `calculate_token_cost()`: Calculates LLM usage costs

---

## Error Handling & Logging

### Error Categories
```python
class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""
    
    def __init__(self):
        self.classification_errors = []  # LLM classification failures
        self.authentication_errors = []  # OAuth failures
        self.validation_errors = []      # Category validation issues
        self.processing_errors = []      # General processing errors
        self.total_cost = 0.0           # Track total LLM costs
        self.total_tokens = 0            # Track total token usage
```

### Error Tracking Methods
- `log_classification_error()`: Records classification failures
- `log_authentication_error()`: Records OAuth/auth failures
- `log_validation_error()`: Records category validation issues
- `log_processing_error()`: Records general processing errors
- `accumulate_costs()`: Tracks token usage and costs

### Logging Strategy
- **Console Logging**: Minimal progress updates
- **Execution Log**: Detailed operation tracking
- **Error Log**: Structured error recording
- **Cost Tracking**: Real-time usage monitoring

---

## Input/Output Specifications

### Input Format (from Stage 5)
```json
{
  "schema_version": "1.0",
  "processing_timestamp": "2024-01-15T10:30:00",
  "records": [
    {
      "ticker": "RY-CA",
      "fiscal_year": 2024,
      "fiscal_quarter": "Q1",
      "section_type": "Management Discussion",
      "speaker_block_id": 1,
      "speaker": "David McKay, CEO",
      "paragraph_content": "Thank you and good morning...",
      "qa_group_id": null
    },
    {
      "ticker": "RY-CA",
      "fiscal_year": 2024,
      "fiscal_quarter": "Q1",
      "section_type": "Investor Q&A",
      "speaker_block_id": 42,
      "speaker": "John Smith, Analyst",
      "paragraph_content": "My first question is about...",
      "qa_group_id": 1
    }
  ]
}
```

### Output Format (Stage 6)
```json
{
  "schema_version": "1.0",
  "processing_timestamp": "2024-01-15T11:45:00",
  "stage": "stage_06_llm_classification",
  "description": "Financial content classification using LLM",
  "records": [
    {
      // All original fields preserved
      "ticker": "RY-CA",
      "fiscal_year": 2024,
      "fiscal_quarter": "Q1",
      "section_type": "Management Discussion",
      "speaker_block_id": 1,
      
      // New Stage 6 fields
      "category_type": ["Revenue & Growth", "Geographic Performance"],
      "category_type_confidence": 0.95,
      "category_type_method": "speaker_block_windowing"
    }
  ]
}
```

### Failed Transcripts Format
```json
{
  "timestamp": "2024-01-15T11:45:00",
  "total_failed": 2,
  "failed_transcripts": [
    {
      "transcript": "ABC-US_2024_Q1",
      "timestamp": "2024-01-15T11:30:00",
      "reason": "Processing failed - see error logs"
    }
  ]
}
```

---

## Performance & Optimization

### Token Usage Optimization
- **Window Processing**: MD classified in 5-paragraph windows
- **Conversation Batching**: Entire Q&A groups in single calls
- **Context Management**: Limited prior speaker block context
- **Structured Prompts**: CO-STAR framework for efficiency

### Processing Efficiency
- **Per-Transcript OAuth**: Prevents mid-transcript auth failures
- **Incremental Saving**: No data loss on failures
- **Parallel Processing**: Multiple speaker blocks/Q&A groups per transcript
- **Rate Limiting**: 1-second pause between transcripts

### Cost Management
- **Real-Time Tracking**: Monitors usage during execution
- **Configurable Rates**: Adjustable token pricing
- **Per-Transcript Costs**: Individual transcript cost tracking
- **Summary Reporting**: Total costs in execution summary

---

## Common Issues & Solutions

### OAuth Token Expiration
- **Issue**: Token expires during processing
- **Solution**: Per-transcript refresh implemented

### Invalid Categories
- **Issue**: LLM returns categories not in allowed set
- **Solution**: Automatic validation and correction

### Large Speaker Blocks
- **Issue**: Speaker blocks with many paragraphs
- **Solution**: Window-based processing with context

### Failed Classifications
- **Issue**: LLM call fails or returns invalid data
- **Solution**: Fallback to "Other" category

---

## Development & Testing

### Development Mode
```yaml
dev_mode: true
dev_max_transcripts: 2  # Limits processing for testing
```

### Testing Approach
1. Enable dev_mode in configuration
2. Process limited transcripts
3. Review classifications
4. Check cost tracking
5. Validate incremental saving

### Key Testing Areas
- Category validation and correction
- OAuth token refresh per transcript
- Incremental saving functionality
- Failed transcript tracking
- Cost calculation accuracy

---

## Maintenance & Operations

### Daily Operations
1. Monitor OAuth authentication success
2. Check classification error rates
3. Review token usage and costs
4. Validate output completeness
5. Check failed transcript counts

### Configuration Updates
- Adjust window sizes for better performance
- Update token cost rates as needed
- Add/modify financial categories
- Fine-tune LLM parameters

### Error Recovery
1. Failed transcripts saved separately
2. Can reprocess individual transcripts
3. OAuth issues auto-retry
4. Comprehensive error logging

---

## Integration Points

### Dependencies on Previous Stages
- **Stage 5 Output**: Requires stage_05_qa_paired_content.json
- **Q&A Groups**: Uses qa_group_id for conversation grouping
- **Speaker Blocks**: Uses speaker_block_id for MD grouping

### Data Flow to Next Stages
- **Preserved Fields**: All Stage 5 fields maintained
- **New Fields**: category_type, category_type_confidence, category_type_method
- **Classification Coverage**: All records classified (no gaps)

---

## Version History

### Version 2.0 (2025-07-30)
- Complete rebuild using Stage 5 architecture patterns
- Per-transcript OAuth refresh
- Incremental saving after each transcript
- Failed transcript tracking
- Enhanced error logging with categorization
- Preserved Stage 6 classification logic and prompts

### Version 1.0 (Original)
- Initial implementation
- Basic classification functionality
- Batch processing approach