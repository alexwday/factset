# Stage 5: Q&A Boundary Detection & Conversation Pairing

> **Version**: 1.0 | **Created**: 2025-07-30  
> **Purpose**: Detect conversation boundaries and pair questions with answers using speaker block analysis and LLM-based boundary detection
> **Script**: `database_refresh/05_qa_pairing/main_qa_pairing.py`

---

## Project Context

- **Stage**: Stage 5 - Q&A Boundary Detection & Conversation Pairing
- **Primary Purpose**: Process Stage 4 validated content to identify Q&A conversation boundaries and group related question-answer exchanges
- **Pipeline Position**: Between Stage 4 (LLM Classification) and Stage 6 (Additional Processing)
- **Production Status**: PRODUCTION READY ✅

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: OpenAI API for LLM-based boundary detection
- **Authentication**: OAuth 2.0 for LLM API + Corporate proxy (MAPLE domain) + NAS NTLM v2
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: JSON-based configuration from NAS

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
import re                          # Pattern matching (unused in current version)
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
# Required .env variables (15 required)
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
CONFIG_PATH=                # NAS configuration path
CLIENT_MACHINE_NAME=
LLM_CLIENT_ID=              # OAuth client ID for LLM API
LLM_CLIENT_SECRET=          # OAuth client secret for LLM API
```

---

## Architecture & Design

### Core Design Patterns
- **Speaker Block Analysis**: Groups paragraphs by speaker to maintain conversation context
- **Sliding Window Processing**: Analyzes configurable-size windows of speaker blocks
- **LLM Boundary Detection**: Uses GPT-4 to identify conversation boundaries with structured tool calls
- **Per-Transcript OAuth**: Refreshes OAuth token for each transcript to prevent expiration
- **Progressive Fallbacks**: LLM detection → XML type attributes → Conservative grouping
- **Minimal Schema Extension**: Only adds 3 fields (qa_group_id, qa_group_confidence, qa_group_method)

### File Structure
```
database_refresh/05_qa_pairing/
├── main_qa_pairing.py          # Primary execution script
├── CLAUDE.md                   # This context file
└── visualize_qa_pairing.py     # Visualization tool for Q&A pairing results
```

### Key Components
1. **OAuth Manager**: Handles token acquisition and per-transcript refresh
2. **Speaker Block Processor**: Groups paragraphs into speaker blocks
3. **Sliding Window Engine**: Manages dynamic window analysis
4. **LLM Boundary Detector**: Detects conversation endpoints using structured prompts
5. **Validation System**: Validates proposed Q&A groupings
6. **Fallback Manager**: Handles cases where LLM detection fails
7. **Cost Tracker**: Monitors token usage and costs

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for Stage 5
stage_05_qa_pairing:
  description: "Q&A boundary detection and conversation pairing"
  input_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_04_classified_content.json"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"
  output_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  dev_mode: true
  dev_max_transcripts: 2
  
  # Sliding window configuration
  window_size: 10                    # Number of speaker blocks to analyze at once
  max_held_blocks: 50               # Maximum blocks to accumulate before forcing breakpoint
  max_consecutive_skips: 5          # Maximum consecutive skip decisions before forcing breakpoint
  max_validation_retries_per_position: 2  # Retries for validation at same position
  
  # LLM configuration
  llm_config:
    model: "gpt-4o-2024-08-06"
    temperature: 0.0
    max_tokens: 100
    cost_per_1k_prompt_tokens: 0.0025
    cost_per_1k_completion_tokens: 0.01
    timeout: 30
    max_retries: 3
    retry_delay: 1.0
```

### Validation Requirements
- **Schema Validation**: Validates all required stage_05_qa_pairing parameters exist
- **Security Validation**: NAS path validation, file path security checks
- **Configuration Validation**: Window sizes, retry limits, cost tracking parameters

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Environment Setup**: Validate environment variables and establish NAS connection
2. **Configuration Loading**: Load and validate configuration from NAS
3. **SSL/Proxy Setup**: Configure SSL certificates and corporate proxy
4. **LLM Client Setup**: Initialize OpenAI client with OAuth authentication
5. **Content Loading**: Load stage_04_classified_content.json from Stage 4
6. **Transcript Grouping**: Group records by transcript identifier
7. **Development Mode**: Apply transcript limits if in development mode
8. **Per-Transcript Processing**:
   - Refresh OAuth token for transcript
   - Group records by speaker blocks
   - Apply sliding window analysis
   - Detect Q&A boundaries using LLM
   - Validate proposed groupings
   - Apply fallback strategies if needed
9. **Output Generation**: Save stage_05_qa_paired_content.json
10. **Error Tracking**: Save failed transcripts to separate file
11. **Log Cleanup**: Save execution logs and cleanup

### Key Business Rules
- **Speaker Block Integrity**: Maintains speaker continuity within blocks
- **Conversation Boundaries**: Detects where one analyst's Q&A ends and another begins
- **Dynamic Context**: Extends context window back to question start when needed
- **Minimal Schema Changes**: Only adds qa_group_id field to records
- **Complete Coverage**: Ensures all Q&A section records get assignments
- **Cost Management**: Tracks and reports token usage and costs

### Sliding Window Algorithm
```
1. Start with initial window of N speaker blocks
2. Ask LLM: "Does current analyst's turn end in this window?"
3. If "skip": Add more blocks and repeat
4. If "breakpoint": Create Q&A group, move to next analyst
5. Apply safety limits (max blocks, max skips)
6. Validate each proposed grouping
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

### Data Processing Functions
- `group_records_by_transcript()`: Groups records by transcript ID
- `group_records_by_speaker_block()`: Creates speaker blocks from paragraphs
- `process_qa_boundaries_sliding_window()`: Main sliding window processor
- `apply_qa_assignments_to_records()`: Adds qa_group_id to records

### LLM Interaction Functions
- `detect_analyst_breakpoint()`: LLM call for boundary detection
- `validate_analyst_assignment()`: LLM call for grouping validation
- `create_breakpoint_detection_prompt()`: CO-STAR prompt generation
- `create_validation_prompt()`: Validation prompt generation

### Utility Functions
- `format_speaker_block_content()`: Formats speaker content with role indicators
- `calculate_token_cost()`: Cost calculation based on token usage
- `extract_transcript_metadata()`: Extracts company/title information

---

## Error Handling & Logging

### Error Categories
```python
class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""
    
    def __init__(self):
        self.boundary_detection_errors = []  # LLM detection failures
        self.authentication_errors = []      # OAuth failures
        self.validation_errors = []          # Grouping validation failures
        self.processing_errors = []          # General processing errors
        self.total_cost = 0.0               # Track total LLM costs
        self.total_tokens = 0                # Track total token usage
```

### Error Tracking Methods
- `log_boundary_error()`: Records LLM boundary detection failures
- `log_authentication_error()`: Records OAuth/auth failures
- `log_validation_error()`: Records validation failures
- `log_processing_error()`: Records general processing errors
- `update_usage()`: Tracks token usage and costs

### Logging Strategy
- **Console Logging**: Minimal progress updates
- **Execution Log**: Detailed operation tracking
- **Error Log**: Structured error recording
- **Cost Tracking**: Real-time usage monitoring

---

## Input/Output Specifications

### Input Format (from Stage 4)
```json
[
  {
    "ticker": "RY-CA",
    "fiscal_year": 2024,
    "fiscal_quarter": "Q1",
    "transcript_key": "RY-CA_2024_Q1",
    "section_name": "Q&A",
    "speaker_block_id": 42,
    "speaker": "John Smith",
    "speaker_type": "analyst",
    "paragraph_sequence": 1,
    "paragraph_content": "Thank you. My first question is about...",
    "paragraph_type": "question",
    "management_discussion_flag": false,
    "investor_qa_flag": true,
    "classification_confidence": 0.95,
    "classification_method": "llm_progressive_3"
  }
]
```

### Output Format (Stage 5)
```json
[
  {
    // All original fields preserved
    "ticker": "RY-CA",
    "fiscal_year": 2024,
    "fiscal_quarter": "Q1",
    "transcript_key": "RY-CA_2024_Q1",
    "section_name": "Q&A",
    "speaker_block_id": 42,
    "speaker": "John Smith",
    "speaker_type": "analyst",
    "paragraph_sequence": 1,
    "paragraph_content": "Thank you. My first question is about...",
    
    // New Stage 5 fields
    "qa_group_id": 1  // Groups related Q&A exchanges
  }
]
```

### Q&A Group Structure (Internal)
```json
{
  "qa_group_id": 1,
  "qa_group_confidence": 0.95,
  "qa_group_method": "llm_breakpoint",  // or "fallback_xml_type", "fallback_conservative"
  "speaker_blocks": [
    // Array of speaker blocks in this Q&A group
  ]
}
```

---

## Performance & Optimization

### Token Usage Optimization
- **Paragraph Truncation**: Limits to 750 chars per paragraph
- **Selective Context**: Only includes necessary speaker blocks
- **Structured Prompts**: Uses CO-STAR framework for efficiency
- **Tool Calling**: Structured responses minimize tokens

### Processing Efficiency
- **Per-Transcript OAuth**: Prevents mid-transcript auth failures
- **Sliding Windows**: Processes incrementally vs entire transcript
- **Early Termination**: Safety limits prevent runaway processing
- **Batch Operations**: Groups related operations

### Cost Management
- **Real-Time Tracking**: Monitors usage during execution
- **Configurable Rates**: Adjustable token pricing
- **Usage Reporting**: Detailed cost breakdown in logs
- **No Budget Limits**: Processes all content (monitor manually)

---

## Common Issues & Solutions

### OAuth Token Expiration
- **Issue**: Token expires during long transcript processing
- **Solution**: Per-transcript token refresh implemented

### Large Context Windows
- **Issue**: Some Q&A exchanges span many speaker blocks
- **Solution**: Dynamic window expansion with safety limits

### Ambiguous Boundaries
- **Issue**: Unclear where one analyst ends and another begins
- **Solution**: Validation step after detection

### Missing Assignments
- **Issue**: Some speaker blocks not assigned to Q&A groups
- **Solution**: Comprehensive fallback strategies

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
3. Review output structure
4. Validate Q&A groupings
5. Check cost tracking

### Visualization Tool
- **Script**: `visualize_qa_pairing.py`
- **Purpose**: Visual representation of Q&A groupings
- **Usage**: Helps validate boundary detection accuracy

---

## Maintenance & Operations

### Daily Operations
1. Monitor OAuth authentication success
2. Check token usage and costs
3. Review failed transcript counts
4. Validate output completeness

### Configuration Updates
- Adjust window_size for better detection
- Update token cost rates as needed
- Modify safety limits based on usage
- Fine-tune LLM parameters

### Error Recovery
1. Failed transcripts saved separately
2. Can reprocess individual transcripts
3. OAuth issues auto-retry
4. Comprehensive error logging

---

## Integration Points

### Dependencies on Previous Stages
- **Stage 4 Output**: Requires stage_04_classified_content.json
- **Classification Flags**: Uses investor_qa_flag to identify Q&A content
- **Speaker Metadata**: Relies on speaker_type and speaker_block_id

### Data Flow to Next Stages
- **Preserved Fields**: All Stage 4 fields maintained
- **New Field**: qa_group_id for Q&A relationship mapping
- **Grouped Content**: Related Q&A exchanges now identifiable

---

## Version History

### Version 1.0 (2025-07-30)
- Initial production release
- Speaker block-based sliding window analysis
- LLM boundary detection with fallbacks
- Per-transcript OAuth refresh
- Comprehensive error handling
- Cost tracking and reporting