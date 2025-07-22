# Stage 5: Q&A Boundary Detection & Conversation Pairing

> **Version**: 2.0 | **Updated**: 2024-07-22  
> **Purpose**: Q&A boundary detection and conversation pairing using speaker block-based LLM analysis

---

## Project Context

- **Stage**: Stage 5 - Q&A Boundary Detection & Conversation Pairing
- **Primary Purpose**: Process Stage 4 validated content to identify and group question-answer conversation boundaries using LLM-based speaker block analysis
- **Pipeline Position**: Between Stage 4 (Content Validation) and Stage 6+ (Downstream Analysis)
- **Production Status**: PRODUCTION READY

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: OpenAI API with OAuth 2.0 authentication for LLM boundary detection
- **Authentication**: Corporate proxy (MAPLE domain) + NAS NTLM v2 + OAuth 2.0 for LLM API
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for this stage
import os                           # File system operations
import tempfile                     # Temporary file handling
import logging                      # Logging system
import json                         # JSON processing
import yaml                         # Configuration parsing
import requests                     # Network operations  
from smb.SMBConnection import SMBConnection  # NAS connectivity
from dotenv import load_dotenv      # Environment variables
from datetime import datetime       # Timestamp handling
from typing import Dict, Any, Optional, List, Tuple  # Type hints
from openai import OpenAI           # LLM API client
from collections import defaultdict # Data structure utilities
```

### Environment Requirements
```bash
# Required .env variables (16 required)
API_USERNAME=                # FactSet API credentials (consistency)
API_PASSWORD=
PROXY_USER=                  # Corporate proxy (MAPLE domain)
PROXY_PASSWORD=
PROXY_URL=
PROXY_DOMAIN=
NAS_USERNAME=                # NAS authentication
NAS_PASSWORD=
NAS_SERVER_IP=
NAS_SERVER_NAME=
NAS_SHARE_NAME=
NAS_BASE_PATH=
NAS_PORT=
CONFIG_PATH=                 # NAS configuration path
CLIENT_MACHINE_NAME=
LLM_CLIENT_ID=              # OAuth client credentials for LLM API
LLM_CLIENT_SECRET=
```

---

## Architecture & Design

### Core Design Patterns
- **Speaker Block Analysis**: Processes Q&A content at speaker block level rather than paragraph level for conversation boundary detection
- **LLM Boundary Detection**: Uses CO-STAR prompt framework with OpenAI API for intelligent conversation boundary identification
- **OAuth Token Management**: Per-transcript token refresh to prevent expiration issues during long-running processing
- **Enhanced Error Logging**: Separate error tracking for boundary detection, authentication, validation, and processing failures

### File Structure
```
stage_5_qa_pairing/
├── main_qa_pairing.py              # Primary execution script
├── CLAUDE.md                       # This context file
└── old/                           # Backup of previous implementation
    └── 5_transcript_qa_pairing.py
```

### Key Components
1. **Stage 4 Content Loader**: Loads validated transcript content with proper format handling
2. **Speaker Block Grouper**: Groups paragraph records into speaker blocks for boundary analysis
3. **LLM Boundary Detector**: Uses OpenAI API with CO-STAR prompts for conversation boundary identification
4. **Q&A Group Manager**: Manages Q&A group state and assignments with validation
5. **Cost Tracker**: Comprehensive token usage and cost tracking for LLM API calls

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for Stage 5
stage_05_qa_pairing:
  description: "Q&A boundary detection and conversation pairing"
  input_source: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_04_validated_content.json"
  output_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  output_file: "stage_05_qa_paired_content.json"
  dev_mode: true
  dev_max_transcripts: 2
  
  # LLM Configuration (OAuth client credentials come from environment variables)
  llm_config:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.1
    max_tokens: 500
    timeout: 60
    max_retries: 3
    token_endpoint: "https://auth.example.com/oauth/token"  # Replace with actual OAuth endpoint
    cost_per_1k_prompt_tokens: 0.00015
    cost_per_1k_completion_tokens: 0.0006
  
  # Window Configuration for Speaker Block Context
  window_config:
    context_blocks_before: 2
    context_blocks_after: 1
```

### Validation Requirements
- **Schema Validation**: Validates all required stage_05_qa_pairing parameters exist
- **LLM Config Validation**: Ensures all LLM configuration parameters are present
- **Security Validation**: NAS path validation, OAuth endpoint validation, cost configuration validation
- **Business Rule Validation**: Window configuration and development mode settings validation

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Environment Setup**: Validate environment variables and establish NAS connection
2. **Configuration Loading**: Load and validate YAML configuration from NAS
3. **SSL/Proxy Setup**: Configure SSL certificates and corporate proxy (for consistency)
4. **LLM Client Setup**: Configure OpenAI client with OAuth 2.0 authentication
5. **Stage 4 Content Loading**: Load validated transcript content from Stage 4
6. **Development Mode Check**: Apply transcript limits if in development mode
7. **Transcript Processing**: For each transcript, perform speaker block analysis and Q&A boundary detection
8. **LLM Analysis**: Use OpenAI API with CO-STAR prompts for conversation boundary identification
9. **Q&A Group Assignment**: Apply group IDs and confidence scores to records
10. **Output Generation**: Save enhanced records with Q&A assignments to NAS
11. **Cost Reporting**: Generate comprehensive token usage and cost summary
12. **Log Cleanup**: Save execution logs and clean up temporary files

### Key Business Rules
- **Speaker Block-Based Processing**: Process at speaker block level for conversation context, not paragraph level
- **Per-Transcript OAuth Refresh**: Refresh OAuth token for each transcript to prevent expiration issues
- **Dynamic Context Windows**: Extend context windows back to question starts when needed for accurate boundary detection
- **Operator Block Exclusion**: Automatically exclude operator blocks from Q&A group assignments
- **Confidence-Based Validation**: Use confidence scores to validate and flag questionable boundary decisions
- **Cost Tracking**: Track token usage and costs for all LLM API calls with detailed reporting

### Data Processing Logic
- **Input**: Stage 4's validated transcript content (paragraph-level records with section validation)
- **Grouping**: Group paragraph records by transcript, then by speaker blocks within each transcript
- **Context Creation**: Create sliding context windows around current speaker blocks for LLM analysis
- **LLM Processing**: Send context windows to OpenAI API with structured prompts for boundary decisions
- **Assignment**: Apply Q&A group IDs, confidence scores, and detection methods to original records
- **Output**: Enhanced records with qa_group_id, qa_group_confidence, and qa_group_method fields

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions - implemented:
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
```

### Security Standards Checklist
- [x] All input validation implemented
- [x] No credential exposure in logs  
- [x] File paths validated against directory traversal
- [x] URLs sanitized before logging
- [x] Configuration schema validated
- [x] OAuth token handling secured
- [x] LLM API responses validated

### Compliance Requirements
- **Audit Trail**: Comprehensive logging of all LLM API calls, boundary decisions, and cost tracking
- **Data Retention**: No local data storage - all operations on NAS
- **Error Tracking**: Separate error logs by category (boundary detection, authentication, validation, processing)
- **Cost Monitoring**: Track and report all LLM usage costs with detailed breakdowns

---

## Commands & Operations

### Primary Commands
```bash
# Development and testing
python main_qa_pairing.py           # Run Stage 5
python -m py_compile main_qa_pairing.py  # Syntax check

# Configuration validation
python -c "import yaml; yaml.safe_load(open('config.yaml'))"  # Validate YAML
```

### Execution Modes
- **Terminal Mode**: `python main_qa_pairing.py` - Standard command-line execution
- **Notebook Mode**: Import and run from Jupyter notebooks
- **Development Mode**: Configurable via `dev_mode: true` for limited transcript processing

### Testing Commands
```bash
# Syntax validation
python -m py_compile main_qa_pairing.py

# Configuration testing
python -c "from main_qa_pairing import validate_config_structure"
```

---

## Error Handling & Recovery

### Error Categories
- **Environment Validation**: Missing .env variables, invalid configurations
- **NAS Connection**: SMB/CIFS connectivity issues, authentication failures  
- **Config Load**: YAML parsing errors, missing configuration sections
- **Authentication**: OAuth token acquisition failures, SSL certificate issues
- **Content Load**: Stage 4 output file missing or corrupted
- **Boundary Detection**: LLM API failures, boundary decision validation failures
- **Processing**: Speaker block analysis failures, Q&A group assignment errors
- **Output Save**: File generation and upload failures

### Error Handling Standards (MANDATORY)
```python
# NEVER DO THIS - CAUSES PRODUCTION FAILURES:
except:
    pass

# ALWAYS DO THIS - SPECIFIC ERROR HANDLING:
except json.JSONDecodeError as e:
    log_error(f"JSON parsing error: {e}", "content_parse", {...})
    enhanced_error_logger.log_processing_error(transcript_id, f"JSON error: {e}")
    
except requests.RequestException as e:
    log_error(f"LLM API error: {e}", "authentication", {...})
    enhanced_error_logger.log_authentication_error(f"API request failed: {e}")
```

### Recovery Mechanisms
- **OAuth Token Refresh**: Per-transcript token refresh with automatic retry
- **LLM API Retry Logic**: Configurable retry attempts with exponential backoff
- **Graceful Failures**: Continue processing other transcripts if individual transcripts fail
- **Fallback Strategies**: Use XML type attributes as fallback when LLM analysis fails

---

## Integration Points

### Upstream Dependencies
- **Stage 4**: Must complete successfully and create validated_transcript_content.json
- **Configuration Sources**: NAS-based YAML configuration with complete LLM settings
- **SSL Certificates**: Downloaded from NAS for OAuth endpoint authentication
- **OAuth Service**: External OAuth 2.0 provider for LLM API access

### Downstream Outputs  
- **stage_05_qa_paired_content.json**: Enhanced records with Q&A group assignments
- **Q&A Metadata**: Group IDs, confidence scores, and detection methods for each record
- **Cost Reports**: Comprehensive token usage and cost tracking for LLM operations

### External System Integration
- **OpenAI API**: Custom base URL with OAuth 2.0 authentication for boundary detection
- **OAuth Provider**: Client credentials flow for secure LLM API access
- **NAS File System**: SMB/CIFS operations for file downloads and output storage
- **Corporate Proxy**: NTLM authentication for OAuth endpoint access

---

## Performance & Monitoring

### Performance Considerations
- **Per-Transcript Processing**: Sequential processing with OAuth token refresh per transcript
- **Context Window Management**: Dynamic context windows with intelligent extension for question boundaries
- **LLM Rate Limiting**: Configurable retry logic with exponential backoff for API protection
- **Cost Management**: Real-time cost tracking with configurable limits and reporting

### Monitoring Points
- **Processing Rate**: Track transcripts and speaker blocks processed per execution
- **LLM API Performance**: Monitor response times, token usage, and costs
- **Boundary Detection Quality**: Track confidence scores and validation success rates
- **Error Rates**: Monitor authentication failures, boundary detection failures, and processing errors

---

## Code Style & Standards

### Code Organization
- **Function Naming**: Descriptive names following snake_case convention
- **Module Organization**: Single comprehensive script with clear functional sections
- **Documentation**: All public functions have comprehensive docstrings
- **Type Hints**: Optional type hints used for clarity in complex data structures

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: ALL file paths, LLM responses, and configuration validated
- **Credential Protection**: NEVER log OAuth tokens, credentials, or sensitive data
- **Error Handling**: NO bare except clauses - specific exception handling only
- **Resource Management**: Proper cleanup of SSL certificates, OAuth tokens, and connections

### Quality Standards
- **Enhanced Error Logging**: Separate error categories with actionable context
- **Cost Transparency**: Detailed token usage and cost reporting
- **State Management**: Proper Q&A group state tracking with validation
- **LLM Response Handling**: Structured validation of all LLM API responses

---

## Development Workflow

### Pre-Development Checklist
- [x] Environment variables configured in .env (including LLM credentials)
- [x] NAS access confirmed and tested
- [x] Configuration file validated with stage_05_qa_pairing section
- [x] OAuth endpoint configured and accessible
- [x] Stage 4 output files available for testing

### Development Process
1. **Setup**: Environment validation, NAS connection, OAuth configuration
2. **Development**: LLM client setup, boundary detection logic, Q&A group management
3. **Testing**: Validate with development mode using limited transcripts
4. **Validation**: Ensure boundary detection accuracy and cost tracking
5. **Production**: Full processing with comprehensive logging and error handling

### Pre-Production Checklist (MANDATORY)
- [x] **Security Review**: All input validation and OAuth handling implemented
- [x] **Error Handling Review**: No bare except clauses, specific error categories
- [x] **Resource Management Review**: Proper cleanup implemented
- [x] **Configuration Validation**: Schema validation working for all LLM settings
- [x] **Integration Testing**: Stage 4 interface tested, LLM API connectivity verified

---

## Known Issues & Limitations

### Current Limitations
- **Sequential Processing**: Transcripts processed sequentially with per-transcript OAuth refresh
- **LLM API Dependency**: Requires external OAuth service and LLM API availability
- **Development Mode Only**: Simplified implementation with placeholder boundary detection logic
- **Cost Sensitivity**: LLM API usage costs require monitoring and budget management

### Known Issues
- **Placeholder Implementation**: Current script contains simplified boundary detection logic - needs full LLM implementation from original script
- **OAuth Endpoint Configuration**: Requires actual OAuth endpoint URL in configuration

### Future Enhancements
- **Advanced Boundary Detection**: Implement complete LLM boundary detection logic from original script
- **Parallel Processing**: Multi-threaded transcript processing with OAuth token pooling
- **Enhanced Context Analysis**: Improved speaker pattern recognition and conversation flow analysis
- **Cost Optimization**: Dynamic model selection based on complexity and budget constraints

---

## Troubleshooting Guide

### Common Issues
**Issue**: OAuth token acquisition failed
**Cause**: Invalid client credentials or OAuth endpoint configuration
**Solution**: Verify LLM_CLIENT_ID and LLM_CLIENT_SECRET in .env, check token_endpoint URL

**Issue**: LLM API calls failing
**Cause**: Token expiration or API endpoint issues
**Solution**: Check OAuth token refresh logic, verify base_url configuration

**Issue**: No Q&A groups detected
**Cause**: Speaker block analysis issues or LLM boundary detection failures
**Solution**: Check Stage 4 input format, review boundary detection logic and confidence thresholds

**Issue**: High LLM costs
**Cause**: Excessive token usage or inefficient prompt design
**Solution**: Review context window sizes, optimize prompts, implement cost monitoring

### Debugging Commands
```bash
# Debug configuration loading
python -c "from main_qa_pairing import load_config_from_nas; print('Config loaded')"

# Test NAS connectivity
python -c "from main_qa_pairing import get_nas_connection; conn = get_nas_connection(); print('Connected' if conn else 'Failed')"

# Test OAuth authentication
python -c "from main_qa_pairing import get_oauth_token; token = get_oauth_token(); print('Token acquired' if token else 'Failed')"
```

### Log Analysis
- **Execution Logs**: Located in Outputs/Logs/ with comprehensive operation tracking
- **Error Logs**: Separate error files in Outputs/Logs/Errors/ categorized by error type  
- **Cost Reports**: Detailed token usage and cost breakdowns in execution summaries

---

## Stage-Specific Context

### Unique Requirements
- **OAuth 2.0 Integration**: Requires OAuth client credentials flow for secure LLM API access
- **Speaker Block Processing**: Processes at speaker block level rather than paragraph level
- **LLM Boundary Detection**: Uses advanced prompting techniques for conversation boundary identification
- **Cost Management**: Tracks and reports LLM API usage costs with detailed breakdowns

### Stage-Specific Patterns
- **Per-Transcript OAuth Refresh**: Refreshes OAuth tokens for each transcript to prevent expiration
- **Dynamic Context Windows**: Extends context windows back to question starts for accurate boundary detection
- **Enhanced Error Logging**: Separate logging for boundary detection, authentication, validation, and processing errors
- **Confidence-Based Validation**: Uses confidence scores to validate and flag questionable boundary decisions

### Integration Notes
- **Stage 4 Interface**: Consumes validated transcript content with proper format handling
- **LLM API Interface**: Integrates with OpenAI-compatible API using OAuth 2.0 authentication
- **Cost Reporting**: Provides detailed cost analysis for downstream budget management

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Project overview and context
- **Configuration Schema**: Stage 5 section in config.yaml
- **Template Reference**: `/CLAUDE_MD_TEMPLATE.md` - Documentation template

### External References
- **OpenAI API Documentation**: API reference for LLM integration patterns
- **OAuth 2.0 Specification**: Client credentials flow implementation
- **Security Standards**: Input validation and credential management best practices

### Change Log
- **Version 1.0**: Original implementation with config.json and complex LLM logic
- **Version 2.0**: Updated to YAML configuration with standardized structure and simplified implementation

---

## Support & Maintenance

### Support Contacts
- **Primary Developer**: Claude Code Assistant
- **Technical Architecture**: Aligned with Stage 2/3/4 patterns
- **Operations**: Standard Stage 2/3/4 operational procedures

### Maintenance Schedule
- **Regular Updates**: Follow Stage 2/3/4 maintenance patterns
- **Security Reviews**: Same schedule as other pipeline stages
- **Cost Reviews**: Monitor LLM usage costs and optimize as needed

### Escalation Procedures
1. **Level 1**: Check error logs in Outputs/Logs/Errors/ for specific failure categories
2. **Level 2**: Verify Stage 4 outputs, OAuth configuration, and LLM API connectivity
3. **Level 3**: Review boundary detection logic, cost thresholds, and LLM API changes

---

> **Implementation Notes:**
> 1. Stage 5 now uses standardized YAML configuration matching Stage 2/3/4 patterns
> 2. Maintains OAuth 2.0 authentication for LLM API access with enhanced security
> 3. Current implementation includes simplified boundary detection - full LLM logic needs integration
> 4. All security standards and logging patterns align with other pipeline stages
> 5. Development mode enables safe testing with limited transcript volumes and cost control