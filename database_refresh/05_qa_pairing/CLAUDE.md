# CLAUDE.md for Stage 5: Q&A Boundary Detection & Conversation Pairing

> **Template Version**: 1.0 | **Created**: 2024-07-22  
> **Purpose**: Q&A conversation boundary detection using LLM-powered speaker block analysis

---

## Project Context
<!-- Q&A boundary detection and conversation pairing -->
- **Stage**: Stage 5 - Q&A Boundary Detection & Conversation Pairing
- **Primary Purpose**: Identify and group question-answer conversation boundaries using LLM analysis
- **Pipeline Position**: Processes Stage 4 validated content to create Q&A conversation groups
- **Production Status**: PRODUCTION READY ✅

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: OpenAI API with function calling for LLM boundary detection
- **Authentication**: OAuth 2.0 client credentials flow with per-transcript token refresh
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for Stage 5
import openai                       # LLM API integration
import requests                     # OAuth token management
import pysmb                        # NAS connectivity
import yaml                         # Configuration parsing
import python-dotenv               # Environment variables
import json                         # Data processing
import re                           # Speaker pattern matching
from collections import defaultdict # Data grouping
from typing import Dict, List, Optional, Tuple, Any
```

### Environment Requirements
```bash
# Required .env variables (16 total)
API_USERNAME=                # FactSet API credentials (for future integration)
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
NAS_PORT=
CONFIG_PATH=                 # NAS configuration path
CLIENT_MACHINE_NAME=
LLM_CLIENT_ID=              # LLM OAuth credentials
LLM_CLIENT_SECRET=
```

---

## Architecture & Design

### Core Design Patterns
- **Speaker Block Analysis**: Processes speaker blocks rather than individual paragraphs for conversation context
- **Dynamic Context Windows**: Extends context back to question starts for better boundary decisions
- **State Machine Approach**: Maintains comprehensive Q&A state with real-time validation and auto-correction
- **Progressive Fallback Strategy**: LLM detection → XML type fallback → conservative grouping

### File Structure
```
database_refresh/05_qa_pairing/
├── main_qa_pairing.py              # Primary execution script
├── CLAUDE.md                      # This context file
└── main_qa_pairing_placeholder_backup.py  # Backup of old placeholder version
```

### Key Components
1. **LLM Boundary Detection**: Uses OpenAI API with CO-STAR prompting for intelligent conversation boundary analysis
2. **Speaker Block Processing**: Groups paragraph records by speaker blocks for contextual analysis
3. **OAuth Token Management**: Per-transcript token refresh to prevent expiration during long processing runs
4. **Enhanced Error Logging**: Categorized error tracking (boundary detection, authentication, validation, processing)

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for Stage 5
stage_05_qa_pairing:
  description: "Q&A boundary detection and conversation pairing"
  input_data_path: "path/to/stage_04_validated_content.json"
  output_data_path: "Outputs/Data/stage_05_qa_pairing"
  output_logs_path: "Outputs/Logs"
  dev_mode: true
  dev_max_transcripts: 2
  
  llm_config:
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o-mini"
    temperature: 0.1
    max_tokens: 500
    timeout: 60
    max_retries: 3
    token_endpoint: "https://auth.example.com/oauth/token"
    cost_per_1k_prompt_tokens: 0.00015
    cost_per_1k_completion_tokens: 0.0006
  
  window_config:
    context_blocks_before: 2
    context_blocks_after: 1
```

### Validation Requirements
- **Schema Validation**: Validates all required configuration sections and parameters
- **Security Validation**: Validates NAS paths and prevents directory traversal
- **Business Rule Validation**: Ensures LLM config parameters are within acceptable ranges

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **Environment Setup**: Validate environment variables, establish NAS connection, load YAML configuration
2. **SSL & Authentication**: Download SSL certificate from NAS, obtain OAuth token for LLM API
3. **Content Loading**: Load Stage 4 validated content with paragraph-level records grouped by speaker blocks
4. **Transcript Processing**: Process each transcript individually with per-transcript OAuth refresh
5. **Q&A Boundary Detection**: Use LLM analysis to identify conversation boundaries between speaker blocks
6. **Group Assignment**: Apply Q&A group IDs to paragraph records based on detected boundaries
7. **Output Generation**: Save enhanced records with Q&A assignments to NAS

### Key Business Rules
- **Speaker Block Focus**: Analyze conversation at speaker block level, not individual paragraphs
- **Per-Transcript OAuth**: Refresh OAuth token for each transcript to eliminate expiration issues
- **Q&A Section Filtering**: Only process records with `section_type == "Investor Q&A"`
- **Operator Exclusion**: Automatically exclude operator blocks from Q&A group assignments
- **Progressive Group IDs**: Assign sequential group IDs (1, 2, 3, etc.) to conversation pairs

### Data Processing Logic
- **Input**: Stage 4 validated content with paragraph records, speaker block groupings, section classifications
- **Processing**: Speaker block analysis with dynamic context windows, LLM boundary detection, state management
- **Output**: Enhanced records with 3 new fields: `qa_group_id`, `qa_group_confidence`, `qa_group_method`
- **Validation**: Real-time Q&A state validation with auto-correction for inconsistent decisions

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions implemented:
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

### Compliance Requirements
- **Audit Trail**: Comprehensive execution logs with timestamps and decision tracking
- **Data Retention**: All logs uploaded to NAS with permanent retention
- **Error Tracking**: Categorized error logs (boundary detection, authentication, validation, processing)

---

## Commands & Operations

### Primary Commands
```bash
# Development and testing
python main_qa_pairing.py                    # Run Stage 5
python -m py_compile main_qa_pairing.py      # Syntax check
pylint main_qa_pairing.py                    # Linting (if configured)

# Configuration validation
python -c "import yaml; yaml.safe_load(open('../config.yaml'))"  # Validate YAML
```

### Execution Modes
- **Terminal Mode**: `python main_qa_pairing.py` - Standard command-line execution
- **Notebook Mode**: Import and run from Jupyter notebooks
- **Development Mode**: Limited transcript processing via `dev_mode: true`

### Testing Commands
```bash
# Syntax validation
python3 -m py_compile main_qa_pairing.py

# Configuration testing
python3 -c "import ast; ast.parse(open('main_qa_pairing.py').read()); print('✅ Syntax is valid')"
```

---

## Error Handling & Recovery

### Error Categories
- **Boundary Detection Errors**: LLM analysis failures, speaker block processing issues
- **Authentication Errors**: OAuth token failures, SSL certificate issues  
- **Validation Errors**: Q&A group assignment validation failures, state inconsistencies
- **Processing Errors**: General transcript processing failures, network issues

### Error Handling Standards (MANDATORY)
```python
# NEVER DO THIS - CAUSES PRODUCTION FAILURES:
except:
    pass

# ALWAYS DO THIS - SPECIFIC ERROR HANDLING:
except (OSError, FileNotFoundError) as e:
    log_error(f"File operation failed: {e}", "file_operation", {...})
    
except (requests.ConnectionError, requests.Timeout) as e:
    log_error(f"Network error: {e}", "network", {...})
```

### Recovery Mechanisms
- **OAuth Token Retry**: Automatic token refresh on authentication failures
- **XML Fallback**: Falls back to XML type attributes when LLM analysis fails
- **Conservative Grouping**: Final fallback using speaker pattern heuristics
- **Error Reporting**: Comprehensive error logs uploaded to NAS with recovery instructions

---

## Integration Points

### Upstream Dependencies
- **Previous Stage**: Stage 4 validated content with paragraph records and speaker block groupings
- **External APIs**: OpenAI API for LLM boundary detection
- **Configuration Sources**: NAS-based YAML configuration

### Downstream Outputs  
- **Next Stage**: Enhanced records with Q&A group assignments for further analysis
- **File Outputs**: JSON files with original records plus Q&A fields
- **Log Outputs**: Execution logs, error logs, and cost tracking reports

### External System Integration
- **OpenAI API**: OAuth 2.0 authentication, function calling, cost tracking
- **NAS File System**: SMB/CIFS operations for config loading and output storage
- **Corporate Proxy**: NTLM authentication for external API access

---

## Performance & Monitoring

### Performance Considerations
- **Per-Transcript OAuth**: Prevents token expiration but adds authentication overhead
- **Memory Management**: Processes transcripts sequentially to manage memory usage
- **Rate Limiting**: Respects LLM API rate limits with timeout configurations
- **Resource Cleanup**: Proper SSL certificate and connection cleanup

### Monitoring Points
- **Token Usage**: Real-time tracking of prompt/completion tokens and costs
- **Processing Time**: Per-transcript timing metrics
- **Error Rates**: Categorized error tracking with recovery success rates
- **Q&A Detection Quality**: Confidence scores and method distribution

---

## Code Style & Standards

### Code Organization
- **Function Naming**: Descriptive names following snake_case convention
- **Class Structure**: EnhancedErrorLogger for categorized error tracking
- **Module Organization**: Single-file standalone script with clear function grouping
- **Documentation**: Comprehensive docstrings for all major functions

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: All paths, URLs, and configuration validated
- **Credential Protection**: OAuth tokens never logged, URLs sanitized
- **Error Handling**: Specific exception handling with detailed logging
- **Resource Management**: Proper cleanup of SSL certificates and connections

### Quality Standards
- **Line Length**: Generally kept under 100 characters
- **Function Length**: Complex functions broken into sub-functions
- **Complexity**: State management isolated for clarity
- **Documentation**: All public functions have comprehensive docstrings

---

## Development Workflow

### Pre-Development Checklist
- [x] Environment variables configured in .env
- [x] NAS access confirmed and tested
- [x] Configuration file validated
- [x] Dependencies installed and verified

### Development Process
1. **Setup**: Environment variables, NAS connectivity, YAML configuration
2. **Development**: Modular function development with comprehensive error handling
3. **Testing**: Syntax validation, configuration testing, integration testing
4. **Validation**: Q&A output validation, cost tracking verification
5. **Deployment**: Production deployment with full logging

### Pre-Production Checklist (MANDATORY)
- [x] **Security Review**: All input validation implemented
- [x] **Error Handling Review**: No bare except clauses
- [x] **Resource Management Review**: Proper cleanup implemented
- [x] **Configuration Validation**: Schema validation working
- [x] **Integration Testing**: LLM API and NAS integration tested

---

## Known Issues & Limitations

### Current Limitations
- **Language Dependency**: Assumes English content patterns for speaker role detection
- **LLM API Dependency**: Requires external LLM API availability and proper authentication
- **Processing Speed**: Sequential per-transcript processing limits throughput for large volumes

### Known Issues
- **None Currently**: All major issues resolved during restoration from original implementation
- **Token Cost**: LLM usage generates per-token costs that scale with transcript volume
- **OAuth Expiration**: Long-running processes may require additional token management

### Future Enhancements
- **Parallel Processing**: Multi-threaded transcript processing for improved performance
- **Caching Strategy**: Cache similar speaker block patterns to reduce LLM API calls
- **Multi-Language Support**: Extend operator detection and speaker patterns to other languages

---

## Troubleshooting Guide

### Common Issues
**Issue**: Q&A sections not receiving qa_group_id assignments
**Cause**: Placeholder logic was being used instead of full LLM boundary detection
**Solution**: Restore full LLM boundary detection logic from original implementation

**Issue**: OAuth token expiration during long processing runs
**Cause**: Single token used for entire processing session
**Solution**: Per-transcript token refresh implemented to prevent expiration

**Issue**: Operator blocks receiving Q&A group assignments
**Cause**: Insufficient operator detection patterns
**Solution**: Enhanced operator detection using both speaker names and content patterns

### Debugging Commands
```bash
# Debug configuration loading
python -c "from main_qa_pairing import load_config_from_nas; print('Config validation passed')"

# Test NAS connectivity
python -c "from main_qa_pairing import get_nas_connection; conn = get_nas_connection(); print('NAS connected' if conn else 'NAS failed')"

# Validate environment
python -c "from main_qa_pairing import validate_environment_variables; validate_environment_variables(); print('Environment validated')"
```

### Log Analysis
- **Execution Logs**: Located at `Outputs/Logs/` on NAS with timestamp and processing details
- **Error Logs**: Categorized error logs at `Outputs/Logs/Errors/` with recovery instructions
- **Debug Logs**: Detailed speaker block decisions and LLM interaction logs

---

## Stage-Specific Context

### Unique Requirements
- **Speaker Block Analysis**: Processes conversation at speaker block level for better context
- **Dynamic Context Windows**: Context windows extend back to question starts when needed
- **Real-time State Management**: Q&A state tracking with auto-correction for inconsistencies
- **Progressive Group Assignment**: Sequential Q&A group IDs (1, 2, 3, etc.) for conversation pairs

### Stage-Specific Patterns
- **CO-STAR Prompting**: Context, Objective, Style, Tone, Audience, Response framework for LLM
- **Function Calling Schema**: Dynamic tools that change based on current Q&A state
- **Per-Transcript OAuth**: Token refresh for each transcript to prevent expiration issues
- **Enhanced Speaker Formatting**: Role indicators ([ANALYST], [EXECUTIVE], [OPERATOR]) for LLM clarity

### Integration Notes
- **Previous Stage Interface**: Consumes Stage 4 validated content with paragraph-level records
- **Next Stage Interface**: Produces enhanced records with Q&A group assignments
- **Sequential Processing**: Processes transcripts one at a time for memory management and state clarity

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Project overview and pipeline context
- **Configuration Schema**: `config.yaml` - YAML configuration with stage_05_qa_pairing section
- **Template Documentation**: `CLAUDE_MD_TEMPLATE.md` - Universal template for pipeline stages

### External References
- **OpenAI API Documentation**: Function calling, authentication, and best practices
- **Anthropic CLAUDE.md Best Practices**: Official documentation for context files
- **SMB/CIFS Protocol**: For NAS integration and file operations

### Change Log
- **Version 1.0**: Initial implementation with placeholder Q&A logic
- **Version 2.0**: Restored full LLM boundary detection logic with speaker block analysis
- **Version 2.1**: Enhanced error handling and OAuth token management

---

## Support & Maintenance

### Support Contacts
- **Primary Developer**: Implementation team
- **Technical Lead**: Pipeline architecture team
- **Operations Team**: Production support team

### Maintenance Schedule
- **Regular Updates**: Configuration updates as needed for new transcript patterns
- **Security Reviews**: Quarterly security reviews for credential management
- **Performance Reviews**: Monthly performance reviews for cost optimization

### Escalation Procedures
1. **Level 1**: Check logs for error categories, validate configuration and environment
2. **Level 2**: Analyze LLM API integration, OAuth token management, NAS connectivity
3. **Level 3**: Deep analysis of Q&A boundary detection logic, state management issues

---

> **Stage 5 Implementation Notes:**
> - Full LLM boundary detection logic restored from original implementation
> - Per-transcript OAuth refresh eliminates token expiration issues
> - Enhanced speaker block analysis with dynamic context windows
> - Comprehensive error handling with categorized logging
> - Production-ready with complete fallback strategies