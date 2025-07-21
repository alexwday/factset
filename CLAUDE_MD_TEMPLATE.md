# CLAUDE.md Universal Template for FactSet Pipeline Stages

> **Template Version**: 1.0 | **Created**: 2024-07-21  
> **Purpose**: Standardized CLAUDE.md template for all pipeline stages based on official Anthropic best practices

---

## Project Context
<!-- Brief overview of this specific stage's role in the broader pipeline -->
- **Stage**: [Stage Number and Name]
- **Primary Purpose**: [One-line description of stage purpose]
- **Pipeline Position**: [Where this fits in the overall workflow]
- **Production Status**: [DEVELOPMENT/TESTING/PRODUCTION READY]

---

## Tech Stack & Dependencies

### Core Technologies
- **Language**: Python 3.8+
- **Primary Framework**: [Main framework/SDK used]
- **Authentication**: [Authentication method - OAuth, Basic Auth, etc.]
- **Storage**: NAS (SMB/CIFS) with NTLM v2 authentication
- **Configuration**: YAML-based configuration from NAS

### Required Dependencies
```python
# Core dependencies for this stage
import fds.sdk.EventsandTranscripts  # [if applicable]
import requests                      # Network operations
import pysmb                        # NAS connectivity
import yaml                         # Configuration parsing
import python-dotenv               # Environment variables
# Stage-specific imports here
```

### Environment Requirements
```bash
# Required .env variables (list all 14+ required)
API_USERNAME=                # FactSet API credentials
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
CONFIG_PATH=                 # NAS configuration path
CLIENT_MACHINE_NAME=
LLM_CLIENT_ID=              # [if applicable for LLM stages]
LLM_CLIENT_SECRET=
```

---

## Architecture & Design

### Core Design Patterns
- **[Pattern 1]**: [Description and rationale]
- **[Pattern 2]**: [Description and rationale]
- **[Pattern 3]**: [Description and rationale]

### File Structure
```
stage_X_[name]/
├── [main_script].py         # Primary execution script
├── CLAUDE.md               # This context file
├── config_schema.yaml      # [if applicable] Configuration schema
├── tests/                  # [if applicable] Test files
└── docs/                   # [if applicable] Additional documentation
```

### Key Components
1. **[Component 1 Name]**: [Purpose and functionality]
2. **[Component 2 Name]**: [Purpose and functionality]
3. **[Component 3 Name]**: [Purpose and functionality]

---

## Configuration Management

### Configuration Schema
```yaml
# NAS configuration structure for this stage
stage_[X]:
  [setting_category_1]:
    [specific_setting]: [description]
    [specific_setting]: [description]
  [setting_category_2]:
    [specific_setting]: [description]
```

### Validation Requirements
- **Schema Validation**: [Describe validation patterns]
- **Security Validation**: [Describe security checks]
- **Business Rule Validation**: [Describe business logic checks]

---

## Business Logic & Workflow

### Primary Workflow Steps
1. **[Step 1]**: [Description of first major step]
2. **[Step 2]**: [Description of second major step]
3. **[Step 3]**: [Description of third major step]
4. **[Step 4]**: [Description of fourth major step]
5. **[Continue as needed]**

### Key Business Rules
- **[Rule 1]**: [Description and rationale]
- **[Rule 2]**: [Description and rationale]
- **[Rule 3]**: [Description and rationale]

### Data Processing Logic
- **Input**: [Describe input data/sources]
- **Processing**: [Describe transformation logic]
- **Output**: [Describe output format/destination]
- **Validation**: [Describe validation checks]

---

## Security & Compliance

### Security Requirements (MANDATORY)
```python
# Required security functions - copy these patterns:
def validate_file_path(path: str) -> bool:
    """Prevent directory traversal attacks."""
    
def validate_nas_path(path: str) -> bool:
    """Ensure safe NAS paths only."""
    
def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
```

### Security Standards Checklist
- [ ] All input validation implemented
- [ ] No credential exposure in logs  
- [ ] File paths validated against directory traversal
- [ ] URLs sanitized before logging
- [ ] Configuration schema validated

### Compliance Requirements
- **Audit Trail**: [Describe logging requirements]
- **Data Retention**: [Describe data retention policies]
- **Error Tracking**: [Describe error logging requirements]

---

## Commands & Operations

### Primary Commands
```bash
# Development and testing
python [main_script].py                    # Run the stage
python -m py_compile [main_script].py      # Syntax check
pylint [main_script].py                    # Linting (if configured)

# Configuration validation
python -c "import yaml; yaml.safe_load(open('config.yaml'))"  # Validate YAML
```

### Execution Modes
- **Terminal Mode**: `python [script].py` - Standard command-line execution
- **Notebook Mode**: Import and run from Jupyter notebooks
- **Scheduled Mode**: [If applicable] Automated execution requirements

### Testing Commands
```bash
# Unit testing (if available)
python -m pytest tests/

# Integration testing (if available)  
python [test_script].py

# Configuration testing
python [config_validation_script].py
```

---

## Error Handling & Recovery

### Error Categories
- **[Category 1]**: [Description and recovery approach]
- **[Category 2]**: [Description and recovery approach]  
- **[Category 3]**: [Description and recovery approach]

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
- **Retry Logic**: [Describe retry patterns and limits]
- **Fallback Strategies**: [Describe fallback approaches]
- **Error Reporting**: [Describe error notification methods]

---

## Integration Points

### Upstream Dependencies
- **Previous Stage**: [If applicable] Output from Stage X-1
- **External APIs**: [List external API dependencies]
- **Configuration Sources**: NAS-based YAML configuration

### Downstream Outputs  
- **Next Stage**: [If applicable] Input for Stage X+1
- **File Outputs**: [Describe file outputs and formats]
- **Log Outputs**: [Describe logging outputs]

### External System Integration
- **FactSet API**: [If applicable] Specific endpoints and methods
- **NAS File System**: SMB/CIFS operations and paths
- **Corporate Proxy**: NTLM authentication requirements
- **LLM APIs**: [If applicable] Authentication and usage patterns

---

## Performance & Monitoring

### Performance Considerations
- **Rate Limiting**: [Describe rate limiting strategies]
- **Memory Management**: [Describe memory usage patterns]
- **Connection Management**: [Describe connection pooling/reuse]
- **Resource Cleanup**: [Describe cleanup procedures]

### Monitoring Points
- **Execution Metrics**: [Key metrics to track]
- **Error Rates**: [Error thresholds and alerting]
- **Performance Metrics**: [Performance KPIs]
- **Resource Utilization**: [Resource monitoring requirements]

---

## Code Style & Standards

### Code Organization
- **Function Naming**: [Describe naming conventions]
- **Class Structure**: [If applicable] Class organization patterns
- **Module Organization**: [Describe module structure]
- **Documentation**: [Documentation requirements]

### Security Standards (NON-NEGOTIABLE)
- **Input Validation**: ALL inputs must be validated
- **Credential Protection**: NEVER log credentials or sensitive data
- **Error Handling**: NO bare except clauses allowed
- **Resource Management**: Proper cleanup in finally blocks

### Quality Standards
- **Line Length**: Maximum 100 characters
- **Function Length**: Maximum 50 lines per function
- **Complexity**: Maximum cyclomatic complexity of 10
- **Documentation**: All public functions must have docstrings

---

## Development Workflow

### Pre-Development Checklist
- [ ] Environment variables configured in .env
- [ ] NAS access confirmed and tested
- [ ] Configuration file validated
- [ ] Dependencies installed and verified

### Development Process
1. **Setup**: [Environment setup steps]
2. **Development**: [Development workflow steps]
3. **Testing**: [Testing procedures]
4. **Validation**: [Validation requirements]
5. **Deployment**: [Deployment procedures]

### Pre-Production Checklist (MANDATORY)
- [ ] **Security Review**: All input validation implemented
- [ ] **Error Handling Review**: No bare except clauses
- [ ] **Resource Management Review**: Proper cleanup implemented
- [ ] **Configuration Validation**: Schema validation working
- [ ] **Integration Testing**: All external systems tested

---

## Known Issues & Limitations

### Current Limitations
- **[Limitation 1]**: [Description and workaround]
- **[Limitation 2]**: [Description and workaround]
- **[Limitation 3]**: [Description and workaround]

### Known Issues
- **[Issue 1]**: [Description and status]
- **[Issue 2]**: [Description and status]
- **[Issue 3]**: [Description and status]

### Future Enhancements
- **[Enhancement 1]**: [Description and priority]
- **[Enhancement 2]**: [Description and priority]
- **[Enhancement 3]**: [Description and priority]

---

## Troubleshooting Guide

### Common Issues
**Issue**: [Problem description]
**Cause**: [Root cause]
**Solution**: [Step-by-step resolution]

**Issue**: [Problem description]
**Cause**: [Root cause]
**Solution**: [Step-by-step resolution]

### Debugging Commands
```bash
# Debug configuration loading
python -c "from [module] import load_config; print(load_config())"

# Test NAS connectivity
python -c "from [module] import test_nas_connection; test_nas_connection()"

# Validate environment
python -c "from [module] import validate_environment; validate_environment()"
```

### Log Analysis
- **Execution Logs**: Location and interpretation
- **Error Logs**: Location and interpretation  
- **Debug Logs**: Location and interpretation

---

## Stage-Specific Context

### Unique Requirements
- **[Requirement 1]**: [Description specific to this stage]
- **[Requirement 2]**: [Description specific to this stage]
- **[Requirement 3]**: [Description specific to this stage]

### Stage-Specific Patterns
- **[Pattern 1]**: [Implementation pattern unique to this stage]
- **[Pattern 2]**: [Implementation pattern unique to this stage]
- **[Pattern 3]**: [Implementation pattern unique to this stage]

### Integration Notes
- **Previous Stage Interface**: [How this stage consumes previous output]
- **Next Stage Interface**: [How this stage produces output for next stage]
- **Parallel Processing**: [If applicable] Parallel execution considerations

---

## Documentation & References

### Internal Documentation
- **Main Project CLAUDE.md**: `/CLAUDE.md` - Project overview and context
- **Configuration Schema**: [Location of schema documentation]
- **API Documentation**: [Location of internal API docs]

### External References
- **FactSet SDK Documentation**: [Relevant SDK documentation links]
- **Anthropic CLAUDE.md Best Practices**: [Official documentation references]
- **Security Standards**: [Relevant security documentation]

### Change Log
- **Version 1.0**: Initial implementation
- **[Future versions]**: [Track major changes]

---

## Support & Maintenance

### Support Contacts
- **Primary Developer**: [Contact information]
- **Technical Lead**: [Contact information]
- **Operations Team**: [Contact information]

### Maintenance Schedule
- **Regular Updates**: [Update frequency and process]
- **Security Reviews**: [Security review schedule]
- **Performance Reviews**: [Performance review schedule]

### Escalation Procedures
1. **Level 1**: [Initial troubleshooting steps]
2. **Level 2**: [Advanced troubleshooting and escalation]
3. **Level 3**: [Critical issue escalation procedures]

---

> **Template Usage Notes:**
> 1. Replace all bracketed placeholders with stage-specific content
> 2. Remove sections not applicable to your stage
> 3. Add stage-specific sections as needed
> 4. Maintain consistent formatting and structure across all stages
> 5. Update version number when making significant changes
> 6. Commit all CLAUDE.md files to version control for team sharing
