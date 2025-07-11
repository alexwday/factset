# Stage 0 Bulk Refresh - Development Lessons & Standards

## Script Status: PRODUCTION READY ✅
**Last Updated**: 2025-07-11  
**Security Review**: PASSED  
**Critical Issues**: ALL RESOLVED (9/9)

## Purpose
Stage 0 performs initial bulk download of ALL historical earnings transcripts from 2023-present for monitored financial institutions. This is a one-time setup operation that establishes the complete historical baseline.

## Critical Lessons Learned

### **🚨 NEVER REPEAT THESE PATTERNS**

#### 1. Security Vulnerabilities (CRITICAL)
- **NEVER** log server IPs, URLs with auth tokens, or credentials
- **NEVER** skip input validation on file paths (directory traversal risk)
- **NEVER** use hardcoded domain names or sensitive constants
- **ALWAYS** validate and sanitize ALL external inputs

#### 2. Error Handling Anti-Patterns (CRITICAL)
- **NEVER** use bare `except:` clauses - they mask critical failures
- **NEVER** use generic `except Exception:` without specific exception types
- **NEVER** silence errors in cleanup code
- **ALWAYS** log specific error details with appropriate levels

#### 3. Resource Management Issues (HIGH)
- **NEVER** close and immediately reopen connections
- **NEVER** use recursive functions without depth limits
- **NEVER** access files while they may be in use (race conditions)
- **ALWAYS** use proper cleanup sequences and context managers

#### 4. Variable Scope Problems (HIGH)
- **NEVER** access global variables without declaring them as global
- **NEVER** assume variables are in scope - always validate
- **ALWAYS** declare global variables at function start

## Mandatory Security Framework

Every script MUST include these validation functions:

```python
def validate_file_path(path: str) -> bool:
    """Validate file path for security - prevents directory traversal."""
    if not path or not isinstance(path, str):
        return False
    if '..' in path or path.startswith('/'):
        return False
    # Additional validation...

def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure - ensures safe paths only."""
    # Implementation details...

def validate_api_response_structure(response) -> bool:
    """Validate API response structure before processing."""
    # Implementation details...

def sanitize_url_for_logging(url: str) -> str:
    """Remove auth tokens from URLs before logging."""
    # Implementation details...
```

## Mandatory Error Handling Patterns

**Replace these dangerous patterns:**
```python
# NEVER DO THIS:
except:
    pass
    
except Exception:
    return False
```

**With specific exception handling:**
```python
# ALWAYS DO THIS:
except (OSError, FileNotFoundError) as e:
    logger.error(f"File operation failed: {e}")
    return False
    
except (requests.ConnectionError, requests.Timeout) as e:
    logger.warning(f"Network error (retryable): {e}")
    raise RetryableError(e)
```

## Configuration Validation Requirements

ALL configuration MUST be validated with:
- Schema structure validation
- Data type validation  
- Range checking for numeric values
- Format validation for dates/URLs
- Security validation for paths

```python
def validate_config_schema(config: Dict[str, Any]) -> None:
    """Comprehensive configuration validation."""
    # Must validate ALL parameters before use
    # Must check data types, ranges, formats
    # Must validate security-sensitive paths
```

## Development Checklist ✅

Before ANY script deployment, verify:

### **Security Review (MANDATORY)**
- [ ] All input validation implemented
- [ ] No credential exposure in logs
- [ ] File paths validated against directory traversal
- [ ] URLs sanitized before logging
- [ ] Configuration schema validated

### **Error Handling Review (MANDATORY)**
- [ ] No bare `except:` clauses anywhere
- [ ] Specific exception types for each operation
- [ ] Appropriate logging levels used
- [ ] Error context preserved in logs

### **Resource Management Review (MANDATORY)**
- [ ] No connection open/close/reopen patterns
- [ ] Proper cleanup in finally blocks
- [ ] No race conditions in file operations
- [ ] Context managers used where appropriate

### **Code Quality Review (MANDATORY)**
- [ ] Global variables properly declared
- [ ] Function parameter validation
- [ ] Type hints for all function signatures
- [ ] Comprehensive docstrings

## Testing Requirements

Before deployment, MUST test:
1. **Malformed input handling** - script should fail gracefully
2. **Network failure scenarios** - proper retry and error handling  
3. **Authentication failures** - clear error messages, no retries
4. **File system errors** - proper cleanup and error reporting
5. **Configuration errors** - validation catches all issues

## Performance Standards

- API rate limiting: 2-second delays minimum
- Connection reuse: Single connection per session
- Memory efficiency: Stream large files, don't load entirely
- Error recovery: Distinguish retryable vs permanent failures

## Audit Requirements

For financial data processing:
- ALL operations must be logged with timestamps
- Error conditions must be captured with full context  
- Authentication events must be auditable
- Data integrity checks must be verifiable

## File Structure Standards

```
stage_X_description/
├── CLAUDE.md           # This file - lessons and standards
├── X_script_name.py    # Main script following all standards
├── config_schema.json  # Configuration schema definition
└── tests/             # Comprehensive test suite
    ├── test_validation.py
    ├── test_security.py
    └── test_integration.py
```

## Critical Dependencies

Ensure these are available and validated:
- Environment variables with format validation
- NAS connectivity with authentication
- API credentials with proper scoping
- SSL certificates with integrity checking
- Configuration files with schema validation

## Migration Notes

When moving from working script to stage-based:
1. Copy ALL validation functions
2. Implement ALL error handling patterns  
3. Add ALL security validations
4. Test ALL failure scenarios
5. Verify ALL configuration validation

## Success Metrics

A properly implemented script should:
- Never crash from invalid input
- Provide clear error messages for all failure modes
- Complete successfully in production environment
- Pass all security validation checks
- Handle network/auth failures gracefully
- Maintain data integrity under all conditions

---

**Remember**: These standards exist because we found these exact issues in production code. Following them prevents critical failures and security vulnerabilities.