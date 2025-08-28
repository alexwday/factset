# 🚀 Stage 4: Transcript Content Structure Validation - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Data Validation Pipeline Stage
- Stack: Python 3.11+, SMB connections, JSON processing
- Architecture: Standalone validation script with NAS integration
- Focus: Transcript content structure validation for financial data
- **Stage Purpose**: Validates that extracted transcript content has exactly 2 sections (MANAGEMENT DISCUSSION SECTION and Q&A)

## 🚨 CRITICAL RULES
- ALWAYS validate transcript structure before downstream processing
- NEVER process transcripts with invalid section structure
- MUST handle validation errors gracefully and log all issues
- Monitor validation success rates continuously
- Handle financial transcript data with appropriate security measures

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/stage-04-*, bugfix/stage-04-*, hotfix/stage-04-*
- Commit format: conventional commits preferred
- Push workflow: commit → push → CI checks → PR
- Merge strategy: squash and merge for features

### Code Quality Standards
```bash
# Pre-commit checks
python -m black . --line-length 88
python -m isort . --profile black
python -m pylint database_refresh/04_validate_structure/ --min-score 8.0
python -m flake8 database_refresh/04_validate_structure/ --max-line-length 88
```

### Testing Strategy
- Local: Unit tests with mocked NAS connections
- Integration: Full Stage 3 → Stage 4 pipeline tests
- Coverage: Minimum 80% for validation logic
- Validation-specific: Test edge cases with various section structures

## 🤖 STAGE 4 CONFIGURATION

### Validation Logic
- **Input**: Stage 3 extracted content (JSON format)
- **Process**: Group records by transcript, validate section structure
- **Output**: Valid transcripts passed to next stage, invalid transcripts logged
- **Requirements**: Exactly 2 sections with specific names

### Processing Pipeline Integration
- **Upstream**: Stage 3 (Extract Content)
- **Downstream**: Stage 5 (QA Pairing)
- **Data Format**: JSON records with section-level transcript content
- **Validation Rules**: Configurable expected section names

## 📁 PROJECT STRUCTURE
```
database_refresh/04_validate_structure/
├── main_structure_validation.py    # Main validation script
├── CLAUDE.md                      # This configuration file
└── [logs/]                        # Runtime logs (created in NAS)
```

## 🔧 ESSENTIAL COMMANDS

### Development
```bash
# Environment setup (from project root)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Load environment variables
source .env  # or export required variables

# Run Stage 4 validation
cd database_refresh/04_validate_structure
python main_structure_validation.py
```

### Stage-Specific Operations
```bash
# Development mode (limited transcripts)
# Set dev_mode: true in config.yaml

# Manual validation testing
python -c "
import main_structure_validation as val
# Test specific validation functions
"

# Check NAS connectivity
python -c "
from main_structure_validation import get_nas_connection
conn = get_nas_connection()
print('NAS Connected:' if conn else 'NAS Failed')
"
```

### Debugging and Monitoring
```bash
# Check validation logs on NAS
# Location: stage_04_validate_structure.output_logs_path

# Monitor validation success rate
grep "validation PASSED" logs/stage_04_*.json | wc -l
grep "validation FAILED" logs/stage_04_*.json | wc -l

# Check for specific validation errors
grep "validation_errors" logs/stage_04_*.json
```

## 🔗 INTEGRATIONS

### Data Sources
- **Input**: Stage 3 extracted content from NAS
- **Configuration**: YAML config file with validation rules
- **SSL Certificates**: Corporate SSL certificate for secure connections

### Processing Dependencies
- **SMB Connection**: NAS file system access
- **Environment Variables**: Credentials and connection details
- **YAML Configuration**: Expected section names and paths

### External Systems
- **NAS Storage**: Input/output data and logging
- **Corporate Proxy**: Network access configuration
- **SSL Infrastructure**: Certificate-based authentication

## 📋 CURRENT FOCUS
- Transcript structure validation implementation
- Section name matching against expected values
- Invalid transcript handling and logging
- Output format consistency for downstream stages

## ⚠️ KNOWN ISSUES
- Path validation for NAS file operations
- Error handling for malformed transcript data
- Development mode limiting for testing efficiency

## 🚫 DO NOT MODIFY
- Environment variable structure and naming
- NAS path construction logic
- SSL certificate handling mechanisms
- Expected section validation logic (without config updates)

## 💡 DEVELOPMENT NOTES

### Validation Logic
```python
# Key validation function: validate_transcript_structure()
# Input: List of transcript records, transcript identifier
# Output: Validation result dictionary with pass/fail status
# Logic: Group by section_name, verify exactly 2 sections match expected names
```

### Configuration Structure
```yaml
stage_04_validate_structure:
  description: "Validates transcript content structure"
  input_data_path: "path/to/stage3/output"
  output_data_path: "path/to/stage4/output" 
  output_logs_path: "path/to/stage4/logs"
  expected_sections:
    - "MANAGEMENT DISCUSSION SECTION"
    - "Q&A"
  dev_mode: false
  dev_max_files: 2
```

### Data Flow
1. **Load**: Stage 3 extracted content from NAS
2. **Group**: Records by transcript (ticker_year_quarter)
3. **Validate**: Each transcript against expected section structure
4. **Split**: Valid vs invalid transcripts
5. **Output**: Valid transcripts for next stage, invalid for review

## 🔍 CODE CONVENTIONS
- Follow existing error logging patterns with structured error details
- Use transcript_key format: `{ticker}_{fiscal_year}_{fiscal_quarter}`
- Maintain consistency with NAS path handling across stages
- Include detailed validation results in logging
- Preserve Stage 3 JSON format for downstream compatibility

## 📊 DATA HANDLING
- **Input Format**: JSON array of transcript content records
- **Record Structure**: Each record has section_name field for validation
- **Output Format**: Valid records in Stage 3 format for seamless pipeline flow
- **Error Records**: Invalid transcripts saved separately for manual review
- **Logging**: Comprehensive validation results with error details

### Key Functions Reference
- `validate_transcript_structure()` - Main validation logic at line 531
- `process_validation_results()` - Separates valid/invalid at line 608  
- `save_validation_outputs()` - Saves results to NAS at line 635
- `load_extracted_content()` - Loads Stage 3 data at line 501

### Development Workflow
1. Modify validation rules in config.yaml if needed
2. Test with dev_mode enabled for quick iteration
3. Monitor validation logs for success/failure patterns
4. Review invalid transcript outputs for data quality issues
5. Ensure valid outputs maintain Stage 3 format compatibility