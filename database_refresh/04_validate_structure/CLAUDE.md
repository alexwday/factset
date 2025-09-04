# 🚀 Stage 4: Transcript Structure Validation - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Earnings Call Structure Validation Pipeline
- Stack: Python 3.11+, SMB connections, JSON processing
- Architecture: Standalone validation script (main_structure_validation.py) with configurable section requirements
- Focus: Validate transcript structure for proper section organization
- **Stage Purpose**: Ensures transcripts have exactly 2 sections with expected names (configurable via YAML)
- **Grouping Strategy**: Groups records by full transcript key including version and event IDs

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
- **Input**: Stage 3 extracted paragraph-level content (JSON format - direct list of records)
- **Process**: Group records by full transcript key (ticker_quarter_year_type_eventid_versionid)
- **Output**: Both valid and invalid transcripts saved separately in Stage 3 format
- **Requirements**: Exactly 2 sections matching expected_sections from config
- **Section Names**: Configurable via config.yaml (expected_sections array)
- **Grouping Key**: Full composite key including version and event IDs (line 864)

### Processing Pipeline Integration
- **Upstream**: Stage 3 (Extract Content) - paragraph-level JSON records
- **Downstream**: Stage 5 (QA Pairing) - only validated transcripts
- **Data Format**: JSON array of paragraph records with section information
- **Validation Rules**: Check section count (must be 2) and names match config
- **Version Handling**: Each version/event ID combination treated as separate transcript

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
- Transcript structure validation with full version tracking
- Extensive grouping analysis and diagnostic output (lines 834-985)
- Base transcript analysis for multiple versions detection
- Detailed file consolidation tracking and discrepancy reporting
- Both valid and invalid transcripts saved in Stage 3 format

## ⚠️ KNOWN ISSUES
- Files with zero paragraphs from Stage 3 are missing (dropped before Stage 4)
- Discrepancy between unique files and transcript groups indicates missing files
- Development mode uses `dev_max_files` parameter (not `dev_max_transcripts`)
- Monitored institutions loaded separately from monitored_institutions.yaml

## 🚫 DO NOT MODIFY
- Environment variable structure and naming
- NAS path construction logic
- SSL certificate handling mechanisms
- Expected section validation logic (without config updates)

## 💡 DEVELOPMENT NOTES

### Validation Logic Implementation
```python
# Key validation function: validate_transcript_structure() (line 549)
# Input: List of transcript records grouped by full transcript key
# Output: Validation result dictionary with:
#   - transcript_key: full composite key (ticker_quarter_year_type_eventid_versionid)
#   - is_valid: boolean pass/fail status
#   - validation_errors: list of specific error messages
#   - sections_found: actual section names found
#   - expected_sections: sections from config
#   - section_counts: number of records per section
#   - total_records: total paragraph records in transcript
# Logic: 
#   1. Group records by section_name field (lines 559-565)
#   2. Check exactly 2 sections exist (lines 579-583)
#   3. Verify sections match expected_sections from config (lines 586-599)
#   4. Log detailed validation results for debugging
```

### Configuration Structure
```yaml
stage_04_validate_structure:
  description: "Validates transcript content structure"
  input_data_path: "path/to/stage_03_extracted_content.json"
  output_data_path: "path/to/stage4/output" 
  output_logs_path: "path/to/stage4/logs"
  expected_sections:  # REQUIRED: Must have exactly 2 section names
    - "Management Discussion Section"
    - "Q&A"
  dev_mode: false
  dev_max_files: 2  # Limit for development testing (NOT dev_max_transcripts)

# Note: monitored_institutions loaded from separate monitored_institutions.yaml file
```

### Data Flow
1. **Load**: Stage 3 extracted content from NAS (load_extracted_content at line 519)
2. **Analyze**: Input files and consolidation patterns (lines 836-845)
3. **Group**: Records by full transcript key: `ticker_quarter_year_type_eventid_versionid` (line 864)
4. **Track**: Base transcripts and multiple versions (lines 904-943)
5. **Validate**: Each transcript against expected section structure (validate_transcript_structure at line 549)
6. **Diagnose**: File discrepancies and missing records (lines 958-969)
7. **Split**: Valid vs invalid transcripts (process_validation_results at line 626)
8. **Output**: Both valid and invalid saved in Stage 3 format (save_validation_outputs at line 653)

## 🔍 CODE CONVENTIONS
- Follow existing error logging patterns with structured error details
- Use full transcript_key format: `{ticker}_{quarter}_{year}_{type}_{eventid}_{versionid}`
- Base transcript key for version analysis: `{ticker}_{quarter}_{year}`
- Maintain consistency with NAS path handling across stages
- Include detailed validation results and grouping analysis in logging
- Preserve Stage 3 JSON format (direct list) for downstream compatibility

## 📊 DATA HANDLING
- **Input Format**: JSON array of transcript content records (direct list from Stage 3)
- **Record Structure**: Each record has section_name field for validation
- **Output Format**: Valid records in Stage 3 format (stage_04_validated_content.json)
- **Invalid Records**: Saved separately as stage_04_invalid_content.json
- **Logging**: Comprehensive validation results with grouping analysis

### Key Functions Reference
- `main()` (line 737): Main orchestration function
- `load_extracted_content()` (line 519): Loads Stage 3 JSON data from NAS
- `validate_transcript_structure()` (line 549): Main validation logic
- `process_validation_results()` (line 626): Separates valid/invalid transcripts
- `save_validation_outputs()` (line 653): Saves both valid and invalid outputs

### Grouping Analysis Features (lines 834-985)
- Tracks unique files from Stage 3 input
- Groups by full transcript key (includes version/event IDs)
- Identifies base transcripts with multiple versions
- Detects file consolidation discrepancies
- Reports missing files (likely zero-paragraph files from Stage 3)
- Provides detailed diagnostic summary

### Development Workflow
1. Modify validation rules in config.yaml if needed
2. Test with dev_mode enabled (dev_max_files parameter) for quick iteration
3. Monitor validation logs for success/failure patterns
4. Review invalid transcript outputs for data quality issues
5. Ensure valid outputs maintain Stage 3 format compatibility

## 📈 DIAGNOSTIC OUTPUT

### Grouping Analysis Output
The script provides extensive diagnostic information:

```
TRANSCRIPT GROUPING ANALYSIS
============================================
📁 Unique files from Stage 3 output: X
📊 Unique transcript keys after grouping: Y
📎 Grouping key: ticker_quarter_year_type_eventid_versionid

# Discrepancy Detection
⚠️ DISCREPANCY: X files resulted in Y transcript groups
   This means Z files are missing from Stage 3 output!

# Version Analysis
🔄 Base transcripts with multiple versions: N
📈 Version distribution (by base transcript)
📝 Transcript type distribution

# Diagnostic Summary
🔍 DIAGNOSTIC SUMMARY:
  Stage 2 → Stage 3: Expected X files to produce content
  Stage 3 → Stage 4: Received records from X files
  Stage 4 Grouping: Created Y transcript groups
```

### Output Files
- **stage_04_validated_content.json**: Valid transcripts in Stage 3 format
- **stage_04_invalid_content.json**: Invalid transcripts for review
- **stage_04_validate_structure_validation_TIMESTAMP.json**: Execution log with grouping details
- **stage_04_validate_structure_validation_errors_TIMESTAMP.json**: Error log if failures occur

### Stage Summary Structure
The stage summary includes comprehensive grouping details:
```python
{
    "grouping_details": {
        "unique_files_from_stage3": int,
        "unique_transcript_versions": int,
        "grouping_key": str,
        "base_transcripts": int,
        "base_transcripts_with_multiple_versions": int,
        "version_distribution": dict,
        "transcript_type_distribution": dict
    }
}
```