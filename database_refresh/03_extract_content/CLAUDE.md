# 🚀 Stage 3: Content Extraction & Paragraph-Level Breakdown Configuration

## 🎯 PROJECT CONTEXT
- Type: XML Transcript Content Extraction with Speaker Attribution
- Stack: Python 3.11+, XML parsing (ElementTree), NAS/SMB integration
- Architecture: Stage 3 of earnings transcript processing pipeline (main_content_extraction.py)
- Focus: Extract paragraph-level content with speaker identification and Q&A flags from all transcripts

## 🚨 CRITICAL RULES
- ALWAYS validate XML structure before processing
- NEVER commit API keys or credentials to repository
- MUST handle XML parsing errors gracefully and continue processing
- Monitor NAS connection stability continuously
- Handle financial transcript data with appropriate security measures
- Validate JSON output structure before saving (validate_and_preview_json at line 868)
- Track and report files with zero paragraphs separately (these are NOT saved to output)

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
python -m pylint database_refresh/03_extract_content/ --min-score 8.0
python -m flake8 database_refresh/03_extract_content/ --max-line-length 88
```

### Testing Strategy
- Local: Test XML parsing with sample files
- Integration: Full NAS connection and file processing tests
- Data validation: Verify extracted paragraph structure
- Error handling: Test malformed XML and connection failures

## 🤖 STAGE 3 CONFIGURATION

### Core Functionality
- Process ALL XML transcripts from Stage 2's processing queue (not just earnings calls)
- Extract paragraph-level content with speaker identification and attribution
- Parse participant metadata from XML and format speaker strings (format_speaker_string at line 718)
- Determine Q&A flags based on XML speaker type attributes (determine_qa_flag at line 743)
- Generate structured JSON output with one record per paragraph
- Track duplicate files and version information for all transcripts
- Monitor and report files that produce zero paragraphs (these are excluded from output)

### Key Processing Steps
1. Load processing queue from Stage 2 output (load_processing_queue at line 590)
2. Load monitored_institutions.yaml separately if available (lines 231-245)
3. Extract metadata from both file record and file path (extract_metadata_from_file_path at line 524)
4. Check for duplicate files in processing queue (lines 1051-1066)
5. Parse XML transcripts with namespace handling (parse_transcript_xml at line 620)
6. Extract participants using 'id' attribute (not 'pid') for mapping (line 654)
7. Process ALL sections found in XML (not limited to specific sections)
8. Generate paragraph-level records with speaker block IDs (extract_transcript_paragraphs at line 754)
9. Track extensive file statistics including version information (lines 1069-1320)
10. Validate JSON structure before saving (validate_and_preview_json at line 868)
11. Save only files with paragraphs to output (save_extracted_content at line 933)

## 📁 PROJECT STRUCTURE
```
database_refresh/03_extract_content/
  main_content_extraction.py    # Main processing script
  CLAUDE.md                     # This configuration file
```

## 🔧 ESSENTIAL COMMANDS

### Development
```bash
# Environment setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Load environment variables
source .env  # or set environment variables

# Run Stage 3 content extraction
python database_refresh/03_extract_content/main_content_extraction.py
```

### Testing & Debugging
```bash
# Test XML parsing locally
python -c "
import xml.etree.ElementTree as ET
with open('sample.xml', 'rb') as f:
    root = ET.fromstring(f.read())
    # Handle namespace detection
    namespace = ''
    if root.tag.startswith('{'):
        namespace = root.tag.split('}')[0] + '}'
    print(f'Root tag: {root.tag}')
    print(f'Namespace: {namespace}')
"

# Validate JSON structure
python -c "
import json
with open('stage_03_extracted_content.json', 'r') as f:
    data = json.load(f)
    print(f'Records: {len(data)}')
    if data:
        print('First record fields:', list(data[0].keys()))
"
```

### Configuration Management
```bash
# Check required environment variables
python -c "
import os
required = ['API_USERNAME', 'API_PASSWORD', 'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP']
missing = [var for var in required if not os.getenv(var)]
print('Missing vars:', missing if missing else 'None')
"
```

## 🔗 INTEGRATIONS

### Data Sources
- Stage 2 processing queue (JSON format)
- XML transcript files on NAS
- Configuration YAML from NAS
- SSL certificates for secure connections

### Output Data
- Structured JSON with paragraph-level records
- Execution logs with detailed processing information
- Error logs for troubleshooting and monitoring

### NAS/SMB Integration
- SMB connection for file access
- Recursive directory creation
- Secure file upload/download operations
- SSL certificate management

## 📋 CURRENT FOCUS
- Processing all transcript types from Stage 2 queue
- Tracking duplicate files and multiple versions per transcript
- Comprehensive file statistics and version tracking
- Separating files with content from those with zero paragraphs
- Detailed execution logging with paragraph distribution analysis

## ⚠️ KNOWN ISSUES
- XML files may have inconsistent namespace usage (handled by auto-detection)
- Some transcript files may have missing participant information (defaults to "Unknown Speaker")
- Network connectivity issues can cause NAS timeouts
- Large XML files may require memory optimization
- Processing queue may contain duplicate file paths (detected and skipped)
- Some transcripts produce zero paragraphs and are excluded from output

## 🚫 DO NOT MODIFY
- NAS connection credentials in environment variables
- SSL certificate files (temporary files are cleaned up automatically)
- Processing queue format from Stage 2
- XML namespace handling logic without thorough testing

## 💡 DEVELOPMENT NOTES
- Use ElementTree for XML parsing with automatic namespace detection (lines 629-637)
- Clean text content to prevent JSON encoding issues (escape quotes, remove newlines)
- Maintain speaker block IDs to preserve conversation flow for Stage 5 (incremented per speaker block)
- Format speaker strings by combining name, title, and affiliation (format_speaker_string at line 718)
- Handle missing participant data by defaulting to "Unknown Speaker"
- Process ALL transcripts from Stage 2's queue (not filtered for earnings calls)
- Each paragraph becomes a separate JSON record for downstream processing
- Files with zero paragraphs are tracked but NOT included in output JSON
- Extensive duplicate detection and version tracking implemented (lines 1051-1320)

## 🔍 CODE CONVENTIONS
- Follow existing logging patterns (log_console, log_execution, log_error)
- Use NAS utility functions for file operations
- Validate file paths and data before processing
- Clean up temporary files (SSL certificates) after execution
- Maintain consistent error handling across functions

## 📊 DATA HANDLING

### Input Data Format
- Stage 2 processing queue: JSON array of file records (ALL transcript types)
- File records include: file_path, date_last_modified, institution_id, ticker
- XML transcripts: FactSet format with meta, participants, and body sections
- Configuration: YAML with stage-specific parameters loaded from NAS
- Monitored institutions: Loaded separately from monitored_institutions.yaml if available

### Output Data Format
```json
{
  "file_path": "Data/2024/Q1/Banks/JPM_JPMorgan/JPM_Q1_2024_E1_12345_1.xml",
  "date_last_modified": "2024-01-01T00:00:00",
  "institution_id": "54321",
  "ticker": "JPM",
  "filename": "JPM_Q1_2024_E1_12345_1.xml",
  "fiscal_year": "2024",
  "fiscal_quarter": "Q1",
  "institution_type": "Banks",
  "company_name": "JPMorgan",
  "transcript_type": "E1",
  "event_id": "12345",
  "version_id": "1",
  "title": "Q1 2024 Earnings Call",
  "section_id": 1,
  "section_name": "Management Discussion",
  "paragraph_id": 1,
  "speaker_block_id": 1,
  "question_answer_flag": null,
  "speaker": "Jamie Dimon, Chairman and CEO",
  "paragraph_content": "Thank you for joining us today..."
}
```
Note: Files that produce zero paragraphs are NOT included in the output JSON file.

### Key Data Fields
- **Metadata**: Passed from Stage 2 file record (institution_id, ticker, date_last_modified)
- **File Path Data**: Extracted from path structure (fiscal_year, fiscal_quarter, institution_type)
- **Filename Data**: Parsed from filename (ticker, quarter, year, transcript_type, event_id, version_id)
- **Speaker Information**: Parsed from XML participants section using 'id' attribute
- **Speaker Formatting**: Name, title, and affiliation combined into readable string
- **Content Structure**: Section, speaker block, and paragraph hierarchy maintained
- **Q&A Flags**: Derived from XML speaker type attributes (q=question, a=answer, else null)
- **Text Cleaning**: Special characters escaped for JSON compatibility (quotes, newlines, tabs)

## 🔧 CONFIGURATION PARAMETERS

### Required Environment Variables
```bash
# API Credentials
API_USERNAME=your_api_username
API_PASSWORD=your_api_password

# Proxy Configuration
PROXY_USER=your_proxy_user
PROXY_PASSWORD=your_proxy_password
PROXY_URL=proxy.company.com:8080
PROXY_DOMAIN=MAPLE

# NAS Configuration
NAS_USERNAME=your_nas_username
NAS_PASSWORD=your_nas_password
NAS_SERVER_IP=192.168.1.100
NAS_SERVER_NAME=nas-server
NAS_SHARE_NAME=share_name
NAS_BASE_PATH=base/path
NAS_PORT=445

# Configuration Path
CONFIG_PATH=path/to/config.yaml
CLIENT_MACHINE_NAME=client-machine

# SSL Certificate
SSL_CERT_FILE=path/to/certificate.cer
```

### Stage Configuration (config.yaml)
```yaml
stage_03_extract_content:
  description: "XML transcript content extraction and paragraph breakdown"
  input_queue_path: "path/to/stage_02_processing_queue.json"
  output_logs_path: "logs/stage_03"
  output_data_path: "data/stage_03"
  dev_mode: false
  dev_max_files: 2
```

## 🐛 DEBUGGING TIPS

### Common Issues
1. **XML Parsing Errors**: Check for malformed XML or encoding issues
2. **NAS Connection Failures**: Verify credentials and network connectivity
3. **Memory Issues**: Monitor large XML file processing
4. **JSON Encoding Errors**: Check for special characters in text content

### Debugging Commands
```bash
# Check XML structure
xmllint --format sample.xml

# Test NAS connectivity
python -c "
from smb.SMBConnection import SMBConnection
conn = SMBConnection('user', 'pass', 'client', 'server')
print('Connected:', conn.connect('ip', 445))
"

# Check for duplicate files in processing queue
python -c "
import json
from collections import Counter
with open('stage_02_processing_queue.json', 'r') as f:
    queue = json.load(f)
    paths = [r['file_path'] for r in queue]
    duplicates = {p: c for p, c in Counter(paths).items() if c > 1}
    print(f'Duplicates found: {len(duplicates)}')
"
```

## 📈 PERFORMANCE OPTIMIZATION

### Best Practices
- Process files in batches to manage memory usage
- Use streaming XML parsing for very large files
- Implement connection pooling for NAS operations
- Cache frequently accessed configuration data
- Monitor processing time per file for bottleneck identification
- Skip duplicate files detected in processing queue
- Track version information for all files (both with and without content)

### Development Mode
- Set `dev_mode: true` in configuration for limited file processing
- Use `dev_max_files` to control batch size during development
- Enable detailed logging for debugging purposes

### File Statistics Tracking
The code tracks comprehensive statistics including:
- **Paragraph Distribution**: How many files have 0, 1-50, 51-100, etc. paragraphs
- **Version Tracking**: Multiple versions of the same transcript (ticker_year_quarter)
- **Content Status**: Which files produce content vs zero paragraphs
- **Duplicate Detection**: Identifies and skips duplicate file paths in queue
- **Type Distribution**: Files grouped by transcript type (E1, Corrected, etc.)
- **Ticker Coverage**: Number of unique tickers processed

## 🔑 KEY FUNCTIONS

### Core Processing Functions
- `main()` (line 973): Main orchestration function
- `load_processing_queue()` (line 590): Loads Stage 2 processing queue from NAS
- `process_transcript_file()` (line 821): Processes single transcript file
- `extract_metadata_from_file_path()` (line 524): Extracts metadata from file path structure
- `parse_transcript_xml()` (line 620): Parses XML with namespace handling
- `extract_transcript_paragraphs()` (line 754): Extracts paragraph-level content
- `format_speaker_string()` (line 718): Formats speaker information into readable string
- `determine_qa_flag()` (line 743): Determines Q&A flag from XML attributes
- `validate_and_preview_json()` (line 868): Validates JSON structure before saving
- `save_extracted_content()` (line 933): Saves extracted content to NAS

### Utility Functions
- `setup_logging()` (line 34): Sets up console logging
- `validate_environment_variables()` (line 131): Validates required env vars
- `get_nas_connection()` (line 173): Creates SMB connection to NAS
- `load_config_from_nas()` (line 214): Loads configuration from NAS
- `setup_ssl_certificate()` (line 319): Sets up SSL certificate
- `setup_proxy_configuration()` (line 351): Configures proxy settings
- `save_logs_to_nas()` (line 78): Saves execution and error logs