# 🚀 FactSet Project - Stage 3: Content Extraction Configuration

## 🎯 PROJECT CONTEXT
- Type: XML Transcript Content Extraction & Paragraph-Level Breakdown
- Stack: Python 3.11+, XML parsing, NAS/SMB integration
- Architecture: Stage 3 of multi-stage financial data processing pipeline
- Focus: Parse XML transcripts and extract structured paragraph-level content

## 🚨 CRITICAL RULES
- ALWAYS validate XML structure before processing
- NEVER commit API keys or credentials to repository
- MUST handle XML parsing errors gracefully
- Monitor NAS connection stability continuously
- Handle financial transcript data with appropriate security measures
- Validate JSON output structure before saving

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
- Process XML transcripts from Stage 2's processing queue
- Extract paragraph-level content with speaker identification
- Parse participant metadata and Q&A flagging
- Generate structured JSON output for downstream stages

### Key Processing Steps
1. Load processing queue from Stage 2 output
2. Extract metadata from file paths (ticker, quarter, year, etc.)
3. Parse XML transcripts with namespace handling
4. Extract participants, sections, and speaker blocks
5. Generate paragraph-level records with speaker information
6. Validate and save JSON output to NAS

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
    print(f'Root tag: {root.tag}')
"

# Validate JSON structure
python -c "
import json
with open('output.json', 'r') as f:
    data = json.load(f)
    print(f'Records: {len(data)}')
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
- XML transcript parsing and content extraction
- Paragraph-level data structuring with speaker identification
- Q&A flag determination from XML attributes
- Robust error handling for malformed XML files

## ⚠️ KNOWN ISSUES
- XML files may have inconsistent namespace usage
- Some transcript files may have missing participant information
- Network connectivity issues can cause NAS timeouts
- Large XML files may require memory optimization

## 🚫 DO NOT MODIFY
- NAS connection credentials in environment variables
- SSL certificate files (temporary files are cleaned up automatically)
- Processing queue format from Stage 2
- XML namespace handling logic without thorough testing

## 💡 DEVELOPMENT NOTES
- Use ElementTree for XML parsing with namespace detection
- Clean text content to prevent JSON encoding issues
- Maintain speaker block relationships for downstream processing
- Log detailed execution information for debugging
- Handle missing or malformed participant data gracefully

## 🔍 CODE CONVENTIONS
- Follow existing logging patterns (log_console, log_execution, log_error)
- Use NAS utility functions for file operations
- Validate file paths and data before processing
- Clean up temporary files (SSL certificates) after execution
- Maintain consistent error handling across functions

## 📊 DATA HANDLING

### Input Data Format
- Stage 2 processing queue: JSON array of file records
- XML transcripts: FactSet format with meta, participants, and body sections
- Configuration: YAML with stage-specific parameters

### Output Data Format
```json
{
  "file_path": "path/to/transcript.xml",
  "date_last_modified": "2024-01-01T00:00:00",
  "filename": "TICKER_Q1_2024_Corrected_12345_1.xml",
  "fiscal_year": "2024",
  "fiscal_quarter": "Q1",
  "ticker": "TICKER",
  "company_name": "Company Name",
  "transcript_type": "Corrected",
  "event_id": "12345",
  "version_id": "1",
  "title": "Company Q1 2024 Earnings Call",
  "section_id": 1,
  "section_name": "Management Discussion",
  "paragraph_id": 1,
  "speaker_block_id": 1,
  "question_answer_flag": "answer",
  "speaker": "John Doe, CEO",
  "paragraph_content": "Thank you for joining us today..."
}
```

### Key Data Fields
- **Metadata**: Extracted from file path structure
- **Speaker Information**: From XML participants section
- **Content Structure**: Section, speaker block, and paragraph hierarchy
- **Q&A Flags**: Derived from XML speaker type attributes

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

# Validate environment
python database_refresh/03_extract_content/main_content_extraction.py --dry-run
```

## 📈 PERFORMANCE OPTIMIZATION

### Best Practices
- Process files in batches to manage memory usage
- Use streaming XML parsing for very large files
- Implement connection pooling for NAS operations
- Cache frequently accessed configuration data
- Monitor processing time per file for bottleneck identification

### Development Mode
- Set `dev_mode: true` in configuration for limited file processing
- Use `dev_max_files` to control batch size during development
- Enable detailed logging for debugging purposes