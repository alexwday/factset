# FactSet Database Refresh Pipeline

A comprehensive multi-stage data processing pipeline for financial earnings transcript acquisition, analysis, and AI-powered content enhancement.

## Overview

This pipeline processes financial earnings call transcripts through 8 sequential stages, from initial data acquisition to AI-enhanced analysis with relationship scoring for improved retrieval-augmented generation (RAG) applications.

### Pipeline Architecture

```
📥 Data Acquisition          🔄 Processing & Validation     🤖 AI Enhancement
┌─────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Stage 0: Historical │────▶│ Stage 2: Database Sync  │────▶│ Stage 5: Q&A Pairing   │
│ Stage 1: Daily Sync │     │ Stage 3: Content Extract│     │ Stage 6: Classification │
└─────────────────────┘     │ Stage 4: Structure Valid│     │ Stage 7: Summarization │
                            └─────────────────────────┘     │ Stage 8: Relationships │
                                                            └─────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- FactSet API credentials
- NAS/SMB access credentials
- LLM API credentials (OpenAI-compatible)
- Corporate SSL certificates

### Environment Setup

```bash
# Clone and setup environment
git clone <repository-url>
cd factset/database_refresh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Required Environment Variables

```bash
# FactSet API
API_USERNAME=your_factset_username
API_PASSWORD=your_factset_password

# Corporate Proxy
PROXY_USER=your_proxy_user
PROXY_PASSWORD=your_proxy_password
PROXY_URL=proxy.company.com:8080
PROXY_DOMAIN=DOMAIN

# NAS Storage
NAS_USERNAME=nas_user
NAS_PASSWORD=nas_password
NAS_SERVER_IP=192.168.1.100
NAS_SERVER_NAME=NAS_SERVER
NAS_SHARE_NAME=shared_folder
NAS_BASE_PATH=/base/path
NAS_PORT=445

# LLM API (for stages 5-7)
LLM_CLIENT_ID=your_llm_client_id
LLM_CLIENT_SECRET=your_llm_client_secret

# Configuration
CONFIG_PATH=config/config.yaml
CLIENT_MACHINE_NAME=YOUR_MACHINE
```

### Running the Pipeline

```bash
# Run individual stages
cd 01_download_daily
python main_daily_sync.py

# Or run multiple stages sequentially
./run_pipeline.sh  # If available
```

## Stage Details

### Stage 0: Historical Download
**Purpose**: Download 3-year rolling window of historical earnings transcripts  
**Input**: FactSet API, monitored institutions list  
**Output**: XML transcripts organized by year/quarter/company  
**Key Features**:
- 3-year rolling window calculation
- Title validation ("Qx 20xx Earnings Call")
- NAS directory structure creation
- SSL certificate handling

```bash
cd 00_download_historical
python main_historical_sync.py
```

### Stage 1: Daily Sync
**Purpose**: Daily incremental transcript synchronization  
**Input**: FactSet API date-based queries  
**Output**: New/updated transcripts  
**Key Features**:
- Configurable sync date ranges
- Version management
- Delta detection
- Rate limiting with exponential backoff

```bash
cd 01_download_daily
python main_daily_sync.py
```

### Stage 2: Database Sync
**Purpose**: File synchronization and delta detection  
**Input**: NAS file system inventory  
**Output**: Processing and removal queues  
**Key Features**:
- Comprehensive file scanning
- Change detection without selection
- Processing queue generation
- Self-contained operations

```bash
cd 02_database_sync
python main_sync_updates.py
```

### Stage 3: Content Extraction
**Purpose**: Parse XML transcripts into structured content  
**Input**: Stage 2 processing queue  
**Output**: JSON records with paragraph-level data  
**Key Features**:
- XML namespace handling
- Speaker identification
- Q&A flag determination
- Content structure validation

```bash
cd 03_extract_content
python main_content_extraction.py
```

### Stage 4: Structure Validation
**Purpose**: Validate transcript section structure  
**Input**: Stage 3 extracted content  
**Output**: Valid transcripts for downstream processing  
**Key Features**:
- Section count validation (exactly 2)
- Expected section name matching
- Invalid transcript filtering
- Processing queue refinement

```bash
cd 04_validate_structure
python main_structure_validation.py
```

### Stage 5: Q&A Pairing
**Purpose**: LLM-based Q&A boundary detection and conversation pairing  
**Input**: Stage 4 validated content  
**Output**: Q&A groups with conversation boundaries  
**Key Features**:
- Sliding window analysis
- LLM boundary detection
- Two-phase validation
- Memory-efficient processing

```bash
cd 05_qa_pairing
python main_qa_pairing.py
```

### Stage 6: LLM Classification
**Purpose**: Financial content classification using LLM  
**Input**: Stage 5 Q&A paired content  
**Output**: Content with financial category assignments  
**Key Features**:
- CO-STAR prompt methodology
- Speaker-block windowing
- Category validation
- Confidence scoring

```bash
cd 06_llm_classification
python main_llm_classification.py
```

### Stage 7: LLM Summarization
**Purpose**: Generate paragraph-level summaries for retrieval optimization  
**Input**: Stage 6 classified content  
**Output**: Content with paragraph summaries  
**Key Features**:
- Q&A conversation summaries
- Management Discussion speaker block summaries
- Sliding window context
- Reranking-optimized outputs

```bash
cd 07_llm_summarization
python main_llm_summarization.py
```

### Stage 8: Speaker Block Relationships
**Purpose**: Score speaker block relationships for context expansion  
**Input**: Stage 7 summarized content  
**Output**: Content with relationship context flags  
**Key Features**:
- Boolean context flags
- Enhanced prompting
- Inclusive relationship scoring
- RAG retrieval optimization

```bash
cd 08_speaker_block_relationships
python main_speaker_block_relationships.py
```

## Configuration

The pipeline uses a shared `config.yaml` file stored on NAS, containing stage-specific parameters:

```yaml
# Example configuration structure
stage_01_download_daily:
  sync_date_range: 7  # Days to sync
  
stage_05_qa_pairing:
  window_size: 10
  max_held_blocks: 50
  
stage_06_llm_classification:
  llm_config:
    model: "gpt-4-turbo"
    temperature: 0.1
  financial_categories:
    - name: "Revenue"
      description: "Revenue recognition and sales figures"
      
# ... additional stage configurations
```

## Development Mode

Most stages support development mode for testing with limited data:

```yaml
stage_XX_name:
  dev_mode: true
  dev_max_files: 2        # For file-based stages
  dev_max_transcripts: 2  # For transcript-based stages
```

## Monitoring and Debugging

### Logs
Each stage generates comprehensive logs saved to NAS:
- **Execution logs**: Detailed operational information
- **Error logs**: Categorized error tracking
- **Cost tracking**: LLM token usage and costs (stages 5-7)

### Common Debug Commands

```bash
# Test NAS connectivity
python -c "from smb.SMBConnection import SMBConnection; print('SMB available')"

# Check environment variables
python -c "
import os
required = ['API_USERNAME', 'NAS_USERNAME', 'LLM_CLIENT_ID']
missing = [var for var in required if not os.getenv(var)]
print('Missing:', missing if missing else 'None')
"

# Validate stage configuration
python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
print('Stages configured:', len([k for k in config.keys() if k.startswith('stage_')]))
"
```

### Performance Monitoring

- **Memory usage**: Monitor with sliding windows and configurable limits
- **API costs**: Track LLM token usage across stages 5-7
- **Processing time**: Log stage execution duration
- **Error rates**: Monitor validation and processing success rates

## Data Flow

### Input Data
- **FactSet API**: Earnings transcript metadata and content
- **XML Transcripts**: Structured financial earnings call data
- **Configuration**: YAML-based stage parameters

### Output Data
- **Structured JSON**: Paragraph-level transcript content
- **Enhanced Metadata**: Speaker information, categories, summaries
- **Relationship Data**: Context expansion flags for RAG applications

### Security
- **Credential Management**: Environment variables only
- **Path Validation**: Directory traversal prevention
- **SSL/TLS**: Certificate-based API authentication
- **Data Privacy**: Financial data security compliance

## Contributing

1. **Branch Naming**: `feature/stage-XX-description` or `bugfix/stage-XX-issue`
2. **Commit Format**: `Stage X: description` preferred
3. **Testing**: Include stage-specific tests for new features
4. **Documentation**: Update CLAUDE.md files for significant changes

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Check environment variables
   - Verify SSL certificate setup
   - Test OAuth token acquisition

2. **NAS Connection Issues**
   - Verify network connectivity
   - Check SMB credentials
   - Test path permissions

3. **LLM API Errors**
   - Monitor token limits
   - Check OAuth credential refresh
   - Verify SSL certificates

4. **Memory Issues**
   - Adjust sliding window sizes
   - Enable development mode
   - Monitor processing limits

### Getting Help

- Check stage-specific CLAUDE.md files for detailed configuration
- Review execution logs on NAS for error details
- Test individual components before full pipeline runs
- Use development mode for cost-effective debugging

## License

[Add your license information here]

## Contact

[Add contact information for the development team]