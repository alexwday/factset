# FactSet Database Refresh Pipeline

A comprehensive multi-stage data processing pipeline for financial earnings transcript acquisition, analysis, and AI-powered content enhancement.

## Overview

This pipeline processes financial earnings call transcripts starting at Stage 2 (database sync), then proceeds through AI-enhanced analysis and master database consolidation.

### Pipeline Architecture

```
📥 Ingestion (external)      🔄 Processing & Validation     🤖 AI Enhancement           📦 Consolidation
┌─────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐     ┌──────────────┐
│ NAS XML Dataset     │────▶│ Stage 2: Database Sync  │────▶│ Stage 5: Q&A Pairing   │────▶│ Stage 9:     │
│ (pre-populated)     │     │ Stage 3: Content Extract│     │ Stage 6: Classification │     │ Master DB    │
└─────────────────────┘     │ Stage 4: Structure Valid│     │ Stage 7: Summarization │     │ & Archive    │
                            └─────────────────────────┘     │ Stage 8: Embeddings    │     └──────────────┘
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

# LLM API (for stages 5-8)
LLM_CLIENT_ID=your_llm_client_id
LLM_CLIENT_SECRET=your_llm_client_secret

# Configuration
CONFIG_PATH=config/config.yaml
CLIENT_MACHINE_NAME=YOUR_MACHINE
```

### Running the Pipeline

```bash
# Run individual stages
cd 02_database_sync
python main_sync_updates.py

# Or run multiple stages sequentially
./run_pipeline.sh  # If available
```

### Timer-Safe Controls

When stages are scheduled independently (for example every 10 minutes), the ETL now uses a shared NAS control file:

- `.../Outputs/Refresh/pipeline_control_flags.json`

Behavior:

- Stage 2 acquires a run lock. If another run is active, it exits as `skipped_overlap`.
- Stages 3-8 check the control file and skip when the run is not ready, upstream failed, or Stage 2 had no files to process.
- Stage 9 acts as finalizer: it archives refresh outputs, force-cleans refresh stage files, and clears the run lock.

This prevents overlap, avoids downstream "missing input" cascades, and ensures cleanup after no-op/failure runs.

## Stage Details

### Deprecated Stages 0-1
Stage 0 (`00_download_historical`) and Stage 1 (`01_download_daily`) have been archived under `database_refresh/deprecated_stages/` and are no longer part of the active pipeline.

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

### Stage 8: Embeddings Generation
**Purpose**: Generate vector embeddings for semantic search and RAG  
**Input**: Stage 7 summarized content  
**Output**: Enhanced records with 3072-dimensional embeddings  
**Key Features**:
- Intelligent text chunking (>1000 tokens → ~500 token chunks)
- tiktoken with hybrid fallback for token counting
- Full 3072-dimensional vectors (text-embedding-3-large)
- Incremental saving with OAuth refresh per transcript

```bash
cd 08_embeddings_generation
python main_embeddings_generation.py
```

### Stage 9: Master Consolidation
**Purpose**: Consolidate processed records into master database and create archives
**Input**: Stage 8 embeddings and Stage 2 removal queue
**Output**: Updated master database and timestamped refresh archive
**Key Features**:
- Incremental master database updates
- Deletion support for outdated records
- Memory-efficient streaming CSV processing
- Deduplication by file_path as unique key
- Archive creation with timestamp for audit trails

```bash
cd 09_master_consolidation
python main_master_consolidation.py
```

## Configuration

The pipeline uses a shared `config.yaml` file stored on NAS, containing stage-specific parameters:

```yaml
# Example configuration structure
stage_02_database_sync:
  input_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Data"
  
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
- **Cost tracking**: LLM token usage and costs (stages 5-8)

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
- **API costs**: Track LLM token usage across stages 5-8
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
- **Vector Embeddings**: 3072-dimensional vectors for semantic search
- **Token Metrics**: Paragraph and block-level token counts for optimization

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
