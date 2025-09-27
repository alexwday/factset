# 🚀 FactSet Database Refresh Pipeline - AI Development Configuration

## 🎯 PROJECT CONTEXT
- Type: Multi-Stage Financial Data Processing Pipeline
- Stack: Python 3.11+, FactSet SDK, LLM Integration, SMB/NAS, XML Processing
- Architecture: 9-stage sequential processing pipeline with LLM-enhanced analysis
- Focus: Earnings transcript acquisition, processing, and AI-powered content enhancement

## 🚨 CRITICAL RULES
- ALWAYS validate AI outputs before deployment
- NEVER commit API keys, OAuth tokens, or credentials
- MUST pass all quality checks before commits
- Monitor token usage and costs continuously across all LLM stages
- Handle financial data with appropriate security measures
- Validate all NAS paths to prevent directory traversal attacks
- MUST handle authentication token refresh for long-running processes

## 🛠️ DEVELOPMENT WORKFLOW

### Git Integration
- Branch pattern: feature/stage-XX-*, bugfix/stage-XX-*, hotfix/stage-XX-*
- Commit format: "Stage X: [description]" preferred
- Push workflow: commit → push → CI checks → PR
- Merge strategy: squash and merge for features

### Code Quality Standards
```bash
# Pre-commit checks for entire pipeline
python -m black database_refresh/ --line-length 88
python -m isort database_refresh/ --profile black
python -m pylint database_refresh/ --min-score 8.0
python -m flake8 database_refresh/ --max-line-length 88

# Manual quality check
make lint  # If available
```

### Testing Strategy
- Local: Stage-specific unit tests with mocked dependencies
- Integration: Full pipeline tests with staging data
- Coverage: Minimum 80% for new code
- AI-specific: Validate LLM output quality and financial accuracy

## 🤖 AI/ML CONFIGURATION

### Model Setup
- Primary: claude-3-5-sonnet (latest) for classification and summarization
- Authentication: OAuth client credentials flow with per-transcript refresh
- Temperature: 0.1 for factual analysis, 0.3 for summarization tasks
- Context management: Optimized for financial earnings transcript processing

### Processing Pipeline Architecture
```
Stage 0: Historical Download → Stage 1: Daily Sync → Stage 2: Database Sync →
Stage 3: Content Extraction → Stage 4: Structure Validation → Stage 5: Q&A Pairing →
Stage 6: LLM Classification → Stage 7: LLM Summarization → Stage 8: Embeddings Generation →
Stage 9: Master Consolidation & Archive
```

## 📁 PROJECT STRUCTURE
```
database_refresh/
├── 00_download_historical/     # Historical transcript acquisition (3-year window)
├── 01_download_daily/          # Daily incremental sync
├── 02_database_sync/           # File synchronization and delta detection
├── 03_extract_content/         # XML parsing and paragraph extraction
├── 04_validate_structure/      # Transcript structure validation
├── 05_qa_pairing/             # Q&A boundary detection and conversation pairing
├── 06_llm_classification/     # LLM-based financial content classification
├── 07_llm_summarization/      # LLM-based content summarization
├── 08_embeddings_generation/  # Vector embeddings for semantic search/RAG
├── 09_master_consolidation/   # Master database update and archive management
├── config.yaml                # Shared configuration file (on NAS)
└── CLAUDE.md                  # This file
```

## 🔧 ESSENTIAL COMMANDS

### Environment Setup
```bash
# Global environment setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Load environment variables
source .env  # or export required variables

# Stage-specific execution pattern
cd database_refresh/0X_stage_name
python main_*.py
```

### Pipeline Operations
```bash
# Run complete pipeline (sequential execution)
for stage in 00_download_historical 01_download_daily 02_database_sync 03_extract_content 04_validate_structure 05_qa_pairing 06_llm_classification 07_llm_summarization 08_embeddings_generation 09_master_consolidation; do
    echo "Running Stage: $stage"
    cd database_refresh/$stage
    python main_*.py
    cd ../..
done

# Individual stage execution
python database_refresh/03_extract_content/main_content_extraction.py

# Development mode (most stages support)
# Set dev_mode: true and dev_max_files/dev_max_transcripts in config.yaml
```

### Debugging and Monitoring
```bash
# Check environment variables
python -c "
import os
required = ['API_USERNAME', 'API_PASSWORD', 'NAS_USERNAME', 'NAS_PASSWORD', 'LLM_CLIENT_ID', 'LLM_CLIENT_SECRET']
missing = [var for var in required if not os.getenv(var)]
print('Missing vars:', missing if missing else 'None')
"

# Test NAS connectivity
python -c "from smb.SMBConnection import SMBConnection; print('SMB module available')"

# Check stage outputs
ls -la /path/to/nas/outputs/stage_*
```

## 🔗 INTEGRATIONS

### Data Sources
- **FactSet APIs**: Events and Transcripts SDK for earnings data
- **NAS Storage**: SMB/CIFS protocol for centralized file operations
- **LLM Services**: OpenAI-compatible APIs with OAuth authentication
- **Corporate Infrastructure**: Proxy authentication, SSL certificates

### Processing Dependencies
- **SMB/NAS Connection**: All stages require NAS access for config and data
- **SSL Certificates**: Downloaded from NAS for secure API communications
- **OAuth Authentication**: LLM stages require client credentials flow
- **Proxy Configuration**: Corporate proxy with domain authentication

### Key Dependencies
```python
# Core dependencies across stages
import yaml                      # Configuration management
import fds.sdk.EventsandTranscripts  # FactSet API access
from smb.SMBConnection import SMBConnection  # NAS connectivity
from dotenv import load_dotenv     # Environment variables
import openai                    # LLM integration (stages 5-7)
import xml.etree.ElementTree     # XML parsing
import json                      # Data serialization
```

## 📋 STAGE-BY-STAGE OVERVIEW

### Stage 0: Historical Download (00_download_historical)
- **Purpose**: Download 3-year rolling window of historical transcripts
- **Key Features**: FactSet API integration, NAS storage organization, title filtering
- **Output**: XML transcripts organized by year/quarter/company structure
- **Critical Logic**: "Qx 20xx Earnings Call" title validation, 3-year window calculation

### Stage 1: Daily Sync (01_download_daily)
- **Purpose**: Daily incremental transcript synchronization
- **Key Features**: Date-based API queries, configurable sync ranges, delta detection
- **Output**: New/updated transcripts added to NAS structure
- **Critical Logic**: Date range calculation, version management, rate limiting

### Stage 2: Database Sync (02_database_sync)
- **Purpose**: Comprehensive file synchronization between NAS and master database
- **Key Features**: Delta detection, processing queue generation, self-contained operation
- **Output**: Processing and removal queues for downstream stages
- **Critical Logic**: File inventory comparison, change detection without selection logic

### Stage 3: Content Extraction (03_extract_content)
- **Purpose**: Parse XML transcripts and extract paragraph-level content
- **Key Features**: XML namespace handling, speaker identification, structured output
- **Output**: JSON records with paragraph-level content and metadata
- **Critical Logic**: Participant extraction, Q&A flag determination, content cleaning

### Stage 4: Structure Validation (04_validate_structure)
- **Purpose**: Validate transcript structure (Management Discussion + Q&A sections)
- **Key Features**: Section validation, invalid transcript filtering
- **Output**: Valid transcripts for downstream processing
- **Critical Logic**: Exactly 2 sections required with specific names

### Stage 5: Q&A Pairing (05_qa_pairing)
- **Purpose**: Q&A boundary detection and conversation pairing using LLM
- **Key Features**: Sliding window analysis, LLM boundary detection, conversation grouping
- **Output**: Q&A groups with proper conversation boundaries
- **Critical Logic**: Two-phase LLM processing, memory-efficient sliding windows

### Stage 6: LLM Classification (06_llm_classification)
- **Purpose**: Classify content into financial categories using LLM
- **Key Features**: CO-STAR prompts, speaker-block windowing, category validation
- **Output**: Classified content with category assignments and confidence scores
- **Critical Logic**: Management Discussion windowing, Q&A conversation-level classification

### Stage 7: LLM Summarization (07_llm_summarization)
- **Purpose**: Generate paragraph-level summaries for retrieval optimization
- **Key Features**: Q&A conversation summaries, speaker block summaries, context windows
- **Output**: Enhanced records with paragraph_summary fields
- **Critical Logic**: Sliding window context, reranking-optimized summaries

### Stage 8: Embeddings Generation (08_embeddings_generation)
- **Purpose**: Generate vector embeddings for semantic search and RAG applications
- **Key Features**: Intelligent chunking, tiktoken with fallback, 3072-dimensional vectors
- **Output**: Enhanced records with embeddings, token counts, and chunk information
- **Critical Logic**: Text chunking for >1000 tokens, OAuth refresh per transcript, incremental saving

### Stage 9: Master Consolidation (09_master_consolidation)
- **Purpose**: Consolidate processed records into master database and archive refresh folder
- **Key Features**: Incremental master updates, deletion support, archive creation, memory-efficient streaming
- **Output**: Updated master_embeddings.csv and timestamped refresh archive
- **Critical Logic**: Deduplication by file_path, streaming CSV processing, atomic updates

## ⚠️ KNOWN ISSUES
- **Authentication**: OAuth tokens may expire during long processing runs (mitigated by per-transcript refresh)
- **Memory Management**: Large transcripts require sliding window approaches and memory limits
- **Network Connectivity**: NAS timeouts and proxy authentication complexity
- **SSL Certificates**: Corporate certificates require proper setup for LLM API access
- **Rate Limiting**: API throttling during high-volume periods requires exponential backoff
- **Title Validation**: Strict format requirements may reject valid transcripts with variations

## 🚫 DO NOT MODIFY
- Environment variable validation logic across all stages
- NAS path security validation functions (prevent directory traversal)
- SSL certificate handling mechanisms
- OAuth authentication flows and token management
- File naming conventions: ticker_quarter_year_transcripttype_eventid_versionid.xml
- Configuration structure validation

## 💡 DEVELOPMENT NOTES

### Pipeline Data Flow
1. **Acquisition** (Stages 0-2): Download and synchronize transcript files
2. **Extraction** (Stages 3-4): Parse content and validate structure
3. **Enhancement** (Stages 5-8): AI-powered analysis, classification, summarization, and embedding

### Configuration Management
- **Shared Config**: config.yaml on NAS contains all stage-specific parameters
- **Environment Variables**: Credentials and connection details only
- **SSL Certificates**: Downloaded from NAS at runtime for secure connections
- **Development Mode**: Most stages support dev_mode for limited processing

### Error Handling Patterns
- **Enhanced Logging**: Categorized error types (authentication, validation, processing)
- **Incremental Saving**: Prevent data loss during processing interruptions
- **Graceful Degradation**: Continue processing with logged errors for non-critical failures
- **Retry Logic**: Exponential backoff for API calls and network operations

## 🔍 CODE CONVENTIONS
- Use `global logger, config` for shared state across stages
- Prefix NAS utility functions with `nas_`
- Implement retry logic with exponential backoff for external calls
- Sanitize URLs before logging to remove credentials
- Use descriptive variable names for financial concepts (ticker, quarter, fiscal_year)
- Follow existing stage numbering and naming conventions
- Include comprehensive logging for audit trails

## 📊 DATA HANDLING
- **Security**: Credentials never logged, URLs sanitized, path validation
- **Validation**: XML structure, JSON format, and financial data validation
- **Error Handling**: Comprehensive logging with contextual details
- **File Organization**: Consistent directory structure across all stages
- **Audit Trail**: Complete execution logging for compliance and debugging
- **Performance**: Stream-based processing for large datasets

## 🔒 SECURITY CONSIDERATIONS
- **Credentials**: Environment variables only, never in code or logs
- **Path Validation**: Prevent directory traversal attacks on all file operations
- **SSL/TLS**: Certificate-based authentication for all external API calls
- **Proxy Authentication**: Secure domain authentication with credential escaping
- **Data Privacy**: Financial data handling with appropriate security measures
- **Access Control**: NAS permissions and authentication validation

## 💰 COST MANAGEMENT
- **Token Tracking**: Detailed usage monitoring for all LLM stages (5-8)
- **Cost Calculation**: Per-transcript cost breakdown with accumulation
- **Development Limits**: dev_mode and file/transcript limits for testing
- **Rate Limiting**: Prevent excessive API usage with configurable delays
- **Optimization**: Sliding windows and context management to reduce token usage
- **Embeddings**: Batch processing and incremental saving to minimize API calls

## 🚀 PERFORMANCE OPTIMIZATIONS
- **Incremental Processing**: Stage-by-stage with intermediate outputs
- **Memory Management**: Sliding windows and configurable limits
- **Parallel Processing**: Multiple independent operations where possible
- **Connection Pooling**: Reuse NAS and API connections
- **Caching**: Configuration and frequently accessed data

## 📈 MONITORING AND MAINTENANCE
- **Execution Logs**: Comprehensive logging to NAS with timestamps
- **Error Categorization**: Separate error types for targeted troubleshooting
- **Cost Tracking**: LLM usage monitoring with budget alerts
- **Quality Metrics**: Validation success rates and processing statistics
- **Pipeline Health**: Stage completion rates and error frequency monitoring

## 🎯 CURRENT DEVELOPMENT FOCUS
- **Stage 8 Complete**: Embeddings generation with tiktoken fallback mechanism
- **Pipeline Complete**: All 8 stages fully operational for production use
- **Performance**: Memory optimization and cost reduction across LLM stages
- **Error Handling**: Enhanced logging and recovery mechanisms
- **Production Ready**: Full pipeline from data acquisition to embeddings generation