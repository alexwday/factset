# Stage 09: Master Database Consolidation & Archive

## Overview
This final stage consolidates all processed records into a master database and creates an archive of the refresh folder. It handles the full lifecycle of data management: removing outdated records, adding new processed data, maintaining a cumulative master database, and archiving each refresh cycle for audit trails.

**Primary Inputs**:
- `stage_02_removal_queue.json` - Files to remove from master
- `stage_08_embeddings.csv` - Newly processed records with embeddings

**Primary Output**:
- `master_embeddings.csv` - Cumulative master database
- `refresh_archive_YYYY-MM-DD_HH-MM-SS.zip` - Archive of current refresh

## Key Features
- **Incremental Updates**: Maintains cumulative master database across runs
- **Deletion Support**: Removes outdated/deleted transcripts from master
- **Streaming Processing**: Handles large CSV files efficiently with chunking
- **Deduplication**: Uses file_path as unique key for record management
- **Archive Creation**: Zips refresh folder with timestamp for audit trail
- **Safety Features**: Optional cleanup with configurable deletion
- **Memory Efficient**: Streams large CSVs without loading entire file

## Process Flow

### 1. Load Removal Queue
- Reads `stage_02_removal_queue.json` from Stage 2
- Extracts file_path values to identify records for deletion
- Creates a set for efficient O(1) lookup during filtering

### 2. Process Master Database
- **If Master Exists**:
  - Streams existing master CSV
  - Filters out records matching removal queue
  - Preserves all other records
- **If No Master**:
  - Creates new master from Stage 8 output
  - Uses fieldnames from new data

### 3. Append New Records
- Loads Stage 8 output (`stage_08_embeddings.csv`)
- Appends all new records to master
- Maintains field order consistency

### 4. Create Archive
- Zips all stage output files (stages 2-8) from refresh folder
- Names archive with timestamp: `refresh_archive_YYYY-MM-DD_HH-MM-SS.zip`
- Stores in configured archive directory
- Preserves complete audit trail

### 5. Optional Cleanup
- Can delete refresh files after successful archive (disabled by default)
- Safety feature prevents accidental data loss
- Enable via `delete_refresh_after_archive: true` in config

## Configuration

Stage 9 configuration in `config.yaml`:

```yaml
stage_09_master_consolidation:
  description: "Master database consolidation and archive"

  # Input paths
  stage_02_removal_queue: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_02_removal_queue.json"
  stage_08_output_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_08_embeddings.csv"

  # Output paths
  master_database_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Database/master_embeddings.csv"
  archive_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Archives"

  # Operational paths
  refresh_folder_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"

  # Processing configuration
  chunk_size: 10000  # Process CSVs in chunks for memory efficiency
  archive_enabled: true  # Create archive of refresh folder
  delete_refresh_after_archive: false  # Safety: don't auto-delete
```

## Usage

### Basic Execution
```bash
cd database_refresh/09_master_consolidation
python main_master_consolidation.py
```

### Full Pipeline Execution
```bash
# After running stages 0-8, complete with:
cd database_refresh/09_master_consolidation
python main_master_consolidation.py
```

The script will:
1. Connect to NAS using environment variables
2. Load removal queue from Stage 2
3. Update master database (remove old, add new)
4. Create timestamped archive of refresh folder
5. Save execution logs to NAS

### Environment Variables Required
- `NAS_USERNAME`: NAS username
- `NAS_PASSWORD`: NAS password
- `NAS_SERVER_IP`: NAS server IP address
- `NAS_SERVER_NAME`: NAS server name
- `NAS_SHARE_NAME`: NAS share name
- `NAS_PORT`: NAS port (default: 445)
- `CONFIG_PATH`: Path to config.yaml on NAS
- `CLIENT_MACHINE_NAME`: SMB client identification

## Master Database Management

### Record Deduplication
- Uses `file_path` (or `filename`) as unique identifier
- Removal queue matches against this key
- Ensures no duplicate transcripts in master

### Incremental Processing
- First run: Creates new master from Stage 8 output
- Subsequent runs:
  1. Remove outdated records (from removal queue)
  2. Append new processed records
  3. Maintain cumulative dataset

### CSV Structure
Master CSV maintains all fields from Stage 8:
- Core identifiers (filename, event_id, ticker, etc.)
- Content fields (paragraph_content, summaries)
- Classifications and categories
- Token counts and metadata
- Embeddings (as JSON strings in cells)

## Archive Management

### Archive Contents
Each archive contains:
- `stage_02_process_queue.json` - Files processed
- `stage_02_removal_queue.json` - Files removed
- `stage_03_extracted_content.json` - Extracted content
- `stage_04_valid_transcripts.json` - Validated transcripts
- `stage_05_qa_paired_content.json` - Q&A paired data
- `stage_06_classified_content.json` - Classified content
- `stage_07_summarized_content.json` - Summarized data
- `stage_08_embeddings.csv` - Final embeddings

### Archive Naming
Format: `refresh_archive_YYYY-MM-DD_HH-MM-SS.zip`
- Timestamp ensures uniqueness
- Chronological ordering in file system
- Easy identification of refresh cycles

### Storage Considerations
- Archives are compressed with ZIP_DEFLATED
- Typical compression ratio: 60-80% for JSON/CSV
- Preserves complete processing history
- Enables audit trails and debugging

## Performance Optimization

### Memory Management
- **Streaming Processing**: Processes CSV row-by-row
- **Chunk Size**: Configurable batch processing (default: 10,000 rows)
- **In-Memory Buffering**: Uses StringIO for efficient I/O
- **No Full File Loading**: Never loads entire master into memory

### Efficiency Features
- Set-based lookups for O(1) removal checking
- Single-pass filtering and appending
- Compressed archive creation
- Minimal network round trips

### Scalability
- Handles master databases >1GB efficiently
- Streams processing for unlimited size
- Archives compress to ~40% of original size
- Incremental updates minimize processing time

## Error Handling

### Failure Scenarios
1. **No Stage 8 Output**: Logs warning, exits gracefully
2. **Missing Removal Queue**: Proceeds with append-only
3. **Archive Creation Failure**: Logs error, preserves data
4. **Master Save Failure**: Raises exception, maintains old master

### Recovery Features
- Atomic operations (all-or-nothing updates)
- Old master preserved until new one saved
- Detailed error logging for troubleshooting
- Archive creation independent of master update

## Monitoring

### Execution Statistics
The script reports:
- Original record count in master
- Records removed (from removal queue)
- New records added (from Stage 8)
- Final record count
- Master database file size
- Archive creation status
- Total execution time

### Log Files
Saves detailed logs to NAS:
- `stage_09_master_consolidation_YYYY-MM-DD_HH-MM-SS.json`
- Error logs in `Errors/` subdirectory if failures occur

## Safety Features

### Data Protection
- **No Auto-Delete**: Refresh folder preserved by default
- **Archive First**: Always archives before any cleanup
- **Atomic Updates**: Master updated in single operation
- **Backup Via Archives**: Complete history preserved

### Configuration Safeguards
```yaml
delete_refresh_after_archive: false  # Must explicitly enable
archive_enabled: true  # Archiving on by default
```

## Re-run Capability

### How It Enables Re-runs
1. **Stage 2** reads master database to detect changes
2. **Stage 2** generates processing queues based on deltas
3. **Stages 3-8** process only new/modified files
4. **Stage 9** updates master with results
5. **Next run** starts with updated master

### Benefits
- Avoids reprocessing unchanged transcripts
- Reduces API costs for LLM operations
- Maintains complete historical record
- Enables incremental daily updates

## Integration with Pipeline

### Pipeline Flow Completion
```
Stage 0-1: Data Acquisition
    ↓
Stage 2: Delta Detection (reads master)
    ↓
Stage 3-8: Processing Pipeline
    ↓
Stage 9: Master Update (writes master)
    ↓
[Ready for next run]
```

### Master Database Lifecycle
1. **Initial State**: No master exists
2. **First Run**: Master created from Stage 8
3. **Daily Runs**: Master incrementally updated
4. **Deletions**: Outdated records removed
5. **Archives**: Historical snapshots preserved

## Next Steps

After Stage 9 completion:
1. Master database ready for:
   - Semantic search applications
   - RAG system integration
   - Analytics and reporting
   - ML model training

2. Archives available for:
   - Audit trails
   - Historical analysis
   - Debugging and recovery
   - Compliance requirements

3. Pipeline ready for:
   - Next scheduled run
   - Incremental updates
   - New transcript processing
   - Continuous operation

## Dependencies
```bash
pip install pyyaml
pip install pysmb
pip install python-dotenv
```

## Troubleshooting

### Common Issues

1. **Large Master Files**
   - Solution: Increase chunk_size in config
   - Consider: Implement database backend for >5GB

2. **Archive Creation Slow**
   - Check: Network bandwidth to NAS
   - Consider: Local temp file creation

3. **Memory Issues**
   - Verify: Streaming mode active
   - Check: chunk_size not too large

4. **Duplicate Records**
   - Ensure: file_path used consistently as key
   - Check: Stage 2 removal queue generation