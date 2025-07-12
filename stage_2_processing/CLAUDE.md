# Stage 2 Transcript Consolidation - Development Documentation

## Purpose
Stage 2 identifies the optimal single transcript per company per fiscal quarter/year and manages the master database inventory. This creates a curated, consolidated view of transcripts for downstream processing while maintaining comprehensive change tracking.

## Architecture Overview

### Core Objectives
1. **File Selection**: Apply 3-tier priority system to select one optimal transcript per company+quarter+year
2. **Master Database**: Create/maintain centralized JSON inventory of selected transcripts
3. **Delta Detection**: Compare NAS files vs database to identify processing changes
4. **Processing Queues**: Generate files for Stage 3 processing and database cleanup

### Key Differences from Stage 0/1
- **Stage 0/1**: Download and organize transcripts by fiscal quarter/year
- **Stage 2**: Consolidate to single best transcript and manage master inventory
- **Focus**: File selection, database management, and change detection (no AI processing)

## File Selection Algorithm

### Three-Tier Priority System

#### **Tier 1: Transcript Type Priority**
Configurable priority order (default: `["Corrected", "Raw", "NearRealTime"]`):
1. **Corrected** (Priority 1): Manually edited for accuracy - highest quality
2. **Raw** (Priority 2): Original transcript as received from earnings call
3. **NearRealTime** (Priority 3): Real-time capture during live call

#### **Tier 2: Version ID Priority**
Within same transcript type:
- Highest `version_id` number wins (latest vendor update)
- Uses `parse_version_from_filename()` from Stage 1
- Groups by version-agnostic key to identify same transcript

#### **Tier 3: Date Modified Priority**
Final tiebreaker:
- Latest `date_last_modified` timestamp wins
- Handles edge cases where multiple unique files exist

### Selection Logic Flow
```python
for company+quarter+year:
    for transcript_type in priority_order:
        files = get_files_in_folder(transcript_type)
        if files:
            best_file = apply_version_and_date_selection(files)
            return best_file  # First priority type with files wins
    return None  # No transcripts found
```

## Master Database Schema

### JSON Structure
```json
{
  "schema_version": "2.0",
  "last_updated": "2024-07-12T10:30:00Z",
  "transcripts": [
    {
      "fiscal_year": "2024",
      "fiscal_quarter": "Q1",
      "institution_type": "Canadian",
      "ticker": "RY-CA",
      "company_name": "Royal Bank of Canada",
      "transcript_type": "Corrected",
      "event_id": "12345",
      "report_id": "67890",
      "version_id": "1",
      "file_path": "Data/2024/Q1/Canadian/RY-CA_Royal_Bank_of_Canada/Corrected/RY-CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml",
      "filename": "RY-CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml",
      "date_last_modified": "2024-01-25T15:30:00Z",
      "selection_priority": 1,
      "version_agnostic_key": "RY-CA_2024-01-25_Earnings_Corrected_12345_67890"
    }
  ]
}
```

### Database Location
- **Path**: `{NAS_BASE_PATH}/Outputs/master_database.json`
- **Backup Strategy**: Timestamped logs maintain audit trail
- **Schema Evolution**: Version field supports future schema changes

## Configuration Requirements

### Stage 2 Configuration Section
Add to existing `config.json`:
```json
{
  "stage_2": {
    "master_database_path": "Outputs/master_database.json",
    "refresh_output_path": "Outputs/Refresh/",
    "transcript_type_priority": ["Corrected", "Raw", "NearRealTime"],
    "include_xml_content": false
  },
  // ... rest of existing config unchanged
}
```

### Configuration Parameters
- `master_database_path`: Location of master database JSON file
- `refresh_output_path`: Directory for processing queue outputs
- `transcript_type_priority`: Ordered list defining Tier 1 selection priority
- `include_xml_content`: Future expansion (currently false for performance)

## Implementation Details

### NAS Scanning Function
```python
def scan_nas_for_transcripts(nas_conn):
    """Scan all Data/YYYY/QX/Type/Company/TranscriptType/ folders"""
    # Traverses enhanced folder structure from Stage 0/1
    # Applies 3-tier selection to each company+quarter+year
    # Returns dictionary keyed by comparison_key
```

### Delta Detection Function
```python
def detect_changes(nas_listing, database_listing):
    """Compare NAS inventory vs database inventory"""
    comparison_key = f"{ticker}_{fiscal_year}_{fiscal_quarter}"
    # Compares file_path and date_last_modified
    # Returns (files_to_process, files_to_remove)
```

### Database Management
```python
def load_master_database(nas_conn):
    """Load existing database or create new one"""
    # Handles JSON corruption gracefully
    # Creates new database if missing
    
def update_master_database(database, files_to_process, files_to_remove):
    """Atomic database update"""
    # Remove outdated records first
    # Add new records second
    # Maintains referential integrity
```

## Code Reuse from Stage 1

### Functions Copied Unchanged
All critical Stage 1 infrastructure copied exactly:

#### **NAS Operations** (Lines 155-411 equivalent)
```python
get_nas_connection()           # SMB connection management
nas_path_join(*parts)         # Safe path construction
nas_file_exists()             # File existence checking
nas_list_files()              # XML file listing
nas_list_directories()        # Directory scanning
nas_download_file()           # File download
nas_upload_file()             # File upload
get_file_modified_time()      # File timestamp retrieval
```

#### **Security & Validation** (Lines 413-487 equivalent)
```python
validate_file_path()          # Directory traversal prevention
validate_nas_path()           # NAS path validation
sanitize_url_for_logging()    # Credential protection
```

#### **Version Management** (Lines 685-848 equivalent)
```python
parse_version_from_filename()          # Extract version numbers
get_version_agnostic_key_from_filename() # Create duplicate detection keys
```

#### **Configuration Management** (Lines 179-311 equivalent)
```python
load_stage_config()           # Download config from NAS
validate_config_schema()      # Comprehensive validation
```

#### **Error Logging** (Lines 91-143 equivalent)
```python
class EnhancedErrorLogger:
    # Extended with Stage 2 specific error types:
    # - selection_errors: File priority resolution conflicts
    # - database_errors: Schema validation, corruption
    # - comparison_errors: Delta detection failures
```

### Stage 2 Specific Enhancements

#### **Extended Error Types**
- `selection_errors`: Priority resolution conflicts requiring manual review
- `database_errors`: Schema validation failures, corruption detection
- `comparison_errors`: Delta detection logic failures

#### **New Functions**
- `scan_nas_for_transcripts()`: Comprehensive NAS traversal with selection
- `select_best_file_from_type()`: 3-tier file selection algorithm
- `get_priority_score()`: Transcript type priority scoring
- `database_to_comparison_format()`: Database format conversion
- `detect_changes()`: Delta detection and change tracking
- `save_processing_queues()`: Output queue generation

## Processing Flow

### Step-by-Step Execution

#### **Step 1: NAS Discovery**
1. Scan `Data/YYYY/QX/Type/Company/` folder structure
2. For each company+quarter+year combination:
   - Check transcript type folders in priority order
   - Apply version management within selected type
   - Select single best file using 3-tier priority
3. Build comprehensive NAS inventory

#### **Step 2: Database Management**
1. Load existing `master_database.json` or create new
2. Validate schema and handle corruption gracefully
3. Convert to comparison format for delta detection

#### **Step 3: Delta Detection**
1. Compare NAS inventory vs database using comparison keys
2. Identify files to process (new/changed)
3. Identify files to remove (deleted/outdated)
4. Log all changes with detailed context

#### **Step 4: Database Update**
1. Remove outdated records from master database
2. Add new selected files to master database
3. Maintain referential integrity throughout

#### **Step 5: Output Generation**
1. Save updated master database to NAS
2. Generate `files_to_process.json` for Stage 3
3. Generate `files_to_remove.json` for cleanup
4. Upload all files to `Outputs/Refresh/`

## Output Structure

### Processing Queue Files
```
Outputs/Refresh/
├── files_to_process.json    # New/changed transcripts for AI processing
├── files_to_remove.json     # Database records to delete
└── master_database.json     # Updated master inventory (backup)
```

### File Format Examples

#### **files_to_process.json**
```json
{
  "timestamp": "2024-07-12T15:30:00Z",
  "total_files": 5,
  "files": [
    {
      "fiscal_year": "2024",
      "fiscal_quarter": "Q2",
      "ticker": "RY-CA",
      "file_path": "Data/2024/Q2/Canadian/RY-CA_Royal_Bank_of_Canada/Corrected/...",
      "filename": "RY-CA_2024-07-15_Earnings_Corrected_54321_98765_2.xml",
      "transcript_type": "Corrected",
      "selection_priority": 1
    }
  ]
}
```

#### **files_to_remove.json**
```json
{
  "timestamp": "2024-07-12T15:30:00Z",
  "total_files": 2,
  "files": [
    {
      "ticker": "TD-CA",
      "fiscal_year": "2024",
      "fiscal_quarter": "Q1",
      "reason": "file_deleted_from_nas"
    }
  ]
}
```

## Error Handling & Monitoring

### Error Categories

#### **Selection Errors**
- Multiple files after 3-tier selection (unusual but handled)
- Priority resolution conflicts requiring manual review
- Version parsing failures

#### **Database Errors**
- JSON corruption or schema validation failures
- Database load/save failures
- Record integrity violations

#### **Comparison Errors**
- Missing required fields during delta detection
- Timestamp comparison failures
- Key generation errors

### Error Recovery
- Graceful degradation: Continue processing other companies
- Detailed error logging with actionable guidance
- Operator notifications for critical failures requiring intervention

## Performance Characteristics

### Typical Execution
- **Empty Run**: 30-60 seconds (no changes)
- **Normal Updates**: 2-5 minutes (5-20 changed files)
- **Full Rebuild**: 10-15 minutes (new database creation)

### Scalability
- Linear scaling with number of companies and quarters
- Optimized NAS traversal using directory listing
- Memory efficient (streams processing, no bulk loading)

## Security & Standards Compliance

### All Stage 0/1 Security Standards Maintained
- ✅ Input validation for all file paths and parameters
- ✅ Directory traversal attack prevention
- ✅ URL sanitization in logs (credentials protected)
- ✅ Configuration schema validation
- ✅ Safe NAS path construction and validation
- ✅ Error handling without credential exposure

### Stage 2 Specific Security
- Database integrity validation before/after updates
- Atomic database operations (prevent corruption)
- Processing queue validation before output

## Integration with Pipeline

### Upstream Dependencies
- **Stage 0**: Initial bulk download and folder structure creation
- **Stage 1**: Daily incremental updates maintaining same structure

### Downstream Integration
- **Stage 3**: Will consume `files_to_process.json` for AI processing
- **Reporting**: Master database provides comprehensive inventory
- **Analytics**: Processing queues enable change tracking and metrics

## Common Scenarios

### New Database Creation
- First run: All transcripts added to `files_to_process.json`
- Master database created with complete inventory
- Normal delta detection for subsequent runs

### Version Updates
- Vendor version ID changes detected automatically
- Old version records removed from database
- New version added to processing queue

### Transcript Type Upgrades
- Raw → Corrected upgrades detected via priority system
- Database automatically updated to reference higher priority file
- Old type record marked for removal

### File Deletions
- Missing files detected during NAS scan
- Database records marked for removal
- Clean deletion from master inventory

## Maintenance & Operations

### Log Files
- Uploaded to `Outputs/Logs/stage_2_consolidation_log_YYYY-MM-DD_HH-MM-SS.log`
- Error logs saved separately by type for targeted investigation
- Audit trail maintains complete processing history

### Database Backup
- Master database automatically backed up during updates
- Timestamped error logs provide restoration points
- Schema versioning supports future upgrades

### Monitoring Points
1. **Selection conflicts**: Monitor `selection_errors` for unusual patterns
2. **Database health**: Watch for `database_errors` indicating corruption
3. **Processing volume**: Track `files_to_process` counts for capacity planning
4. **Completion rates**: Monitor end-to-end execution success

## Future Enhancements

### Potential Improvements
1. **Parallel NAS scanning**: Multi-threaded directory traversal
2. **Incremental scanning**: Track last scan timestamps to reduce work
3. **Database compression**: Optimize storage for large inventories
4. **XML content inclusion**: Add transcript content when needed

### Configuration Extensions
1. **Custom priority rules**: Company-specific transcript type preferences
2. **Quality filters**: Exclude transcripts below quality thresholds
3. **Date range limits**: Focus consolidation on specific time periods

## Critical Reminders

### Development Standards
1. **NEVER** modify security validation functions from Stage 1
2. **ALWAYS** maintain atomic database operations
3. **NEVER** skip file path validation
4. **ALWAYS** log selection decisions for audit trail
5. **NEVER** expose credentials in logs or error messages

### Operational Guidelines
1. Safe to re-run (idempotent operations)
2. Database corruption handling built-in
3. Processing queues always reflect current state
4. Error recovery enables partial completion

---

Stage 2 successfully bridges Stage 0/1 outputs with Stage 3 processing needs while maintaining all security, reliability, and audit standards established in the pipeline.