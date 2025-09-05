# Stage 2: Database Sync - Output Field Documentation

## Output Files

### 1. `stage_02_process_queue.json`
Files to be processed by downstream stages (earnings calls only).

**Fields (from lines 929-934):**
- **file_path**: Full NAS path to the transcript XML file - identifies the exact file location
- **date_last_modified**: ISO 8601 timestamp when file was last modified - used for change detection
- **institution_id**: ID from monitored_institutions config lookup - links to institution metadata (can be null)  
- **ticker**: Stock ticker extracted from filename (first part before underscore) - identifies the company (can be null)

### 2. `stage_02_removal_queue.json`
Files to be removed from the master database.

**Fields (from lines 976-978):**
- **file_path**: Full NAS path to the file - identifies what to remove
- **date_last_modified**: ISO 8601 timestamp from existing database record - preserves historical metadata
