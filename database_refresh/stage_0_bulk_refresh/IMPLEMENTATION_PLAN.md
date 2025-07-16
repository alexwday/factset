# Stage 0 Implementation Plan - Filename & Folder Structure Changes

## Overview
This document outlines the implementation plan for updating Stage 0 with new filename format, simplified folder structure, NAS scanning workflow, and earnings call title validation.

## Requirements Summary

### 1. New Filename Format
**Current**: `{ticker}_{date}_{event_type}_{transcript_type}_{event_id}_{report_id}_{version_id}.xml`
**New**: `{ticker}_{transcript_type}_{quarter}_{year}_{event_id}_{version_id}.xml`

**Example**: `RY-CA_Corrected_Q1_2024_12345_1.xml`

### 2. Simplified Folder Structure
**Current**: `Data/YYYY/QX/Type/Company/TranscriptType/filename.xml`
**New**: `Data/YYYY/QX/Type/Company/filename.xml`

**Rationale**: Transcript type now in filename, eliminates need for separate subfolders

### 3. NAS Scanning & Version Management Workflow
**New Process**:
1. **Initial NAS Scan**: Parse entire `Data/` folder structure once to build current inventory
2. **Company Processing**: For each monitored company:
   - Compare API results with NAS inventory by `ticker + event_id`
   - If higher version found → download new, archive old version
   - If new event_id found → validate earnings call title, then store
3. **Archive Management**: Move old versions to `{NAS_BASE_PATH}/Outputs/Archive/`

### 4. Earnings Call Title Validation
**Requirement**: Only process files with exact title format `"Qx 20xx Earnings Call"`
- **Accept**: "Q1 2024 Earnings Call", "Q3 2023 Earnings Call"
- **Reject**: "Q1 2024 Earnings Call - Fixed Income", "First Quarter 2024 Earnings Call"
- **Action**: Reject and log (info level) files that don't match exact pattern

### 5. Business Rules
- **Multiple Events**: If multiple valid earnings calls per quarter, take latest from API and archive NAS version
- **Final State**: Always 1 file per company folder (latest version of latest event)
- **API Priority**: API is source of truth - always trust API over existing NAS files

## Implementation Phases

### Phase 1: Update Filename Functions

#### Functions to Modify:
1. **`create_filename(transcript_data, target_ticker)`**
   - Input: Add quarter, year parameters (from XML parsing)
   - Output: `{ticker}_{transcript_type}_{quarter}_{year}_{event_id}_{version_id}.xml`

2. **`create_version_agnostic_key(transcript_data, target_ticker)`**
   - Output: `{ticker}_{transcript_type}_{quarter}_{year}_{event_id}` (no version_id)

3. **`parse_version_from_filename(filename)`**
   - Update for new format: version_id at position -1 after splitting by "_"

4. **`get_version_agnostic_key_from_filename(filename)`**
   - Update for new format: remove last part (version_id)

#### New Functions to Add:
5. **`parse_filename_metadata(filename) -> Dict[str, str]`**
   ```python
   # Parse: RY-CA_Corrected_Q1_2024_12345_1.xml
   # Return: {
   #   'ticker': 'RY-CA',
   #   'transcript_type': 'Corrected', 
   #   'quarter': 'Q1',
   #   'year': '2024',
   #   'event_id': '12345',
   #   'version_id': '1'
   # }
   ```

### Phase 2: Update Folder Structure

#### Functions to Modify:
1. **`create_enhanced_directory_structure()`**
   - Remove transcript_type from path construction
   - New path: `Data/YYYY/QX/Type/Company/` (no TranscriptType subfolder)

2. **`get_existing_files_with_version_management()`**
   - Update to scan company folder directly (no transcript type subfolders)
   - Update path construction logic

#### New Functions to Add:
3. **`create_archive_directory_structure()`**
   - Create Archive folder structure: `Archive/YYYY-MM-DD/`
   - For timestamped archival of old versions

4. **`archive_old_file(nas_conn, old_file_path, archive_reason)`**
   - Move file from Data/ to Archive/ with timestamp
   - Log archival action with reason

### Phase 3: Add NAS Scanning

#### New Functions to Add:
1. **`scan_existing_nas_inventory(nas_conn) -> Dict[str, Dict[str, Any]]`**
   ```python
   # Scan all Data/YYYY/QX/Type/Company/ folders
   # Parse filenames using parse_filename_metadata()
   # Return dictionary keyed by f"{ticker}_{event_id}"
   # Value contains: file_path, metadata, modified_time
   ```

2. **`build_nas_comparison_key(ticker, event_id) -> str`**
   ```python
   # Standard key format for comparing NAS vs API
   return f"{ticker}_{event_id}"
   ```

3. **`compare_versions(nas_version_id, api_version_id) -> str`**
   ```python
   # Compare version IDs, return "nas_newer", "api_newer", or "same"
   # Handle string/int conversion safely
   ```

### Phase 4: Add Title Validation

#### New Functions to Add:
1. **`validate_earnings_call_title(xml_content) -> Tuple[bool, str, str, str]`**
   ```python
   # Parse XML title and validate exact format
   # Return: (is_valid, title, quarter, year)
   # Only accept: "Q1 2024 Earnings Call" format
   # Reject: anything with suffixes or different formats
   ```

2. **`extract_quarter_year_from_title(title) -> Tuple[str, str]`**
   ```python
   # Extract Q1, Q2, Q3, Q4 and 20XX from valid title
   # Return: ("Q1", "2024")
   ```

#### Enhanced Error Logging:
3. **Add to `EnhancedErrorLogger`**:
   ```python
   def log_title_rejection(self, ticker: str, filename: str, title: str, reason: str):
       """Log rejected files due to title validation (info level)."""
   ```

### Phase 5: Update Main Processing Logic

#### Functions to Modify:
1. **`process_bank()` - Major Refactor**
   - **Step 1**: Get NAS inventory for this ticker
   - **Step 2**: Process each API transcript:
     - Check if event_id exists in NAS inventory
     - If exists: compare versions, archive old if API newer
     - If new: validate title, download if valid earnings call
   - **Step 3**: Download and save to simplified folder structure

2. **`download_transcript_with_enhanced_structure()` - Update**
   - Remove quarter/year parsing from XML (now from title validation)
   - Use simplified folder structure (no transcript type subfolder)
   - Use new filename format

3. **`main()` - Add NAS Scanning**
   - Add initial NAS scan before company processing
   - Pass NAS inventory to each company processing call

#### New Processing Functions:
4. **`process_transcript_event(nas_conn, transcript, ticker, nas_inventory, configuration, proxy_user, proxy_password) -> bool`**
   ```python
   # Process single transcript event
   # Handle version comparison, title validation, download
   # Return success/failure status
   ```

## Implementation Order

### Phase 1: Filename Functions ✅ (Start Here)
- Low risk changes to filename generation and parsing
- Can be tested independently
- Foundation for all other changes

### Phase 2: Folder Structure 
- Modify directory creation and file location logic
- Update existing file scanning logic
- Test with simplified paths

### Phase 3: NAS Scanning
- Add comprehensive NAS inventory scanning
- Build comparison and version management logic
- Test with existing data

### Phase 4: Title Validation
- Add XML title parsing and validation
- Integrate rejection logging
- Test with various title formats

### Phase 5: Main Logic Integration
- Integrate all changes into main processing flow
- Update company processing workflow
- End-to-end testing

## Testing Strategy

### Unit Testing
- Test all filename parsing functions with various formats
- Test title validation with valid/invalid examples
- Test version comparison logic

### Integration Testing
- Test NAS scanning with existing data structure
- Test archival process with real files
- Test end-to-end workflow with single company

### Validation Testing
- Verify only valid earnings calls are processed
- Verify version management works correctly
- Verify simplified folder structure is maintained

## Risk Mitigation

### Backup Strategy
- Archive functionality preserves old versions
- Log all changes for audit trail
- Can revert by restoring from Archive/

### Gradual Rollout
- Test with single company first
- Validate output before processing all companies
- Monitor error logs for unexpected issues

### Error Handling
- Enhanced error logging for all rejection reasons
- Graceful failure for parsing errors
- Continue processing other companies if one fails

## Expected Outcomes

### File Organization
- **Before**: Multiple transcript type subfolders per company
- **After**: Single file per company (latest version of latest earnings event)

### Performance Improvements
- Single NAS scan vs multiple scans per company
- Reduced folder traversal complexity
- Faster version comparison through filename parsing

### Data Quality
- Only valid earnings calls processed
- Always latest version maintained
- Clear audit trail of all changes

## Next Steps

1. **Approve Phase 1**: Review and approve filename function changes
2. **Implement Phase 1**: Update all filename-related functions
3. **Test Phase 1**: Validate filename generation and parsing
4. **Continue Phases**: Move through phases 2-5 systematically