# Stage 3: XML Content Extraction & Paragraph-Level Breakdown - Context for Claude

## Overview
Stage 3 processes XML transcripts from Stage 2's processing queue and extracts paragraph-level content with speaker attribution. Each paragraph becomes a database record with complete metadata preservation and structured content breakdown.

## Current Status ✅ PRODUCTION READY
- **Script**: `3_transcript_content_extraction.py` - Fully functional with all security standards
- **Input**: `files_to_process.json` from Stage 2 output
- **Output**: `extracted_transcript_sections.json` with paragraph-level database records
- **Features**: Development mode, complete field preservation, speaker attribution, Q&A detection
- **Issues Resolved**: Filename parsing with underscore-containing tickers (RY_CA vs RY-CA)

## Key Business Logic

### Input Processing
1. **Reads Stage 2 Output**: Loads `files_to_process.json` from `Outputs/Refresh/`
2. **Development Mode**: Configurable via `dev_mode: true` to process only 1-2 files during testing
3. **Field Preservation**: All original Stage 2 fields are maintained in each paragraph record

### XML Content Extraction
1. **Downloads XML Files**: Retrieves transcript files from NAS using paths from Stage 2
2. **Parses XML Structure**: Extracts metadata, participants, sections, and speaker content
3. **Speaker Attribution**: Creates formatted strings like "Dave McKay, President & CEO, Royal Bank of Canada"
4. **Q&A Detection**: Identifies question vs answer paragraphs from XML speaker types

### Output Structure
Each paragraph becomes a database record with:
- **All Stage 2 fields**: ticker, fiscal_year, filename, etc. (complete preservation)
- **section_id**: Sequential section identifier (1, 2, 3...) for section ordering
- **section_name**: "Presentation", "Q&A Session", etc.
- **paragraph_id**: Sequential numbering across entire transcript for reconstruction
- **speaker_block_id**: Sequential speaker block identifier (1, 2, 3...) for grouping speaker statements
- **question_answer_flag**: null, "question", or "answer" from XML
- **speaker**: Formatted speaker string with name, title, affiliation
- **paragraph_content**: Clean paragraph text

## Technical Implementation

### XML Processing Logic
- **Reuses HTML Viewer Approach**: Adapts sophisticated XML parsing from `transcript_html_viewer.py`
- **Namespace Handling**: Robust detection and handling of XML namespaces
- **Speaker Mapping**: Matches speaker IDs to participant details
- **Sequential Ordering**: Maintains paragraph order for document reconstruction

### Development Mode
- **Configuration**: `"dev_mode": true` in config.json
- **File Limiting**: `"dev_max_files": 2` for iterative testing
- **Production Switch**: Set `"dev_mode": false` for full processing

### Error Handling
- **Enhanced Error Logger**: Separate JSON files for parsing, download, filesystem, validation, and processing errors
- **Graceful Failures**: Continues processing other files if individual files fail
- **Comprehensive Logging**: Detailed audit trail with timestamps

## Critical Issues Resolved

### 1. File Path Duplication (Fixed 2024-07-12)
**Problem**: Stage 3 was prepending `NAS_BASE_PATH` to `file_path`, but Stage 2 already provides complete paths
**Solution**: Use `file_path` directly without additional path construction
**Impact**: Resolved XML download failures

### 2. Filename Parsing with Underscore Tickers (Fixed 2024-07-12)
**Problem**: Tickers contain underscores (RY_CA), causing incorrect array positions during filename parsing
**Root Cause**: 
- Expected: `RY-CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml` (7 parts)
- Actual: `RY_CA_2024-01-25_Earnings_Corrected_12345_67890_1.xml` (8 parts)
**Solution**: Corrected parsing positions:
- `event_id = parts[5]` (not parts[4])
- `report_id = parts[6]` (not parts[5])
**Impact**: Fixed incorrect event_id values showing as transcript_type

### 3. Enhanced Data Structure for Stage 4 Integration (Fixed 2024-07-13)
**Enhancement**: Added section and speaker block tracking for LLM classification
**Changes**:
- Added `section_id`: Sequential section identifier (1, 2, 3...)
- Renamed `paragraph_order` to `paragraph_id` for consistency
- Enhanced `speaker_block_id`: Sequential speaker block identifier (1, 2, 3...)
**Impact**: Enables section reconstruction and speaker block grouping for downstream LLM classification

## Configuration

### Stage 3 Config Section
```json
"stage_3": {
  "dev_mode": true,
  "dev_max_files": 2,
  "input_source": "Outputs/Refresh/files_to_process.json",
  "output_file": "extracted_transcript_sections.json",
  "output_path": "Outputs/Refresh",
  "description": "XML content extraction and paragraph-level breakdown"
}
```

### Key Settings
- **dev_mode**: Enable/disable development mode
- **dev_max_files**: Limit files processed during development
- **input_source**: Path to Stage 2 processing queue
- **output_file**: Name for paragraph-level content records
- **output_path**: Directory for output file storage

## Security Standards Compliance

### Input Validation ✅
- **File Path Validation**: All paths validated against directory traversal attacks
- **NAS Path Security**: Safe path construction and validation
- **XML Parsing**: Robust error handling for malformed XML

### Error Handling ✅
- **Specific Exceptions**: No bare `except:` clauses
- **Error Context**: Detailed error information preserved
- **Graceful Degradation**: Continues processing on individual file failures

### Resource Management ✅
- **Connection Management**: Proper NAS connection handling
- **File Cleanup**: Temporary files properly cleaned up
- **Memory Management**: Efficient processing of large XML files

## Development Standards

### Code Quality Requirements
1. **Field Preservation**: All Stage 2 fields MUST be preserved in output records
2. **Sequential Ordering**: Paragraph order MUST be maintained for document reconstruction
3. **Speaker Attribution**: Speaker information MUST be properly formatted and mapped
4. **Error Logging**: All failures MUST be logged with actionable error messages
5. **Development Mode**: MUST support limited file processing for testing

### Testing Protocol
1. **Development Mode**: Test with 1-2 files using `dev_mode: true`
2. **Field Validation**: Verify all Stage 2 fields are preserved
3. **Content Accuracy**: Validate speaker attribution and paragraph extraction
4. **Error Handling**: Test with malformed XML and network failures
5. **Production Mode**: Full processing with `dev_mode: false`

## Integration with Pipeline

### Stage Dependencies
- **Input**: Stage 2 must complete successfully and create `files_to_process.json`
- **Output**: Creates `extracted_transcript_sections.json` for Stage 4+ consumption
- **NAS Integration**: Downloads XML files using paths from Stage 2

### Data Flow
1. **Stage 2 → Stage 3**: Processing queue with optimal transcript selections
2. **Stage 3 → Stage 4+**: Paragraph-level database records ready for analysis
3. **Error Tracking**: Enhanced error logs for operational monitoring

## Common Issues & Solutions

### Issue: "Failed to download XML file"
**Cause**: File path duplication or invalid NAS paths
**Solution**: Verify Stage 2 provides correct file paths; check NAS connectivity

### Issue: "Event ID showing as transcript type"
**Cause**: Filename parsing with underscore-containing tickers
**Solution**: Use corrected parsing positions (parts[5] for event_id in 8-part filenames)

### Issue: "No paragraphs extracted"
**Cause**: XML structure changes or namespace issues
**Solution**: Review XML parsing logic; check for namespace variations

### Issue: "Speaker attribution missing"
**Cause**: Participant mapping failures or missing speaker IDs
**Solution**: Validate XML participant structure; check speaker ID mapping

## Performance Considerations

### Optimization Strategies
- **Development Mode**: Use for iterative testing to avoid processing large volumes
- **Error Recovery**: Continue processing other files if individual files fail
- **Memory Management**: Process files individually to avoid memory buildup
- **NAS Efficiency**: Batch file operations where possible

### Monitoring Points
- **Processing Rate**: Track files processed per minute
- **Error Rates**: Monitor parsing and download failure rates
- **Output Quality**: Validate paragraph extraction completeness
- **Resource Usage**: Monitor memory and network utilization

## Future Enhancements

### Potential Improvements
1. **Parallel Processing**: Process multiple files concurrently
2. **Content Validation**: Enhanced XML structure validation
3. **Speaker Detection**: Improved speaker role identification
4. **Content Enrichment**: Additional metadata extraction from XML
5. **Performance Optimization**: Caching and batch processing improvements

### Integration Opportunities
1. **Direct Database Storage**: Write to database instead of JSON files
2. **Real-time Processing**: Process files as they become available
3. **Content Analysis**: Basic sentiment or topic analysis during extraction
4. **Quality Metrics**: Content quality scoring and validation

## References

### Related Documentation
- **Main README**: Pipeline overview and Stage 3 usage instructions
- **Stage 2 CLAUDE.md**: Upstream processing and file selection logic
- **transcript_html_viewer.py**: XML processing reference implementation
- **Security Standards**: Pipeline-wide security and validation requirements

### Key Files
- **Main Script**: `3_transcript_content_extraction.py`
- **Configuration**: `config.json` (stage_3 section)
- **Input**: `Outputs/Refresh/files_to_process.json`
- **Output**: `Outputs/Refresh/extracted_transcript_sections.json`
- **Error Logs**: `Outputs/Logs/Errors/stage_3_*_errors.json`