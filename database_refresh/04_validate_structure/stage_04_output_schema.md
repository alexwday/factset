# Stage 4: Structure Validation - Output Field Documentation

## Output Files

### 1. `stage_04_validated_content.json`
Records from transcripts that passed validation (exactly 2 expected sections).

**Format:** Array of paragraph records  
**Content:** Filtered subset of Stage 3 records where transcript structure is valid

**All Fields (preserved from Stage 3):**
- **file_path**: Full NAS path to the transcript XML file
- **date_last_modified**: ISO 8601 timestamp when file was last modified
- **institution_id**: ID from monitored_institutions config (can be null)
- **ticker**: Stock ticker identifying the company
- **filename**: Name of the XML file
- **fiscal_year**: Year from directory structure (Data/YYYY/...)
- **fiscal_quarter**: Quarter from directory structure (Data/YYYY/QX/...)
- **institution_type**: Type from directory structure (e.g., "Banks")
- **company_name**: Company name from config lookup (can be null)
- **transcript_type**: Type from filename (e.g., "Corrected", "Final")
- **event_id**: Event ID from filename
- **version_id**: Version ID from filename
- **title**: Title extracted from XML meta section
- **section_id**: Sequential number of the section (1, 2, 3...)
- **section_name**: Name of the section from XML
- **paragraph_id**: Global sequential paragraph number across entire transcript
- **speaker_block_id**: Sequential ID for speaker blocks
- **question_answer_flag**: "q" for question, "a" for answer, null otherwise
- **speaker**: Formatted speaker string (name, title, affiliation)
- **paragraph_content**: Cleaned paragraph text with escaped quotes

### 2. `stage_04_invalid_content.json`  
Records from transcripts that failed validation.

**Format:** Array of paragraph records  
**Content:** Filtered subset of Stage 3 records where transcript structure is invalid

**All Fields (preserved from Stage 3):**
- Same complete field list as above - all Stage 3 fields are preserved
- Records are from transcripts that don't have exactly 2 expected sections

## Validation Process (from lines 549-623)

### Validation Result Structure (internal only, not saved to output):
```json
{
  "transcript_key": "ticker_quarter_year_type_eventid_versionid",
  "is_valid": true/false,
  "validation_errors": ["list of error messages"],
  "sections_found": ["actual section names"],
  "expected_sections": ["configured expected sections"],
  "section_counts": {"section_name": record_count},
  "total_records": number_of_records
}
```

### Validation Rules (from lines 579-599):
1. **Section Count**: Must have exactly 2 sections
2. **Section Names**: Must match expected_sections from config
3. **No Extra Sections**: No unexpected sections allowed

### Grouping Key (from line 670):
Records are grouped by composite key: `{ticker}_{fiscal_quarter}_{fiscal_year}_{transcript_type}_{event_id}_{version_id}`

**Notes:**
- Stage 4 acts as a filter - no new fields are added
- Valid and invalid outputs maintain Stage 3 format for downstream compatibility
- Each version/event ID combination is validated separately
- Records from the same transcript stay together (all valid or all invalid)