# Stage 5: Q&A Pairing - Output Field Documentation

## Output Files

### 1. `stage_05_qa_paired_content.json`
Records with Q&A conversation group assignments added.

**Format:** Enhanced Stage 4 records with qa_group_id field added  
**Content:** All Stage 4 records with Q&A boundary detection applied

**All Fields:**
- **file_path**: Full NAS path to the transcript XML file
- **date_last_modified**: ISO 8601 timestamp when file was last modified
- **institution_id**: ID from monitored_institutions config (can be null)
- **ticker**: Stock ticker identifying the company
- **filename**: Name of the XML file
- **fiscal_year**: Year from directory structure
- **fiscal_quarter**: Quarter from directory structure
- **institution_type**: Type from directory structure (e.g., "Banks")
- **company_name**: Company name from config lookup (can be null)
- **transcript_type**: Type from filename (e.g., "Corrected", "Final")
- **event_id**: Event ID from filename
- **version_id**: Version ID from filename
- **title**: Title extracted from XML meta section
- **section_id**: Sequential number of the section
- **section_name**: Name of the section from XML
- **paragraph_id**: Global sequential paragraph number
- **speaker_block_id**: Sequential ID for speaker blocks
- **question_answer_flag**: "q" for question, "a" for answer, null otherwise
- **speaker**: Formatted speaker string (name, title, affiliation)
- **paragraph_content**: Cleaned paragraph text
- **qa_group_id** (added in Stage 5, from lines 1856-1872):
  - For Q&A section records: Integer ID identifying the Q&A conversation group (1, 2, 3...)
  - For Q&A section records without assignment: `null` (unassigned blocks)
  - For non-Q&A section records: `null` (Management Discussion sections)

### 2. `stage_05_qa_pairing_failed_transcripts.json`
Original records for transcripts that failed processing.

**Format:** Array of paragraph records  
**Content:** Records from transcripts where Q&A pairing failed

**All Fields:**
- Same as Stage 4 input (all fields listed above except qa_group_id)
- No modifications made to failed transcript records

## Q&A Group Structure (internal processing, from lines 1647-1736)

### Q&A Group Object (created during processing):
```json
{
  "qa_group_id": 1,
  "start_block_id": 25,
  "end_block_id": 28,
  "confidence": 1.0,
  "speaker_blocks": [/* array of speaker blocks */]
}
```

**Confidence Levels:**
- `1.0`: Normal sliding window with validation passed
- `0.8`: Memory limit reached (max_held_blocks)
- `0.7`: Consecutive skip limit reached
- `0.5`: Validation failed but proceeding anyway

## Processing Logic (from lines 1802-1879)

### Field Assignment Rules:
1. **Q&A Section Records** (section_name == "Q&A"):
   - If speaker_block_id maps to a Q&A group: `qa_group_id` = group number
   - If no mapping exists: `qa_group_id` = null (unassigned)

2. **Non-Q&A Section Records** (e.g., "Management Discussion"):
   - Always: `qa_group_id` = null

### Notes:
- Q&A groups are numbered sequentially starting from 1
- Each speaker_block_id can belong to only one Q&A group
- Unassigned Q&A blocks indicate boundary detection gaps
- The qa_group_id links related question-answer conversations together
- All original Stage 4 fields are preserved without modification