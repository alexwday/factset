# Stage 7: LLM Summarization - Output Field Documentation

## Output Files

### 1. `stage_07_summarized_content.json`
Records with block-level summaries added.

**Format:** Enhanced Stage 6 records with block_summary field added  
**Content:** All Stage 6 records with LLM-generated summaries applied

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
- **qa_group_id**: Q&A conversation group ID (from Stage 5)
- **classification_ids**: Array of numeric category IDs (from Stage 6)
- **classification_names**: Array of category names (from Stage 6)
- **block_summary** (added in Stage 7):
  - Type: string or null
  - Purpose: Block-level summary for reranking and retrieval optimization
  - Assignment logic varies by section type

### 2. `stage_07_failed_transcripts.json`
Metadata about transcripts that failed processing.

**Format:** JSON object with failure information  
**Content:** Summary of failed transcripts (not the actual records)

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_failed": 2,
  "failed_transcript_ids": ["transcript_key_1", "transcript_key_2"],
  "failed_transcripts": [
    {
      "transcript_id": "transcript_key_1",
      "error": "Processing error message"
    }
  ]
}
```

## Summary Assignment Logic

### Q&A Sections (from lines 1014-1016, 1041-1042, 1054-1055):
- **Grouped by**: `qa_group_id`
- **Summary scope**: Entire Q&A conversation
- **Assignment**: Same summary applied to ALL paragraphs in the Q&A group
- **On success**: LLM-generated conversational summary
- **On failure**: `null` value for all records in group

### Management Discussion Sections (from lines 1183-1184, 1209-1210, 1222-1223):
- **Grouped by**: `speaker_block_id`
- **Summary scope**: Individual speaker block
- **Context**: Uses previous 2 speaker block summaries as context
- **Assignment**: Same summary applied to ALL paragraphs in the speaker block
- **On success**: LLM-generated speaker block summary
- **On failure**: `null` value for all records in block

### Other Sections (from lines 1239-1249):
- **Processing**: Individual record level
- **Summary format**: `[SECTION_NAME] {first_100_chars}...`
- **Purpose**: Basic identification for non-standard sections
- **Always succeeds**: Simple truncation, no LLM call

### Failed Transcripts (from lines 1274-1275):
- **All records**: Receive `block_summary = null`
- **Original data**: Preserved unchanged except for null summary

## Summary Content Structure

### Q&A Summaries Include:
- Speaker attribution (analyst vs executive)
- Key questions asked
- Main points from answers
- Quantitative data mentioned
- Forward-looking statements

### Management Discussion Summaries Include:
- Speaker name and role
- Main topics covered
- Key financial metrics
- Strategic initiatives mentioned
- Position in presentation (opening/middle/closing)

### Notes:
- Block-level summarization reduces redundancy (all paragraphs in a block share the same summary)
- Summaries optimized for post-retrieval reranking
- Null values indicate processing failures but preserve data integrity
- Original Stage 6 fields remain unchanged
- The field is named `block_summary` (not `paragraph_summary`) to reflect block-level application