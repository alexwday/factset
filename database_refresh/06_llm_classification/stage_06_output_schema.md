# Stage 6: LLM Classification - Output Field Documentation

## Output Files

### 1. `stage_06_classified_content.json`
Records with financial category classifications added.

**Format:** Enhanced Stage 5 records with classification fields added  
**Content:** All Stage 5 records with LLM-based financial categorization applied

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
- **classification_ids** (added in Stage 6, from lines 1302, 1133):
  - Array of numeric category IDs (0-22) from CATEGORY_REGISTRY
  - Multiple categories can be assigned to each record
  - Category 0 indicates "Non-Relevant" content
  - Applied at speaker block level for both MD and Q&A sections
- **classification_names** (added in Stage 6, from lines 1303, 1134):
  - Array of category names corresponding to classification_ids
  - Human-readable names from CATEGORY_REGISTRY
  - Example: ["Revenue", "Guidance", "Market Conditions"]

### 2. `stage_06_failed_transcripts.json`
Metadata about transcripts that failed processing.

**Format:** JSON object with failure information  
**Content:** Summary of failed transcripts (not the actual records)

```json
{
  "timestamp": "2024-01-01T12:00:00",
  "total_failed": 2,
  "failed_transcripts": ["transcript_key_1", "transcript_key_2"]
}
```

## Category System (from line 41, CATEGORY_REGISTRY)

### Category Structure:
- Categories are loaded from `financial_categories.yaml` or config.yaml
- Each category has:
  - `id`: Numeric identifier (0-22)
  - `name`: Category name
  - `description`: Category description
- Category 0 is reserved for "Non-Relevant" content

### Classification Logic (from lines 1089-1305):
1. **Management Discussion**: Each speaker block classified independently
   - Context from previous speaker blocks provided to LLM
   - Same classifications applied to all paragraphs within a speaker block

2. **Q&A Sections**: Each Q&A group classified as a unit
   - All records in same qa_group_id get same classifications
   - Unpaired Q&A records classified at speaker block level

3. **Fallback**: Category 0 assigned if classification fails or no categories identified

### Notes:
- Multiple categories can be assigned to each record (array format)
- All paragraphs within a speaker block share the same classifications
- All records in a Q&A conversation group share the same classifications
- The actual implementation uses single-pass classification (not two-pass as mentioned in docs)
- Classification is performed at the speaker block level, not individual paragraph level