# Stage 3: Content Extraction - Output Field Documentation

## Output File

### `stage_03_extracted_content.json`
Paragraph-level content extracted from XML transcripts.

**Fields (from lines 793-801, 836-842, and 570-579):**

#### Base Fields (from Stage 2 input):
- **file_path**: Full NAS path to the transcript XML file - source file location
- **date_last_modified**: ISO 8601 timestamp when file was last modified - from Stage 2 queue
- **institution_id**: ID from monitored_institutions config - links to institution metadata (can be null)
- **ticker**: Stock ticker from Stage 2 or extracted from filename - identifies the company

#### Extracted Metadata (from file path and filename):
- **filename**: Name of the XML file - extracted from file path
- **fiscal_year**: Year from directory structure (Data/YYYY/...) - identifies fiscal period
- **fiscal_quarter**: Quarter from directory structure (Data/YYYY/QX/...) - identifies fiscal quarter
- **institution_type**: Type from directory structure (e.g., "Banks") - categorizes institution
- **company_name**: Company name from monitored_institutions config lookup - human-readable name (can be null)
- **transcript_type**: Type from filename (e.g., "Corrected", "Final") - version type
- **event_id**: Event ID from filename - unique event identifier
- **version_id**: Version ID from filename - version number

#### XML Content Fields:
- **title**: Title extracted from XML meta section - transcript title

#### Paragraph-Level Fields:
- **section_id**: Sequential number of the section (1, 2, 3...) - identifies section order
- **section_name**: Name of the section from XML - describes section content
- **paragraph_id**: Global sequential paragraph number across entire transcript - unique paragraph ID
- **speaker_block_id**: Sequential ID for speaker blocks - groups paragraphs by same speaker
- **question_answer_flag**: "q" for question, "a" for answer, null otherwise - from XML speaker type
- **speaker**: Formatted speaker string (name, title, affiliation) - identifies who is speaking
- **paragraph_content**: Cleaned paragraph text with escaped quotes and no newlines - actual content

**Notes:**
- Only files with at least one paragraph are included in output
- Files producing zero paragraphs are tracked in logs but excluded from JSON
- Each paragraph becomes a separate record in the JSON array
- Text is cleaned: quotes escaped, newlines/tabs replaced with spaces