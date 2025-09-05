# Stage 8: Embeddings Generation - Output Field Documentation

## Output Files

### 1. `stage_08_embeddings.csv` (or `.json` if configured)
Records with embeddings and chunking metadata added.

**Format:** Enhanced Stage 7 records with embeddings and chunk-level organization  
**Content:** Chunk-level records with formatted text and embeddings, where chunks may contain one or more paragraphs

**All Fields (in CSV column order):**
- **file_path**: Full NAS path to the transcript XML file
- **filename**: Name of the XML file
- **date_last_modified**: ISO 8601 timestamp when file was last modified
- **title**: Title extracted from XML meta section
- **transcript_type**: Type from filename (e.g., "Corrected", "Final", "E1")
- **event_id**: Event ID from filename
- **version_id**: Version ID from filename
- **fiscal_year**: Year from directory structure
- **fiscal_quarter**: Quarter from directory structure
- **institution_type**: Type from directory structure (e.g., "Banks")
- **institution_id**: ID from monitored_institutions config (can be null)
- **ticker**: Stock ticker identifying the company
- **company_name**: Company name from config lookup (can be null)
- **section_name**: Name of the section ("MANAGEMENT DISCUSSION SECTION" or "Q&A")
- **speaker_block_id**: Sequential ID for speaker blocks
- **qa_group_id**: Q&A conversation group ID (from Stage 5, only for Q&A sections)
- **classification_ids**: Array of classification IDs from Stage 6 (JSON array in CSV)
- **classification_names**: Array of classification names from Stage 6 (JSON array in CSV)
- **block_summary**: Block-level summary from Stage 7
- **chunk_id**: Sequential chunk number within the speaker block or Q&A group
- **chunk_tokens**: Number of tokens in this chunk (calculated with tiktoken or fallback)
- **chunk_content**: Formatted text with headers that gets embedded
- **chunk_paragraph_ids**: List of original paragraph IDs in this chunk (JSON array in CSV)
- **chunk_embedding**: 3072-dimensional vector (JSON array in CSV)

**Additional Fields in Enhanced Records (internal processing):**
- **paragraph_tokens**: Tokens in the chunk (same as chunk_tokens)
- **block_tokens**: Total tokens in the speaker block
- **total_chunks**: Total number of chunks created from this block
- **block_level_chunk**: Boolean indicating if entire block kept as single chunk
- **paragraphs_in_chunk**: Count of paragraphs merged into this chunk
- **paragraph_chunk**: For split paragraphs, format "X/Y" indicating chunk X of Y

### 2. `stage_08_failed_transcripts_[timestamp].json`
Metadata about transcripts that failed processing.

**Format:** JSON object with failure information  
**Content:** Summary of failed transcripts (not the actual records)

```json
{
  "stage": "08_embeddings_generation",
  "timestamp": "2024-01-01T12:00:00",
  "total_failed": 2,
  "failed_transcripts": [
    {
      "transcript": "ticker_eventid.xml",
      "timestamp": "2024-01-01T12:00:00",
      "error": "Processing error message",
      "processing_time": "0:00:30.123456"
    }
  ]
}
```

## Chunking and Embedding Logic

### Content Formatting
Before embedding, content is formatted with appropriate headers:

**Management Discussion Format:**
```
[Speaker Name, Title, Company]
Paragraph 1...
Paragraph 2...
```

**Q&A Format:**
```
[Q&A Group 123]

[Analyst Name, Company]
Question paragraph...

[Executive Name, Title]
Answer paragraph...
```

### Hierarchical Chunking Strategy:
1. **Management Discussion**: Group by `speaker_block_id`
2. **Q&A Sections**: Group by `qa_group_id` first to preserve conversations
3. **Token calculation**: Count tokens for formatted content using tiktoken (or fallback)
4. **Chunking decision**:
   - If ≤ 1000 tokens: Keep as single chunk
   - If > 1000 tokens: Split intelligently

### Management Discussion Chunking:
- **Single chunk**: When speaker block ≤ 1000 tokens
- **Multi-chunk**: Split at paragraph boundaries when > 1000 tokens
- **Smart merging**: Final chunks < 300 tokens merged with previous chunk
- **Headers preserved**: Each chunk includes speaker header

### Q&A Section Chunking:
- **Primary grouping**: By `qa_group_id` to keep conversations together
- **Single chunk**: Entire Q&A group when ≤ 1000 tokens
- **Multi-chunk**: Split at speaker block boundaries when possible
- **Headers preserved**: Q&A Group header and speaker subheaders in each chunk

### Embedding Generation (from lines 912-944):
- **Model**: text-embedding-3-large
- **Dimensions**: 3072 (full model dimensions)
- **Batch processing**: 50 texts per API call
- **Retry logic**: 3 attempts with exponential backoff
- **OAuth refresh**: Per transcript to prevent token expiration

## Token Counting Methods

### Primary Method - Tiktoken (from lines 740-754):
- **Encoding**: cl100k_base (GPT-4/Claude compatible)
- **Accuracy**: Exact token counts
- **Availability**: Requires tiktoken package

### Fallback Method - Estimation (from lines 756-794):
- **When used**: If tiktoken unavailable or fails
- **Algorithm**: Weighted average of three methods:
  - Refined character-based: len/3.5 (50% weight)
  - Standard character-based: len/4.0 (30% weight)  
  - Word-based: words * 1.3 (20% weight)
- **Safety buffer**: +10% for chunking decisions
- **Accuracy**: ±15% of actual token count

## Classification Handling:
- **Input**: Arrays of `classification_ids` and `classification_names` from Stage 7
- **Processing**: No aggregation needed - Stage 6 provides block-level classifications
- **Consistency**: All paragraphs within a block share the same classifications
- **Output**: Direct pass-through of classification arrays from Stage 7

## Output Format Details

### CSV Format (Default, from lines 564-687):
- **Advantages**: Efficient appending, lower memory usage, compatible with data tools
- **Embedding storage**: JSON string in last column
- **List fields**: Stored as JSON arrays (paragraph_ids, category_ids, etc.)
- **Headers**: Written only on first batch

### JSON Format (Optional, from lines 689-738):
- **Structure**: Array of objects
- **Embedding storage**: Native array of floats
- **Incremental writing**: Appends without closing bracket until complete
- **Final step**: Closes array after all transcripts processed

## Processing Statistics

### Enhanced Error Logger tracks (from lines 90-177):
- **embedding_errors**: Failed embedding generation attempts
- **chunking_errors**: Text chunking failures
- **authentication_errors**: OAuth/SSL issues
- **processing_errors**: General transcript processing errors
- **validation_errors**: Data validation issues
- **total_embeddings**: Count of successful embeddings
- **total_chunks**: Count of chunks created
- **using_fallback_tokenizer**: Whether tiktoken or fallback was used

## Notes:
- **One record per chunk**: Each output record represents a chunk, not a paragraph
- **Chunk may contain multiple paragraphs**: When paragraphs are merged, chunk_paragraph_ids lists all included
- **All Stage 7 fields preserved**: Through spread operator on first record (all identical within block)
- **Block-level consistency**: All paragraphs in a speaker_block_id share same field values except paragraph_content
- **Embeddings shared within chunks**: All merged paragraphs get same embedding vector
- **Traceability maintained**: chunk_paragraph_ids preserves mapping to original paragraphs