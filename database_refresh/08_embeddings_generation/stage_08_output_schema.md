# Stage 8: Embeddings Generation - Output Field Documentation

## Output Files

### 1. `stage_08_embeddings.csv` (or `.json` if configured)
Records with embeddings and chunking metadata added.

**Format:** Enhanced Stage 7 records with embeddings and chunk-level organization  
**Content:** Chunk-level records with embeddings, where chunks may contain one or more paragraphs

**All Fields (in CSV column order):**
- **file_path**: Full NAS path to the transcript XML file
- **filename**: Name of the XML file
- **date_last_modified**: ISO 8601 timestamp when file was last modified
- **event_id**: Event ID from filename
- **version_id**: Version ID from filename
- **title**: Title extracted from XML meta section
- **fiscal_year**: Year from directory structure
- **fiscal_quarter**: Quarter from directory structure
- **institution_type**: Type from directory structure (e.g., "Banks")
- **institution_id**: ID from monitored_institutions config (can be null)
- **ticker**: Stock ticker identifying the company
- **company_name**: Company name from config lookup (can be null)
- **section_id**: Sequential number of the section
- **section_name**: Name of the section from XML
- **speaker_block_id**: Sequential ID for speaker blocks
- **speaker_block_tokens**: Total tokens in the speaker block (calculated in Stage 8)
- **speaker_name**: Formatted speaker string (name, title, affiliation)
- **qa_group_id**: Q&A conversation group ID (from Stage 5)
- **primary_category_id**: Most common category ID from aggregated classifications
- **primary_category_name**: Most common category name from aggregated classifications
- **secondary_category_ids**: List of other unique category IDs (JSON array in CSV)
- **secondary_category_names**: List of other unique category names (JSON array in CSV)
- **block_summary**: Block-level summary from Stage 7
- **chunk_id**: Sequential chunk number within the speaker block
- **chunk_content**: The actual text that was embedded (may combine multiple paragraphs)
- **chunk_tokens**: Number of tokens in this chunk (calculated with tiktoken or fallback)
- **chunk_paragraph_ids**: List of paragraph IDs included in this chunk (JSON array in CSV)
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

### Hierarchical Chunking Strategy (from lines 1047-1334):
1. **Group by speaker_block_id**: All records grouped by speaker blocks first
2. **Token calculation**: Count tokens for each block using tiktoken (or fallback estimation)
3. **Chunking decision**:
   - If block ≤ 1000 tokens: Keep as single chunk
   - If block > 1000 tokens: Split intelligently

### Single Block Chunking (from lines 1138-1175):
- **When applied**: Block total tokens ≤ chunk_threshold (1000)
- **Result**: One chunk = entire speaker block
- **Fields preserved**: All Stage 7 fields from first record (all identical within block)
- **Merged data**: All paragraph_content concatenated with "\n\n"
- **Categories**: Aggregated from all paragraphs in block
- **Output**: Single embedding for entire block

### Multi-Chunk Splitting (from lines 1177-1332):
- **When applied**: Block total tokens > chunk_threshold
- **Strategy**: 
  - Try to keep paragraphs together
  - Target ~500 tokens per chunk
  - Split within paragraphs only when single paragraph > 1000 tokens
- **Paragraph boundaries**: Respected when possible
- **Final chunk handling**: Merged with previous if < 300 tokens

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

## Category Aggregation (from lines 969-1010):
- **Input**: Arrays of classification_ids and classification_names from Stage 7
- **Processing**: Count occurrences across all paragraphs in chunk
- **Primary category**: Most common category (by frequency)
- **Secondary categories**: All other unique categories
- **Output**: Separated into primary and secondary fields

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