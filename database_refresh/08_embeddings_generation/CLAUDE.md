# Stage 08: Embeddings Generation

## Overview
This stage generates vector embeddings from Stage 7's summarized content for semantic search and retrieval-augmented generation (RAG).

## Key Features
- **Token Calculation**: Uses tiktoken to calculate tokens for paragraphs and speaker/QA blocks
- **Intelligent Chunking**: Breaks large paragraphs (>1000 tokens) into ~500 token chunks
- **Embeddings Generation**: Creates full 3072-dimensional vectors using OpenAI's text-embedding-3-large
- **PostgreSQL Integration**: Uses halfvec (16-bit float) type for efficient storage of 3072 dimensions
- **Storage Optimization**: 50% reduction in storage compared to standard vectors while maintaining quality

## Process Flow

### 1. Token Calculation
- Calculates `paragraph_tokens` for each paragraph using tiktoken
- Aggregates `block_tokens` for:
  - MD sections: Total tokens per speaker block
  - Q&A sections: Total tokens per qa_id

### 2. Text Chunking
- **Threshold**: 1000 tokens triggers chunking
- **Target Size**: 500 tokens per chunk
- **Minimum Final Chunk**: 300 tokens (merges with previous if smaller)
- **Breaking Strategy**: Finds sentence boundaries for natural breaks

### 3. Embedding Generation
- **Model**: text-embedding-3-large
- **Dimensions**: 3072 (full model dimensions)
- **Storage Type**: halfvec (16-bit floats)
- **Benefits**: Better semantic representation with full dimensions
- **Rate Limiting**: Pauses every 100 embeddings
- **Retry Logic**: 3 attempts with exponential backoff

## Database Schema

```sql
CREATE TABLE stage_08_embeddings (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(50),
    file_name TEXT,
    section_type VARCHAR(10),
    speaker_id INTEGER,
    speaker_name TEXT,
    speaker_title TEXT,
    qa_id INTEGER,
    question_speaker_name TEXT,
    answer_speaker_name TEXT,
    paragraph_id VARCHAR(100),
    paragraph_sequence INTEGER,
    paragraph_text TEXT,
    paragraph_summary TEXT,
    paragraph_tokens INTEGER,      -- New: tokens in original paragraph
    block_tokens INTEGER,          -- New: total tokens in speaker/QA block
    chunk_id INTEGER,              -- New: chunk number (1, 2, 3...)
    chunk_text TEXT,               -- New: actual text of chunk
    chunk_tokens INTEGER,          -- New: tokens in this chunk
    embedding halfvec(3072),       -- New: embedding vector (halfvec for 3072 dims)
    embedding_model VARCHAR(100),  -- New: model used (text-embedding-3-large)
    embedding_dimensions INTEGER,  -- New: vector dimensions (3072)
    processed_at_stage_7 TIMESTAMP,
    processed_at_stage_8 TIMESTAMP,
    UNIQUE(paragraph_id, chunk_id)
);
```

## Usage

### Basic Execution
```bash
cd database_refresh/08_embeddings_generation
python main_embeddings_generation.py
```

### With Options
```bash
# Process limited events
python main_embeddings_generation.py --limit 10

# Enable debug logging
python main_embeddings_generation.py --debug

# Custom config
python main_embeddings_generation.py --config ../config.yaml
```

## Configuration
Uses the same `config.yaml` as other stages with OpenAI API configuration:

```yaml
openai:
  api_key: your-api-key-here
  
database:
  host: localhost
  port: 5432
  name: factset_transcripts
  user: your-user
  password: your-password
```

### Embedding Configuration
- **Model**: text-embedding-3-large
- **Dimensions**: 3072 (full dimensions)
- **Storage**: halfvec (16-bit precision)
- **Max supported**: 4000 dimensions with halfvec

## Output Statistics
The script provides detailed statistics:
- Paragraphs processed
- Chunks created
- Embeddings generated
- Embedding failures
- Average chunks per paragraph

## Vector Search Queries

### Find Similar Content
```sql
-- Find top 5 most similar chunks to a query
-- Note: Cast query embedding to halfvec(3072) for comparison
SELECT 
    chunk_text,
    1 - (embedding <=> '[query_embedding]'::halfvec(3072)) AS similarity
FROM stage_08_embeddings
ORDER BY embedding <=> '[query_embedding]'::halfvec(3072)
LIMIT 5;
```

### Aggregate by Speaker Block
```sql
-- Get all chunks for a speaker block
SELECT *
FROM stage_08_embeddings
WHERE event_id = 'EVENT123' 
  AND section_type = 'MD'
  AND speaker_id = 1
ORDER BY paragraph_sequence, chunk_id;
```

## Performance Considerations

### Rate Limiting
- Pauses for 1 second every 100 embeddings
- Exponential backoff on API errors

### Chunking Efficiency
- Only chunks paragraphs over 1000 tokens
- Minimizes API calls by batching appropriately
- Preserves semantic coherence in chunks

### Database Optimization
- Uses HNSW index with halfvec_cosine_ops for better performance
- Composite unique constraint on (paragraph_id, chunk_id)
- Indexes on event_id for fast lookups
- 50% storage reduction with halfvec vs standard vectors

## Error Handling
- Retries embedding generation 3 times with exponential backoff
- Logs all failures to `stage_08_embeddings.log`
- Continues processing even if individual embeddings fail
- Database transactions ensure data consistency

## Dependencies
```bash
pip install psycopg2-binary
pip install openai
pip install tiktoken
pip install numpy
pip install pyyaml
pip install tqdm
```

## PostgreSQL Requirements
Ensure pgvector extension is installed (version 0.7.0+ required for halfvec):
```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify halfvec support
SELECT * FROM pg_available_extensions WHERE name = 'vector';
-- Should show version 0.7.0 or higher
```

### Why halfvec?
- **Full Dimensions**: Supports all 3072 dimensions from text-embedding-3-large
- **Storage Efficiency**: 50% less storage than standard vectors
- **Performance**: 30% faster HNSW index builds
- **Quality**: Minimal loss in precision for similarity search
- **Capacity**: Supports up to 4000 dimensions (vs 2000 for standard vectors)

## Next Steps
The embeddings generated in this stage enable:
1. Semantic search across earnings transcripts
2. RAG (Retrieval-Augmented Generation) for Q&A systems
3. Similar content discovery
4. Context expansion for LLM applications