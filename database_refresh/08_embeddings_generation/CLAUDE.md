# Stage 08: Embeddings Generation

## Overview
This stage generates vector embeddings from the original paragraph text (with Stage 7's summaries preserved) for semantic search and retrieval-augmented generation (RAG). While Stage 7 adds summaries for display and reranking, Stage 8 creates embeddings from the full original content to capture complete semantic meaning.

**Input**: `Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_07_summarized_content.json` (contains both original text and summaries)
**Output**: `Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_08_embeddings.json`

Architecture exactly matches Stage 7 with NAS operations, incremental saving, and OAuth refresh.

## Key Features
- **Token Calculation**: Uses tiktoken to calculate tokens for paragraphs and speaker/QA blocks
- **Intelligent Chunking**: Breaks large paragraphs (>1000 tokens) into ~500 token chunks
- **Batch Processing**: Sends embeddings in batches of 50 texts for efficient API usage
- **Embeddings Generation**: Creates full 3072-dimensional vectors using OpenAI's text-embedding-3-large
- **JSON Input/Output**: Reads from Stage 7 JSON and outputs enhanced JSON with embeddings
- **Full Semantic Representation**: Leverages all 3072 dimensions for maximum embedding quality

## Process Flow

### 1. Token Calculation
- Calculates `paragraph_tokens` for each paragraph using tiktoken (with fallback)
- **Fallback Mechanism**: If tiktoken fails, uses hybrid estimation:
  - Refined character-based: ~3.5 chars/token (50% weight)
  - Standard character-based: ~4 chars/token (30% weight)
  - Word-based: ~1.3 tokens/word (20% weight)
  - Adds 10% safety buffer for chunking decisions
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
- **Batch Processing**: 50 texts per API call (reduces latency by 50x)
- **Storage Type**: vector (32-bit floats)
- **Benefits**: Maximum semantic representation with full dimensions
- **Rate Limiting**: Pauses every 100 embeddings
- **Retry Logic**: 3 attempts with exponential backoff

## Output JSON Structure

Each record in the output contains all fields from Stage 7 plus:

```json
{
  // All original Stage 7 fields preserved...
  "paragraph_content": "...",     // Original content from Stage 3 (what gets embedded)
  "block_summary": "...",        // Summary from Stage 7 (for display/reranking)
  
  // New Stage 8 fields:
  "paragraph_tokens": 850,        // Tokens in original paragraph
  "block_tokens": 2500,           // Total tokens in speaker/QA block
  "chunk_id": 1,                  // Chunk number (1, 2, 3...)
  "total_chunks": 2,              // Total chunks for this paragraph
  "chunk_text": "...",            // Actual text chunk being embedded
  "chunk_tokens": 425,            // Tokens in this chunk
  "embedding": [0.123, -0.456, ...] // 3072-dimensional vector of chunk_text
}
```

## Usage

### Basic Execution
```bash
cd database_refresh/08_embeddings_generation
python main_embeddings_generation.py
```

The script will:
1. Connect to NAS using environment variables
2. Load configuration from NAS
3. Download Stage 7 output from NAS
4. Process each transcript with OAuth token refresh
5. Save results incrementally to NAS
6. Save logs and error reports to NAS

## Configuration
Configuration is loaded from NAS at runtime from `config.yaml`:

```yaml
stage_08_embeddings_generation:
  description: "Generate vector embeddings from Stage 7 summarized content"
  input_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_07_summarized_content.json"
  output_data_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
  output_logs_path: "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs"
  dev_mode: true
  dev_max_transcripts: 10
  
  # Embedding configuration
  embedding_config:
    model: "text-embedding-3-large"
    dimensions: 3072
    chunk_threshold: 1000
    target_chunk_size: 500
    min_final_chunk: 300
    rate_limit_pause: 100
    token_refresh_interval: 500
  
  # LLM configuration (same OAuth as Stage 07)
  llm_config:
    base_url: "https://api.openai.com/v1"
    timeout: 30
    max_retries: 3
    token_endpoint: "https://auth.example.com/oauth/token"
```

### Authentication & NAS Access

**Environment Variables Required**:
- `NAS_SHARE_NAME`: NAS share name
- `NAS_USER`: NAS username
- `NAS_PASSWORD`: NAS password
- `NAS_DOMAIN`: NAS domain (optional)
- `LLM_CLIENT_ID`: OAuth client ID
- `LLM_CLIENT_SECRET`: OAuth client secret

**Features**:
- OAuth 2.0 client credentials flow (same as Stage 07)
- SSL certificate downloaded from NAS
- Automatic OAuth token refresh per transcript

### Embedding Configuration
- **Model**: text-embedding-3-large
- **Dimensions**: 3072 (full dimensions)
- **Storage**: vector (standard 32-bit precision)
- **Requirements**: PostgreSQL with pgvector 0.5.0+

## Output Statistics
The script provides detailed statistics:
- Paragraphs processed
- Chunks created
- Embeddings generated
- Embedding failures
- Average chunks per paragraph

## Working with Embeddings

### Loading and Searching Embeddings
```python
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
with open('stage_08_embeddings.json', 'r') as f:
    embeddings_data = json.load(f)

# Extract embeddings and texts
embeddings = np.array([record['embedding'] for record in embeddings_data])
texts = [record['chunk_text'] for record in embeddings_data]

# Search for similar content
query_embedding = generate_embedding("your search query")  # Use same model
similarities = cosine_similarity([query_embedding], embeddings)[0]
top_5_indices = np.argsort(similarities)[-5:][::-1]

for idx in top_5_indices:
    print(f"Similarity: {similarities[idx]:.3f}")
    print(f"Text: {texts[idx][:200]}...\n")
```

## Performance Considerations

### Rate Limiting
- Pauses for 1 second every 100 embeddings
- Exponential backoff on API errors

### Chunking Efficiency
- Only chunks paragraphs over 1000 tokens
- Minimizes API calls by batching appropriately
- Preserves semantic coherence in chunks

### Processing Optimization
- Batched processing with progress tracking
- Automatic OAuth token refresh every 500 embeddings
- Rate limiting with configurable pauses
- Dev mode for testing with limited transcripts

## Error Handling
- Retries embedding generation 3 times with exponential backoff
- Logs all failures to `stage_08_embeddings.log`
- Continues processing even if individual embeddings fail
- Database transactions ensure data consistency

## Dependencies
```bash
pip install pyyaml
pip install pysmb
pip install python-dotenv
pip install openai
pip install tiktoken  # Optional but recommended for accurate token counting
pip install requests
```

### Token Counting Fallback
The system includes a robust fallback mechanism if tiktoken is unavailable:
- **Primary Method**: tiktoken with cl100k_base encoding (GPT-4/Claude compatible)
- **Fallback Method**: Hybrid estimation algorithm that:
  - Combines character, word, and refined character counting
  - Adds 10% safety buffer for chunking decisions
  - Logs warnings when using fallback
  - Continues processing without interruption
- **Impact of Fallback**:
  - Token counts will be approximate (±15% accuracy)
  - Chunking decisions may be slightly less optimal
  - Embeddings still generate correctly
  - All features remain functional

## Storage Requirements

### Embedding Storage
- **Full Dimensions**: All 3072 dimensions from text-embedding-3-large
- **Precision**: 32-bit floats for maximum accuracy
- **Size per embedding**: ~12KB (3072 × 4 bytes)
- **JSON file size estimates**:
  - 1,000 embeddings: ~15 MB
  - 10,000 embeddings: ~150 MB
  - 100,000 embeddings: ~1.5 GB

## Next Steps
The embeddings generated in this stage enable:
1. Semantic search across earnings transcripts
2. RAG (Retrieval-Augmented Generation) for Q&A systems
3. Similar content discovery
4. Context expansion for LLM applications