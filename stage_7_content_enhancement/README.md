# Stage 7: Content Enhancement System

## Overview

Stage 7 enhances earnings transcript paragraphs with condensed summaries and importance scoring using a sophisticated sliding window approach. Optimizes content for retrieval systems while maintaining full context awareness.

## Features

- **Sliding Window Processing**: Uses previous speaker block context while processing current block in 5-paragraph batches
- **Dual Enhancement**: Provides both condensed summaries and relative importance scoring
- **Progressive Context**: Builds context within speaker blocks using previously enhanced content
- **OAuth Authentication**: Per-transcript token refresh for reliable API access
- **Cost Tracking**: Comprehensive token usage and cost monitoring
- **Production Ready**: Follows established Stage 4/5/6 patterns with comprehensive error handling

## Input/Output

- **Input**: `detailed_classified_sections.json` from Stage 6
- **Output**: `enhanced_transcript_sections.json` with paragraph-level enhancements
- **Fields Added**: `paragraph_summary`, `paragraph_importance`

## Configuration

Add the stage_7 section to your main `config.json`:

```json
{
  "stage_7": {
    "dev_mode": true,
    "dev_max_transcripts": 2,
    "input_source": "Outputs/Refresh/detailed_classified_sections.json",
    "output_file": "enhanced_transcript_sections.json",
    "output_path": "Outputs/Refresh",
    "processing_config": {
      "batch_size": 5,
      "max_previous_speaker_blocks": 2,
      "min_paragraph_length": 50
    },
    "llm_config": {
      "token_endpoint": "https://oauth2.example.com/token",
      "base_url": "https://api.example.com/v1",
      "model": "gpt-4-turbo",
      "temperature": 0.3,
      "max_tokens": 2000,
      "timeout": 90,
      "max_retries": 3,
      "cost_per_1k_prompt_tokens": 0.03,
      "cost_per_1k_completion_tokens": 0.06
    }
  }
}
```

## Usage

### Development Mode
```bash
python stage_7_content_enhancement/7_transcript_content_enhancement.py
```

### Production Mode
Set `dev_mode: false` in configuration, then run the same command.

## Architecture

### Sliding Window Approach

1. **Previous Speaker Blocks**: Shows summaries from last 2 speaker blocks for context
2. **Current Speaker Block**: Processes in 5-paragraph batches with full context
3. **Progressive Enhancement**: Uses previously enhanced paragraphs within current block
4. **Context Separation**: Clear distinction between content to process vs contextual information

### Enhancement Output

- **Summary**: Condensed 1-2 sentence summary optimized for retrieval
- **Importance**: Relative scoring (0.0-1.0) within speaker block context

## Performance

- **Processing**: ~1 LLM call per 5-paragraph batch
- **Token Usage**: 400-600 tokens per batch
- **Cost**: ~$0.015-$0.025 per batch (5 paragraphs)
- **Rate Limiting**: 1-second delays between transcripts

## Error Handling

- **OAuth Refresh**: Per-transcript token refresh prevents expiration
- **Content Filtering**: Skips paragraphs shorter than minimum length
- **Validation**: Comprehensive response validation with retry logic
- **Null Assignment**: Sets null values for failed enhancements

## Integration

Stage 7 seamlessly integrates with the existing pipeline:
- Preserves all Stage 6 fields
- Adds minimal 2-field enhancement
- Maintains complete dataset integrity
- Follows established Stage 4/5/6 patterns

For detailed technical documentation, see `CLAUDE.md`.