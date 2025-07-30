# Stage 7: LLM-Based Content Summarization

## Overview
Stage 7 processes Stage 6's classified content to create comprehensive summaries optimized for retrieval systems. It uses the production-ready architecture from Stage 5/6 with Stage 7's summarization logic.

## Configuration Required
This stage requires a configuration section in your NAS config.yaml file. See `stage_07_config_sample.yaml` for the required structure.

### Key Configuration Parameters:
- `input_data_path`: Path to Stage 6's output (stage_06_classified_content.json)
- `output_data_path`: Where to save summarized content
- `dev_mode`: Enable/disable development mode
- `dev_max_transcripts`: Number of transcripts to process in dev mode
- `llm_config`: LLM API settings including model, temperature, token costs

## Features
- **Per-Transcript OAuth Refresh**: Prevents token expiration during processing
- **Incremental Saving**: Results saved after each transcript (JSON array format)
- **Enhanced Error Logging**: Categorized error tracking
- **Failed Transcript Tracking**: Separate file for failed transcripts
- **Cost Tracking**: Real-time monitoring of LLM usage and costs
- **Development Mode**: Test with limited transcripts

## Summarization Approach
- **Management Discussion**: Single comprehensive summary for entire section
- **Q&A Groups**: Single summary per conversation (all paragraphs in group)
- **Other Content**: Basic template summaries

## Output
Creates `stage_07_summarized_content.json` with:
- All original fields from Stage 6
- `paragraph_summary`: Comprehensive summary text
- `paragraph_importance`: Importance score (0.0-1.0)
- `summary_method`: Method used (qa_group_llm, management_llm, basic_template, failed)

## Running Stage 7
```bash
python database_refresh/07_llm_summarization/main_llm_summarization.py
```

## Prerequisites
- Stage 6 must have completed successfully
- Configuration section must be added to NAS config.yaml
- All environment variables must be set (.env file)
- LLM API credentials (LLM_CLIENT_ID, LLM_CLIENT_SECRET)