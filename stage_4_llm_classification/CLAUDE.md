# Stage 4: LLM-Based Transcript Classification - Context for Claude

## Overview
Stage 4 processes Stage 3 output to add LLM-based section type classification. Uses a sophisticated 3-level progressive classification system to identify "Management Discussion" vs "Investor Q&A" content with high accuracy and comprehensive cost tracking.

## Current Status ✅ PRODUCTION READY
- **Script**: `4_transcript_llm_classification.py` - Complete 3-level classification system
- **Input**: `extracted_transcript_sections.json` from Stage 3 output
- **Output**: `classified_transcript_sections.json` with section type classifications
- **Features**: OAuth authentication, SSL support, progressive classification, comprehensive cost tracking, optimized content handling

## Key Business Logic

### 3-Level Progressive Classification System

**Level 1: Direct Section Classification**
- Analyzes entire section content using CO-STAR prompts
- Classifies as "Management Discussion", "Investor Q&A", "Mixed", or "Administrative"
- If uniform classification with high confidence → COMPLETE
- If mixed or low confidence → proceed to Level 2

**Level 2: Breakpoint Detection**
- Creates indexed list of speaker blocks within mixed section
- **Single API call** to identify exact speaker block where transition occurs
- Prompt: "At which speaker block does this transition from Management Discussion to Q&A?"
- Apply uniform classification before/after breakpoint
- If clear breakpoint with high confidence → COMPLETE
- If unclear or low confidence → proceed to Level 3

**Level 3: Contextual Individual Classification**
- Classify each speaker block individually with surrounding context
- Include up to 3 previous speaker blocks + 1 future speaker block
- Use prior classifications for consistency
- Maintains coherent narrative flow

### LLM Integration Architecture

**OAuth 2.0 Authentication Flow**
```python
# Client credentials grant flow
auth_data = {
    'grant_type': 'client_credentials',
    'client_id': LLM_CLIENT_ID,
    'client_secret': LLM_CLIENT_SECRET
}
# Use OAuth token as OpenAI API key
client = OpenAI(api_key=oauth_token, base_url=custom_base_url)
```

**SSL Certificate Management**
- Downloads certificate from NAS at runtime (like FactSet API)
- Sets SSL_CERT_FILE environment variable
- Validates certificate chain for secure connections

**Function Calling Schema**
Uses structured JSON responses with confidence scoring:
```json
{
  "classification": "Management Discussion|Investor Q&A",
  "confidence": 0.92,
  "reasoning": "Brief explanation of decision"
}
```

### CO-STAR Prompt Framework

**Context (C)**: Institution, fiscal period, section details
**Objective (O)**: Clear classification goals
**Style (S)**: Decisive, pattern-based analysis
**Tone (T)**: Professional and analytical
**Audience (A)**: Financial analysts and systems
**Response (R)**: Structured function calling

**Example Level 1 Prompt Structure**:
```xml
<context>
  <institution>Royal Bank of Canada</institution>
  <fiscal_period>2024 Q1</fiscal_period>
  <section_name>Prepared Remarks</section_name>
</context>

<objective>
Classify this earnings call section as Management Discussion, 
Investor Q&A, Mixed, or Administrative.
</objective>
```

## Output Schema

### Enhanced Records
Stage 4 adds exactly 3 new fields to each Stage 3 record:

```json
{
  // Stage 3 fields (preserved)
  "section_id": 2,
  "speaker_block_id": 23,
  "paragraph_id": 145,
  "section_name": "Q&A Session",
  "speaker": "Dave McKay, President & CEO, Royal Bank of Canada",
  "question_answer_flag": "answer",
  "paragraph_content": "Thank you for the question...",
  
  // New Stage 4 fields
  "section_type": "Investor Q&A",
  "section_type_confidence": 0.92,
  "section_type_method": "breakpoint_detection"
}
```

### Classification Methods
- **"section_uniform"**: Level 1 uniform classification
- **"breakpoint_detection"**: Level 2 transition point identification  
- **"contextual_individual"**: Level 3 speaker block analysis

### Output File Structure
```json
{
  "schema_version": "1.0",
  "processing_timestamp": "2024-07-13T10:30:00Z",
  "total_records": 1247,
  "classification_summary": {
    "management_discussion": 456,
    "investor_qa": 791,
    "other": 0
  },
  "records": [...]
}
```

## Configuration

### Stage 4 Config Section
```json
"stage_4": {
  "dev_mode": true,
  "dev_max_transcripts": 2,
  "input_source": "Outputs/Refresh/extracted_transcript_sections.json",
  "output_file": "classified_transcript_sections.json", 
  "output_path": "Outputs/Refresh",
  "llm_config": {
    "provider": "openai_custom",
    "model": "gpt-4-turbo",
    "temperature": 0.1,
    "max_tokens": 1000,
    "base_url": "https://your-llm-api.com/v1",
    "token_endpoint": "https://oauth.your-api.com/token",
    "timeout": 60,
    "max_retries": 3,
    "cost_per_1k_prompt_tokens": 0.03,
    "cost_per_1k_completion_tokens": 0.06
  },
  "classification_thresholds": {
    "min_confidence": 0.7
  },
  "content_limits": {
    "max_paragraph_chars": 750
  }
}
```

### Environment Variables (Additional)
```bash
# LLM Authentication
LLM_CLIENT_ID=your_client_id
LLM_CLIENT_SECRET=your_client_secret
```

## Technical Implementation

### Progressive Classification Logic
```python
def classify_transcript_sections(transcript_records):
    # Group by sections
    for section_records in sections:
        # Level 1: Direct classification
        level1_result = classify_section_level_1(section_records)
        
        if uniform_and_confident(level1_result):
            apply_uniform_classification(section_records, level1_result)
            continue
            
        # Level 2: Breakpoint detection
        level2_result = classify_section_level_2(section_records)
        
        if clear_breakpoint(level2_result):
            apply_breakpoint_classification(section_records, level2_result)
            continue
            
        # Level 3: Individual classification
        level3_result = classify_section_level_3(section_records)
        apply_individual_classifications(section_records, level3_result)
```

### Error Handling & Recovery
- **OAuth Token Refresh**: Automatic refresh for each transcript (eliminates expiration issues)
- **Rate Limiting**: 1-second delays between transcripts
- **Circuit Breaker**: Retry logic with exponential backoff
- **Fallback Classifications**: Default to "Management Discussion" on errors
- **Cost Tracking**: Real-time token usage and cost monitoring with detailed summaries

### Cost Tracking & Budget Management
- **Real-time Cost Calculation**: Per-call cost display for Level 1, 2, and 3 classifications
- **Configurable Token Rates**: Set cost_per_1k_prompt_tokens and cost_per_1k_completion_tokens in config
- **Cumulative Cost Tracking**: Accumulates total cost and token usage throughout execution
- **Final Cost Summary**: Detailed execution report with total cost, token usage, and efficiency metrics
- **Example Output**: "Total tokens used: 125,847 | Total LLM cost: $4.2314 | Average cost per 1K tokens: $0.0336"

### Performance Optimization
- **Batch Processing**: Groups records by transcript for efficiency
- **Progressive Efficiency**: 70-80% resolved at Level 1, 15-20% at Level 2, 5-10% at Level 3
- **API Cost Minimization**: Only escalates to higher levels when needed
- **Content Optimization**: Full sections sent with paragraph-level character limits (750 chars per paragraph)
- **Development Mode**: Limits transcripts processed for testing
- **Cost Tracking**: Configurable token costs with real-time budget monitoring

## Security & Standards Compliance

### Authentication Security ✅
- **OAuth Tokens**: Never logged, securely stored in memory
- **SSL/TLS**: Certificate chain validation, TLS 1.2 minimum
- **Input Sanitization**: All data validated before API calls
- **Error Handling**: No credential exposure in logs

### Resource Management ✅
- **Connection Management**: Proper OAuth token lifecycle
- **Memory Management**: Efficient processing of large datasets
- **File Cleanup**: SSL certificates and temp files properly cleaned

### Error Logging ✅
- **Specific Error Types**: LLM, authentication, classification, processing
- **Context Preservation**: Detailed error information with recovery guidance
- **Cost Tracking**: Real-time token usage and cost monitoring with detailed summaries

## Integration with Pipeline

### Stage Dependencies
- **Input**: Stage 3 must complete successfully and create `extracted_transcript_sections.json`
- **Output**: Creates `classified_transcript_sections.json` for downstream consumption
- **Configuration**: Extends existing config.json with stage_4 section

### Data Flow
1. **Stage 3 → Stage 4**: Paragraph-level records with section/speaker structure
2. **Stage 4 Processing**: 3-level progressive LLM classification
3. **Stage 4 Output**: Enhanced records with section_type classifications

## Common Usage Scenarios

### Development Testing
```bash
# Set dev_mode: true, dev_max_transcripts: 2
python stage_4_llm_classification/4_transcript_llm_classification.py
```

### Production Processing  
```bash
# Set dev_mode: false
python stage_4_llm_classification/4_transcript_llm_classification.py
```

### Classification Quality Review
- Review records with `section_type_confidence < 0.7`
- Check `section_type_method` distribution for processing insights
- Monitor error logs for systematic issues

## Performance Characteristics

### Typical Execution
- **Level 1 Success**: 1 API call per section (70-80% of cases)
- **Level 2 Success**: 2 API calls per section (15-20% of cases)  
- **Level 3 Required**: 2 + N API calls per section (5-10% of cases)
- **Average Cost**: ~1.5 API calls per section across all levels
- **Token Usage**: 5,000-15,000 tokens per call (increased from previous 300-500 for better accuracy)
- **Cost Summary**: Final execution report shows total tokens, total cost, and average cost per 1K tokens

### Scalability
- **Linear Scaling**: Performance scales with number of sections and paragraphs
- **Rate Limiting**: Built-in throttling prevents API overload
- **Batch Processing**: Efficient grouping minimizes overhead
- **Development Mode**: Safe testing with limited transcript processing

## Quality Assurance

### Confidence Scoring
- **High Confidence** (>0.9): Auto-accept classifications
- **Medium Confidence** (0.7-0.9): Monitor for patterns
- **Low Confidence** (<0.7): Review threshold for manual validation

### Validation Methods
- **Cross-Section Consistency**: Ensure logical classification patterns
- **Speaker Role Validation**: Check classifications align with speaker types
- **Transition Logic**: Verify breakpoint detections make sense
- **Method Distribution**: Monitor which levels are used most frequently

## Future Enhancements

### Potential Improvements
1. **Confidence Calibration**: Fine-tune thresholds based on accuracy metrics
2. **Caching System**: Store classifications for repeated content
3. **Batch API Calls**: Group multiple sections for efficiency
4. **Model Fine-tuning**: Train on domain-specific examples
5. **Real-time Processing**: Process transcripts as they become available

### Integration Opportunities
1. **Feedback Loop**: Learn from manual corrections
2. **Quality Metrics**: Detailed accuracy and consistency scoring
3. **Comparative Analysis**: Cross-institution classification patterns
4. **Temporal Analysis**: Track classification changes over time

## Troubleshooting

### Common Issues

**"OAuth token acquisition failed"**
- Check LLM_CLIENT_ID and LLM_CLIENT_SECRET environment variables
- Verify token_endpoint URL in configuration
- Check SSL certificate setup

**"SSL certificate setup failed"**
- Ensure certificate.cer exists in NAS Inputs/certificate/ folder
- Check NAS connectivity and file permissions
- Verify certificate is valid and not expired

**"Low confidence classifications"**
- Review transcript content quality and structure
- Check for unusual section naming or speaker patterns
- Consider adjusting classification thresholds

**"High API error rates"**
- Check OAuth token refresh frequency
- Monitor rate limiting and add delays if needed
- Verify LLM service availability and quotas

## References

### Related Documentation
- **Stage 3 CLAUDE.md**: Input data structure and processing flow
- **Main README**: Pipeline overview and Stage 4 usage instructions
- **Configuration Schema**: Complete config.json structure with LLM settings

### Key Files
- **Main Script**: `4_transcript_llm_classification.py`
- **Configuration**: `config.json` (stage_4 section)
- **Input**: `Outputs/Refresh/extracted_transcript_sections.json`
- **Output**: `Outputs/Refresh/classified_transcript_sections.json`
- **Error Logs**: `Outputs/Logs/Errors/stage_4_*_errors.json`