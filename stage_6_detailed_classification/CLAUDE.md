# Stage 6: Detailed Classification System - Context for Claude

## Overview
Stage 6 processes Stage 5 output to add detailed financial category classification for both Management Discussion sections and Q&A conversations. Uses sophisticated speaker block windowing for Management Discussion and complete conversation analysis for Q&A groups.

## Current Status ✅ PRODUCTION READY
- **Script**: `6_transcript_detailed_classification.py` - Complete dual-approach classification system
- **Input**: `qa_paired_transcript_sections.json` from Stage 5 output
- **Output**: `detailed_classified_sections.json` with detailed financial classifications
- **Features**: OAuth authentication, SSL support, speaker block windowing, conversation analysis, comprehensive cost tracking

## Key Business Logic

### Dual Classification Approach

**Management Discussion Processing:**
- **Speaker Block Context**: Prior speaker blocks shown as 750-char previews with classifications
- **Current Speaker Block**: Always shown in full (all paragraphs, complete content)
- **5-Paragraph Windows**: Process P1-5, then P6-10, etc. within each speaker block
- **Progressive Context**: Each call sees previous classifications within current speaker block
- **LLM Only**: No fallback classifications - comprehensive LLM classification

**Q&A Group Processing:**
- **Complete Conversations**: Uses Stage 5 qa_group_id boundaries
- **Single LLM Call**: Classifies entire Q&A group in one go
- **Original Sequence**: All paragraphs in conversation order
- **Conversation-Level**: Apply classification to entire group

### Financial Content Categories

**10 Core Categories:**
1. **Financial Performance & Results**: Revenue, earnings per share, profitability metrics (ROE/ROA), net income, general financial health indicators
2. **Credit Quality & Risk Management**: Loan loss provisions, non-performing loans, charge-offs, delinquency trends, reserve coverage ratios, portfolio risk assessment
3. **Capital & Regulatory Management**: Capital adequacy ratios (CET1, Tier 1, leverage), stress test results, capital allocation decisions, regulatory requirements
4. **Strategic Initiatives & Transformation**: Digital transformation, technology investments, M&A activity, market expansion, new products/services, efficiency programs
5. **Market Environment & Outlook**: Macroeconomic commentary, interest rate environment, competitive landscape, industry trends, forward guidance/outlook
6. **Operating Efficiency & Expenses**: Operating costs, expense management, efficiency ratios, technology spend, personnel costs, productivity initiatives
7. **Asset & Liability Management**: Net interest income/margin dynamics, deposit trends and costs, loan yields, funding strategies, balance sheet optimization
8. **Non-Interest Revenue & Segments**: Fee-based income streams, trading revenues, wealth management, investment banking, divisional performance
9. **ESG & Sustainability**: Environmental commitments, social responsibility initiatives, governance improvements, climate risk management, sustainability reporting
10. **Insurance Operations**: Insurance-specific metrics including premiums, underwriting results, combined ratios, catastrophe losses, reserve development

### LLM Integration Architecture

**OAuth 2.0 Authentication Flow**
```python
# Per-transcript token refresh eliminates expiration issues
def refresh_oauth_token_for_transcript() -> bool:
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client()
        return llm_client is not None
```

**SSL Certificate Management**
- Downloads certificate from NAS at runtime (same as Stage 4/5)
- Sets SSL_CERT_FILE environment variable
- Validates certificate chain for secure connections

**Function Calling Schemas**
Two different schemas for the two processing approaches:
```json
{
  "management_discussion": {
    "paragraph_classifications": [
      {
        "paragraph_number": 1,
        "categories": ["Financial Performance & Results", "Market Environment & Outlook"],
        "confidence": 0.87
      }
    ]
  },
  "qa_conversation": {
    "conversation_classification": {
      "categories": ["Credit Quality & Risk Management", "Capital & Regulatory Management"],
      "confidence": 0.92
    }
  }
}
```

### CO-STAR Prompt Framework

**Management Discussion Prompts**:
- Full category descriptions embedded in system prompt
- Context shows prior speaker block previews (750 chars) with classifications
- Current speaker block shown in full with classification window highlighted
- No minimum/maximum category requirements - applies all relevant categories

**Q&A Conversation Prompts**:
- Complete conversation context with speaker role indicators
- Category descriptions focused on conversation-level analysis
- Analyzes both analyst questions and management responses
- Single classification applied to entire conversation

## Output Schema

### Enhanced Records
Stage 6 adds exactly 1 new field to each Stage 5 record (following Stage 4/5 pattern):

```json
{
  // Stage 5 fields (preserved)
  "section_id": 2,
  "speaker_block_id": 23,
  "paragraph_id": 145,
  "section_name": "Q&A Session",
  "speaker": "Dave McKay, President & CEO, Royal Bank of Canada",
  "section_type": "Investor Q&A",
  "section_type_confidence": 0.92,
  "section_type_method": "breakpoint_detection",
  "qa_group_id": 5,
  "qa_group_confidence": 0.89,
  "qa_group_method": "llm_detection",
  
  // New Stage 6 field
  "detailed_classification": {
    "categories": ["Credit Quality & Risk Management", "Capital & Regulatory Management"],
    "confidence": 0.87,
    "method": "speaker_block_windowing"  // or "complete_conversation"
  }
}
```

### Classification Methods
- **"speaker_block_windowing"**: Management Discussion paragraph-level classification with speaker block context
- **"complete_conversation"**: Q&A group-level classification of entire conversations

### Output File Structure
```json
{
  "schema_version": "1.0",
  "processing_timestamp": "2024-07-13T10:30:00Z",
  "total_records": 1247,
  "classification_summary": {
    "management_discussion_classified": 456,
    "qa_groups_classified": 23,
    "total_with_classifications": 791
  },
  "cost_summary": {
    "total_cost": 4.2314,
    "total_tokens": 125847,
    "total_api_calls": 89,
    "average_cost_per_call": 0.0475,
    "average_tokens_per_call": 1414
  },
  "records": [...]
}
```

## Configuration

### Stage 6 Config Section
```json
"stage_6": {
  "dev_mode": true,
  "dev_max_transcripts": 2,
  "input_source": "Outputs/Refresh/qa_paired_transcript_sections.json",
  "output_file": "detailed_classified_sections.json",
  "output_path": "Outputs/Refresh",
  "processing_config": {
    "md_paragraph_window_size": 5,
    "prior_block_preview_chars": 750,
    "max_speaker_blocks_context": 2
  },
  "llm_config": {
    "token_endpoint": "https://oauth2.example.com/token",
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4-turbo",
    "temperature": 0.1,
    "max_tokens": 1500,
    "timeout": 90,
    "max_retries": 3,
    "cost_per_1k_prompt_tokens": 0.03,
    "cost_per_1k_completion_tokens": 0.06
  },
  "description": "Detailed paragraph-level classification for Management Discussion and Q&A topic analysis"
}
```

### Environment Variables
Uses same environment variables as Stage 4/5:
```bash
# LLM Authentication
LLM_CLIENT_ID=your_client_id
LLM_CLIENT_SECRET=your_client_secret

# All other variables same as Stage 4/5
```

## Technical Implementation

### Management Discussion Processing Logic
```python
def process_management_discussion_section(md_records):
    # Group by speaker blocks
    speaker_blocks = defaultdict(list)
    for record in sorted(md_records, key=lambda x: x["paragraph_id"]):
        speaker_blocks[record["speaker_block_id"]].append(record)
    
    for block_id, block_records in speaker_blocks.items():
        # Process in 5-paragraph windows
        previous_classifications = []
        
        for window_start in range(0, len(block_records), 5):
            # Get prior blocks context (750-char previews)
            prior_blocks_context = get_prior_blocks_context(speaker_blocks, block_id)
            
            # Format context with full current block + windowed classification scope
            context = format_management_discussion_context(
                current_block_records=block_records,
                current_paragraph_window=paragraph_window,
                prior_blocks_context=prior_blocks_context,
                previous_classifications=previous_classifications
            )
            
            # Single LLM call for 5-paragraph window
            # Apply classifications and update previous_classifications
```

### Q&A Group Processing Logic
```python
def process_qa_group(qa_group_records):
    # Format complete conversation context
    conversation_context = format_qa_group_context(qa_group_records)
    
    # Single LLM call for entire conversation
    response = llm_client.chat.completions.create(...)
    
    # Apply same classification to all paragraphs in group
    for record in qa_group_records:
        record["detailed_classification"] = conversation_classification
```

### Context Formatting Functions

**Management Discussion Context**:
```python
def format_management_discussion_context(current_block_records, current_paragraph_window, 
                                       prior_blocks_context, previous_classifications):
    # Prior speaker blocks (750-char previews with classifications)
    # Current speaker block (full content always)
    # Show ALL paragraphs: [CLASSIFIED], [TO_CLASSIFY], [FUTURE]
    # Highlight current classification window
```

**Q&A Conversation Context**:
```python
def format_qa_group_context(qa_group_records):
    # Complete conversation with speaker role indicators
    # [ANALYST QUESTION], [MANAGEMENT RESPONSE], [OTHER]
    # All paragraphs in conversation order
```

### Error Handling & Recovery
- **OAuth Token Refresh**: Per-transcript refresh (eliminates expiration issues)
- **Rate Limiting**: 1-second delays between transcripts
- **No Fallback Classifications**: LLM-only approach, sets to null on failure
- **Cost Tracking**: Real-time token usage and cost monitoring with detailed summaries
- **Enhanced Error Logging**: Separate categories for LLM, authentication, classification, and processing errors

### Cost Tracking & Budget Management
- **Real-time Cost Calculation**: Per-call cost display for both MD and Q&A processing
- **Configurable Token Rates**: Set cost_per_1k_prompt_tokens and cost_per_1k_completion_tokens in config
- **Cumulative Cost Tracking**: Accumulates total cost and token usage throughout execution
- **Detailed Final Summary**: Enhanced execution report with total and average metrics
- **Per-Call Metrics**: Average cost per call and average tokens per call

## Security & Standards Compliance

### Authentication Security ✅
- **OAuth Tokens**: Never logged, securely stored in memory
- **SSL/TLS**: Certificate chain validation, TLS 1.2 minimum
- **Input Sanitization**: All data validated before API calls
- **Error Handling**: No credential exposure in logs

### Resource Management ✅
- **Connection Management**: Proper OAuth token lifecycle with per-transcript refresh
- **Memory Management**: Efficient processing of large datasets
- **File Cleanup**: SSL certificates and temp files properly cleaned

### Error Logging ✅
- **Specific Error Types**: LLM, authentication, classification, processing
- **Context Preservation**: Detailed error information with recovery guidance
- **Cost Tracking**: Real-time token usage and cost monitoring with comprehensive summaries

## Integration with Pipeline

### Stage Dependencies
- **Input**: Stage 5 must complete successfully and create `qa_paired_transcript_sections.json`
- **Output**: Creates `detailed_classified_sections.json` for downstream consumption
- **Configuration**: Extends existing config.json with stage_6 section

### Data Flow
1. **Stage 5 → Stage 6**: Paragraph-level records with section types and Q&A group boundaries
2. **Stage 6 Processing**: Dual-approach detailed classification (MD windowing + Q&A conversation)
3. **Stage 6 Output**: Enhanced records with detailed financial category classifications

## Common Usage Scenarios

### Development Testing
```bash
# Set dev_mode: true, dev_max_transcripts: 2
python stage_6_detailed_classification/6_transcript_detailed_classification.py
```

### Production Processing  
```bash
# Set dev_mode: false
python stage_6_detailed_classification/6_transcript_detailed_classification.py
```

### Classification Quality Review
- Review records with `detailed_classification.confidence < 0.7`
- Check `detailed_classification.method` distribution for processing insights
- Monitor category distribution for content analysis patterns

## Performance Characteristics

### Typical Execution
- **Management Discussion**: 1 LLM call per 5-paragraph window
- **Q&A Groups**: 1 LLM call per complete conversation
- **Token Usage**: 3,000-5,000 tokens per call (higher due to detailed categories)
- **Average Cost**: ~$0.10-$0.20 per call
- **Per-Transcript Tracking**: Real-time logging of processing time, cost, and tokens

### Scalability
- **Linear Scaling**: Performance scales with number of paragraphs and conversations
- **Rate Limiting**: Built-in throttling prevents API overload
- **Development Mode**: Safe testing with limited transcript processing
- **Cost Monitoring**: Comprehensive tracking prevents budget overruns

## Quality Assurance

### Confidence Scoring
- **High Confidence** (>0.9): Auto-accept classifications
- **Medium Confidence** (0.7-0.9): Monitor for patterns
- **Low Confidence** (<0.7): Review threshold for manual validation

### Validation Methods
- **Category Distribution**: Monitor which categories are most/least used
- **Multi-Category Analysis**: Track paragraphs/conversations with multiple categories
- **Method Consistency**: Ensure appropriate method assignment (windowing vs conversation)
- **Confidence Patterns**: Analyze confidence distributions for calibration

## Future Enhancements

### Potential Improvements
1. **Category Refinement**: Adjust categories based on usage patterns and feedback
2. **Confidence Calibration**: Fine-tune thresholds based on accuracy metrics
3. **Batch Processing**: Group similar content for efficiency
4. **Caching System**: Store classifications for repeated content patterns
5. **Progressive Classification**: Start with broad categories, then refine

### Integration Opportunities
1. **Analytics Dashboard**: Visualize category distributions and trends
2. **Comparative Analysis**: Cross-institution category pattern analysis
3. **Temporal Analysis**: Track category focus changes over time
4. **Quality Metrics**: Detailed accuracy and consistency scoring

## Troubleshooting

### Common Issues

**"OAuth token acquisition failed"**
- Check LLM_CLIENT_ID and LLM_CLIENT_SECRET environment variables
- Verify token_endpoint URL in configuration
- Check SSL certificate setup

**"Classification window processing failed"**
- Check Management Discussion content structure
- Verify speaker block grouping is correct
- Monitor for unusually long paragraphs affecting token limits

**"Q&A conversation classification failed"**
- Verify Q&A group boundaries from Stage 5
- Check conversation length doesn't exceed token limits
- Ensure proper speaker role identification

**"High classification costs"**
- Monitor token usage per call in logs
- Consider adjusting max_tokens in configuration
- Review category complexity and prompt length

## References

### Related Documentation
- **Stage 5 CLAUDE.md**: Input data structure with Q&A group boundaries
- **Stage 4 CLAUDE.md**: LLM integration patterns and OAuth authentication
- **Main README**: Pipeline overview and Stage 6 usage instructions

### Key Files
- **Main Script**: `6_transcript_detailed_classification.py`
- **Configuration**: `config.json` (stage_6 section)
- **Input**: `Outputs/Refresh/qa_paired_transcript_sections.json`
- **Output**: `Outputs/Refresh/detailed_classified_sections.json`
- **Error Logs**: `Outputs/Logs/Errors/stage_6_*_errors.json`