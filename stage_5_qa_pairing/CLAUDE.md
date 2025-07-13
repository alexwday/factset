# Stage 5: Q&A Pairing System - Context for Claude

## Overview
Stage 5 processes Stage 4 output to identify and group question-answer conversation boundaries using state-driven speaker block analysis with automatic group ID management. Creates Q&A relationship mappings while maintaining paragraph-level granularity.

## Current Status ✅ PRODUCTION READY (Latest Updates 2024-07-13)
- **Script**: `5_transcript_qa_pairing.py` - Complete Q&A boundary detection system
- **Input**: `classified_transcript_sections.json` from Stage 4 output  
- **Output**: `qa_paired_transcript_sections.json` with Q&A group assignments
- **Recent Major Improvements**:
  - **State-Driven Architecture**: Dynamic tool definitions based on current group state
  - **Automatic Group ID Assignment**: Sequential numbering prevents invalid sequences
  - **Enhanced Operator Detection**: Content pattern matching excludes operator transitions
  - **Real-Time Validation**: Built-in state machine with auto-correction and retry logic
  - **Simplified Prompting**: Speaker pattern focus vs complex thematic analysis
  - **Clean Logging**: Eliminated false warnings about decision inconsistencies
  - **Fixed Q&A Groups Counting**: Proper per-transcript accumulation (replaced buggy set-based deduplication)
  - **Enhanced Execution Metrics**: Added timing tracking and per-file cost/timing metrics
  - **Aligned Summary Format**: Consistent execution summary format with Stage 4

## Key Business Logic

### State-Driven Speaker Block Analysis (2024-07-13 Architecture)
Stage 5 uses a sophisticated state machine to ensure consistent Q&A group formation:
- **Speaker Block Focus**: All paragraphs spoken by one person in sequence
- **State Awareness**: System knows if group is active/inactive and valid decision options
- **Automatic ID Assignment**: Sequential group numbering (1,2,3...) prevents invalid sequences
- **Real-Time Validation**: Invalid LLM responses caught and retried with feedback
- **Enhanced Operator Detection**: Content pattern matching for transition statements
- **Complete Exchange Capture**: Analyst question → executive response → follow-up → closing

### Sliding Window Architecture
```python
SPEAKER_BLOCK_WINDOW_CONFIG = {
    "context_blocks_before": 3,    # Previous speaker blocks for pattern recognition
    "context_blocks_after": 2,     # Future speaker blocks for boundary validation
    "dynamic_extension": True,     # Extend window back to question start when needed
    "full_paragraphs": True        # No character limits - send complete content
}
```

**Dynamic Window Extension Logic**:
- **Standard Window**: 3 blocks before + current + 2 blocks after
- **Extended Window**: When analyzing question end, includes ALL blocks back to question start
- **No Overlap**: Uses complete speaker blocks, no overlap needed between windows

### Q&A Group Detection Process

**1. Filter to Q&A Sections**
- Only processes speaker blocks where `section_type = "Investor Q&A"`
- Leverages Stage 4 section classification results

**2. Sequential Speaker Block Analysis**
- Analyzes each speaker block for boundary decisions
- Uses sliding window context with dynamic extension
- Makes single LLM call per speaker block

**3. Boundary Decision Types**
- **group_start**: Speaker block begins new Q&A exchange
- **group_continue**: Speaker block continues current Q&A exchange  
- **group_end**: Speaker block ends current Q&A exchange
- **standalone**: Speaker block is isolated (rare in Q&A sections)

**4. Automatic Group ID Assignment & Formation**
- **System-Controlled IDs**: LLM only decides action (start/continue/end), system assigns sequential IDs
- **State Machine Enforcement**: Prevents invalid transitions with real-time correction
- **Group Formation**: Consecutive speaker blocks with same auto-assigned `qa_group_id`
- **Aggregated Confidence**: Weighted average across group decisions
- **Operator Exclusion**: Standalone blocks skipped in group formation to prevent false warnings

## Major Architectural Improvements (2024-07-13)

### 1. State-Driven Decision Logic ✅
**Problem Solved**: Inconsistent LLM decisions and duplicate group endings
**Solution**: Dynamic tool definitions that change based on current state

```python
# Current State Awareness
if current_group_active:
    valid_options = "CONTINUE current group OR END current group"
    status_enum = ["group_continue", "group_end", "standalone"]
else:
    valid_options = "START new group OR mark as STANDALONE"
    status_enum = ["group_start", "standalone"]
```

**Benefits**:
- LLM physically cannot choose invalid transitions
- Clear decision options eliminate confusion
- State machine prevents duplicate endings

### 2. Automatic Group ID Management ✅
**Problem Solved**: Invalid ID sequences like 1,3,2,3 causing validation warnings
**Solution**: System-controlled sequential assignment

```python
# BEFORE: LLM chose arbitrary IDs
"qa_group_id": {"type": "integer"}  # Could be anything!

# AFTER: System assigns automatically  
# (qa_group_id removed from LLM schema entirely)
def assign_group_id_to_decision(decision, qa_state):
    if status == "group_start":
        return qa_state["next_group_id"]  # 1, 2, 3, 4...
```

**Benefits**:
- Guaranteed sequential numbering (1,2,3,4...)
- No more invalid sequences or gaps
- Real-time ID validation

### 3. Enhanced Operator Detection ✅
**Problem Solved**: Operators saying "Thank you, next question is from X" tagged with Q&A groups
**Solution**: Content pattern matching with precedence over pleasantries

```python
operator_content_patterns = [
    "next question", "question comes from", "question is from",
    "please hold", "concludes our question", "end of q&a"
]

# Special case: "thank you" + "next question" = operator
if "thank you" in content and "next question" in content:
    return True  # Operator, not pleasantries
```

**Benefits**:
- All operator transitions properly excluded
- No false Q&A group assignments
- Clean separation of call management vs Q&A content

### 4. Real-Time Validation & Auto-Correction ✅
**Problem Solved**: False warnings about "decision inconsistencies"
**Solution**: Built-in validation with single retry and auto-correction

```python
def validate_llm_decision(decision, qa_state):
    if validation_error:
        return retry_with_validation_feedback()  # Single retry
    
    # Auto-assign proper group ID
    return assign_group_id_to_decision(decision, qa_state)
```

**Benefits**:
- Invalid responses caught and retried immediately
- Auto-correction for edge cases
- Clean logs with only genuine issues

### 5. Simplified Prompting Strategy ✅
**Problem Solved**: Complex thematic analysis causing conflicting guidance
**Solution**: Focus on speaker patterns and complete exchange capture

```python
# BEFORE: Complex thematic analysis
"Look for theme changes, topic shifts, content coherence..."

# AFTER: Simple speaker pattern focus
"ANALYST speakers start Q&A, EXECUTIVE speakers respond, 
OPERATOR speakers manage call flow - always STANDALONE"
```

**Benefits**:
- Consistent decision making
- Eliminates conflicting instructions
- Better capture of complete exchanges

## LLM Integration Architecture

### Per-Transcript OAuth Refresh
```python
def refresh_oauth_token_for_transcript() -> bool:
    """Refresh OAuth token for each new transcript."""
    new_token = get_oauth_token()
    if new_token:
        oauth_token = new_token
        llm_client = setup_llm_client()  # Update client with fresh token
        return llm_client is not None
    return False
```

**Refresh Strategy**:
- Fresh OAuth token for every transcript (eliminates expiration issues)
- More frequent than Stage 4's every-5-transcripts approach
- Proper error handling if token refresh fails

### Sophisticated Context Formatting
```python
def format_speaker_block_context(window_blocks, current_block_index):
    """
    Create structured context with clear decision boundaries:
    
    === PRIOR SPEAKER BLOCK CONTEXT ===
    SPEAKER BLOCK 15: [Full content]
    SPEAKER BLOCK 16: [Full content]
    
    === CURRENT SPEAKER BLOCK (DECISION POINT) ===
    **ANALYZE THIS BLOCK FOR Q&A BOUNDARY DECISION:**
    SPEAKER BLOCK 17: [Full content]
    
    === FUTURE SPEAKER BLOCK CONTEXT ===
    SPEAKER BLOCK 18: [Full content]
    SPEAKER BLOCK 19: [Full content]
    """
```

### Enhanced CO-STAR Prompt Framework (Updated 2024-07-13)
**Context (C)**: State-aware earnings call analysis with current group status
**Objective (O)**: Make single decision based on current state (continue/end vs start/standalone)
**Style (S)**: Focus on speaker patterns and complete exchange capture
**Tone (T)**: Conservative - capture complete analyst→executive conversations
**Audience (A)**: Financial analysts requiring accurate Q&A relationship mapping
**Response (R)**: Simplified function calling with automatic ID assignment

**Key Prompt Improvements**:
- **State Context**: Shows current active group and valid decision options
- **Speaker Pattern Focus**: Clear analyst/executive/operator role guidance
- **Exchange Completion Rules**: Specific guidance on when to continue vs end groups
- **Operator Handling**: Explicit instructions to mark operators as standalone
- **Concise Reasoning**: 100-character limit for faster response generation

### Dynamic Function Calling Schema (Updated 2024-07-13)
```python
# Schema changes based on current state!
def create_qa_boundary_detection_tools(qa_state):
    if current_group_active:
        # Can only continue, end, or mark standalone
        status_enum = ["group_continue", "group_end", "standalone"]
        description = f"Continue or end active Q&A group {current_group_id}"
    else:
        # Can only start new or mark standalone
        status_enum = ["group_start", "standalone"]
        description = "Start new Q&A group or mark operator as standalone"
    
    # qa_group_id REMOVED - system assigns automatically!
    return {
        "qa_group_decision": {
            "current_block_id": int,
            "group_status": status_enum,  # Dynamic based on state!
            "confidence_score": float,
            "reasoning": str  # Max 100 chars for efficiency
        }
    }
```

**Key Schema Improvements**:
- **Dynamic Status Options**: LLM can only choose valid transitions
- **Removed Group ID**: System assigns automatically, prevents invalid sequences
- **State-Aware Descriptions**: Tool description changes based on current context
- **Simplified Response**: Fewer required fields, faster generation

## Output Schema

### Minimal Field Addition
Stage 5 adds exactly **3 new fields** to each Stage 4 record:

```json
{
  // Stage 4 fields (preserved)
  "section_id": 2,
  "speaker_block_id": 23,
  "paragraph_id": 145,
  "section_type": "Investor Q&A",
  "section_type_confidence": 0.92,
  "section_type_method": "section_uniform",
  
  // New Stage 5 fields (MINIMAL ADDITION)
  "qa_group_id": 1,                    // Groups related Q&A exchanges
  "qa_group_confidence": 0.87,         // Aggregated confidence across group
  "qa_group_method": "llm_detection"   // Detection method used
}
```

### Q&A Group Assignment Logic
- **Q&A Sections Only**: Only records with `section_type = "Investor Q&A"` get assignments
- **Null Values**: Non-Q&A sections get `null` for all three Q&A fields
- **Group Consistency**: All paragraphs within same speaker block range get same Q&A group ID

### Confidence Aggregation Strategy
```python
def calculate_qa_group_confidence(block_decisions):
    """
    Weighted average based on decision criticality:
    - Group start/end decisions: Higher weight (0.4 each)
    - Group continuation decisions: Lower weight (0.2 / num_continuations)
    """
```

**Example**: Q&A group spanning 4 speaker blocks
- Block 15 (group_start): confidence 0.9, weight 0.4 
- Block 16 (group_continue): confidence 0.8, weight 0.1
- Block 17 (group_continue): confidence 0.8, weight 0.1  
- Block 18 (group_end): confidence 0.9, weight 0.4
- **Final confidence**: (0.9×0.4 + 0.8×0.1 + 0.8×0.1 + 0.9×0.4) = 0.88

### Q&A Group Methods
- **"llm_detection"**: High confidence LLM boundary detection (≥0.8)
- **"llm_detection_medium_confidence"**: Medium confidence LLM detection (0.6-0.79)
- **"xml_fallback"**: Fallback to XML type attributes when LLM fails

## Configuration

### Stage 5 Config Section
```json
"stage_5": {
  "dev_mode": true,
  "dev_max_transcripts": 2,
  "input_source": "Outputs/Refresh/classified_transcript_sections.json",
  "output_file": "qa_paired_transcript_sections.json",
  "output_path": "Outputs/Refresh",
  "window_config": {
    "context_blocks_before": 3,
    "context_blocks_after": 2,
    "dynamic_extension": true,
    "full_paragraphs": true
  },
  "llm_config": {
    "provider": "openai_custom",
    "model": "gpt-4-turbo", 
    "temperature": 0.1,
    "max_tokens": 1500,
    "base_url": "https://your-llm-api.com/v1",
    "token_endpoint": "https://oauth.your-api.com/token",
    "timeout": 60,
    "max_retries": 3,
    "cost_per_1k_prompt_tokens": 0.03,
    "cost_per_1k_completion_tokens": 0.06
  }
}
```

**Key Configuration Differences from Stage 4**:
- **Cost Tracking for Reporting**: Includes cost per token configuration for usage reporting (no budget limits)
- **Larger max_tokens**: 1500 vs 1000 (longer context windows)
- **Window Configuration**: Speaker block-based window settings

## Error Handling & Fallback Strategies

### Failure Categories and Recovery

#### **1. LLM Decision Inconsistencies**
```python
def detect_decision_inconsistencies(decisions):
    """
    Detect logical inconsistencies:
    - Group end followed by same group continuation
    - Group start without proper sequence
    """
```

**Recovery**: Extended context window retry, then XML fallback

#### **2. OAuth Token Failures** 
```python
def refresh_oauth_token_for_transcript():
    """Per-transcript token refresh with failure handling."""
```

**Recovery**: Return original records without Q&A assignments

#### **3. Q&A Group Validation Issues**
```python
def validate_qa_group_completeness(qa_groups, speaker_blocks):
    """
    Ensure logical question-answer patterns:
    - Groups have both analyst and executive speakers
    - Proper speaker role distribution
    """
```

**Recovery**: Conservative XML-based grouping

#### **4. Speaker Block Context Issues**
**Problem**: Q&A group spans beyond available context window
**Detection**: Question start earlier than available context
**Recovery**: Dynamic window extension back to question start

### Fallback Hierarchy
1. **Primary**: LLM speaker block boundary detection
2. **Secondary**: Extended context window retry for inconsistencies  
3. **Tertiary**: XML type attribute grouping (`type="q"`, `type="a"`)
4. **Final**: Return original records with null Q&A assignments

### XML Fallback Strategy
```python
def apply_xml_fallback_grouping(speaker_blocks):
    """
    Conservative grouping using XML type attributes:
    - type="question" starts new Q&A group
    - type="answer" continues current group
    - Confidence set to 0.5 for all fallback groups
    """
```

## Technical Implementation

### Core Processing Flow
```python
def process_transcript_qa_pairing(transcript_records, transcript_id):
    # 1. Refresh OAuth token for transcript
    refresh_oauth_token_for_transcript()
    
    # 2. Group records by speaker blocks  
    speaker_blocks = group_records_by_speaker_block(transcript_records)
    
    # 3. Process Q&A boundaries with fallbacks
    qa_groups = process_qa_boundaries_with_fallbacks(speaker_blocks, transcript_id)
    
    # 4. Apply Q&A assignments to paragraph records
    enhanced_records = apply_qa_assignments_to_records(transcript_records, qa_groups)
    
    return enhanced_records
```

### Dynamic Context Window Logic
```python
def create_speaker_block_window(current_block_index, speaker_blocks, qa_state):
    """
    Standard: current_index - 3 to current_index + 2
    Extended: question_start_index to current_index + 2 (for question endings)
    """
    if qa_state and qa_state.get("extends_question_start"):
        question_start_index = qa_state.get("question_start_index", 0)
        start_index = max(0, question_start_index)
    else:
        start_index = max(0, current_block_index - 3)
```

### Enhanced Error Logging
- **Boundary Detection Errors**: Failed LLM analysis with speaker block context
- **Authentication Errors**: OAuth token refresh failures
- **Validation Errors**: Q&A group consistency issues  
- **Processing Errors**: General transcript processing failures

Each error type saved to separate JSON files with recovery guidance.

## Integration with Pipeline

### Stage Dependencies
- **Input**: Stage 4 must complete successfully with section type classifications
- **Requirement**: Records must have `section_type = "Investor Q&A"` for processing
- **Output**: Enhanced records with Q&A relationship metadata for downstream analysis

### Data Flow
1. **Stage 4 → Stage 5**: Paragraph records with section classifications
2. **Stage 5 Processing**: Speaker block Q&A boundary detection  
3. **Stage 5 Output**: Enhanced records with Q&A group assignments

### Backward Compatibility
- **Preserves All Fields**: All Stage 4 fields maintained exactly
- **Additive Only**: Only adds 3 new fields, no modifications
- **Null Handling**: Non-Q&A sections get null Q&A assignments

## Common Usage Scenarios

### Development Testing
```bash
# Set dev_mode: true, dev_max_transcripts: 2
python stage_5_qa_pairing/5_transcript_qa_pairing.py
```

### Production Processing
```bash  
# Set dev_mode: false
python stage_5_qa_pairing/5_transcript_qa_pairing.py
```

### Q&A Quality Review
- Review records with `qa_group_confidence < 0.7`
- Check `qa_group_method` distribution (prefer "llm_detection")
- Monitor error logs for systematic boundary detection issues

## Performance Characteristics

### Typical Execution
- **1 LLM call per speaker block** in Q&A sections
- **Per-transcript OAuth refresh** (eliminates token expiration)
- **Dynamic context windows** (3-7 speaker blocks depending on extension)
- **Conservative fallbacks** (prefer XML grouping over failed LLM analysis)
- **Comprehensive Timing Tracking**: Start/end time with execution duration reporting
- **Per-File Metrics**: Real-time logging of processing time, cost, tokens, and Q&A groups per transcript
- **Enhanced Execution Summary**: Includes total execution time, average time per transcript, and average cost per transcript

### Scalability
- **Linear with Q&A speaker blocks**: Performance scales with Q&A content volume
- **Per-transcript isolation**: Each transcript processed independently
- **Memory efficient**: Processes one transcript at a time
- **Development mode**: Safe testing with transcript limits

## Quality Assurance

### Confidence Scoring
- **High Confidence** (≥0.8): Auto-accept Q&A groupings (`llm_detection`)
- **Medium Confidence** (0.6-0.79): Monitor patterns (`llm_detection_medium_confidence`)
- **Low Confidence** (<0.6): XML fallback used (`xml_fallback`)

### Validation Methods
- **Decision Consistency**: Check for logical boundary sequences
- **Group Completeness**: Ensure analyst-executive speaker patterns
- **Speaker Role Validation**: Verify speakers align with question/answer roles
- **Method Distribution**: Monitor LLM vs fallback usage rates

## Future Enhancements

### Potential Improvements
1. **Multi-turn Conversation Tracking**: Better handling of back-and-forth exchanges
2. **Speaker Role Learning**: Fine-tune speaker type detection
3. **Confidence Calibration**: Optimize confidence thresholds based on accuracy
4. **Batch Processing**: Group multiple speaker blocks for efficiency
5. **Context Caching**: Cache speaker block analysis for repeated content

### Integration Opportunities  
1. **Retrieval Enhancement**: Use Q&A groups for coherent response generation
2. **Analytics Dashboard**: Q&A pattern analysis across institutions
3. **Content Summarization**: Q&A-aware summarization for key exchanges
4. **Temporal Analysis**: Track Q&A pattern changes over time

## Troubleshooting

### Common Issues (Updated 2024-07-13)

**"Group ID mismatch in decision sequence" (RESOLVED)**
- **Root Cause**: Standalone operator blocks with qa_group_id=None triggered false warnings
- **Fix Applied**: Enhanced group formation logic skips standalone blocks entirely
- **Result**: Clean processing logs with no false warnings

**"Invalid ID sequences like 1,3,2,3" (RESOLVED)**
- **Root Cause**: LLM was choosing group IDs freely without sequential logic
- **Fix Applied**: Automatic sequential ID assignment (1,2,3,4...)
- **Result**: Guaranteed proper group numbering

**"Operators tagged with Q&A groups" (RESOLVED)**
- **Root Cause**: "Thank you, next question" caught by pleasantries vs operator detection
- **Fix Applied**: Enhanced operator content pattern matching with precedence
- **Result**: All operator transitions properly excluded

**"OAuth token refresh failed for transcript"**
- Check LLM_CLIENT_ID and LLM_CLIENT_SECRET environment variables
- Verify token_endpoint URL in stage_5 configuration
- Check SSL certificate setup and validity

**"No Q&A sections found in transcript"**
- Verify Stage 4 section classification completed successfully
- Check that input data has `section_type = "Investor Q&A"` records
- Review Stage 4 classification accuracy for transcript

**"Invalid LLM decision - retry with validation feedback"**
- System automatically retries with state-specific feedback
- Monitor if single retry resolves the issue
- Check for systematic prompt interpretation issues

**"High XML fallback usage"**
- Review LLM model performance and prompt effectiveness
- Check for systematic speaker block formatting issues
- Consider adjusting confidence thresholds

### Resolved Issues Archive

#### Issue: Duplicate Group Endings (2024-07-13)
**Symptoms**: Multiple consecutive "group_end" decisions for same group
**Root Cause**: LLM marked executive answer as "group_end", then analyst "thank you" as another "group_end"
**Solution**: State machine prevents duplicate endings, enhanced operator detection
**Status**: ✅ RESOLVED

#### Issue: Invalid Group ID Sequences (2024-07-13)
**Symptoms**: Group IDs like 1,3,2,3 causing validation warnings
**Root Cause**: LLM freely choosing group IDs without sequential logic
**Solution**: Automatic sequential assignment, removed ID choice from LLM
**Status**: ✅ RESOLVED

#### Issue: False Validation Warnings (2024-07-13)
**Symptoms**: "Decision inconsistencies detected" despite correct output
**Root Cause**: Post-processing validation didn't account for real-time corrections
**Solution**: Smart validation distinguishes auto-corrections from real errors
**Status**: ✅ RESOLVED

#### Issue: Q&A Groups Under-Counting (2024-07-13)
**Symptoms**: Summary showed only 3 Q&A groups when 5 transcripts had 3 groups each (should be 15)
**Root Cause**: Set-based deduplication treated Group 1 from different transcripts as same group
**Solution**: Proper per-transcript accumulation replaces buggy set-based counting
**Status**: ✅ RESOLVED

#### Issue: Missing Execution Time Tracking (2024-07-13)
**Symptoms**: Stage 5 lacked timing metrics unlike Stage 4
**Root Cause**: No start/end time tracking implemented
**Solution**: Added comprehensive timing with execution duration and per-file averages
**Status**: ✅ RESOLVED

## References

### Related Documentation
- **Stage 4 CLAUDE.md**: Section classification input requirements
- **Main README**: Pipeline overview and Stage 5 integration
- **Configuration Schema**: Complete config.json structure with stage_5 settings

### Key Files
- **Main Script**: `5_transcript_qa_pairing.py`
- **Configuration**: `config.json` (stage_5 section)
- **Input**: `Outputs/Refresh/classified_transcript_sections.json`
- **Output**: `Outputs/Refresh/qa_paired_transcript_sections.json`
- **Error Logs**: `Outputs/Logs/Errors/stage_5_*_errors.json`