# Stage 5 Q&A Pairing: Example LLM Prompt

## Scenario Context
This example shows what the LLM sees when analyzing speaker block 23 in a Royal Bank of Canada earnings call. The system has detected an active Q&A group (ID: 3) and is deciding whether the current executive response should continue or end the group.

---

## Complete LLM Prompt

**CONTEXT**: You are analyzing earnings call speaker blocks to determine Q&A conversation boundaries. Each speaker block contains one person's complete statement with their role and content clearly marked. Your goal is to capture complete conversational exchanges between question askers and institution responders.

**CURRENT STATE**:
- Q&A Group Active: YES
- Active Group ID: 3
- Last Decision: group_continue
- Next Block Preview: Thank you, that's very helpful

**LOOK-AHEAD ALERT**: Next block contains closing pleasantries - consider including in current group if active.

**OBJECTIVE**: For the current speaker block (marked as DECISION POINT), determine:
1. Does this block start a new Q&A group?
2. Does this block continue an existing Q&A group?  
3. Does this block end the current Q&A group?
4. What is the complete Q&A group span (start_block_id to end_block_id)?

**STYLE**: Analyze complete speaker blocks as units. Focus on complete conversational exchanges:
- **Complete Exchanges**: Capture entire back-and-forth between question asker and institution responder
- **Related Follow-ups**: If follow-up question is on the same theme/topic, include in current Q&A group
- **New Themes**: If follow-up question introduces different theme/topic, start new Q&A group
- **Speaker Roles**: [ANALYST] ask questions, [EXECUTIVE] answer, [OPERATOR] manage call flow
- **Operator Exclusion**: [OPERATOR] blocks should NEVER be assigned Q&A group IDs - they are call management only

**TONE**: Conservative and methodical. Prioritize:
1. Complete conversational exchanges over artificial splits
2. Thematic coherence - related questions/answers stay together
3. Clear boundaries when themes change or new participants enter
4. Operator blocks remain unassigned (no Q&A group)

**AUDIENCE**: Financial analysts who need accurate question-answer relationship mapping for research and analysis.

**RESPONSE**: Use the analyze_speaker_block_boundaries function with structured output.

=== PRIOR SPEAKER BLOCK CONTEXT ===

SPEAKER BLOCK 20:
  Speaker: Mike Mayo - Wells Fargo Securities [ANALYST - typically asks questions]
  XML Role: question
  Section Type: Investor Q&A
  Content:
    • Thanks Dave. Two questions on capital allocation. First, you mentioned the CET1 ratio target of 12.5%. 
    • Given the strong capital generation we're seeing, are you considering raising that target?
    • And second, on the dividend policy, should we expect any changes to the current approach?

SPEAKER BLOCK 21:
  Speaker: Dave McKay - CEO, Royal Bank of Canada [EXECUTIVE - typically provides answers]
  XML Role: answer
  Section Type: Investor Q&A
  Content:
    • Thanks Mike. On the CET1 ratio, we're comfortable with our current 12.5% target.
    • We've built significant capital buffers and continue to generate strong organic capital.
    • The regulatory environment remains stable, so we don't see a need to raise that target at this time.

SPEAKER BLOCK 22:
  Speaker: Dave McKay - CEO, Royal Bank of Canada [EXECUTIVE - typically provides answers]
  XML Role: answer
  Section Type: Investor Q&A
  Content:
    • Regarding the dividend policy, we remain committed to our progressive dividend approach.
    • We've increased the dividend for 13 consecutive years and expect to continue that track record.
    • Our payout ratio remains within our target range of 40-50%, giving us flexibility for future increases.

=== CURRENT SPEAKER BLOCK (DECISION POINT) ===
**ANALYZE THIS BLOCK FOR Q&A BOUNDARY DECISION:**

SPEAKER BLOCK 23:
  Speaker: Dave McKay - CEO, Royal Bank of Canada [EXECUTIVE - typically provides answers]
  XML Role: answer
  Section Type: Investor Q&A
  Content:
    • So in summary Mike, we're well positioned from both a capital and dividend perspective.
    • The 12.5% CET1 target gives us the right balance between growth investment and shareholder returns.
    • We'll continue to evaluate our capital allocation strategy as market conditions evolve.

=== FUTURE SPEAKER BLOCK CONTEXT ===

SPEAKER BLOCK 24:
  Speaker: Mike Mayo - Wells Fargo Securities [ANALYST - typically asks questions]
  XML Role: general
  Section Type: Investor Q&A
  Content:
    • Thank you, that's very helpful.

SPEAKER BLOCK 25:
  Speaker: Operator [OPERATOR - manages call flow]
  XML Role: general
  Section Type: Investor Q&A
  Content:
    • Thank you. Our next question comes from Meny Grauman with Scotiabank.

**CRITICAL ANALYSIS INSTRUCTIONS**:

**STATE-AWARE DECISION LOGIC (MANDATORY)**:
- If NO active group: Look for question starts or operator management
- If active group: Look for continuation, completion, or premature interruption
- If group just ended: Don't create duplicate endings - look for new starts or pleasantries
- Use next block preview to make better ending decisions

**OPERATOR BLOCK HANDLING (MANDATORY)**:
- If current block speaker contains "operator" → assign qa_group_id: null, group_status: "standalone"
- Operator blocks manage call flow, introduce speakers, handle technical issues
- NEVER assign operator blocks to Q&A groups - they interrupt but don't participate in Q&A

**CONVERSATION FLOW PRIORITIES**:
1. **Complete Exchange Capture**: Ensure full question→answer→follow-up sequences stay together
2. **Thematic Coherence**: 
   - Same topic/theme follow-ups → continue current Q&A group
   - Different topic/theme questions → start new Q&A group
3. **Natural Boundaries with Look-ahead**:
   - "Thank you" + next block is operator/new speaker → end current group
   - "Thank you" + next block continues theme → include thanks in group
   - New analyst introduction → start new group (unless thanking previous)
   - Operator transitions → natural break points

**CLOSING PLEASANTRIES HANDLING**:
- If current block contains "thank you", "appreciate", "helpful" AND active group exists:
  - Check next block: if operator/new topic → include thanks and end group
  - If unclear → include thanks in current group for completeness
- Avoid creating separate groups for brief thanks - attach to preceding Q&A

**SPEAKER TRANSITION ANALYSIS**:
- Analyst→Executive: Typically question→answer flow (continue group)
- Executive→Analyst (same person): Often clarification/follow-up (continue if same theme)
- Executive→Analyst (different person): Usually new question (evaluate theme)
- Any→Operator: Call management, not Q&A content (operator gets no assignment)
- Operator→Any: Introduction to new exchange (next block likely starts new group)

**RESPONSE FORMAT**:
- Keep reasoning brief (max 100 characters): focus on key decision factors
- Examples: "Analyst Q start", "Exec continues A", "Theme shift to guidance", "Operator transition"

**CONFIDENCE SCORING GUIDANCE**:
- High (0.8-1.0): Clear speaker patterns, obvious theme boundaries, definitive operator blocks
- Medium (0.6-0.79): Some ambiguity in theme continuation or speaker roles
- Low (0.4-0.59): Unclear boundaries, mixed themes, uncertain speaker roles
- Very Low (<0.4): Highly ambiguous content requiring XML fallback

---

## Expected LLM Response

The LLM would use the `analyze_speaker_block_boundaries` function to respond with something like:

```json
{
  "qa_group_decision": {
    "current_block_id": 23,
    "qa_group_id": 3,
    "qa_group_start_block": 20,
    "qa_group_end_block": 24,
    "group_status": "group_continue",
    "confidence_score": 0.85,
    "reasoning": "Exec summary response, next block has thanks - continue to include pleasantries",
    "continue_to_next_block": true
  }
}
```

## Analysis Rationale

**Why This Decision Makes Sense**:
1. **Active Group Context**: System knows group 3 is active (started with Mike Mayo's question)
2. **Content Analysis**: Current block is Dave McKay summarizing his answers to Mike's capital questions
3. **Look-ahead Intelligence**: Next block is Mike saying "thank you" - should be included in the exchange
4. **Complete Exchange**: The full conversation (Q → A1 → A2 → A3 → Thanks) belongs together
5. **State Awareness**: Since group is active and next block has pleasantries, continue group to capture complete exchange

**The Result**: Blocks 20-24 form a complete Q&A group covering Mike Mayo's capital allocation questions and Dave McKay's comprehensive responses, including the closing pleasantries.