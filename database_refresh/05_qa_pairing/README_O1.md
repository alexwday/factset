# Stage 5 Q&A Pairing - O1 Model Support

## Files Overview

- **`main_qa_pairing.py`** - Original version using function calling (GPT-4o, GPT-4o-mini)
- **`main_qa_pairing_o1.py`** - O1 model version with text-based responses
- **`config.yaml`** - Shared configuration file (works with both versions)

## O1 Model Limitations

O1 models have the following limitations compared to GPT-4o models:
- ❌ No function calling / tools support
- ❌ No temperature parameter
- ❌ No max_tokens parameter  
- ❌ No system messages
- ⏱️ Longer response times (up to 60+ seconds)
- 💰 Higher cost per token

## How to Test O1 Models

### 1. Update Model in Configuration
In `config.yaml`, change the model:
```yaml
llm_config:
  model: "o1-mini"  # or "o1-preview"
  # temperature and max_tokens will be automatically ignored
```

### 2. Run O1 Version
```bash
python database_refresh/05_qa_pairing/main_qa_pairing_o1.py
```

The O1 script automatically ignores unsupported parameters like `temperature` and `max_tokens`.

## Key Differences in O1 Version

### API Call Changes
```python
# Original (GPT-4o)
response = llm_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    tools=create_analyst_session_boundary_tool(),
    tool_choice="required",
    temperature=0.1,
    max_tokens=300
)

# O1 Version  
response = llm_client.chat.completions.create(
    model="o1-mini",
    messages=[{"role": "user", "content": prompt}]
    # No tools, temperature, max_tokens supported
)
```

### Response Parsing Changes
```python
# Original (function calling)
decision = result["decision"]

# O1 Version (text parsing)
response_text = response.choices[0].message.content.strip().lower()
if response_text == "continue":
    decision = "continue"
elif response_text == "end":
    decision = "end"
```

### Prompt Changes
```xml
<!-- Original -->
<output_format>
Make your decision by calling the boundary_decision function with either "continue" or "end".
</output_format>

<!-- O1 Version -->
<output_format>
Respond with exactly one word: either "continue" or "end".
Do not include any other text, explanations, or reasoning. Just the single word decision.
</output_format>
```

## Cost Comparison

| Model | Prompt Tokens (per 1K) | Completion Tokens (per 1K) |
|-------|------------------------|----------------------------|
| GPT-4o-mini | $0.00015 | $0.0006 |
| O1-mini | $0.003 | $0.012 |
| O1-preview | $0.015 | $0.060 |

O1 models are **20-100x more expensive** than GPT-4o-mini, so use sparingly for testing.

## Switching Back to Original

To return to the original GPT-4o version:
1. Change the model back to `"gpt-4o-mini"` in `config.yaml`
2. Run `main_qa_pairing.py` instead of `main_qa_pairing_o1.py`