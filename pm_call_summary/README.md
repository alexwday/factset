# PM Call Summary Pipeline

This pipeline processes earnings transcripts from the XML files to generate concise PM-focused summaries and insights.

## Pipeline Stages

### Stage 1: XML Processing & Content Extraction
- **Purpose**: Load and parse XML transcripts for PM call summary generation
- **Script**: `stage_1_xml_processing/1_pm_xml_processing.py`
- **Input**: XML transcript files from database refresh pipeline
- **Output**: Structured transcript data ready for summarization

### Stage 2: PM Summary Generation
- **Purpose**: Generate concise PM-focused summaries from transcripts
- **Script**: `stage_2_pm_summary/2_pm_summary_generation.py`
- **Input**: Processed transcript data
- **Output**: PM call summaries with key insights

### Stage 3: Key Metrics Extraction
- **Purpose**: Extract and structure key financial metrics and guidance
- **Script**: `stage_3_metrics_extraction/3_pm_metrics_extraction.py`
- **Input**: Transcript content and summaries
- **Output**: Structured metrics and guidance data

## Configuration

The pipeline uses `config.json` in this directory for configuration. Key settings include:
- Input XML file paths
- Output directory structure
- LLM API settings (if applicable)
- Processing parameters

## Usage

```bash
# From project root directory
python pm_call_summary/stage_1_xml_processing/1_pm_xml_processing.py
python pm_call_summary/stage_2_pm_summary/2_pm_summary_generation.py
python pm_call_summary/stage_3_metrics_extraction/3_pm_metrics_extraction.py
```

## Dependencies

Uses the same requirements.txt as the main project. Key dependencies:
- xml.etree.ElementTree (built-in)
- pandas
- python-dotenv
- pysmb (for NAS access)
- openai (for LLM-based summarization)