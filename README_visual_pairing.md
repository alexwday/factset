# Visual Sentence Pairing Tool

A production-ready tool for manually pairing sentences between PDF and HTML transcripts with visual diff highlighting.

## Installation

```bash
pip install pdfplumber beautifulsoup4
```

## Usage

### With real PDF and HTML files:
```bash
python3 visual_sentence_pairing.py --pdf transcript.pdf --html transcript.html
```

### With custom output file:
```bash
python3 visual_sentence_pairing.py --pdf transcript.pdf --html transcript.html --output my_pairing.html
```

### Demo mode (no files):
```bash
python3 visual_sentence_pairing.py
```

## Features

- **Real PDF extraction** using pdfplumber
- **Real HTML parsing** using BeautifulSoup4
- **Interactive pairing** - click sentences to select and pair them
- **Word-level diff highlighting** - see exactly what changed between versions
- **Order preservation** - sentences stay in original order
- **Export functionality** - save pairs as JSON
- **Multi-sentence pairing** - supports 2:1, 1:2 matches

## Interface

1. **Select sentences** by clicking on them (supports multi-select)
2. **Pair selected** sentences using the "Pair Selected" button
3. **View differences** with word-level highlighting using "Show Differences"
4. **Export results** as JSON when done

## Diff View

The diff view shows:
- **Red highlighting**: Words only in PDF (removed in HTML)
- **Green highlighting**: Words only in HTML (added from PDF)
- **No highlighting**: Common words between both versions
- **Similarity percentage** and word statistics

## File Support

- **PDF**: Uses pdfplumber to extract text from all pages
- **HTML**: Uses BeautifulSoup4 to parse various HTML structures
- **Fallback**: Demo data if files can't be read or dependencies missing