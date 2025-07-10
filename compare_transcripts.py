#!/usr/bin/env python3
"""
Transcript Comparison Tool
Compare content between PDF transcripts and HTML transcripts generated from XML
"""

import sys
import re
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Set
import html
from bs4 import BeautifulSoup

try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")
    pdfplumber = None


class TranscriptExtractor:
    """Extract and normalize transcript content from different formats."""
    
    def __init__(self):
        self.speaker_patterns = [
            r'^([A-Z][a-zA-Z\s\-\.]+)(?:\s*[-–]\s*)?(?:[A-Z][a-zA-Z\s&,\.]+)?:?\s*',
            r'^([A-Z][A-Z\s]+)(?:\s*[-–]\s*)?(?:[A-Z][a-zA-Z\s&,\.]+)?:?\s*',
            r'^\s*([A-Z][a-zA-Z\.\s]+),\s+([A-Z][a-zA-Z\s&,\.]+)\s*:?\s*',
        ]
    
    def extract_from_html(self, html_path: str) -> Dict:
        """Extract content from HTML transcript."""
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Extract metadata
        title = soup.find('h1')
        title_text = title.get_text().strip() if title else ""
        
        # Extract transcript content
        sections = []
        for section_div in soup.find_all('div', class_='section'):
            section_header = section_div.find('div', class_='section-header')
            section_name = section_header.get_text().strip() if section_header else ""
            
            speakers = []
            for speaker_block in section_div.find_all('div', class_='speaker-block'):
                speaker_info = speaker_block.find('div', class_='speaker-info')
                speaker_name = ""
                if speaker_info:
                    name_span = speaker_info.find('span', class_='speaker-name')
                    if name_span:
                        speaker_name = name_span.get_text().strip()
                
                content_div = speaker_block.find('div', class_='content')
                content_paragraphs = []
                if content_div:
                    for p in content_div.find_all('p'):
                        if p.get_text().strip():
                            content_paragraphs.append(p.get_text().strip())
                
                speakers.append({
                    'speaker': speaker_name,
                    'content': ' '.join(content_paragraphs)
                })
            
            sections.append({
                'section': section_name,
                'speakers': speakers
            })
        
        return {
            'title': title_text,
            'sections': sections,
            'raw_text': self._extract_content_text(sections)
        }
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """Extract content from PDF transcript."""
        if pdfplumber:
            return self._extract_with_pdfplumber(pdf_path)
        elif PyPDF2:
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise Exception("No PDF library available. Install pdfplumber or PyPDF2")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict:
        """Extract text using pdfplumber (recommended)."""
        text_lines = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_lines.extend(page_text.split('\n'))
        
        return self._parse_pdf_content(text_lines)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Dict:
        """Extract text using PyPDF2 (fallback)."""
        text_lines = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_lines.extend(page_text.split('\n'))
        
        return self._parse_pdf_content(text_lines)
    
    def _parse_pdf_content(self, text_lines: List[str]) -> Dict:
        """Parse PDF text lines into structured content."""
        # Clean up lines
        cleaned_lines = []
        for line in text_lines:
            line = line.strip()
            if line and not self._is_header_footer(line):
                cleaned_lines.append(line)
        
        # Find title (usually first few lines)
        title = ""
        content_start = 0
        for i, line in enumerate(cleaned_lines[:10]):
            if any(keyword in line.lower() for keyword in ['earnings', 'conference', 'call', 'transcript']):
                title = line
                content_start = i + 1
                break
        
        # Parse speaker segments
        sections = []
        current_section = "Main Content"
        current_speaker = ""
        current_content = []
        speakers = []
        
        for line in cleaned_lines[content_start:]:
            # Check if this line starts a new speaker
            speaker_match = self._extract_speaker_name(line)
            
            if speaker_match:
                # Save previous speaker content
                if current_speaker and current_content:
                    speakers.append({
                        'speaker': current_speaker,
                        'content': ' '.join(current_content)
                    })
                
                # Start new speaker
                current_speaker = speaker_match
                current_content = [line[len(speaker_match):].strip()]
            else:
                # Continue current speaker's content
                if line:
                    current_content.append(line)
        
        # Save last speaker
        if current_speaker and current_content:
            speakers.append({
                'speaker': current_speaker,
                'content': ' '.join(current_content)
            })
        
        sections.append({
            'section': current_section,
            'speakers': speakers
        })
        
        return {
            'title': title,
            'sections': sections,
            'raw_text': self._extract_content_text(sections)
        }
    
    def _extract_speaker_name(self, line: str) -> str:
        """Extract speaker name from line using patterns."""
        for pattern in self.speaker_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        return ""
    
    def _is_header_footer(self, line: str) -> bool:
        """Check if line is likely a header/footer."""
        header_footer_patterns = [
            r'^\s*\d+\s*$',  # Page numbers
            r'^\s*Page\s+\d+',  # Page indicators
            r'^\s*\d{1,2}/\d{1,2}/\d{4}',  # Dates
            r'^\s*\d{1,2}:\d{2}',  # Times
            r'^\s*[A-Z\s]+\s+©',  # Copyright lines
        ]
        
        for pattern in header_footer_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _extract_content_text(self, sections: List[Dict]) -> List[str]:
        """Extract just the content text for comparison."""
        content_lines = []
        for section in sections:
            for speaker in section['speakers']:
                if speaker['content']:
                    content_lines.append(speaker['content'])
        return content_lines


class TranscriptComparator:
    """Compare transcript content between different formats."""
    
    def __init__(self):
        self.extractor = TranscriptExtractor()
    
    def compare_transcripts(self, html_path: str, pdf_path: str) -> Dict:
        """Compare HTML and PDF transcripts."""
        print("Extracting content from HTML...")
        html_data = self.extractor.extract_from_html(html_path)
        
        print("Extracting content from PDF...")
        pdf_data = self.extractor.extract_from_pdf(pdf_path)
        
        print("Normalizing and comparing content...")
        
        # Normalize text for comparison
        html_normalized = self._normalize_text_list(html_data['raw_text'])
        pdf_normalized = self._normalize_text_list(pdf_data['raw_text'])
        
        # Perform comparison
        comparison = self._compare_normalized_content(html_normalized, pdf_normalized)
        
        return {
            'html_data': html_data,
            'pdf_data': pdf_data,
            'comparison': comparison
        }
    
    def _normalize_text_list(self, text_list: List[str]) -> List[str]:
        """Normalize text for comparison."""
        normalized = []
        for text in text_list:
            # Remove extra whitespace, normalize punctuation
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r"[''']", "'", text)
            text = re.sub(r'[–—]', '-', text)
            if text:
                normalized.append(text)
        return normalized
    
    def _compare_normalized_content(self, html_content: List[str], pdf_content: List[str]) -> Dict:
        """Compare normalized content and find differences."""
        # Join content for sequence matching
        html_text = ' '.join(html_content)
        pdf_text = ' '.join(pdf_content)
        
        # Basic statistics
        html_words = html_text.split()
        pdf_words = pdf_text.split()
        
        # Find common and unique content
        html_sentences = self._split_into_sentences(html_text)
        pdf_sentences = self._split_into_sentences(pdf_text)
        
        html_set = set(self._normalize_sentences(html_sentences))
        pdf_set = set(self._normalize_sentences(pdf_sentences))
        
        common_sentences = html_set.intersection(pdf_set)
        html_unique = html_set - pdf_set
        pdf_unique = pdf_set - html_set
        
        # Calculate similarity
        similarity = len(common_sentences) / max(len(html_set), len(pdf_set)) if html_set or pdf_set else 0
        
        return {
            'similarity': similarity,
            'html_word_count': len(html_words),
            'pdf_word_count': len(pdf_words),
            'html_sentence_count': len(html_sentences),
            'pdf_sentence_count': len(pdf_sentences),
            'common_sentences': len(common_sentences),
            'html_unique_sentences': list(html_unique)[:20],  # First 20 for display
            'pdf_unique_sentences': list(pdf_unique)[:20],    # First 20 for display
            'html_unique_count': len(html_unique),
            'pdf_unique_count': len(pdf_unique)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _normalize_sentences(self, sentences: List[str]) -> List[str]:
        """Normalize sentences for comparison."""
        normalized = []
        for sentence in sentences:
            # Convert to lowercase, remove extra whitespace
            sentence = sentence.lower()
            sentence = re.sub(r'\s+', ' ', sentence.strip())
            if len(sentence) > 10:  # Filter out very short sentences
                normalized.append(sentence)
        return normalized
    
    def generate_report(self, comparison_result: Dict, output_path: str) -> None:
        """Generate comparison report."""
        html_data = comparison_result['html_data']
        pdf_data = comparison_result['pdf_data']
        comparison = comparison_result['comparison']
        
        report = f"""
# Transcript Comparison Report

## Overview
- **HTML Title**: {html_data['title']}
- **PDF Title**: {pdf_data['title']}
- **Content Similarity**: {comparison['similarity']:.2%}

## Statistics
| Format | Word Count | Sentence Count | Unique Sentences |
|--------|------------|----------------|------------------|
| HTML   | {comparison['html_word_count']:,} | {comparison['html_sentence_count']:,} | {comparison['html_unique_count']:,} |
| PDF    | {comparison['pdf_word_count']:,} | {comparison['pdf_sentence_count']:,} | {comparison['pdf_unique_count']:,} |

## Content Coverage
- **Common Sentences**: {comparison['common_sentences']:,}
- **HTML-Only Content**: {comparison['html_unique_count']:,} sentences
- **PDF-Only Content**: {comparison['pdf_unique_count']:,} sentences

## Sample HTML-Only Content
{self._format_sentence_list(comparison['html_unique_sentences'])}

## Sample PDF-Only Content
{self._format_sentence_list(comparison['pdf_unique_sentences'])}

## Analysis
"""
        
        if comparison['similarity'] > 0.8:
            report += "✅ **High similarity** - Content is very similar between formats\n"
        elif comparison['similarity'] > 0.6:
            report += "⚠️ **Moderate similarity** - Some differences in content\n"
        else:
            report += "❌ **Low similarity** - Significant differences in content\n"
        
        if comparison['html_unique_count'] > comparison['pdf_unique_count']:
            report += "- HTML version contains more unique content\n"
        elif comparison['pdf_unique_count'] > comparison['html_unique_count']:
            report += "- PDF version contains more unique content\n"
        else:
            report += "- Both versions have similar amounts of unique content\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _format_sentence_list(self, sentences: List[str]) -> str:
        """Format sentence list for report."""
        if not sentences:
            return "None\n"
        
        formatted = ""
        for i, sentence in enumerate(sentences[:10], 1):
            formatted += f"{i}. {sentence[:200]}{'...' if len(sentence) > 200 else ''}\n"
        
        if len(sentences) > 10:
            formatted += f"... and {len(sentences) - 10} more\n"
        
        return formatted


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_transcripts.py <html_file> <pdf_file>")
        print("Example: python compare_transcripts.py transcript.html transcript.pdf")
        sys.exit(1)
    
    html_path = sys.argv[1]
    pdf_path = sys.argv[2]
    
    # Validate files exist
    if not Path(html_path).exists():
        print(f"Error: HTML file '{html_path}' not found")
        sys.exit(1)
    
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    try:
        # Perform comparison
        comparator = TranscriptComparator()
        result = comparator.compare_transcripts(html_path, pdf_path)
        
        # Generate report
        output_path = "transcript_comparison_report.md"
        comparator.generate_report(result, output_path)
        
        print(f"\n✅ Comparison complete!")
        print(f"📊 Similarity: {result['comparison']['similarity']:.2%}")
        print(f"📄 Report saved to: {output_path}")
        
        # Quick summary
        comparison = result['comparison']
        print(f"\n📈 Quick Summary:")
        print(f"   HTML: {comparison['html_word_count']:,} words, {comparison['html_unique_count']:,} unique sentences")
        print(f"   PDF:  {comparison['pdf_word_count']:,} words, {comparison['pdf_unique_count']:,} unique sentences")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()