#!/usr/bin/env python3
"""
Visual Transcript Comparison Tool
Compare PDF and HTML transcripts using sliding window word matching with HTML output
"""

import sys
import re
import html
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import difflib
from enum import Enum
import time

try:
    import pdfplumber
except ImportError:
    print("Warning: pdfplumber not installed. Install with: pip install pdfplumber")
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")
    PyPDF2 = None


@dataclass
class WordMatch:
    """Represents a matched word segment."""
    pdf_start: int
    pdf_end: int
    html_start: int
    html_end: int
    confidence: float


@dataclass
class TextSegment:
    """Represents a segment of text with match status."""
    words: List[str]
    start_idx: int
    end_idx: int
    match_type: str  # 'matched', 'missing', 'extra'
    matched_segment: Optional['TextSegment'] = None


class DiffType(Enum):
    """Types of differences in sentence alignment."""
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"


@dataclass
class SentenceAlignment:
    """Represents an aligned sentence pair or diff."""
    diff_type: DiffType
    pdf_sentences: List[str]
    html_sentences: List[str]
    similarity: float = 0.0


class AdvancedTranscriptExtractor:
    """Enhanced text extraction with better content parsing."""
    
    def __init__(self):
        self.speaker_patterns = [
            r'^([A-Z][a-zA-Z\s\-\.\']+?)(?:\s*[-–:]\s*)',
            r'^([A-Z][A-Z\s]+?)(?:\s*[-–:]\s*)',
            r'^\s*([A-Z][a-zA-Z\.\s]+?),\s+([A-Z][a-zA-Z\s&,\.]+?)\s*:',
            r'^([A-Z]+[a-z]*(?:\s+[A-Z]+[a-z]*)*)\s*[-–:]\s*',
        ]
    
    def extract_pdf_words(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Extract words and raw text from PDF."""
        if pdfplumber:
            return self._extract_pdf_pdfplumber(pdf_path)
        elif PyPDF2:
            return self._extract_pdf_pypdf2(pdf_path)
        else:
            raise Exception("No PDF library available. Install pdfplumber or PyPDF2")
    
    def _extract_pdf_pdfplumber(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Extract using pdfplumber with progress tracking."""
        raw_lines = []
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"    - Processing {total_pages} pages...")
            
            for page_num, page in enumerate(pdf.pages, 1):
                if page_num % 5 == 0 or page_num == total_pages:
                    print(f"      Page {page_num}/{total_pages}")
                
                page_text = page.extract_text()
                if page_text:
                    raw_lines.extend(page_text.split('\n'))
        
        return self._process_pdf_lines(raw_lines)
    
    def _extract_pdf_pypdf2(self, pdf_path: str) -> Tuple[List[str], List[str]]:
        """Extract using PyPDF2 with progress tracking."""
        raw_lines = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"    - Processing {total_pages} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                if page_num % 5 == 0 or page_num == total_pages:
                    print(f"      Page {page_num}/{total_pages}")
                
                page_text = page.extract_text()
                if page_text:
                    raw_lines.extend(page_text.split('\n'))
        
        return self._process_pdf_lines(raw_lines)
    
    def _process_pdf_lines(self, raw_lines: List[str]) -> Tuple[List[str], List[str]]:
        """Process PDF lines into words and clean text."""
        # Filter out headers, footers, page numbers
        content_lines = []
        for line in raw_lines:
            line = line.strip()
            if line and not self._is_metadata_line(line):
                content_lines.append(line)
        
        # Extract content text (skip title/header sections)
        content_start = self._find_content_start(content_lines)
        content_text = ' '.join(content_lines[content_start:])
        
        # Extract words
        words = self._extract_words(content_text)
        
        return words, content_lines[content_start:]
    
    def extract_html_words(self, html_path: str) -> Tuple[List[str], List[str]]:
        """Extract words and content from HTML transcript."""
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Extract all content from speaker blocks
        content_parts = []
        for speaker_block in soup.find_all('div', class_='speaker-block'):
            content_div = speaker_block.find('div', class_='content')
            if content_div:
                for p in content_div.find_all('p'):
                    text = p.get_text().strip()
                    if text:
                        content_parts.append(text)
        
        content_text = ' '.join(content_parts)
        words = self._extract_words(content_text)
        
        return words, content_parts
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract and normalize words from text."""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into words and clean
        words = []
        for word in text.split():
            # Remove punctuation except apostrophes and hyphens
            clean_word = re.sub(r'[^\w\'\-]', '', word)
            if clean_word and len(clean_word) > 1:  # Skip single characters
                words.append(clean_word.lower())
        
        return words
    
    def _is_metadata_line(self, line: str) -> bool:
        """Check if line is metadata (headers, footers, etc.)."""
        metadata_patterns = [
            r'^\s*\d+\s*$',  # Page numbers
            r'^\s*Page\s+\d+',  # Page indicators
            r'^\s*\d{1,2}/\d{1,2}/\d{4}',  # Dates
            r'^\s*\d{1,2}:\d{2}',  # Times
            r'^\s*©.*',  # Copyright
            r'^\s*[A-Z\s]+\s+©',  # Company copyright
            r'^\s*www\.',  # URLs
            r'^\s*http',  # URLs
        ]
        
        for pattern in metadata_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return len(line) < 3  # Very short lines
    
    def _find_content_start(self, lines: List[str]) -> int:
        """Find where actual transcript content starts."""
        for i, line in enumerate(lines):
            # Look for patterns that indicate content start
            if any(indicator in line.lower() for indicator in [
                'good morning', 'good afternoon', 'welcome', 'thank you',
                'operator', 'conference call', 'earnings call'
            ]):
                return max(0, i - 1)  # Include line before if exists
        
        # Fallback: skip obvious header lines
        for i, line in enumerate(lines[:20]):
            if len(line) > 50 and not any(skip in line.lower() for skip in [
                'company', 'quarter', 'fiscal', 'ended'
            ]):
                return i
        
        return min(5, len(lines))  # Default fallback


class SlidingWindowMatcher:
    """Sliding window algorithm for matching word sequences."""
    
    def __init__(self, window_size: int = 6, min_match_threshold: float = 0.7):
        self.window_size = window_size  # Reduced from 8 for better performance
        self.min_match_threshold = min_match_threshold
    
    def find_matches(self, pdf_words: List[str], html_words: List[str]) -> List[WordMatch]:
        """Find matching word segments using sliding window with progress tracking."""
        matches = []
        used_html_indices = set()
        total_windows = len(pdf_words) - self.window_size + 1
        
        print(f"    - Processing {total_windows:,} word windows...")
        start_time = time.time()
        last_progress_time = start_time
        
        for pdf_start in range(total_windows):
            # Progress reporting every 10% or 2 seconds
            current_time = time.time()
            if (current_time - last_progress_time > 2.0) or (pdf_start % max(1, total_windows // 10) == 0):
                progress = (pdf_start / total_windows) * 100
                elapsed = current_time - start_time
                if pdf_start > 0:
                    eta = (elapsed / pdf_start) * (total_windows - pdf_start)
                    print(f"      Progress: {progress:.1f}% ({pdf_start:,}/{total_windows:,}) - ETA: {eta:.1f}s")
                last_progress_time = current_time
            
            pdf_window = pdf_words[pdf_start:pdf_start + self.window_size]
            
            best_match = self._find_best_html_match(
                pdf_window, html_words, used_html_indices
            )
            
            if best_match and best_match['confidence'] >= self.min_match_threshold:
                match = WordMatch(
                    pdf_start=pdf_start,
                    pdf_end=pdf_start + self.window_size,
                    html_start=best_match['start'],
                    html_end=best_match['end'],
                    confidence=best_match['confidence']
                )
                matches.append(match)
                
                # Mark HTML indices as used
                for idx in range(best_match['start'], best_match['end']):
                    used_html_indices.add(idx)
        
        elapsed = time.time() - start_time
        print(f"    - Word matching completed in {elapsed:.2f}s, found {len(matches)} matches")
        
        return self._merge_overlapping_matches(matches)
    
    def _find_best_html_match(self, pdf_window: List[str], html_words: List[str], 
                             used_indices: Set[int]) -> Optional[Dict]:
        """Find the best matching HTML window for a PDF window (optimized)."""
        best_match = None
        best_confidence = 0
        window_size = len(pdf_window)
        
        # Early termination if we find a perfect match
        for html_start in range(len(html_words) - window_size + 1):
            # Skip if any index is already used
            if any(idx in used_indices for idx in range(html_start, html_start + window_size)):
                continue
            
            html_window = html_words[html_start:html_start + window_size]
            confidence = self._calculate_similarity(pdf_window, html_window)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    'start': html_start,
                    'end': html_start + window_size,
                    'confidence': confidence
                }
                
                # Early termination for near-perfect matches
                if confidence >= 0.95:
                    break
        
        return best_match
    
    def _calculate_similarity(self, window1: List[str], window2: List[str]) -> float:
        """Calculate similarity between two word windows."""
        if len(window1) != len(window2):
            return 0.0
        
        exact_matches = sum(1 for w1, w2 in zip(window1, window2) if w1 == w2)
        fuzzy_matches = sum(1 for w1, w2 in zip(window1, window2) 
                          if w1 != w2 and self._fuzzy_match(w1, w2))
        
        total_score = exact_matches + (fuzzy_matches * 0.8)
        return total_score / len(window1)
    
    def _fuzzy_match(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to be considered a match."""
        # Handle common variations
        if abs(len(word1) - len(word2)) > 3:
            return False
        
        # Use sequence matcher for similarity
        similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
        return similarity > 0.8
    
    def _merge_overlapping_matches(self, matches: List[WordMatch]) -> List[WordMatch]:
        """Merge overlapping matches to avoid double-counting."""
        if not matches:
            return matches
        
        # Sort by PDF start position
        matches.sort(key=lambda m: m.pdf_start)
        
        merged = [matches[0]]
        for current in matches[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current.pdf_start < last.pdf_end:
                # Merge if current has better confidence
                if current.confidence > last.confidence:
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged


class SentenceAligner:
    """Align sentences between PDF and HTML using sequence matching."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
    
    def extract_sentences(self, text_lines: List[str]) -> List[str]:
        """Extract sentences from text lines."""
        # Join lines and split into sentences
        full_text = ' '.join(text_lines)
        
        # More sophisticated sentence splitting
        # Split on sentence endings but keep some context
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences that are substantial
            if len(sentence) > 10 and not self._is_metadata_sentence(sentence):
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def _is_metadata_sentence(self, sentence: str) -> bool:
        """Check if sentence is likely metadata."""
        metadata_indicators = [
            r'^\d+\s*$',  # Just numbers
            r'^page\s+\d+',  # Page numbers
            r'©\s*\d{4}',  # Copyright
            r'www\.',  # URLs
            r'http[s]?://',  # URLs
        ]
        
        for pattern in metadata_indicators:
            if re.search(pattern, sentence.lower()):
                return True
        return False
    
    def align_sentences(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Align sentences using sequence matching."""
        # Normalize sentences for comparison
        pdf_normalized = [self._normalize_sentence(s) for s in pdf_sentences]
        html_normalized = [self._normalize_sentence(s) for s in html_sentences]
        
        # Use difflib to find sequence matches
        matcher = difflib.SequenceMatcher(None, pdf_normalized, html_normalized)
        alignments = []
        
        for tag, pdf_start, pdf_end, html_start, html_end in matcher.get_opcodes():
            pdf_group = pdf_sentences[pdf_start:pdf_end]
            html_group = html_sentences[html_start:html_end]
            
            if tag == 'equal':
                # Sentences match
                for pdf_sent, html_sent in zip(pdf_group, html_group):
                    similarity = self._calculate_sentence_similarity(pdf_sent, html_sent)
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.EQUAL,
                        pdf_sentences=[pdf_sent],
                        html_sentences=[html_sent],
                        similarity=similarity
                    ))
            
            elif tag == 'delete':
                # Sentences only in PDF
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.DELETE,
                    pdf_sentences=pdf_group,
                    html_sentences=[],
                    similarity=0.0
                ))
            
            elif tag == 'insert':
                # Sentences only in HTML
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.INSERT,
                    pdf_sentences=[],
                    html_sentences=html_group,
                    similarity=0.0
                ))
            
            elif tag == 'replace':
                # Different sentences - try to pair them
                if len(pdf_group) == 1 and len(html_group) == 1:
                    similarity = self._calculate_sentence_similarity(pdf_group[0], html_group[0])
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.REPLACE,
                        pdf_sentences=pdf_group,
                        html_sentences=html_group,
                        similarity=similarity
                    ))
                else:
                    # Multiple sentences - treat as separate delete and insert
                    if pdf_group:
                        alignments.append(SentenceAlignment(
                            diff_type=DiffType.DELETE,
                            pdf_sentences=pdf_group,
                            html_sentences=[],
                            similarity=0.0
                        ))
                    if html_group:
                        alignments.append(SentenceAlignment(
                            diff_type=DiffType.INSERT,
                            pdf_sentences=[],
                            html_sentences=html_group,
                            similarity=0.0
                        ))
        
        return alignments
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Normalize sentence for comparison."""
        # Convert to lowercase, remove extra whitespace
        normalized = sentence.lower().strip()
        # Remove punctuation and normalize whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences."""
        norm1 = self._normalize_sentence(sent1)
        norm2 = self._normalize_sentence(sent2)
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()


class VisualComparisonGenerator:
    """Generate HTML visual comparison output."""
    
    def __init__(self):
        self.extractor = AdvancedTranscriptExtractor()
        self.matcher = SlidingWindowMatcher()
        self.sentence_aligner = SentenceAligner()
    
    def generate_comparison(self, pdf_path: str, html_path: str, output_path: str):
        """Generate visual comparison HTML with progress tracking."""
        total_start = time.time()
        print("🚀 Starting comprehensive transcript comparison...")
        
        print("\n📄 Extracting PDF content...")
        start_time = time.time()
        pdf_words, pdf_lines = self.extractor.extract_pdf_words(pdf_path)
        print(f"  ✓ PDF extraction completed in {time.time() - start_time:.2f}s")
        print(f"    - Extracted {len(pdf_words):,} words and {len(pdf_lines):,} lines")
        
        print("\n🌐 Extracting HTML content...")
        start_time = time.time()
        html_words, html_lines = self.extractor.extract_html_words(html_path)
        print(f"  ✓ HTML extraction completed in {time.time() - start_time:.2f}s")
        print(f"    - Extracted {len(html_words):,} words and {len(html_lines):,} lines")
        
        print("\n📝 Extracting sentences...")
        start_time = time.time()
        pdf_sentences = self.sentence_aligner.extract_sentences(pdf_lines)
        html_sentences = self.sentence_aligner.extract_sentences(html_lines)
        print(f"  ✓ Sentence extraction completed in {time.time() - start_time:.2f}s")
        print(f"    - PDF: {len(pdf_sentences):,} sentences")
        print(f"    - HTML: {len(html_sentences):,} sentences")
        
        print("\n🔗 Aligning sentences...")
        start_time = time.time()
        sentence_alignments = self.sentence_aligner.align_sentences(pdf_sentences, html_sentences)
        print(f"  ✓ Sentence alignment completed in {time.time() - start_time:.2f}s")
        print(f"    - Created {len(sentence_alignments):,} alignments")
        
        print("\n🔍 Finding word matches...")
        start_time = time.time()
        matches = self.matcher.find_matches(pdf_words, html_words)
        print(f"  ✓ Word matching completed in {time.time() - start_time:.2f}s")
        
        print("\n📊 Analyzing coverage...")
        start_time = time.time()
        analysis = self._analyze_coverage(pdf_words, html_words, matches, sentence_alignments)
        print(f"  ✓ Analysis completed in {time.time() - start_time:.2f}s")
        
        print("\n🎨 Generating HTML output...")
        start_time = time.time()
        html_output = self._generate_html_output(
            pdf_words, html_words, matches, analysis, pdf_path, html_path, sentence_alignments
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_output)
        print(f"  ✓ HTML generation completed in {time.time() - start_time:.2f}s")
        
        total_time = time.time() - total_start
        print(f"\n🎉 Total processing time: {total_time:.2f} seconds")
        
        return analysis
    
    def _analyze_coverage(self, pdf_words: List[str], html_words: List[str], 
                         matches: List[WordMatch], sentence_alignments: List[SentenceAlignment]) -> Dict:
        """Analyze coverage and generate statistics."""
        pdf_matched = set()
        html_matched = set()
        
        for match in matches:
            for i in range(match.pdf_start, match.pdf_end):
                pdf_matched.add(i)
            for i in range(match.html_start, match.html_end):
                html_matched.add(i)
        
        pdf_coverage = len(pdf_matched) / len(pdf_words) if pdf_words else 0
        html_coverage = len(html_matched) / len(html_words) if html_words else 0
        
        # Sentence-level statistics
        equal_sentences = sum(1 for align in sentence_alignments if align.diff_type == DiffType.EQUAL)
        deleted_sentences = sum(len(align.pdf_sentences) for align in sentence_alignments if align.diff_type == DiffType.DELETE)
        inserted_sentences = sum(len(align.html_sentences) for align in sentence_alignments if align.diff_type == DiffType.INSERT)
        replaced_sentences = sum(1 for align in sentence_alignments if align.diff_type == DiffType.REPLACE)
        
        total_pdf_sentences = sum(len(align.pdf_sentences) for align in sentence_alignments)
        total_html_sentences = sum(len(align.html_sentences) for align in sentence_alignments)
        
        sentence_similarity = sum(align.similarity for align in sentence_alignments) / len(sentence_alignments) if sentence_alignments else 0
        
        return {
            'total_matches': len(matches),
            'pdf_word_count': len(pdf_words),
            'html_word_count': len(html_words),
            'pdf_matched_words': len(pdf_matched),
            'html_matched_words': len(html_matched),
            'pdf_coverage': pdf_coverage,
            'html_coverage': html_coverage,
            'avg_confidence': sum(m.confidence for m in matches) / len(matches) if matches else 0,
            'sentence_alignments': len(sentence_alignments),
            'equal_sentences': equal_sentences,
            'deleted_sentences': deleted_sentences,
            'inserted_sentences': inserted_sentences,
            'replaced_sentences': replaced_sentences,
            'total_pdf_sentences': total_pdf_sentences,
            'total_html_sentences': total_html_sentences,
            'sentence_similarity': sentence_similarity
        }
    
    def _generate_html_output(self, pdf_words: List[str], html_words: List[str], 
                             matches: List[WordMatch], analysis: Dict, 
                             pdf_path: str, html_path: str, sentence_alignments: List[SentenceAlignment]) -> str:
        """Generate the HTML comparison output with diff-like sentence alignment."""
        
        # Generate diff table rows
        diff_rows = self._create_diff_rows(sentence_alignments)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript Comparison: PDF vs HTML - Diff View</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e9ecef;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 13px;
            margin-top: 4px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .diff-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .diff-header {{
            background: #495057;
            color: white;
            padding: 15px 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2px;
        }}
        .diff-header-left {{
            text-align: center;
            font-weight: bold;
        }}
        .diff-header-right {{
            text-align: center;
            font-weight: bold;
        }}
        .diff-table {{
            display: table;
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }}
        .diff-row {{
            display: table-row;
        }}
        .diff-cell {{
            display: table-cell;
            padding: 12px 20px;
            vertical-align: top;
            width: 50%;
            border-bottom: 1px solid #e9ecef;
            font-size: 15px;
            line-height: 1.6;
        }}
        .diff-cell:first-child {{
            border-right: 1px solid #e9ecef;
        }}
        .matched-row {{
            background-color: #f8fff8;
        }}
        .matched-row .diff-cell {{
            border-left: 4px solid #28a745;
        }}
        .deleted-row {{
            background-color: #fff5f5;
        }}
        .deleted-row .diff-cell:first-child {{
            border-left: 4px solid #dc3545;
        }}
        .deleted-row .diff-cell:last-child {{
            background-color: #f8f9fa;
            color: #6c757d;
            font-style: italic;
        }}
        .inserted-row {{
            background-color: #fffbf0;
        }}
        .inserted-row .diff-cell:first-child {{
            background-color: #f8f9fa;
            color: #6c757d;
            font-style: italic;
        }}
        .inserted-row .diff-cell:last-child {{
            border-left: 4px solid #ffc107;
        }}
        .replaced-row {{
            background-color: #f0f8ff;
        }}
        .replaced-row .diff-cell {{
            border-left: 4px solid #17a2b8;
        }}
        .similarity-badge {{
            display: inline-block;
            background: #6c757d;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 8px;
        }}
        .high-similarity {{
            background: #28a745;
        }}
        .medium-similarity {{
            background: #ffc107;
            color: #212529;
        }}
        .low-similarity {{
            background: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Transcript Diff Comparison</h1>
        <p><strong>PDF Reference:</strong> {Path(pdf_path).name}</p>
        <p><strong>HTML Comparison:</strong> {Path(html_path).name}</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{analysis['equal_sentences']}</div>
                <div class="stat-label">Matched Sentences</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['deleted_sentences']}</div>
                <div class="stat-label">PDF Only</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['inserted_sentences']}</div>
                <div class="stat-label">HTML Only</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['replaced_sentences']}</div>
                <div class="stat-label">Modified</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['sentence_similarity']:.1%}</div>
                <div class="stat-label">Avg Similarity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['total_pdf_sentences']}</div>
                <div class="stat-label">Total PDF Sentences</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{analysis['total_html_sentences']}</div>
                <div class="stat-label">Total HTML Sentences</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #28a745;"></div>
                <span>Matched</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #dc3545;"></div>
                <span>PDF Only</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #ffc107;"></div>
                <span>HTML Only</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #17a2b8;"></div>
                <span>Modified</span>
            </div>
        </div>
    </div>
    
    <div class="diff-container">
        <div class="diff-header">
            <div class="diff-header-left">📄 PDF Reference</div>
            <div class="diff-header-right">🌐 HTML Comparison</div>
        </div>
        <div class="diff-table">
            {diff_rows}
        </div>
    </div>
</body>
</html>"""
        
        return html_template
    
    def _create_diff_rows(self, sentence_alignments: List[SentenceAlignment]) -> str:
        """Create HTML diff table rows from sentence alignments."""
        rows = []
        
        for alignment in sentence_alignments:
            if alignment.diff_type == DiffType.EQUAL:
                # Matched sentences - show side by side
                pdf_text = html.escape(alignment.pdf_sentences[0]) if alignment.pdf_sentences else ""
                html_text = html.escape(alignment.html_sentences[0]) if alignment.html_sentences else ""
                
                similarity_class = self._get_similarity_class(alignment.similarity)
                similarity_badge = f'<span class="similarity-badge {similarity_class}">{alignment.similarity:.0%}</span>'
                
                rows.append(f'''
                <div class="diff-row matched-row">
                    <div class="diff-cell">{pdf_text}</div>
                    <div class="diff-cell">{html_text}{similarity_badge}</div>
                </div>''')
            
            elif alignment.diff_type == DiffType.DELETE:
                # PDF only content
                for sentence in alignment.pdf_sentences:
                    escaped_sentence = html.escape(sentence)
                    rows.append(f'''
                    <div class="diff-row deleted-row">
                        <div class="diff-cell">{escaped_sentence}</div>
                        <div class="diff-cell">(missing in HTML)</div>
                    </div>''')
            
            elif alignment.diff_type == DiffType.INSERT:
                # HTML only content
                for sentence in alignment.html_sentences:
                    escaped_sentence = html.escape(sentence)
                    rows.append(f'''
                    <div class="diff-row inserted-row">
                        <div class="diff-cell">(not in PDF)</div>
                        <div class="diff-cell">{escaped_sentence}</div>
                    </div>''')
            
            elif alignment.diff_type == DiffType.REPLACE:
                # Modified content - show both versions
                pdf_text = html.escape(alignment.pdf_sentences[0]) if alignment.pdf_sentences else ""
                html_text = html.escape(alignment.html_sentences[0]) if alignment.html_sentences else ""
                
                similarity_class = self._get_similarity_class(alignment.similarity)
                similarity_badge = f'<span class="similarity-badge {similarity_class}">{alignment.similarity:.0%}</span>'
                
                rows.append(f'''
                <div class="diff-row replaced-row">
                    <div class="diff-cell">{pdf_text}</div>
                    <div class="diff-cell">{html_text}{similarity_badge}</div>
                </div>''')
        
        return '\n'.join(rows)
    
    def _get_similarity_class(self, similarity: float) -> str:
        """Get CSS class for similarity score."""
        if similarity >= 0.8:
            return "high-similarity"
        elif similarity >= 0.5:
            return "medium-similarity"
        else:
            return "low-similarity"


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python compare_transcripts_visual.py <pdf_file> <html_file>")
        print("Example: python compare_transcripts_visual.py transcript.pdf transcript_output.html")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    html_path = sys.argv[2]
    
    # Validate files exist
    if not Path(pdf_path).exists():
        print(f"Error: PDF file '{pdf_path}' not found")
        sys.exit(1)
    
    if not Path(html_path).exists():
        print(f"Error: HTML file '{html_path}' not found")
        sys.exit(1)
    
    try:
        # Generate comparison
        generator = VisualComparisonGenerator()
        output_path = "transcript_visual_comparison.html"
        
        analysis = generator.generate_comparison(pdf_path, html_path, output_path)
        
        print(f"\n✅ Visual comparison complete!")
        print(f"📊 PDF Coverage: {analysis['pdf_coverage']:.1%}")
        print(f"🌐 HTML Coverage: {analysis['html_coverage']:.1%}")
        print(f"🎯 Match Confidence: {analysis['avg_confidence']:.1%}")
        print(f"📄 Output: {output_path}")
        print(f"\nOpen {output_path} in your browser to view the detailed comparison!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()