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
class IndexedSentence:
    """Represents a sentence with its original position index."""
    index: int
    text: str
    is_matched: bool = False
    match_group_id: Optional[str] = None

@dataclass
class SentenceMatch:
    """Represents a match between indexed sentences."""
    match_id: str
    pdf_sentences: List[IndexedSentence]
    html_sentences: List[IndexedSentence]
    match_type: str  # 'exact', 'fuzzy', 'partial', 'manual'
    similarity: float = 0.0
    confidence: str = 'high'  # 'high', 'medium', 'low'

@dataclass
class SentenceAlignment:
    """Legacy class - kept for compatibility."""
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
    """Sliding window algorithm for matching word sequences with adaptive sizing."""
    
    def __init__(self, window_size: int = 6, min_match_threshold: float = 0.7):
        self.base_window_size = window_size
        self.min_match_threshold = min_match_threshold
        self.base = 31
        self.mod = 10**9 + 7
        self.similarity_cache = {}  # Cache for word similarity calculations
        self.adaptive_sizing = True  # Enable smart window sizing
        self.max_cache_size = 10000  # Prevent memory issues
    
    def find_matches(self, pdf_words: List[str], html_words: List[str]) -> List[WordMatch]:
        """Find matching word segments using optimized approach with adaptive window sizing."""
        matches = []
        used_html_indices = set()
        
        # Determine optimal window sizes to try
        window_sizes = self._determine_optimal_window_sizes(pdf_words, html_words)
        print(f"    - Using adaptive window sizes: {window_sizes}")
        
        start_time = time.time()
        
        # Try different window sizes, starting with the most likely to succeed
        for window_size in window_sizes:
            window_matches = self._find_matches_for_window_size(
                pdf_words, html_words, window_size, used_html_indices
            )
            matches.extend(window_matches)
            
            # Update used indices
            for match in window_matches:
                for idx in range(match.html_start, match.html_end):
                    used_html_indices.add(idx)
        
        elapsed = time.time() - start_time
        print(f"    - Adaptive matching completed in {elapsed:.2f}s, found {len(matches)} matches")
        
        return self._merge_overlapping_matches(matches)
    
    def _determine_optimal_window_sizes(self, pdf_words: List[str], html_words: List[str]) -> List[int]:
        """Determine optimal window sizes based on text characteristics."""
        if not self.adaptive_sizing:
            return [self.base_window_size]
        
        # Analyze text characteristics
        avg_word_length = sum(len(word) for word in pdf_words[:100]) / min(100, len(pdf_words))
        
        # Financial transcripts often have:
        # - Short phrases: "revenue increased", "Q3 results" (2-4 words)
        # - Medium phrases: "earnings per share", "operating margin improved" (4-6 words)  
        # - Long phrases: "we are pleased to announce", "looking forward to the future" (6-10 words)
        
        window_sizes = []
        
        if avg_word_length > 6:  # Longer words suggest more technical content
            window_sizes = [4, 6, 8]  # Shorter windows for technical terms
        else:  # More conversational content
            window_sizes = [6, 4, 8]  # Standard window first
        
        # Add very short windows for key financial terms
        if not 3 in window_sizes:
            window_sizes.append(3)
            
        return window_sizes
    
    def _find_matches_for_window_size(self, pdf_words: List[str], html_words: List[str], 
                                    window_size: int, used_html_indices: Set[int]) -> List[WordMatch]:
        """Find matches for a specific window size."""
        matches = []
        total_windows = len(pdf_words) - window_size + 1
        
        if total_windows <= 0:
            return matches
        
        print(f"      - Window size {window_size}: {total_windows:,} windows")
        
        # Build hash map for this window size
        html_hash_map = self._build_rolling_hash_map_sized(html_words, window_size)
        
        for pdf_start in range(total_windows):
            pdf_window = pdf_words[pdf_start:pdf_start + window_size]
            
            # Use rolling hash for fast candidate finding
            best_match = self._find_best_html_match_sized(
                pdf_window, html_words, html_hash_map, used_html_indices, window_size
            )
            
            if best_match and best_match['confidence'] >= self.min_match_threshold:
                match = WordMatch(
                    pdf_start=pdf_start,
                    pdf_end=pdf_start + window_size,
                    html_start=best_match['start'],
                    html_end=best_match['end'],
                    confidence=best_match['confidence']
                )
                matches.append(match)
        
        return matches
    
    def _build_rolling_hash_map_sized(self, words: List[str], window_size: int) -> Dict[int, List[int]]:
        """Build a hash map of rolling hashes to positions for O(1) lookup with custom window size."""
        hash_map = {}
        if len(words) < window_size:
            return hash_map
            
        # Precompute powers of base
        powers = [1]
        for i in range(1, window_size):
            powers.append((powers[-1] * self.base) % self.mod)
        
        # Initial hash computation
        current_hash = 0
        for i in range(window_size):
            word_hash = hash(words[i].lower()) % self.mod
            current_hash = (current_hash + word_hash * powers[window_size - 1 - i]) % self.mod
        
        # Store initial hash
        if current_hash not in hash_map:
            hash_map[current_hash] = []
        hash_map[current_hash].append(0)
        
        # Rolling hash for remaining positions
        highest_power = powers[window_size - 1]
        for i in range(window_size, len(words)):
            # Remove leftmost word
            old_word_hash = hash(words[i - window_size].lower()) % self.mod
            current_hash = (current_hash - old_word_hash * highest_power) % self.mod
            
            # Add new word
            new_word_hash = hash(words[i].lower()) % self.mod
            current_hash = (current_hash * self.base + new_word_hash) % self.mod
            
            # Store position
            start_pos = i - window_size + 1
            if current_hash not in hash_map:
                hash_map[current_hash] = []
            hash_map[current_hash].append(start_pos)
        
        return hash_map
    
    def _build_rolling_hash_map(self, words: List[str]) -> Dict[int, List[int]]:
        """Build a hash map of rolling hashes to positions for O(1) lookup."""
        return self._build_rolling_hash_map_sized(words, self.base_window_size)
    
    def _find_best_html_match_sized(self, pdf_window: List[str], html_words: List[str], 
                                   html_hash_map: Dict[int, List[int]], 
                                   used_indices: Set[int], window_size: int) -> Optional[Dict]:
        """Find matches using rolling hash for custom window size."""
        pdf_hash = self._compute_window_hash_sized(pdf_window, window_size)
        
        # Fast hash-based candidate lookup
        candidates = html_hash_map.get(pdf_hash, [])
        
        best_match = None
        best_confidence = 0
        
        # Check exact hash matches first (fastest)
        for html_start in candidates:
            if any(idx in used_indices for idx in range(html_start, html_start + window_size)):
                continue
                
            html_window = html_words[html_start:html_start + window_size]
            confidence = self._calculate_similarity_cached(pdf_window, html_window)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    'start': html_start,
                    'end': html_start + window_size,
                    'confidence': confidence
                }
                
                # Early termination for perfect matches
                if confidence >= 0.98:
                    return best_match
        
        # If no good hash match, skip fuzzy fallback for adaptive sizing (too expensive)
        return best_match
    
    def _compute_window_hash_sized(self, window: List[str], window_size: int) -> int:
        """Compute rolling hash for a single window with custom size."""
        window_hash = 0
        power = 1
        
        for i in range(len(window) - 1, -1, -1):
            word_hash = hash(window[i].lower()) % self.mod
            window_hash = (window_hash + word_hash * power) % self.mod
            power = (power * self.base) % self.mod
        
        return window_hash
    
    def _compute_window_hash(self, window: List[str]) -> int:
        """Compute rolling hash for a single window."""
        window_hash = 0
        power = 1
        
        for i in range(len(window) - 1, -1, -1):
            word_hash = hash(window[i].lower()) % self.mod
            window_hash = (window_hash + word_hash * power) % self.mod
            power = (power * self.base) % self.mod
        
        return window_hash
    
    def _find_best_html_match_optimized(self, pdf_window: List[str], html_words: List[str], 
                                      html_hash_map: Dict[int, List[int]], 
                                      used_indices: Set[int]) -> Optional[Dict]:
        """Find matches using rolling hash for O(1) candidate lookup."""
        pdf_hash = self._compute_window_hash(pdf_window)
        
        # Fast hash-based candidate lookup
        candidates = html_hash_map.get(pdf_hash, [])
        
        best_match = None
        best_confidence = 0
        
        # Check exact hash matches first (fastest)
        for html_start in candidates:
            if any(idx in used_indices for idx in range(html_start, html_start + self.window_size)):
                continue
                
            html_window = html_words[html_start:html_start + self.window_size]
            confidence = self._calculate_similarity_cached(pdf_window, html_window)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    'start': html_start,
                    'end': html_start + self.window_size,
                    'confidence': confidence
                }
                
                # Early termination for perfect matches
                if confidence >= 0.98:
                    return best_match
        
        # If no good hash match, fall back to fuzzy matching for top candidates
        if best_confidence < self.min_match_threshold:
            return self._fuzzy_fallback_search(pdf_window, html_words, used_indices)
        
        return best_match
    
    def _fuzzy_fallback_search(self, pdf_window: List[str], html_words: List[str], 
                              used_indices: Set[int]) -> Optional[Dict]:
        """Fallback fuzzy search for when hash matching fails."""
        best_match = None
        best_confidence = 0
        window_size = len(pdf_window)
        
        # Sample every 5th position for fuzzy matching (much faster)
        step_size = max(1, (len(html_words) - window_size) // 200)  # Limit to ~200 checks
        
        for html_start in range(0, len(html_words) - window_size + 1, step_size):
            if any(idx in used_indices for idx in range(html_start, html_start + window_size)):
                continue
            
            html_window = html_words[html_start:html_start + window_size]
            confidence = self._calculate_similarity_cached(pdf_window, html_window)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    'start': html_start,
                    'end': html_start + window_size,
                    'confidence': confidence
                }
        
        return best_match
    
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
    
    def _calculate_similarity_cached(self, window1: List[str], window2: List[str]) -> float:
        """Calculate similarity with caching for repeated calculations."""
        # Create cache key from window contents
        cache_key = (tuple(w.lower() for w in window1), tuple(w.lower() for w in window2))
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Prevent cache from growing too large
        if len(self.similarity_cache) >= self.max_cache_size:
            # Clear half the cache (simple LRU approximation)
            keys_to_remove = list(self.similarity_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        similarity = self._calculate_similarity(window1, window2)
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _calculate_similarity(self, window1: List[str], window2: List[str]) -> float:
        """Calculate similarity between two word windows with optimizations."""
        if len(window1) != len(window2):
            return 0.0
        
        # Fast path: check if windows are identical
        if window1 == window2:
            return 1.0
        
        # Fast path: check if lowercased windows are identical
        lower1 = [w.lower() for w in window1]
        lower2 = [w.lower() for w in window2]
        if lower1 == lower2:
            return 0.95  # Nearly perfect but not exact case match
        
        exact_matches = 0
        fuzzy_matches = 0
        
        for w1, w2 in zip(window1, window2):
            if w1.lower() == w2.lower():
                exact_matches += 1
            elif self._fuzzy_match_cached(w1, w2):
                fuzzy_matches += 1
        
        total_score = exact_matches + (fuzzy_matches * 0.8)
        return total_score / len(window1)
    
    def _fuzzy_match_cached(self, word1: str, word2: str) -> bool:
        """Cached fuzzy word matching with optimizations."""
        # Normalize case for comparison
        w1, w2 = word1.lower(), word2.lower()
        
        # Create cache key
        cache_key = (w1, w2) if w1 <= w2 else (w2, w1)
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Fast rejection filters
        if abs(len(w1) - len(w2)) > 3:
            self.similarity_cache[cache_key] = False
            return False
        
        # Very short words must be exact
        if len(w1) <= 2 or len(w2) <= 2:
            result = w1 == w2
            self.similarity_cache[cache_key] = result
            return result
        
        # Common financial abbreviations and variations
        financial_equivalents = {
            ('revenue', 'revenues'), ('earning', 'earnings'), ('margin', 'margins'),
            ('q1', 'first'), ('q2', 'second'), ('q3', 'third'), ('q4', 'fourth'),
            ('bn', 'billion'), ('mn', 'million'), ('k', 'thousand'),
            ('ebitda', 'earnings'), ('capex', 'capital'), ('opex', 'operating')
        }
        
        for equiv_pair in financial_equivalents:
            if (w1 in equiv_pair and w2 in equiv_pair):
                self.similarity_cache[cache_key] = True
                return True
        
        # Use sequence matcher for detailed comparison
        similarity = difflib.SequenceMatcher(None, w1, w2).ratio()
        result = similarity > 0.8
        self.similarity_cache[cache_key] = result
        return result
    
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
    """Fresh approach: Index-based sentence alignment with positional fuzzy matching."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.similarity_cache = {}
        self.max_cache_size = 5000
        self.matches = []  # List of SentenceMatch objects
        self.match_counter = 0
    
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
        """Fresh approach: Index-based alignment with fuzzy positional matching."""
        print("    - Using fresh index-based sentence alignment...")
        
        # Reset state
        self.matches = []
        self.match_counter = 0
        
        # Step 1: Create indexed sentences
        pdf_indexed = self._create_indexed_sentences(pdf_sentences, 'PDF')
        html_indexed = self._create_indexed_sentences(html_sentences, 'HTML')
        
        print(f"    - Indexed {len(pdf_indexed)} PDF and {len(html_indexed)} HTML sentences")
        
        # Step 2: Find direct matches first
        self._find_direct_matches(pdf_indexed, html_indexed)
        
        # Step 3: Find fuzzy positional matches (start/middle/end)
        self._find_positional_matches(pdf_indexed, html_indexed)
        
        # Step 4: Assemble multi-sentence matches
        self._assemble_multi_sentence_matches(pdf_indexed, html_indexed)
        
        # Step 5: Convert to legacy format for compatibility
        alignments = self._convert_to_legacy_format(pdf_indexed, html_indexed)
        
        # Store indexed data for interactive UI
        self._last_pdf_indexed = pdf_indexed
        self._last_html_indexed = html_indexed
        
        print(f"    - Created {len(self.matches)} matches, {len(alignments)} final alignments")
        return alignments
    
    def _create_indexed_sentences(self, sentences: List[str], source: str) -> List[IndexedSentence]:
        """Create indexed sentences with original positions."""
        indexed = []
        for i, sentence in enumerate(sentences):
            indexed.append(IndexedSentence(
                index=i,
                text=sentence.strip(),
                is_matched=False,
                match_group_id=None
            ))
        return indexed
    
    def _find_direct_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 2: Find nearly identical sentences first."""
        print("    - Finding direct matches...")
        matched_count = 0
        
        for pdf_sent in pdf_indexed:
            if pdf_sent.is_matched:
                continue
                
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                similarity = self._calculate_sentence_similarity(pdf_sent.text, html_sent.text)
                
                if similarity >= 0.9:  # Very high threshold for direct matches
                    match_id = f"direct_{self.match_counter}"
                    self.match_counter += 1
                    
                    # Mark as matched
                    pdf_sent.is_matched = True
                    pdf_sent.match_group_id = match_id
                    html_sent.is_matched = True
                    html_sent.match_group_id = match_id
                    
                    # Create match
                    self.matches.append(SentenceMatch(
                        match_id=match_id,
                        pdf_sentences=[pdf_sent],
                        html_sentences=[html_sent],
                        match_type='exact',
                        similarity=similarity,
                        confidence='high'
                    ))
                    
                    matched_count += 1
                    break
        
        print(f"      Found {matched_count} direct matches")
    
    def _find_positional_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 3: Find fuzzy matches at sentence start/middle/end."""
        print("    - Finding positional matches...")
        matched_count = 0
        
        # Try to match unmatched sentences to parts of other sentences
        for pdf_sent in pdf_indexed:
            if pdf_sent.is_matched:
                continue
                
            best_match = None
            best_similarity = 0.0
            best_position = None
            
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                # Check if PDF sentence matches start, middle, or end of HTML sentence
                position, similarity = self._check_positional_match(pdf_sent.text, html_sent.text)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_match = html_sent
                    best_similarity = similarity
                    best_position = position
            
            if best_match:
                match_id = f"positional_{self.match_counter}"
                self.match_counter += 1
                
                # Mark as matched
                pdf_sent.is_matched = True
                pdf_sent.match_group_id = match_id
                best_match.is_matched = True
                best_match.match_group_id = match_id
                
                # Create match
                self.matches.append(SentenceMatch(
                    match_id=match_id,
                    pdf_sentences=[pdf_sent],
                    html_sentences=[best_match],
                    match_type=f'fuzzy_{best_position}',
                    similarity=best_similarity,
                    confidence='medium' if best_similarity >= 0.8 else 'low'
                ))
                
                matched_count += 1
        
        print(f"      Found {matched_count} positional matches")
    
    def _check_positional_match(self, pdf_text: str, html_text: str) -> tuple:
        """Check if PDF sentence matches start, middle, or end of HTML sentence."""
        pdf_clean = self._normalize_sentence_robust(pdf_text)
        html_clean = self._normalize_sentence_robust(html_text)
        
        pdf_words = pdf_clean.split()
        html_words = html_clean.split()
        
        if not pdf_words or not html_words:
            return 'none', 0.0
        
        # Check start match
        start_similarity = self._calculate_overlap_similarity(pdf_words, html_words[:len(pdf_words)])
        
        # Check end match
        end_similarity = self._calculate_overlap_similarity(pdf_words, html_words[-len(pdf_words):])
        
        # Check middle match (sliding window)
        middle_similarity = 0.0
        if len(html_words) > len(pdf_words):
            for i in range(len(html_words) - len(pdf_words) + 1):
                window = html_words[i:i + len(pdf_words)]
                similarity = self._calculate_overlap_similarity(pdf_words, window)
                middle_similarity = max(middle_similarity, similarity)
        
        # Return best match
        best_similarity = max(start_similarity, end_similarity, middle_similarity)
        
        if best_similarity == start_similarity:
            return 'start', start_similarity
        elif best_similarity == end_similarity:
            return 'end', end_similarity
        else:
            return 'middle', middle_similarity
    
    def _calculate_overlap_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Calculate similarity between two word lists."""
        if not words1 or not words2:
            return 0.0
        
        matches = 0
        for w1, w2 in zip(words1, words2):
            if w1.lower() == w2.lower():
                matches += 1
            elif self._words_are_similar(w1, w2):
                matches += 0.8
        
        return matches / max(len(words1), len(words2))
    
    def _assemble_multi_sentence_matches(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Step 4: Try to build 2:1, 1:2 matches from unmatched sentences."""
        print("    - Assembling multi-sentence matches...")
        matched_count = 0
        
        # Try to find 2:1 matches (2 PDF sentences -> 1 HTML sentence)
        for i in range(len(pdf_indexed) - 1):
            pdf_sent1 = pdf_indexed[i]
            pdf_sent2 = pdf_indexed[i + 1]
            
            if pdf_sent1.is_matched or pdf_sent2.is_matched:
                continue
            
            combined_pdf = f"{pdf_sent1.text} {pdf_sent2.text}"
            
            for html_sent in html_indexed:
                if html_sent.is_matched:
                    continue
                
                similarity = self._calculate_sentence_similarity(combined_pdf, html_sent.text)
                
                if similarity >= self.similarity_threshold:
                    match_id = f"multi_2to1_{self.match_counter}"
                    self.match_counter += 1
                    
                    # Mark as matched
                    pdf_sent1.is_matched = True
                    pdf_sent1.match_group_id = match_id
                    pdf_sent2.is_matched = True
                    pdf_sent2.match_group_id = match_id
                    html_sent.is_matched = True
                    html_sent.match_group_id = match_id
                    
                    # Create match
                    self.matches.append(SentenceMatch(
                        match_id=match_id,
                        pdf_sentences=[pdf_sent1, pdf_sent2],
                        html_sentences=[html_sent],
                        match_type='multi_2to1',
                        similarity=similarity,
                        confidence='medium' if similarity >= 0.8 else 'low'
                    ))
                    
                    matched_count += 1
                    break
        
        # Try to find 1:2 matches (1 PDF sentence -> 2 HTML sentences)
        for i in range(len(html_indexed) - 1):
            html_sent1 = html_indexed[i]
            html_sent2 = html_indexed[i + 1]
            
            if html_sent1.is_matched or html_sent2.is_matched:
                continue
            
            combined_html = f"{html_sent1.text} {html_sent2.text}"
            
            for pdf_sent in pdf_indexed:
                if pdf_sent.is_matched:
                    continue
                
                similarity = self._calculate_sentence_similarity(pdf_sent.text, combined_html)
                
                if similarity >= self.similarity_threshold:
                    match_id = f"multi_1to2_{self.match_counter}"
                    self.match_counter += 1
                    
                    # Mark as matched
                    pdf_sent.is_matched = True
                    pdf_sent.match_group_id = match_id
                    html_sent1.is_matched = True
                    html_sent1.match_group_id = match_id
                    html_sent2.is_matched = True
                    html_sent2.match_group_id = match_id
                    
                    # Create match
                    self.matches.append(SentenceMatch(
                        match_id=match_id,
                        pdf_sentences=[pdf_sent],
                        html_sentences=[html_sent1, html_sent2],
                        match_type='multi_1to2',
                        similarity=similarity,
                        confidence='medium' if similarity >= 0.8 else 'low'
                    ))
                    
                    matched_count += 1
                    break
        
        print(f"      Found {matched_count} multi-sentence matches")
    
    def _convert_to_legacy_format(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]) -> List[SentenceAlignment]:
        """Step 5: Convert to legacy format, maintaining original order."""
        alignments = []
        
        # Create a mapping of all positions
        max_index = max(len(pdf_indexed), len(html_indexed))
        
        for i in range(max_index):
            pdf_sent = pdf_indexed[i] if i < len(pdf_indexed) else None
            html_sent = html_indexed[i] if i < len(html_indexed) else None
            
            # Check if either sentence is part of a match
            if pdf_sent and pdf_sent.is_matched:
                # Find the match
                for match in self.matches:
                    if any(s.index == pdf_sent.index for s in match.pdf_sentences):
                        # This PDF sentence is part of a match
                        if not any(any(s.index == pdf_sent.index for s in align.pdf_sentences) for align in alignments):
                            # Haven't added this match yet
                            diff_type = DiffType.EQUAL if match.similarity >= 0.9 else DiffType.REPLACE
                            alignments.append(SentenceAlignment(
                                diff_type=diff_type,
                                pdf_sentences=[s.text for s in match.pdf_sentences],
                                html_sentences=[s.text for s in match.html_sentences],
                                similarity=match.similarity
                            ))
                        break
            elif html_sent and html_sent.is_matched:
                # Find the match
                for match in self.matches:
                    if any(s.index == html_sent.index for s in match.html_sentences):
                        # This HTML sentence is part of a match
                        if not any(any(s.index == html_sent.index for s in align.html_sentences) for align in alignments):
                            # Haven't added this match yet
                            diff_type = DiffType.EQUAL if match.similarity >= 0.9 else DiffType.REPLACE
                            alignments.append(SentenceAlignment(
                                diff_type=diff_type,
                                pdf_sentences=[s.text for s in match.pdf_sentences],
                                html_sentences=[s.text for s in match.html_sentences],
                                similarity=match.similarity
                            ))
                        break
            else:
                # Unmatched sentences - add in their original positions
                if pdf_sent and not html_sent:
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.DELETE,
                        pdf_sentences=[pdf_sent.text],
                        html_sentences=[],
                        similarity=0.0
                    ))
                elif html_sent and not pdf_sent:
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.INSERT,
                        pdf_sentences=[],
                        html_sentences=[html_sent.text],
                        similarity=0.0
                    ))
                elif pdf_sent and html_sent and not pdf_sent.is_matched and not html_sent.is_matched:
                    # Both unmatched at same position
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.DELETE,
                        pdf_sentences=[pdf_sent.text],
                        html_sentences=[],
                        similarity=0.0
                    ))
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.INSERT,
                        pdf_sentences=[],
                        html_sentences=[html_sent.text],
                        similarity=0.0
                    ))
        
        return alignments
    
    def _calculate_sentence_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two sentences."""
        return self._calculate_robust_similarity(text1, text2)
    
    
    def _calculate_robust_similarity_cached(self, pdf_sent: str, html_sent: str) -> float:
        """Cached version of robust similarity calculation."""
        # Create cache key
        cache_key = (pdf_sent.strip()[:100], html_sent.strip()[:100])  # Use first 100 chars as key
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Prevent cache from growing too large
        if len(self.similarity_cache) >= self.max_cache_size:
            # Clear half the cache (simple LRU approximation)
            keys_to_remove = list(self.similarity_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.similarity_cache[key]
        
        similarity = self._calculate_robust_similarity(pdf_sent, html_sent)
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def _calculate_robust_similarity(self, pdf_sent: str, html_sent: str) -> float:
        """Calculate similarity with fuzzy matching for common variations."""
        # Normalize both sentences
        pdf_clean = self._normalize_sentence_robust(pdf_sent)
        html_clean = self._normalize_sentence_robust(html_sent)
        
        # Check for substring matches (HTML sentence contained in PDF)
        if html_clean in pdf_clean:
            # Calculate how much of the PDF sentence is the HTML sentence
            html_len = len(html_clean.split())
            pdf_len = len(pdf_clean.split())
            if pdf_len > 0:
                return min(0.95, html_len / pdf_len + 0.3)  # Boost for substring matches
        
        # Check reverse (PDF sentence contained in HTML)
        if pdf_clean in html_clean:
            pdf_len = len(pdf_clean.split())
            html_len = len(html_clean.split())
            if html_len > 0:
                return min(0.95, pdf_len / html_len + 0.3)
        
        # Word-level similarity with fuzzy matching
        pdf_words = set(pdf_clean.split())
        html_words = set(html_clean.split())
        
        if not pdf_words or not html_words:
            return 0.0
        
        # Exact word matches
        exact_matches = pdf_words.intersection(html_words)
        
        # Fuzzy word matches for common variations
        fuzzy_matches = 0
        remaining_pdf = pdf_words - exact_matches
        remaining_html = html_words - exact_matches
        
        for pdf_word in remaining_pdf:
            for html_word in remaining_html:
                if self._words_are_similar(pdf_word, html_word):
                    fuzzy_matches += 1
                    break
        
        total_matches = len(exact_matches) + fuzzy_matches
        total_words = max(len(pdf_words), len(html_words))
        
        return total_matches / total_words if total_words > 0 else 0.0
    
    def _normalize_sentence_robust(self, sentence: str) -> str:
        """Robust normalization handling common variations."""
        normalized = sentence.lower().strip()
        
        # Remove common PDF prefixes (date/time, speaker info)
        # Pattern: date/time followed by dash or colon, then speaker info
        normalized = re.sub(r'^.*?\d{1,2}:\d{2}.*?[-–]\s*', '', normalized)
        normalized = re.sub(r'^.*?\b(am|pm)\b.*?[-–]\s*', '', normalized)
        normalized = re.sub(r'^.*?\b(morning|afternoon|evening)\b.*?[-–]\s*', '', normalized)
        
        # Remove speaker prefixes like "Thanks Asim and"
        normalized = re.sub(r'^(thanks?|thank you|good morning|good afternoon|hello)\s+\w+\s+(and\s+)?', '', normalized)
        
        # Normalize currency symbols and abbreviations
        currency_mappings = {
            'cad': 'dollar',
            'usd': 'dollar', 
            'us': 'dollar',
            '$': 'dollar',
            'cdn': 'dollar',
            'c$': 'dollar',
            'us$': 'dollar'
        }
        
        for old, new in currency_mappings.items():
            normalized = re.sub(r'\b' + re.escape(old) + r'\b', new, normalized)
        
        # Normalize numbers with currency
        normalized = re.sub(r'\bdollar\s*(\d+(?:\.\d+)?)\s*billion\b', r'dollar \1 billion', normalized)
        normalized = re.sub(r'\bdollar\s*(\d+(?:\.\d+)?)\s*million\b', r'dollar \1 million', normalized)
        
        # Remove extra punctuation and whitespace
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _words_are_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to be considered a match."""
        # Handle common variations
        variations = {
            ('reporting', 'reported'),
            ('earnings', 'earning'),
            ('billions', 'billion'),
            ('millions', 'million'),
            ('dollars', 'dollar'),
            ('cad', 'usd', '$', 'dollar', 'cdn'),
            ('quarterly', 'quarter'),
            ('annual', 'yearly'),
        }
        
        # Check if words are in the same variation group
        for variation_group in variations:
            if word1 in variation_group and word2 in variation_group:
                return True
        
        # Check edit distance for similar words
        if abs(len(word1) - len(word2)) <= 2:
            similarity = difflib.SequenceMatcher(None, word1, word2).ratio()
            return similarity >= 0.8
        
        return False
    
    
    def _normalize_sentence(self, sentence: str) -> str:
        """Legacy normalize sentence method (kept for compatibility)."""
        return self._normalize_sentence_robust(sentence)
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Legacy similarity method (kept for compatibility)."""
        return self._calculate_robust_similarity(sent1, sent2)


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
        .interactive-cell {{
            position: relative;
        }}
        .sentence-content {{
            margin-bottom: 8px;
        }}
        .sentence-index {{
            font-size: 11px;
            color: #6c757d;
            font-style: italic;
            margin-bottom: 8px;
        }}
        .match-controls, .unmatched-controls {{
            display: flex;
            gap: 8px;
            align-items: center;
            margin-top: 8px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
            border: 1px solid #e9ecef;
        }}
        .btn-unmatch, .btn-edit-match, .btn-match {{
            background: #007bff;
            color: white;
            border: none;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
        }}
        .btn-unmatch:hover, .btn-edit-match:hover, .btn-match:hover {{
            background: #0056b3;
        }}
        .match-type, .match-target {{
            font-size: 11px;
            padding: 2px 4px;
        }}
        .placeholder {{
            color: #6c757d;
            font-style: italic;
        }}
        .save-controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .btn-save {{
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            cursor: pointer;
            margin-left: 8px;
        }}
        .btn-save:hover {{
            background: #1e7e34;
        }}
        .match-summary {{
            margin-top: 20px;
            padding: 15px;
            background: #e9ecef;
            border-radius: 6px;
        }}
    </style>
    <script>
        let manualMatches = [];
        let modifiedMatches = new Set();
        
        function unmatchRow(rowId) {{
            const row = document.querySelector(`[data-row-id="${{rowId}}"]`);
            if (row) {{
                // Convert to unmatched state
                row.className = 'diff-row deleted-row';
                row.innerHTML = `
                    <div class="diff-cell interactive-cell">
                        <div class="sentence-content">${{row.querySelector('.sentence-content').innerHTML}}</div>
                        <div class="sentence-index">${{row.querySelector('.sentence-index').innerHTML}}</div>
                        <div class="unmatched-controls">
                            <button class="btn-match" onclick="startMatching(${{rowId}}, 'pdf')">🔗 Match</button>
                            <select class="match-target" style="display:none;" onchange="createMatch(${{rowId}}, 'pdf', this.value)">
                                <option value="">Select HTML sentence...</option>
                            </select>
                        </div>
                    </div>
                    <div class="diff-cell placeholder">(manually unmatched)</div>
                `;
                modifiedMatches.add(rowId);
                updateMatchSummary();
            }}
        }}
        
        function editMatch(rowId) {{
            alert('Match editing not yet implemented - use unmatch and re-match for now');
        }}
        
        function updateMatchType(rowId, newType) {{
            modifiedMatches.add(rowId);
            updateMatchSummary();
        }}
        
        function startMatching(rowId, side) {{
            const row = document.querySelector(`[data-row-id="${{rowId}}"]`);
            const select = row.querySelector('.match-target');
            if (select) {{
                select.style.display = 'inline-block';
                // Populate options dynamically based on current unmatched sentences
                // This would require server-side support to get current state
            }}
        }}
        
        function createMatch(rowId, side, targetIndex) {{
            if (!targetIndex) return;
            
            // Record manual match
            manualMatches.push({{
                sourceRow: rowId,
                sourceSide: side,
                targetIndex: parseInt(targetIndex),
                timestamp: new Date().toISOString()
            }});
            
            modifiedMatches.add(rowId);
            updateMatchSummary();
            
            // Visual feedback
            const row = document.querySelector(`[data-row-id="${{rowId}}"]`);
            if (row) {{
                row.style.background = '#d4edda';
                setTimeout(() => {{
                    row.style.background = '';
                }}, 2000);
            }}
        }}
        
        function updateMatchSummary() {{
            let summary = document.querySelector('.match-summary');
            if (!summary) {{
                summary = document.createElement('div');
                summary.className = 'match-summary';
                document.querySelector('.header').appendChild(summary);
            }}
            
            summary.innerHTML = `
                <h4>📝 Manual Changes</h4>
                <p>Modified matches: ${{modifiedMatches.size}}</p>
                <p>Manual matches created: ${{manualMatches.length}}</p>
                <div class="save-controls">
                    <button class="btn-save" onclick="saveMatches()">💾 Save Changes</button>
                    <button class="btn-save" onclick="exportMatches()">📤 Export Matches</button>
                </div>
            `;
        }}
        
        function saveMatches() {{
            const matchData = {{
                manualMatches: manualMatches,
                modifiedRows: Array.from(modifiedMatches),
                timestamp: new Date().toISOString(),
                totalRows: document.querySelectorAll('.diff-row').length
            }};
            
            // Save to localStorage
            localStorage.setItem('transcriptMatches', JSON.stringify(matchData));
            
            // Download as JSON file
            const blob = new Blob([JSON.stringify(matchData, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcript_matches_${{new Date().toISOString().split('T')[0]}}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('✅ Matches saved successfully!');
        }}
        
        function exportMatches() {{
            const allRows = document.querySelectorAll('.diff-row');
            const exportData = [];
            
            allRows.forEach((row, index) => {{
                const rowId = row.getAttribute('data-row-id');
                const matchType = row.getAttribute('data-match-type') || 'unmatched';
                const pdfContent = row.querySelector('.diff-cell:first-child .sentence-content');
                const htmlContent = row.querySelector('.diff-cell:last-child .sentence-content');
                
                exportData.push({{
                    rowId: rowId,
                    matchType: matchType,
                    pdfText: pdfContent ? pdfContent.textContent : '',
                    htmlText: htmlContent ? htmlContent.textContent : '',
                    isModified: modifiedMatches.has(parseInt(rowId))
                }});
            }});
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcript_alignment_export_${{new Date().toISOString().split('T')[0]}}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            alert('📤 Alignment data exported successfully!');
        }}
        
        // Load saved matches on page load
        window.addEventListener('load', function() {{
            const savedMatches = localStorage.getItem('transcriptMatches');
            if (savedMatches) {{
                try {{
                    const data = JSON.parse(savedMatches);
                    manualMatches = data.manualMatches || [];
                    modifiedMatches = new Set(data.modifiedRows || []);
                    updateMatchSummary();
                }} catch (e) {{
                    console.warn('Could not load saved matches:', e);
                }}
            }}
        }});
    </script>
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
        """Create HTML diff table rows with interactive matching capabilities."""
        rows = []
        row_counter = 0
        
        # Get access to the sentence aligner's indexed data if available
        sentence_aligner = self.sentence_aligner
        pdf_indexed = getattr(sentence_aligner, '_last_pdf_indexed', [])
        html_indexed = getattr(sentence_aligner, '_last_html_indexed', [])
        matches = getattr(sentence_aligner, 'matches', [])
        
        for alignment in sentence_alignments:
            row_counter += 1
            
            # Determine row type and styling
            if alignment.diff_type == DiffType.EQUAL or alignment.diff_type == DiffType.REPLACE:
                # Matched sentences - show side by side with interactive controls
                pdf_texts = [html.escape(sent) for sent in alignment.pdf_sentences]
                html_texts = [html.escape(sent) for sent in alignment.html_sentences]
                
                similarity_class = self._get_similarity_class(alignment.similarity)
                similarity_badge = f'<span class="similarity-badge {similarity_class}">{alignment.similarity:.0%}</span>'
                
                # Create interactive controls
                match_controls = f'''
                <div class="match-controls">
                    <button class="btn-unmatch" onclick="unmatchRow({row_counter})">✂️ Unmatch</button>
                    <button class="btn-edit-match" onclick="editMatch({row_counter})">✏️ Edit</button>
                    <select class="match-type" onchange="updateMatchType({row_counter}, this.value)">
                        <option value="1:1" {"selected" if len(pdf_texts) == 1 and len(html_texts) == 1 else ""}>1:1</option>
                        <option value="2:1" {"selected" if len(pdf_texts) == 2 and len(html_texts) == 1 else ""}>2:1</option>
                        <option value="1:2" {"selected" if len(pdf_texts) == 1 and len(html_texts) == 2 else ""}>1:2</option>
                        <option value="3:1" {"selected" if len(pdf_texts) == 3 and len(html_texts) == 1 else ""}>3:1</option>
                        <option value="1:3" {"selected" if len(pdf_texts) == 1 and len(html_texts) == 3 else ""}>1:3</option>
                    </select>
                </div>'''
                
                row_class = "matched-row" if alignment.diff_type == DiffType.EQUAL else "replaced-row"
                
                rows.append(f'''
                <div class="diff-row {row_class}" data-row-id="{row_counter}" data-match-type="{len(pdf_texts)}:{len(html_texts)}">
                    <div class="diff-cell interactive-cell">
                        <div class="sentence-content">{'<br>'.join(pdf_texts)}</div>
                        <div class="sentence-index">PDF: {self._get_sentence_indices(alignment.pdf_sentences, pdf_indexed)}</div>
                    </div>
                    <div class="diff-cell interactive-cell">
                        <div class="sentence-content">{'<br>'.join(html_texts)}</div>
                        <div class="sentence-index">HTML: {self._get_sentence_indices(alignment.html_sentences, html_indexed)}</div>
                        {similarity_badge}
                        {match_controls}
                    </div>
                </div>''')
            
            elif alignment.diff_type == DiffType.DELETE:
                # PDF only content with manual matching option
                for i, sentence in enumerate(alignment.pdf_sentences):
                    escaped_sentence = html.escape(sentence)
                    row_counter += 1 if i > 0 else 0
                    
                    match_controls = f'''
                    <div class="unmatched-controls">
                        <button class="btn-match" onclick="startMatching({row_counter}, 'pdf')">🔗 Match</button>
                        <select class="match-target" style="display:none;" onchange="createMatch({row_counter}, 'pdf', this.value)">
                            <option value="">Select HTML sentence...</option>
                            {self._create_html_options(html_indexed)}
                        </select>
                    </div>'''
                    
                    rows.append(f'''
                    <div class="diff-row deleted-row" data-row-id="{row_counter}" data-side="pdf">
                        <div class="diff-cell interactive-cell">
                            <div class="sentence-content">{escaped_sentence}</div>
                            <div class="sentence-index">PDF: {self._get_sentence_index(sentence, pdf_indexed)}</div>
                            {match_controls}
                        </div>
                        <div class="diff-cell placeholder">(not matched - missing in HTML)</div>
                    </div>''')
            
            elif alignment.diff_type == DiffType.INSERT:
                # HTML only content with manual matching option
                for i, sentence in enumerate(alignment.html_sentences):
                    escaped_sentence = html.escape(sentence)
                    row_counter += 1 if i > 0 else 0
                    
                    match_controls = f'''
                    <div class="unmatched-controls">
                        <button class="btn-match" onclick="startMatching({row_counter}, 'html')">🔗 Match</button>
                        <select class="match-target" style="display:none;" onchange="createMatch({row_counter}, 'html', this.value)">
                            <option value="">Select PDF sentence...</option>
                            {self._create_pdf_options(pdf_indexed)}
                        </select>
                    </div>'''
                    
                    rows.append(f'''
                    <div class="diff-row inserted-row" data-row-id="{row_counter}" data-side="html">
                        <div class="diff-cell placeholder">(not matched - missing in PDF)</div>
                        <div class="diff-cell interactive-cell">
                            <div class="sentence-content">{escaped_sentence}</div>
                            <div class="sentence-index">HTML: {self._get_sentence_index(sentence, html_indexed)}</div>
                            {match_controls}
                        </div>
                    </div>''')
        
        return '\n'.join(rows)
    
    def _get_sentence_indices(self, sentences: List[str], indexed_sentences: List) -> str:
        """Get comma-separated indices for a list of sentences."""
        indices = []
        for sentence in sentences:
            idx = self._get_sentence_index(sentence, indexed_sentences)
            if idx != "?":
                indices.append(idx)
        return ", ".join(indices) if indices else "?"
    
    def _get_sentence_index(self, sentence: str, indexed_sentences: List) -> str:
        """Get the original index of a sentence."""
        try:
            for indexed in indexed_sentences:
                if hasattr(indexed, 'text') and indexed.text == sentence:
                    return str(indexed.index)
            # Fallback to finding by text content
            for i, indexed in enumerate(indexed_sentences):
                text = indexed.text if hasattr(indexed, 'text') else str(indexed)
                if text == sentence:
                    return str(i)
        except:
            pass
        return "?"
    
    def _create_html_options(self, html_indexed: List) -> str:
        """Create HTML options for manual matching."""
        options = []
        for indexed in html_indexed:
            if hasattr(indexed, 'text') and hasattr(indexed, 'index') and hasattr(indexed, 'is_matched'):
                if not indexed.is_matched:
                    text_preview = indexed.text[:50] + "..." if len(indexed.text) > 50 else indexed.text
                    options.append(f'<option value="{indexed.index}">[{indexed.index}] {html.escape(text_preview)}</option>')
        return '\n'.join(options)
    
    def _create_pdf_options(self, pdf_indexed: List) -> str:
        """Create PDF options for manual matching."""
        options = []
        for indexed in pdf_indexed:
            if hasattr(indexed, 'text') and hasattr(indexed, 'index') and hasattr(indexed, 'is_matched'):
                if not indexed.is_matched:
                    text_preview = indexed.text[:50] + "..." if len(indexed.text) > 50 else indexed.text
                    options.append(f'<option value="{indexed.index}">[{indexed.index}] {html.escape(text_preview)}</option>')
        return '\n'.join(options)
    
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