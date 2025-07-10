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
    """Align sentences between PDF and HTML using sequence matching."""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        self.similarity_cache = {}  # Cache for sentence similarity calculations
        self.max_cache_size = 5000  # Smaller cache for sentence-level operations
    
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
        """Align sentences using optimized ordered matching for robust comparison."""
        print("    - Using optimized ordered sentence matching...")
        
        # Use optimized ordered alignment that maintains sequence
        alignments = self._ordered_sequence_alignment(pdf_sentences, html_sentences)
        
        # Second pass: post-processing for multi-sentence matches
        print("    - Post-processing for multi-sentence matches...")
        alignments = self._post_process_alignments(alignments, pdf_sentences, html_sentences)
        
        print(f"    - Created {len(alignments)} final alignments")
        return alignments
    
    def _fuzzy_align_sentences(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Perform fuzzy sentence alignment with substring and similarity matching."""
        alignments = []
        used_pdf_indices = set()
        used_html_indices = set()
        
        total_pdf = len(pdf_sentences)
        print(f"    - Matching {total_pdf:,} PDF sentences against {len(html_sentences):,} HTML sentences...")
        
        # Phase 1: Find exact and near-exact matches
        for pdf_idx, pdf_sent in enumerate(pdf_sentences):
            if pdf_idx in used_pdf_indices:
                continue
            
            # Progress update without newlines
            if pdf_idx % max(1, total_pdf // 20) == 0 or pdf_idx == total_pdf - 1:
                progress = (pdf_idx + 1) / total_pdf * 100
                print(f"\r      Progress: {progress:.1f}% ({pdf_idx + 1:,}/{total_pdf:,})", end="", flush=True)
                
            best_match = None
            best_similarity = 0.0
            best_html_idx = -1
            
            for html_idx, html_sent in enumerate(html_sentences):
                if html_idx in used_html_indices:
                    continue
                
                similarity = self._calculate_robust_similarity(pdf_sent, html_sent)
                
                if similarity > best_similarity and similarity >= 0.6:  # Threshold for matching
                    best_similarity = similarity
                    best_match = html_sent
                    best_html_idx = html_idx
            
            if best_match:
                # Determine match type based on similarity
                if best_similarity >= 0.9:
                    match_type = DiffType.EQUAL
                else:
                    match_type = DiffType.REPLACE
                
                alignments.append(SentenceAlignment(
                    diff_type=match_type,
                    pdf_sentences=[pdf_sent],
                    html_sentences=[best_match],
                    similarity=best_similarity
                ))
                
                used_pdf_indices.add(pdf_idx)
                used_html_indices.add(best_html_idx)
        
        print()  # New line after progress
        
        # Phase 2: Handle unmatched sentences
        for pdf_idx, pdf_sent in enumerate(pdf_sentences):
            if pdf_idx not in used_pdf_indices:
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.DELETE,
                    pdf_sentences=[pdf_sent],
                    html_sentences=[],
                    similarity=0.0
                ))
        
        for html_idx, html_sent in enumerate(html_sentences):
            if html_idx not in used_html_indices:
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.INSERT,
                    pdf_sentences=[],
                    html_sentences=[html_sent],
                    similarity=0.0
                ))
        
        # Sort alignments to maintain original order
        return self._sort_alignments_by_position(alignments, pdf_sentences, html_sentences)
    
    def _ordered_sequence_alignment(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Two-phase ordered sequence alignment that maintains order and handles sentence splits."""
        print(f"    - Using two-phase ordered alignment algorithm...")
        start_time = time.time()
        
        # Phase 1: Initial 1:1 and 1:many/many:1 matching
        initial_alignments = self._phase1_initial_matching(pdf_sentences, html_sentences)
        
        # Phase 2: Improve low-confidence matches by combining adjacent sentences
        final_alignments = self._phase2_improve_matches(initial_alignments, pdf_sentences, html_sentences)
        
        elapsed = time.time() - start_time
        print(f"    - Two-phase alignment completed in {elapsed:.2f}s, {len(final_alignments)} alignments")
        
        return final_alignments
    
    def _phase1_initial_matching(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Phase 1: Sequential matching with lookahead, maintaining strict order."""
        alignments = []
        pdf_idx = 0
        html_idx = 0
        
        print(f"    - Phase 1: Initial matching with lookahead...")
        
        while pdf_idx < len(pdf_sentences) and html_idx < len(html_sentences):
            pdf_sent = pdf_sentences[pdf_idx]
            
            # Try 1:1 matching first
            best_match = self._find_best_sequential_match(
                pdf_sent, html_sentences, html_idx, max_lookahead=5
            )
            
            if best_match['similarity'] >= 0.6:
                # Good 1:1 match found
                # Add any skipped HTML sentences as inserts (maintaining order)
                for skip_idx in range(best_match['offset']):
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.INSERT,
                        pdf_sentences=[],
                        html_sentences=[html_sentences[html_idx + skip_idx]],
                        similarity=0.0
                    ))
                
                # Add the matched pair
                match_type = DiffType.EQUAL if best_match['similarity'] >= 0.9 else DiffType.REPLACE
                alignments.append(SentenceAlignment(
                    diff_type=match_type,
                    pdf_sentences=[pdf_sent],
                    html_sentences=[best_match['sentence']],
                    similarity=best_match['similarity']
                ))
                
                pdf_idx += 1
                html_idx += best_match['offset'] + 1
            else:
                # Try 1:many matching (PDF sentence might match combined HTML sentences)
                combined_match = self._try_one_to_many_match(
                    pdf_sent, html_sentences, html_idx, max_combine=3
                )
                
                if combined_match['similarity'] >= 0.6:
                    # PDF sentence matches multiple HTML sentences
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.REPLACE,
                        pdf_sentences=[pdf_sent],
                        html_sentences=combined_match['sentences'],
                        similarity=combined_match['similarity']
                    ))
                    
                    pdf_idx += 1
                    html_idx += combined_match['count']
                else:
                    # Try many:1 matching (multiple PDF sentences might match one HTML)
                    many_to_one_match = self._try_many_to_one_match(
                        pdf_sentences, pdf_idx, html_sentences[html_idx] if html_idx < len(html_sentences) else "", max_combine=3
                    )
                    
                    if many_to_one_match['similarity'] >= 0.6:
                        # Multiple PDF sentences match one HTML sentence
                        alignments.append(SentenceAlignment(
                            diff_type=DiffType.REPLACE,
                            pdf_sentences=many_to_one_match['sentences'],
                            html_sentences=[html_sentences[html_idx]],
                            similarity=many_to_one_match['similarity']
                        ))
                        
                        pdf_idx += many_to_one_match['count']
                        html_idx += 1
                    else:
                        # No good match - PDF sentence is deleted
                        alignments.append(SentenceAlignment(
                            diff_type=DiffType.DELETE,
                            pdf_sentences=[pdf_sent],
                            html_sentences=[],
                            similarity=0.0
                        ))
                        pdf_idx += 1
        
        # Handle remaining sentences in order
        while pdf_idx < len(pdf_sentences):
            alignments.append(SentenceAlignment(
                diff_type=DiffType.DELETE,
                pdf_sentences=[pdf_sentences[pdf_idx]],
                html_sentences=[],
                similarity=0.0
            ))
            pdf_idx += 1
        
        while html_idx < len(html_sentences):
            alignments.append(SentenceAlignment(
                diff_type=DiffType.INSERT,
                pdf_sentences=[],
                html_sentences=[html_sentences[html_idx]],
                similarity=0.0
            ))
            html_idx += 1
        
        return alignments
    
    def _find_best_sequential_match(self, pdf_sent: str, html_sentences: List[str], 
                                   start_idx: int, max_lookahead: int = 5) -> dict:
        """Find best matching HTML sentence within lookahead window."""
        best_match = {'similarity': 0.0, 'sentence': '', 'offset': 0}
        
        lookahead_limit = min(max_lookahead, len(html_sentences) - start_idx)
        
        for offset in range(lookahead_limit):
            html_sent = html_sentences[start_idx + offset]
            similarity = self._calculate_robust_similarity_cached(pdf_sent, html_sent)
            
            if similarity > best_match['similarity']:
                best_match = {
                    'similarity': similarity,
                    'sentence': html_sent,
                    'offset': offset
                }
                
                # Early termination for excellent matches
                if similarity >= 0.9:
                    break
        
        return best_match
    
    def _try_one_to_many_match(self, pdf_sent: str, html_sentences: List[str], 
                              start_idx: int, max_combine: int = 3) -> dict:
        """Try matching one PDF sentence to multiple combined HTML sentences."""
        best_match = {'similarity': 0.0, 'sentences': [], 'count': 0}
        
        for combine_count in range(2, min(max_combine + 1, len(html_sentences) - start_idx + 1)):
            combined_html_sentences = html_sentences[start_idx:start_idx + combine_count]
            combined_text = ' '.join(combined_html_sentences)
            
            similarity = self._calculate_robust_similarity_cached(pdf_sent, combined_text)
            
            if similarity > best_match['similarity']:
                best_match = {
                    'similarity': similarity,
                    'sentences': combined_html_sentences,
                    'count': combine_count
                }
        
        return best_match
    
    def _try_many_to_one_match(self, pdf_sentences: List[str], start_idx: int, 
                              html_sent: str, max_combine: int = 3) -> dict:
        """Try matching multiple PDF sentences to one HTML sentence."""
        best_match = {'similarity': 0.0, 'sentences': [], 'count': 0}
        
        for combine_count in range(2, min(max_combine + 1, len(pdf_sentences) - start_idx + 1)):
            combined_pdf_sentences = pdf_sentences[start_idx:start_idx + combine_count]
            combined_text = ' '.join(combined_pdf_sentences)
            
            similarity = self._calculate_robust_similarity_cached(combined_text, html_sent)
            
            if similarity > best_match['similarity']:
                best_match = {
                    'similarity': similarity,
                    'sentences': combined_pdf_sentences,
                    'count': combine_count
                }
        
        return best_match
    
    def _phase2_improve_matches(self, alignments: List[SentenceAlignment], 
                               pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Phase 2: Improve low-confidence matches by checking adjacent sentence combinations."""
        print(f"    - Phase 2: Improving {len(alignments)} alignments...")
        
        improved_alignments = []
        i = 0
        
        while i < len(alignments):
            current = alignments[i]
            
            # Only try to improve low-confidence REPLACE matches
            if (current.diff_type == DiffType.REPLACE and 
                current.similarity < 0.8 and 
                len(current.pdf_sentences) == 1 and 
                len(current.html_sentences) == 1):
                
                # Try combining with previous alignment if it's also low confidence
                if (i > 0 and 
                    alignments[i-1].diff_type == DiffType.REPLACE and
                    alignments[i-1].similarity < 0.8):
                    
                    improved_prev = self._try_improve_by_combining(
                        alignments[i-1], current, direction='forward'
                    )
                    
                    if improved_prev and improved_prev.similarity > max(alignments[i-1].similarity, current.similarity) + 0.1:
                        # Replace previous two alignments with improved one
                        improved_alignments[-1] = improved_prev  # Replace the last added (previous)
                        i += 1  # Skip current since it's now combined
                        continue
                
                # Try combining with next alignment if it exists and is low confidence
                if (i + 1 < len(alignments) and 
                    alignments[i+1].diff_type == DiffType.REPLACE and
                    alignments[i+1].similarity < 0.8):
                    
                    improved_next = self._try_improve_by_combining(
                        current, alignments[i+1], direction='forward'
                    )
                    
                    if improved_next and improved_next.similarity > max(current.similarity, alignments[i+1].similarity) + 0.1:
                        # Add improved alignment and skip next
                        improved_alignments.append(improved_next)
                        i += 2  # Skip both current and next
                        continue
            
            # No improvement found, keep original alignment
            improved_alignments.append(current)
            i += 1
        
        return improved_alignments
    
    def _try_improve_by_combining(self, align1: SentenceAlignment, align2: SentenceAlignment, 
                                 direction: str) -> Optional[SentenceAlignment]:
        """Try to improve alignment by combining two adjacent alignments."""
        combined_pdf = align1.pdf_sentences + align2.pdf_sentences
        combined_html = align1.html_sentences + align2.html_sentences
        
        # Try different combination patterns
        combinations = [
            # Combine all PDF sentences with all HTML sentences
            (combined_pdf, combined_html),
            # Try PDF1 with HTML1+HTML2 
            (align1.pdf_sentences, combined_html) if align1.pdf_sentences else None,
            # Try PDF1+PDF2 with HTML1
            (combined_pdf, align1.html_sentences) if align1.html_sentences else None,
        ]
        
        best_improvement = None
        best_similarity = 0.0
        
        for combo in combinations:
            if combo is None:
                continue
                
            pdf_text = ' '.join(combo[0]) if combo[0] else ''
            html_text = ' '.join(combo[1]) if combo[1] else ''
            
            if pdf_text and html_text:
                similarity = self._calculate_robust_similarity_cached(pdf_text, html_text)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_improvement = SentenceAlignment(
                        diff_type=DiffType.EQUAL if similarity >= 0.9 else DiffType.REPLACE,
                        pdf_sentences=combo[0],
                        html_sentences=combo[1],
                        similarity=similarity
                    )
        
        return best_improvement
    
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
    
    def _sort_alignments_by_position(self, alignments: List[SentenceAlignment], 
                                   pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Sort alignments to maintain reasonable order."""
        # Create position mapping
        pdf_positions = {sent: idx for idx, sent in enumerate(pdf_sentences)}
        html_positions = {sent: idx for idx, sent in enumerate(html_sentences)}
        
        def get_sort_key(alignment):
            pdf_pos = min([pdf_positions.get(sent, 999999) for sent in alignment.pdf_sentences] + [999999])
            html_pos = min([html_positions.get(sent, 999999) for sent in alignment.html_sentences] + [999999])
            return min(pdf_pos, html_pos)
        
        return sorted(alignments, key=get_sort_key)
    
    def _post_process_alignments(self, alignments: List[SentenceAlignment], 
                                pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Post-process alignments to handle multi-sentence matches."""
        # Identify unmatched sentences
        unmatched_pdf = []
        unmatched_html = []
        matched_alignments = []
        
        for align in alignments:
            if align.diff_type == DiffType.DELETE:
                unmatched_pdf.extend(align.pdf_sentences)
            elif align.diff_type == DiffType.INSERT:
                unmatched_html.extend(align.html_sentences)
            else:
                matched_alignments.append(align)
        
        print(f"      Found {len(unmatched_pdf)} unmatched PDF and {len(unmatched_html)} unmatched HTML sentences")
        
        # Try to match consecutive unmatched sentences
        new_matches = []
        
        # Phase 1: Try combining consecutive PDF sentences to match HTML
        if unmatched_pdf and unmatched_html:
            print("      Phase 1: Combining PDF sentences to match HTML...")
            new_matches.extend(self._match_combined_sentences(unmatched_pdf, unmatched_html, "pdf_to_html"))
        
        # Phase 2: Try combining consecutive HTML sentences to match PDF
        if unmatched_pdf and unmatched_html:
            print("      Phase 2: Combining HTML sentences to match PDF...")
            new_matches.extend(self._match_combined_sentences(unmatched_pdf, unmatched_html, "html_to_pdf"))
        
        # Remove newly matched sentences from unmatched lists
        for match in new_matches:
            for sent in match.pdf_sentences:
                if sent in unmatched_pdf:
                    unmatched_pdf.remove(sent)
            for sent in match.html_sentences:
                if sent in unmatched_html:
                    unmatched_html.remove(sent)
        
        # Rebuild final alignments
        final_alignments = matched_alignments + new_matches
        
        # Add remaining unmatched sentences
        if unmatched_pdf:
            final_alignments.append(SentenceAlignment(
                diff_type=DiffType.DELETE,
                pdf_sentences=unmatched_pdf,
                html_sentences=[],
                similarity=0.0
            ))
        
        if unmatched_html:
            final_alignments.append(SentenceAlignment(
                diff_type=DiffType.INSERT,
                pdf_sentences=[],
                html_sentences=unmatched_html,
                similarity=0.0
            ))
        
        # Re-sort by position
        return self._sort_alignments_by_position(final_alignments, pdf_sentences, html_sentences)
    
    def _match_combined_sentences(self, unmatched_pdf: List[str], unmatched_html: List[str], 
                                 direction: str) -> List[SentenceAlignment]:
        """Try to match by combining consecutive sentences."""
        new_matches = []
        
        if direction == "pdf_to_html":
            total_html = len(unmatched_html)
            # Try combining 2-3 consecutive PDF sentences to match single HTML sentences
            for html_idx, html_sent in enumerate(unmatched_html[:]):  # Copy to iterate safely
                if html_idx % max(1, total_html // 10) == 0:
                    progress = (html_idx + 1) / total_html * 100
                    print(f"\r        PDF→HTML: {progress:.1f}% ({html_idx + 1}/{total_html})", end="", flush=True)
                
                # Try combining 2 consecutive PDF sentences
                for i in range(len(unmatched_pdf) - 1):
                    combined_pdf = unmatched_pdf[i] + " " + unmatched_pdf[i + 1]
                    similarity = self._calculate_robust_similarity(combined_pdf, html_sent)
                    
                    if similarity >= 0.7:  # Lower threshold for combined matches
                        new_matches.append(SentenceAlignment(
                            diff_type=DiffType.REPLACE if similarity < 0.9 else DiffType.EQUAL,
                            pdf_sentences=[unmatched_pdf[i], unmatched_pdf[i + 1]],
                            html_sentences=[html_sent],
                            similarity=similarity
                        ))
                        break
                
                # Try combining 3 consecutive PDF sentences if no 2-match found
                if html_sent in unmatched_html:  # Still unmatched
                    for i in range(len(unmatched_pdf) - 2):
                        combined_pdf = " ".join([unmatched_pdf[i], unmatched_pdf[i + 1], unmatched_pdf[i + 2]])
                        similarity = self._calculate_robust_similarity(combined_pdf, html_sent)
                        
                        if similarity >= 0.7:
                            new_matches.append(SentenceAlignment(
                                diff_type=DiffType.REPLACE if similarity < 0.9 else DiffType.EQUAL,
                                pdf_sentences=[unmatched_pdf[i], unmatched_pdf[i + 1], unmatched_pdf[i + 2]],
                                html_sentences=[html_sent],
                                similarity=similarity
                            ))
                            break
            if total_html > 0:
                print()  # New line after progress
        
        else:  # html_to_pdf
            total_pdf = len(unmatched_pdf)
            # Try combining 2-3 consecutive HTML sentences to match single PDF sentences
            for pdf_idx, pdf_sent in enumerate(unmatched_pdf[:]):  # Copy to iterate safely
                if pdf_idx % max(1, total_pdf // 10) == 0:
                    progress = (pdf_idx + 1) / total_pdf * 100
                    print(f"\r        HTML→PDF: {progress:.1f}% ({pdf_idx + 1}/{total_pdf})", end="", flush=True)
                
                # Try combining 2 consecutive HTML sentences
                for i in range(len(unmatched_html) - 1):
                    combined_html = unmatched_html[i] + " " + unmatched_html[i + 1]
                    similarity = self._calculate_robust_similarity(pdf_sent, combined_html)
                    
                    if similarity >= 0.7:
                        new_matches.append(SentenceAlignment(
                            diff_type=DiffType.REPLACE if similarity < 0.9 else DiffType.EQUAL,
                            pdf_sentences=[pdf_sent],
                            html_sentences=[unmatched_html[i], unmatched_html[i + 1]],
                            similarity=similarity
                        ))
                        break
            if total_pdf > 0:
                print()  # New line after progress
        
        return new_matches
    
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