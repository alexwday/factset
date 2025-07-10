#!/usr/bin/env python3
"""
Test the integrated sliding window approach in the main script.
This creates a minimal test using the actual updated SentenceAligner.
"""

import sys
import re
import time
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from enum import Enum

class DiffType(Enum):
    EQUAL = "equal"
    DELETE = "delete"
    INSERT = "insert"
    REPLACE = "replace"

@dataclass
class IndexedSentence:
    index: int
    text: str
    is_matched: bool = False
    match_group_id: Optional[str] = None

@dataclass
class SentenceMatch:
    match_id: str
    pdf_sentences: List[IndexedSentence]
    html_sentences: List[IndexedSentence]
    match_type: str
    similarity: float = 0.0
    confidence: str = 'high'

@dataclass
class SentenceAlignment:
    diff_type: DiffType
    pdf_sentences: List[str]
    html_sentences: List[str]
    similarity: float = 0.0

# Import the sliding window methods from our updated main script
class SlidingWindowSentenceAligner:
    """Test wrapper for the integrated sliding window approach."""
    
    def __init__(self, similarity_threshold: float = 0.7, window_size: int = 4, min_match_length: int = 3):
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self.min_match_length = min_match_length
        self.similarity_cache = {}
        self.max_cache_size = 5000
        self.matches = []
        self.match_counter = 0
        self.word_matches = []
    
    def extract_sentences(self, text_lines: List[str]) -> List[str]:
        """Extract sentences from text lines."""
        full_text = ' '.join(text_lines)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', full_text)
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def align_sentences(self, pdf_sentences: List[str], html_sentences: List[str]) -> List[SentenceAlignment]:
        """Advanced sliding window alignment with word-level precision."""
        print("    - Using advanced sliding window sentence alignment...")
        
        # Reset state
        self.matches = []
        self.match_counter = 0
        self.word_matches = []
        
        # Step 1: Create indexed sentences
        pdf_indexed = self._create_indexed_sentences(pdf_sentences, 'PDF')
        html_indexed = self._create_indexed_sentences(html_sentences, 'HTML')
        
        print(f"    - Indexed {len(pdf_indexed)} PDF and {len(html_indexed)} HTML sentences")
        
        # Step 2: Perform sliding window word-level matching
        word_match_results = self._sliding_window_matching(pdf_sentences, html_sentences)
        
        # Step 3: Convert word matches to sentence-level alignments
        self._build_sentence_alignments_from_word_matches(pdf_indexed, html_indexed, word_match_results)
        
        # Step 4: Handle remaining unmatched sentences
        self._handle_unmatched_sentences(pdf_indexed, html_indexed)
        
        # Step 5: Convert to legacy format for compatibility
        alignments = self._convert_to_position_ordered_format(pdf_indexed, html_indexed)
        
        print(f"    - Created {len(self.word_matches)} word matches, {len(self.matches)} sentence matches, {len(alignments)} final alignments")
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
    
    def _sliding_window_matching(self, pdf_sentences: List[str], html_sentences: List[str]) -> Dict:
        """Perform sliding window word-level matching."""
        print("    - Performing sliding window word-level matching...")
        start_time = time.time()
        
        # Tokenize all sentences
        pdf_tokenized = [self._tokenize_sentence(sent, i) for i, sent in enumerate(pdf_sentences)]
        html_tokenized = [self._tokenize_sentence(sent, i) for i, sent in enumerate(html_sentences)]
        
        # Build word position index for fast lookup
        html_word_index = self._build_word_index(html_tokenized)
        
        # Find all word-level matches
        word_matches = []
        total_windows = sum(max(0, len(sent['words']) - self.window_size + 1) for sent in pdf_tokenized)
        processed_windows = 0
        
        for pdf_sent in pdf_tokenized:
            sentence_matches = self._find_word_matches_for_sentence(
                pdf_sent, html_tokenized, html_word_index
            )
            word_matches.extend(sentence_matches)
            
            # Progress tracking
            processed_windows += max(0, len(pdf_sent['words']) - self.window_size + 1)
            if total_windows > 0 and processed_windows % max(1, total_windows // 10) == 0:
                progress = processed_windows / total_windows * 100
                print(f"\r      Progress: {progress:.1f}% ({processed_windows:,}/{total_windows:,})", end="", flush=True)
        
        if total_windows > 0:
            print()  # New line after progress
        
        # Merge overlapping matches
        optimized_matches = self._merge_word_matches(word_matches)
        
        elapsed = time.time() - start_time
        print(f"      Found {len(optimized_matches)} word matches in {elapsed:.2f}s")
        
        # Store for later use
        self.word_matches = optimized_matches
        
        return {
            'word_matches': optimized_matches,
            'pdf_tokenized': pdf_tokenized,
            'html_tokenized': html_tokenized
        }
    
    def _tokenize_sentence(self, sentence: str, sentence_idx: int) -> Dict:
        """Tokenize a sentence into words with position tracking."""
        normalized = re.sub(r'\s+', ' ', sentence.strip().lower())
        words = []
        
        raw_words = normalized.split()
        for word in raw_words:
            clean_word = re.sub(r'[^\w\'-]', '', word)
            if clean_word and len(clean_word) >= 2:
                words.append(clean_word)
        
        return {
            'sentence_idx': sentence_idx,
            'original_text': sentence,
            'normalized_text': normalized,
            'words': words,
            'word_count': len(words)
        }
    
    def _build_word_index(self, tokenized_sentences: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
        """Build an index of word -> [(sentence_idx, word_idx), ...]"""
        from collections import defaultdict
        word_index = defaultdict(list)
        
        for sent in tokenized_sentences:
            for word_idx, word in enumerate(sent['words']):
                word_index[word].append((sent['sentence_idx'], word_idx))
        
        return dict(word_index)
    
    def _find_word_matches_for_sentence(self, pdf_sent: Dict, html_tokenized: List[Dict], 
                                       html_word_index: Dict) -> List[Dict]:
        """Find all word matches for a single PDF sentence using sliding windows."""
        matches = []
        pdf_words = pdf_sent['words']
        
        # Try all possible window positions
        for start_pos in range(len(pdf_words) - self.window_size + 1):
            window = pdf_words[start_pos:start_pos + self.window_size]
            
            # Find potential matches for this window
            candidates = self._find_window_candidates(window, html_word_index)
            
            # For each candidate, try to extend the match
            for html_sent_idx, html_start_pos in candidates:
                html_sent = html_tokenized[html_sent_idx]
                
                # Try to extend the match in both directions
                extended_match = self._extend_word_match(
                    pdf_words, start_pos, start_pos + self.window_size,
                    html_sent['words'], html_start_pos, html_start_pos + self.window_size
                )
                
                if extended_match and extended_match['length'] >= self.min_match_length:
                    # Calculate similarity
                    similarity = self._calculate_word_match_similarity(
                        pdf_words[extended_match['pdf_start']:extended_match['pdf_end']],
                        html_sent['words'][extended_match['html_start']:extended_match['html_end']]
                    )
                    
                    if similarity >= 0.7:  # Similarity threshold
                        matches.append({
                            'pdf_sentence_idx': pdf_sent['sentence_idx'],
                            'html_sentence_idx': html_sent_idx,
                            'pdf_word_start': extended_match['pdf_start'],
                            'pdf_word_end': extended_match['pdf_end'],
                            'html_word_start': extended_match['html_start'],
                            'html_word_end': extended_match['html_end'],
                            'matched_text': ' '.join(pdf_words[extended_match['pdf_start']:extended_match['pdf_end']]),
                            'similarity': similarity,
                            'length': extended_match['length']
                        })
        
        return matches
    
    def _find_window_candidates(self, window: List[str], html_word_index: Dict) -> List[Tuple[int, int]]:
        """Find all potential starting positions for a window in HTML sentences."""
        candidates = []
        
        first_word = window[0]
        if first_word in html_word_index:
            for sent_idx, word_idx in html_word_index[first_word]:
                candidates.append((sent_idx, word_idx))
        
        return candidates
    
    def _extend_word_match(self, pdf_words: List[str], pdf_start: int, pdf_end: int,
                          html_words: List[str], html_start: int, html_end: int) -> Optional[Dict]:
        """Extend a match bidirectionally as far as possible."""
        
        # Extend forward
        while (pdf_end < len(pdf_words) and 
               html_end < len(html_words) and 
               self._words_similar_sliding(pdf_words[pdf_end], html_words[html_end])):
            pdf_end += 1
            html_end += 1
        
        # Extend backward
        while (pdf_start > 0 and 
               html_start > 0 and 
               self._words_similar_sliding(pdf_words[pdf_start - 1], html_words[html_start - 1])):
            pdf_start -= 1
            html_start -= 1
        
        match_length = pdf_end - pdf_start
        
        if match_length >= self.min_match_length:
            return {
                'pdf_start': pdf_start,
                'pdf_end': pdf_end,
                'html_start': html_start,
                'html_end': html_end,
                'length': match_length
            }
        
        return None
    
    def _words_similar_sliding(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough to extend a match."""
        if word1 == word2:
            return True
        
        # Handle common variations
        variations = {
            ('revenue', 'revenues'),
            ('margin', 'margins'),
            ('earning', 'earnings'),
            ('quarter', 'quarterly'),
            ('year', 'yearly', 'annual'),
            ('percent', '%', 'percentage'),
        }
        
        for variant_group in variations:
            if word1 in variant_group and word2 in variant_group:
                return True
        
        # Check edit distance for close matches
        if abs(len(word1) - len(word2)) <= 1 and len(word1) >= 4:
            common_chars = sum(1 for a, b in zip(word1, word2) if a == b)
            similarity = common_chars / max(len(word1), len(word2))
            return similarity >= 0.8
        
        return False
    
    def _calculate_word_match_similarity(self, pdf_words: List[str], html_words: List[str]) -> float:
        """Calculate similarity between two word sequences."""
        if len(pdf_words) != len(html_words):
            return 0.0
        
        matches = sum(1 for w1, w2 in zip(pdf_words, html_words) if self._words_similar_sliding(w1, w2))
        return matches / len(pdf_words)
    
    def _merge_word_matches(self, matches: List[Dict]) -> List[Dict]:
        """Merge overlapping word matches to avoid duplication."""
        if not matches:
            return matches
        
        matches.sort(key=lambda m: (m['pdf_sentence_idx'], m['pdf_word_start'], m['pdf_word_end']))
        
        merged = [matches[0]]
        
        for current in matches[1:]:
            last = merged[-1]
            
            if (current['pdf_sentence_idx'] == last['pdf_sentence_idx'] and
                current['pdf_word_start'] < last['pdf_word_end']):
                if current['similarity'] > last['similarity'] or current['pdf_word_end'] > last['pdf_word_end']:
                    merged[-1] = current
            else:
                merged.append(current)
        
        return merged
    
    def _build_sentence_alignments_from_word_matches(self, pdf_indexed: List[IndexedSentence], 
                                                    html_indexed: List[IndexedSentence], 
                                                    word_match_results: Dict):
        """Build sentence-level alignments from word-level matches."""
        print("    - Building sentence alignments from word matches...")
        
        sentence_match_groups = {}
        
        for word_match in self.word_matches:
            pdf_idx = word_match['pdf_sentence_idx']
            html_idx = word_match['html_sentence_idx']
            key = (pdf_idx, html_idx)
            
            if key not in sentence_match_groups:
                sentence_match_groups[key] = []
            sentence_match_groups[key].append(word_match)
        
        for (pdf_idx, html_idx), word_matches in sentence_match_groups.items():
            total_similarity = sum(wm['similarity'] * wm['length'] for wm in word_matches)
            total_length = sum(wm['length'] for wm in word_matches)
            avg_similarity = total_similarity / total_length if total_length > 0 else 0.0
            
            if avg_similarity >= self.similarity_threshold:
                match_id = f"sliding_{self.match_counter}"
                self.match_counter += 1
                
                pdf_indexed[pdf_idx].is_matched = True
                pdf_indexed[pdf_idx].match_group_id = match_id
                html_indexed[html_idx].is_matched = True
                html_indexed[html_idx].match_group_id = match_id
                
                self.matches.append(SentenceMatch(
                    match_id=match_id,
                    pdf_sentences=[pdf_indexed[pdf_idx]],
                    html_sentences=[html_indexed[html_idx]],
                    match_type='sliding_window',
                    similarity=avg_similarity,
                    confidence='high' if avg_similarity >= 0.9 else 'medium' if avg_similarity >= 0.8 else 'low'
                ))
        
        print(f"      Created {len(self.matches)} sentence matches from word matches")
    
    def _handle_unmatched_sentences(self, pdf_indexed: List[IndexedSentence], html_indexed: List[IndexedSentence]):
        """Handle sentences that weren't matched by sliding window."""
        # Simple fallback for unmatched sentences
        pass
    
    def _convert_to_position_ordered_format(self, pdf_indexed: List[IndexedSentence], 
                                          html_indexed: List[IndexedSentence]) -> List[SentenceAlignment]:
        """Convert to legacy format, maintaining strict positional order for ALL sentences."""
        alignments = []
        
        # Track which sentences have been processed
        pdf_processed = set()
        html_processed = set()
        
        # Process PDF sentences in order
        for pdf_idx in range(len(pdf_indexed)):
            pdf_sent = pdf_indexed[pdf_idx]
            
            if pdf_sent.is_matched and pdf_idx not in pdf_processed:
                # Find the match for this PDF sentence
                match_found = False
                for match in self.matches:
                    if any(s.index == pdf_idx for s in match.pdf_sentences):
                        # Mark all sentences in this match as processed
                        for s in match.pdf_sentences:
                            pdf_processed.add(s.index)
                        for s in match.html_sentences:
                            html_processed.add(s.index)
                        
                        # Add the match
                        diff_type = DiffType.EQUAL if match.similarity >= 0.9 else DiffType.REPLACE
                        alignments.append(SentenceAlignment(
                            diff_type=diff_type,
                            pdf_sentences=[s.text for s in match.pdf_sentences],
                            html_sentences=[s.text for s in match.html_sentences],
                            similarity=match.similarity
                        ))
                        match_found = True
                        break
                
                if not match_found:
                    alignments.append(SentenceAlignment(
                        diff_type=DiffType.DELETE,
                        pdf_sentences=[pdf_sent.text],
                        html_sentences=[],
                        similarity=0.0
                    ))
                    pdf_processed.add(pdf_idx)
            
            elif not pdf_sent.is_matched and pdf_idx not in pdf_processed:
                # Unmatched PDF sentence
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.DELETE,
                    pdf_sentences=[pdf_sent.text],
                    html_sentences=[],
                    similarity=0.0
                ))
                pdf_processed.add(pdf_idx)
        
        # Process HTML sentences that weren't included in matches
        for html_idx in range(len(html_indexed)):
            if html_idx not in html_processed:
                html_sent = html_indexed[html_idx]
                alignments.append(SentenceAlignment(
                    diff_type=DiffType.INSERT,
                    pdf_sentences=[],
                    html_sentences=[html_sent.text],
                    similarity=0.0
                ))
                html_processed.add(html_idx)
        
        # Sort alignments by position
        def get_position_key(alignment):
            pdf_indices = []
            html_indices = []
            
            for pdf_text in alignment.pdf_sentences:
                for i, sent in enumerate(pdf_indexed):
                    if sent.text == pdf_text:
                        pdf_indices.append(i)
                        break
            
            for html_text in alignment.html_sentences:
                for i, sent in enumerate(html_indexed):
                    if sent.text == html_text:
                        html_indices.append(i)
                        break
            
            pdf_min = min(pdf_indices) if pdf_indices else float('inf')
            html_min = min(html_indices) if html_indices else float('inf')
            
            if alignment.diff_type == DiffType.DELETE:
                return (pdf_min, 0)
            elif alignment.diff_type == DiffType.INSERT:
                return (html_min, 1)
            else:
                return (min(pdf_min, html_min), 0)
        
        alignments.sort(key=get_position_key)
        return alignments

def test_integrated_sliding_window():
    """Test the integrated sliding window approach."""
    
    # Test case with cross-sentence matching
    pdf_sentences = [
        "Revenue increased significantly in Q3.",
        "We are pleased with the results.",
        "Operating margins improved by 2%.",
        "Looking forward, we expect continued growth."
    ]
    
    html_sentences = [
        "Revenue increased significantly in Q3, and we are pleased with the results.",
        "Operating margins improved by 2%.",
        "Looking forward, we expect continued growth.",
        "Additional disclosure information."
    ]
    
    print("🧪 Testing Integrated Sliding Window Alignment")
    print("=" * 60)
    
    print("\n📄 PDF Sentences:")
    for i, sent in enumerate(pdf_sentences):
        print(f"  [{i}] {sent}")
    
    print("\n🌐 HTML Sentences:")
    for i, sent in enumerate(html_sentences):
        print(f"  [{i}] {sent}")
    
    print("\n🔗 Running Integrated Sliding Window Alignment...")
    aligner = SlidingWindowSentenceAligner(similarity_threshold=0.6, window_size=4, min_match_length=3)
    alignments = aligner.align_sentences(pdf_sentences, html_sentences)
    
    print(f"\n📊 Results: {len(alignments)} alignments created")
    print("=" * 60)
    
    # Display results
    for i, alignment in enumerate(alignments):
        print(f"\nAlignment {i+1}: {alignment.diff_type.value.upper()}")
        
        if alignment.pdf_sentences:
            print(f"  📄 PDF ({len(alignment.pdf_sentences)}): {'; '.join(alignment.pdf_sentences)}")
        else:
            print(f"  📄 PDF: (none)")
            
        if alignment.html_sentences:
            print(f"  🌐 HTML ({len(alignment.html_sentences)}): {'; '.join(alignment.html_sentences)}")
        else:
            print(f"  🌐 HTML: (none)")
            
        if alignment.similarity > 0:
            print(f"  📈 Similarity: {alignment.similarity:.1%}")
    
    # Show word matches found
    print(f"\n🔍 Word-Level Matches Found:")
    for i, word_match in enumerate(aligner.word_matches):
        print(f"  {i+1}. '{word_match['matched_text']}' (similarity: {word_match['similarity']:.1%})")
        print(f"      PDF[{word_match['pdf_sentence_idx']}] words {word_match['pdf_word_start']}:{word_match['pdf_word_end']}")
        print(f"      HTML[{word_match['html_sentence_idx']}] words {word_match['html_word_start']}:{word_match['html_word_end']}")
    
    print(f"\n🎉 Integrated sliding window test completed!")
    print(f"✅ Successfully found {len(aligner.word_matches)} word-level matches")
    print(f"✅ Created {len(aligner.matches)} sentence-level matches")

if __name__ == "__main__":
    try:
        test_integrated_sliding_window()
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()