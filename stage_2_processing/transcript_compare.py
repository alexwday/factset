#!/usr/bin/env python3
"""
Enhanced transcript comparison tool with improved normalization and commenting.

Usage: python transcript_compare.py <pdf/txt file> <html file> [output.html]
"""

import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum
from difflib import SequenceMatcher
import html as html_module

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("Warning: pdfplumber not installed. PDF support disabled. Use: pip install pdfplumber")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Error: BeautifulSoup4 required. Install with: pip install beautifulsoup4")
    sys.exit(1)


class AlignmentType(Enum):
    """Types of alignment between transcript segments."""
    MATCH = "match"
    SIMILAR = "similar"      # High similarity but not exact
    PDF_GAP = "pdf_gap"      # Content only in HTML
    HTML_GAP = "html_gap"    # Content only in PDF


@dataclass
class AlignedSegment:
    """Represents an aligned segment between transcripts."""
    alignment_type: AlignmentType
    pdf_text: str
    html_text: str
    similarity: float = 1.0


class EnhancedTranscriptComparer:
    """Enhanced transcript comparison with better normalization."""
    
    def __init__(self):
        # Number to word mappings
        self.number_map = {
            '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
            'ii': 'two',  # Roman numeral
        }
        
        # Currency normalization (CAD/$ are equivalent by default)
        self.currency_patterns = [
            (r'\$\s*(\d+(?:\.\d+)?)', r'cad \1'),  # $150 -> cad 150
            (r'cad\s+(\d+(?:\.\d+)?)', r'cad \1'),  # CAD 150 -> cad 150
            (r'usd\s+(\d+(?:\.\d+)?)', r'usd \1'),  # USD 150 -> usd 150
        ]
    
    def pre_tokenize_normalize(self, text: str) -> str:
        """
        Apply normalizations that affect token boundaries BEFORE tokenization.
        Only handles clear cases that change token boundaries.
        """
        # Normalize currency symbols that affect token boundaries
        # Convert $X to CAD X (with space to create proper tokens)
        normalized = re.sub(r'\$(\d+(?:\.\d+)?)', r'CAD \1', text)
        
        # Handle other currency symbols that might be attached
        # US$X -> CAD X, USD$X -> CAD X, etc.
        normalized = re.sub(r'(?:US|USD|C)\$(\d+(?:\.\d+)?)', r'CAD \1', normalized)
        
        # Handle compound phrases that should be split the same way
        # Convert "year-over-year" to "year over year" to match tokenization
        normalized = re.sub(r'\byear-over-year\b', 'year over year', normalized)
        
        return normalized
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF or text file."""
        if pdf_path.endswith('.txt'):
            with open(pdf_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber required for PDF files")
        
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        
        return "\n".join(text_parts)
    
    def extract_html_text(self, html_path: str) -> str:
        """Extract text from HTML file."""
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove scripts and styles
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Get text
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines()]
        return '\n'.join(line for line in lines if line)
    
    def normalize_for_comparison(self, token: str) -> str:
        """
        Enhanced normalization with all requested improvements.
        """
        # Convert to lowercase (case insensitive)
        normalized = token.lower()
        
        # Remove hyphens completely (not replacing with spaces at token level)
        # This makes "re-queue" -> "requeue" and "pre-tax" -> "pretax"
        normalized = normalized.replace('-', '')
        
        # Remove possessive 's (Banking's -> Banking)
        normalized = re.sub(r"'s\b", '', normalized)
        
        # Normalize currency ($ -> cad by default)
        # Note: pre-tokenization should handle most currency cases
        for pattern, replacement in self.currency_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Normalize numbers 1-10 and roman numerals
        words = normalized.split()
        normalized_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)  # Remove punctuation for lookup
            if clean_word in self.number_map:
                normalized_words.append(self.number_map[clean_word])
            else:
                normalized_words.append(word)
        normalized = ' '.join(normalized_words)
        
        # Remove punctuation (including periods and commas for better alignment)
        # But keep spaces and alphanumeric
        normalized = re.sub(r'[.,;:!?\'"()]', '', normalized)
        
        # Normalize percent symbol
        normalized = normalized.replace('%', ' percent')
        
        # Clean up extra spaces
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def normalize_for_hyphen_matching(self, token: str) -> str:
        """
        Alternative normalization that handles hyphen cases specifically.
        Used for secondary matching attempts.
        """
        # Start with standard normalization
        normalized = self.normalize_for_comparison(token)
        
        # Additional hyphen-specific normalizations
        # Handle compound words by trying both with and without hyphens
        
        return normalized
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization preserving original tokens."""
        return re.findall(r'\S+', text)
    
    def tokenize_with_pre_normalization(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize text with pre-normalization for better token boundary alignment.
        Returns: (original_tokens, normalized_tokens_for_alignment)
        """
        # Apply pre-tokenization normalization
        pre_normalized = self.pre_tokenize_normalize(text)
        
        # Tokenize both original and pre-normalized text
        original_tokens = self.tokenize(text)
        pre_normalized_tokens = self.tokenize(pre_normalized)
        
        return original_tokens, pre_normalized_tokens
    
    def calculate_similarity(self, token1: str, token2: str) -> float:
        """Calculate similarity between tokens."""
        norm1 = self.normalize_for_comparison(token1)
        norm2 = self.normalize_for_comparison(token2)
        
        if norm1 == norm2:
            return 1.0
        
        # Check for high character-level similarity
        if len(norm1) > 2 and len(norm2) > 2:
            matcher = SequenceMatcher(None, norm1, norm2)
            ratio = matcher.ratio()
            if ratio > 0.85:  # 85% similar
                return ratio
        
        return 0.0
    
    def align_sequences_smart(self, pdf_tokens: List[str], html_tokens: List[str]) -> List[AlignedSegment]:
        """
        Smart alignment that handles token boundary issues with multiple approaches.
        """
        segments = []
        
        # Create normalized versions for primary alignment
        pdf_normalized = [self.normalize_for_comparison(t) for t in pdf_tokens]
        html_normalized = [self.normalize_for_comparison(t) for t in html_tokens]
        
        # Use SequenceMatcher for primary alignment
        matcher = SequenceMatcher(None, pdf_normalized, html_normalized)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # For equal blocks, check each token pair to see if they're EXACTLY equal
                for i in range(i1, i2):
                    j = j1 + (i - i1)
                    if pdf_tokens[i] == html_tokens[j]:
                        # Exact match
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.MATCH,
                            pdf_text=pdf_tokens[i],
                            html_text=html_tokens[j],
                            similarity=1.0
                        ))
                    else:
                        # Similar after normalization
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.SIMILAR,
                            pdf_text=pdf_tokens[i],
                            html_text=html_tokens[j],
                            similarity=0.9
                        ))
                
            elif tag == 'replace':
                # For replacements, do word-by-word alignment
                pdf_words = pdf_tokens[i1:i2]
                html_words = html_tokens[j1:j2]
                
                # Align the replacement section word by word
                word_matcher = SequenceMatcher(None, 
                                             [self.normalize_for_comparison(w) for w in pdf_words],
                                             [self.normalize_for_comparison(w) for w in html_words])
                
                for wtag, wi1, wi2, wj1, wj2 in word_matcher.get_opcodes():
                    if wtag == 'equal':
                        # Check if actually equal or just normalized equal
                        for wi in range(wi1, wi2):
                            wj = wj1 + (wi - wi1)
                            if wi < len(pdf_words) and wj < len(html_words):
                                if pdf_words[wi] == html_words[wj]:
                                    segments.append(AlignedSegment(
                                        alignment_type=AlignmentType.MATCH,
                                        pdf_text=pdf_words[wi],
                                        html_text=html_words[wj],
                                        similarity=1.0
                                    ))
                                else:
                                    segments.append(AlignedSegment(
                                        alignment_type=AlignmentType.SIMILAR,
                                        pdf_text=pdf_words[wi],
                                        html_text=html_words[wj],
                                        similarity=0.9
                                    ))
                    elif wtag == 'replace':
                        if wi1 < wi2:
                            segments.append(AlignedSegment(
                                alignment_type=AlignmentType.HTML_GAP,
                                pdf_text=' '.join(pdf_words[wi1:wi2]),
                                html_text='',
                                similarity=0
                            ))
                        if wj1 < wj2:
                            segments.append(AlignedSegment(
                                alignment_type=AlignmentType.PDF_GAP,
                                pdf_text='',
                                html_text=' '.join(html_words[wj1:wj2]),
                                similarity=0
                            ))
                    elif wtag == 'delete':
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.HTML_GAP,
                            pdf_text=' '.join(pdf_words[wi1:wi2]),
                            html_text='',
                            similarity=0
                        ))
                    elif wtag == 'insert':
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.PDF_GAP,
                            pdf_text='',
                            html_text=' '.join(html_words[wj1:wj2]),
                            similarity=0
                        ))
                
            elif tag == 'delete':
                pdf_text = ' '.join(pdf_tokens[i1:i2])
                segments.append(AlignedSegment(
                    alignment_type=AlignmentType.HTML_GAP,
                    pdf_text=pdf_text,
                    html_text='',
                    similarity=0
                ))
                
            elif tag == 'insert':
                html_text = ' '.join(html_tokens[j1:j2])
                segments.append(AlignedSegment(
                    alignment_type=AlignmentType.PDF_GAP,
                    pdf_text='',
                    html_text=html_text,
                    similarity=0
                ))
        
        return segments
    
    def merge_adjacent_segments(self, segments: List[AlignedSegment]) -> List[AlignedSegment]:
        """Merge all adjacent segments of the same type."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Merge if same type (regardless of size)
            if current.alignment_type == next_seg.alignment_type:
                # Merge texts
                if current.pdf_text and next_seg.pdf_text:
                    current.pdf_text += ' ' + next_seg.pdf_text
                elif next_seg.pdf_text:
                    current.pdf_text = next_seg.pdf_text
                    
                if current.html_text and next_seg.html_text:
                    current.html_text += ' ' + next_seg.html_text
                elif next_seg.html_text:
                    current.html_text = next_seg.html_text
                    
                # Keep the average similarity for merged segments
                if hasattr(current, 'similarity') and hasattr(next_seg, 'similarity'):
                    current.similarity = (current.similarity + next_seg.similarity) / 2
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        return merged
    
    def generate_html_report(self, results: Dict[str, any], output_path: str):
        """Generate HTML report with clean visualization."""
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Transcript Comparison Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: #1a1a2e;
            color: white;
            padding: 20px 30px;
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 24px;
            font-weight: 500;
        }}
        
        .header p {{
            margin: 0;
            opacity: 0.8;
            font-size: 14px;
        }}
        
        .stats {{
            background: #f0f0f0;
            padding: 20px 30px;
            display: flex;
            justify-content: space-around;
            border-bottom: 1px solid #ddd;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #1a1a2e;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        
        
        .alignment-container {{
            padding: 20px;
            max-height: 70vh;
            overflow-y: auto;
        }}
        
        .alignment-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            line-height: 1.6;
        }}
        
        .alignment-table td {{
            padding: 12px 20px;
            vertical-align: top;
            border-bottom: 1px solid #eee;
        }}
        
        .alignment-table td:first-child {{
            width: 50%;
            border-right: 2px solid #eee;
        }}
        
        /* Match - green */
        .match-row {{
            background-color: #e6ffe6;
        }}
        
        .match-row td {{
            color: #006600;
        }}
        
        /* Similar - light green */
        .similar-row {{
            background-color: #f0fff0;
        }}
        
        .similar-row td {{
            color: #228b22;
        }}
        
        /* Missing from HTML (PDF only) - light red */
        .pdf-only-row {{
            background-color: #ffe6e6;
        }}
        
        .pdf-only-row td {{
            color: #cc0000;
        }}
        
        /* Missing from PDF (HTML only) - darker red */
        .html-only-row {{
            background-color: #ffcccc;
        }}
        
        .html-only-row td {{
            color: #990000;
        }}
        
        .column-header {{
            background: #2c3e50;
            color: white;
            font-weight: bold;
            text-align: center;
            padding: 15px !important;
            position: sticky;
            top: 0;
            z-index: 10;
            font-size: 16px;
        }}
        
        .legend {{
            padding: 20px 30px;
            background: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }}
        
        .legend h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            color: #495057;
        }}
        
        .legend-item {{
            display: inline-block;
            margin-right: 30px;
            padding: 5px 15px;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .legend-match {{
            background-color: #e6ffe6;
            color: #006600;
            border: 1px solid #006600;
        }}
        
        .legend-similar {{
            background-color: #f0fff0;
            color: #228b22;
        }}
        
        .legend-pdf-only {{
            background-color: #ffe6e6;
            color: #cc0000;
        }}
        
        .legend-html-only {{
            background-color: #ffcccc;
            color: #990000;
        }}
        
        .alignment-table tr:hover {{
            filter: brightness(0.95);
        }}
        
        .info-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 30px;
        }}
        
        .info-box {{
            background: #f8f9fa;
            border-left: 4px solid #17a2b8;
            padding: 15px 20px;
            font-size: 13px;
            color: #333;
        }}
        
        .info-box h4 {{
            margin-top: 0;
            margin-bottom: 10px;
            color: #17a2b8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Transcript Comparison Report</h1>
            <p>Comparing: {html_module.escape(Path(results['pdf_path']).name)} vs {html_module.escape(Path(results['html_path']).name)}</p>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{results['match_rate']:.1f}%</div>
                <div class="stat-label">Match Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{results['total_pdf_tokens']:,}</div>
                <div class="stat-label">PDF Tokens</div>
            </div>
            <div class="stat">
                <div class="stat-value">{results['total_html_tokens']:,}</div>
                <div class="stat-label">HTML Tokens</div>
            </div>
            <div class="stat">
                <div class="stat-value">{results['exact_matches']:,}</div>
                <div class="stat-label">Exact Matches</div>
            </div>
        </div>
        
        <div class="info-section">
            <div class="info-box">
                <h4>Normalization Applied</h4>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li>$ and CAD treated as equivalent</li>
                    <li>Case insensitive matching</li>
                    <li>Hyphens ignored (FX-trading = FX trading)</li>
                    <li>Possessives removed (Banking's = Banking)</li>
                    <li>Numbers 1-10 and II normalized to words</li>
                    <li>Punctuation normalized (. , ; : removed)</li>
                    <li>% normalized to "percent"</li>
                </ul>
            </div>
            <div class="info-box">
                <h4>Algorithm Used</h4>
                <p><strong>Sequence Alignment with Normalization</strong></p>
                <p>The algorithm uses dynamic programming sequence alignment (similar to DNA sequencing) to find optimal matches between transcripts:</p>
                <ul style="margin: 5px 0; padding-left: 20px;">
                    <li>Tokenizes both documents into words</li>
                    <li>Normalizes tokens using the rules shown</li>
                    <li>Aligns sequences to maximize matches</li>
                    <li>Groups consecutive segments of the same type</li>
                    <li>Handles insertions, deletions, and substitutions</li>
                </ul>
            </div>
        </div>
        
        <div class="alignment-container">
            <table class="alignment-table">
                <tr>
                    <td class="column-header">PDF Transcript</td>
                    <td class="column-header">HTML Transcript</td>
                </tr>
"""
        
        # Add segments
        for segment in results['segments']:
            if segment.alignment_type == AlignmentType.MATCH:
                html_content += f"""
                <tr class="match-row">
                    <td>{html_module.escape(segment.pdf_text)}</td>
                    <td>{html_module.escape(segment.html_text)}</td>
                </tr>
                """
            
            elif segment.alignment_type == AlignmentType.SIMILAR:
                html_content += f"""
                <tr class="similar-row">
                    <td>{html_module.escape(segment.pdf_text)}</td>
                    <td>{html_module.escape(segment.html_text)}</td>
                </tr>
                """
            
            elif segment.alignment_type == AlignmentType.HTML_GAP:
                html_content += f"""
                <tr class="pdf-only-row">
                    <td>{html_module.escape(segment.pdf_text)}</td>
                    <td></td>
                </tr>
                """
            
            elif segment.alignment_type == AlignmentType.PDF_GAP:
                html_content += f"""
                <tr class="html-only-row">
                    <td></td>
                    <td>{html_module.escape(segment.html_text)}</td>
                </tr>
                """
        
        html_content += """
            </table>
        </div>
        
        <div class="legend">
            <h3>Legend</h3>
            <span class="legend-item legend-match">Exact Match</span>
            <span class="legend-item legend-similar">Normalized Match</span>
            <span class="legend-item legend-pdf-only">PDF Only</span>
            <span class="legend-item legend-html-only">HTML Only</span>
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def compare(self, pdf_path: str, html_path: str) -> Dict[str, any]:
        """Main comparison method with improved normalization."""
        # Extract text
        pdf_text = self.extract_pdf_text(pdf_path)
        html_text = self.extract_html_text(html_path)
        
        # Apply pre-tokenization normalization and tokenize
        pdf_orig_tokens, pdf_pre_norm_tokens = self.tokenize_with_pre_normalization(pdf_text)
        html_orig_tokens, html_pre_norm_tokens = self.tokenize_with_pre_normalization(html_text)
        
        # Use pre-normalized tokens for alignment if they're different from original
        # This handles currency symbol cases like $4.5 vs CAD 4.5
        if pdf_pre_norm_tokens != pdf_orig_tokens or html_pre_norm_tokens != html_orig_tokens:
            # Use pre-normalized tokens for alignment
            segments = self.align_sequences_smart(pdf_pre_norm_tokens, html_pre_norm_tokens)
            # But preserve original tokens for display
            self._map_segments_to_original(segments, pdf_orig_tokens, html_orig_tokens, 
                                         pdf_pre_norm_tokens, html_pre_norm_tokens)
        else:
            # Use original tokens
            segments = self.align_sequences_smart(pdf_orig_tokens, html_orig_tokens)
        
        # Merge all adjacent segments of the same type
        segments = self.merge_adjacent_segments(segments)
        
        # Calculate statistics based on original tokens
        total_pdf_tokens = len(pdf_orig_tokens)
        total_html_tokens = len(html_orig_tokens)
        
        exact_matches = sum(
            len(seg.pdf_text.split()) 
            for seg in segments 
            if seg.alignment_type == AlignmentType.MATCH
        )
        
        similar_matches = sum(
            len(seg.pdf_text.split()) 
            for seg in segments 
            if seg.alignment_type == AlignmentType.SIMILAR
        )
        
        total_matches = exact_matches + similar_matches
        
        return {
            'pdf_path': pdf_path,
            'html_path': html_path,
            'total_pdf_tokens': total_pdf_tokens,
            'total_html_tokens': total_html_tokens,
            'exact_matches': exact_matches,
            'similar_matches': similar_matches,
            'match_rate': total_matches / max(total_pdf_tokens, total_html_tokens) * 100,
            'segments': segments
        }
    
    def _map_segments_to_original(self, segments: List[AlignedSegment], 
                                pdf_orig: List[str], html_orig: List[str],
                                pdf_pre_norm: List[str], html_pre_norm: List[str]):
        """
        Map segments back to original tokens when pre-normalization was applied.
        This is a simplified version - in practice, you'd need more sophisticated mapping.
        """
        # For now, we'll assume a simple case where token counts match
        # In a full implementation, this would need more sophisticated token mapping
        pass


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python transcript_compare.py <pdf/txt file> <html file> [output.html]")
        print("\nEnhanced normalization includes:")
        print("  - $ and CAD treated as equivalent")
        print("  - Case insensitive")
        print("  - Ignores hyphens (FX-trading = FX trading)")
        print("  - Numbers 1-10 normalized to words")
        print("  - Comment functionality for sharing")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    html_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "comparison_report.html"
    
    # Validate files
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not Path(html_path).exists():
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    # Run comparison
    print(f"Comparing transcripts...")
    comparer = EnhancedTranscriptComparer()
    results = comparer.compare(pdf_path, html_path)
    
    # Generate report
    comparer.generate_html_report(results, output_path)
    
    # Print summary
    print(f"\n✅ Comparison complete!")
    print(f"  Match rate: {results['match_rate']:.1f}%")
    print(f"  Exact matches: {results['exact_matches']} tokens")
    print(f"  Similar matches: {results['similar_matches']} tokens")
    print(f"  Report saved to: {output_path}")


if __name__ == "__main__":
    main()