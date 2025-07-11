#!/usr/bin/env python3
"""
Realistic transcript comparison tool.
Only normalizes truly equivalent representations, not guessing at meanings.

Usage: python transcript_compare_realistic.py <pdf/txt file> <html file> [output.html]
"""

import re
import sys
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


class RealisticTranscriptComparer:
    """Transcript comparison with minimal assumptions."""
    
    def __init__(self, custom_equivalents: Dict[str, str] = None):
        """
        Initialize with optional custom equivalents.
        
        Args:
            custom_equivalents: Dict mapping tokens to their equivalent form
                               e.g., {'$': 'CAD', 'USD': 'CAD'} for Canadian context
        """
        self.custom_equivalents = custom_equivalents or {}
    
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
        Minimal normalization - only for truly equivalent representations.
        """
        # Lowercase for case-insensitive comparison
        normalized = token.lower()
        
        # Remove common punctuation that doesn't change meaning
        # Keep periods for decimals, keep $ for currency  
        normalized = re.sub(r'[,;:!?\'"()]', '', normalized)
        
        # Apply custom equivalents if provided
        # e.g., if user specified {'$': 'cad'}, then '$' -> 'cad'
        if normalized in self.custom_equivalents:
            normalized = self.custom_equivalents[normalized]
        
        # Normalize percent symbol to word (this is universally safe)
        normalized = normalized.replace('%', ' percent')
        
        return normalized.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization preserving original tokens."""
        # Split on whitespace, keep everything
        return re.findall(r'\S+', text)
    
    def calculate_similarity(self, token1: str, token2: str) -> float:
        """
        Calculate similarity between tokens.
        Returns 1.0 for exact match, 0-1 for partial match.
        """
        # Normalize for comparison
        norm1 = self.normalize_for_comparison(token1)
        norm2 = self.normalize_for_comparison(token2)
        
        if norm1 == norm2:
            return 1.0
        
        # Check for very similar tokens (1 character difference)
        # This catches OCR errors like "the" vs "tho"
        if abs(len(norm1) - len(norm2)) <= 1:
            # Use simple character-based similarity
            matcher = SequenceMatcher(None, norm1, norm2)
            ratio = matcher.ratio()
            if ratio > 0.8:  # 80% similar
                return ratio
        
        return 0.0
    
    def align_sequences(self, pdf_tokens: List[str], html_tokens: List[str]) -> List[AlignedSegment]:
        """
        Align sequences with similarity scoring.
        """
        segments = []
        
        # Create normalized versions for alignment
        pdf_normalized = [self.normalize_for_comparison(t) for t in pdf_tokens]
        html_normalized = [self.normalize_for_comparison(t) for t in html_tokens]
        
        # Use SequenceMatcher on normalized tokens
        matcher = SequenceMatcher(None, pdf_normalized, html_normalized)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Check if truly equal or just similar
                pdf_text = ' '.join(pdf_tokens[i1:i2])
                html_text = ' '.join(html_tokens[j1:j2])
                
                # Are the original texts exactly the same?
                if pdf_text == html_text:
                    alignment_type = AlignmentType.MATCH
                else:
                    # They're similar after normalization (e.g., "5%" vs "5 percent")
                    alignment_type = AlignmentType.SIMILAR
                
                segments.append(AlignedSegment(
                    alignment_type=alignment_type,
                    pdf_text=pdf_text,
                    html_text=html_text,
                    similarity=1.0
                ))
                
            elif tag == 'replace':
                # Check if it's a small difference that might be similar
                pdf_text = ' '.join(pdf_tokens[i1:i2])
                html_text = ' '.join(html_tokens[j1:j2])
                
                # For single token replacements, check similarity
                if (i2 - i1) == 1 and (j2 - j1) == 1:
                    similarity = self.calculate_similarity(pdf_tokens[i1], html_tokens[j1])
                    if similarity > 0.8:
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.SIMILAR,
                            pdf_text=pdf_text,
                            html_text=html_text,
                            similarity=similarity
                        ))
                        continue
                
                # Otherwise, treat as separate gaps
                if i2 > i1:
                    segments.append(AlignedSegment(
                        alignment_type=AlignmentType.HTML_GAP,
                        pdf_text=pdf_text,
                        html_text='',
                        similarity=0
                    ))
                if j2 > j1:
                    segments.append(AlignedSegment(
                        alignment_type=AlignmentType.PDF_GAP,
                        pdf_text='',
                        html_text=html_text,
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
        """Merge adjacent segments of the same type for cleaner display."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Merge if same type
            if current.alignment_type == next_seg.alignment_type:
                # Combine texts
                if current.pdf_text and next_seg.pdf_text:
                    current.pdf_text += ' ' + next_seg.pdf_text
                elif next_seg.pdf_text:
                    current.pdf_text = next_seg.pdf_text
                    
                if current.html_text and next_seg.html_text:
                    current.html_text += ' ' + next_seg.html_text
                elif next_seg.html_text:
                    current.html_text = next_seg.html_text
                
                # Average similarity for merged segments
                if current.alignment_type == AlignmentType.SIMILAR:
                    current.similarity = (current.similarity + next_seg.similarity) / 2
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        return merged
    
    def generate_html_report(self, results: Dict[str, any], output_path: str):
        """Generate the HTML comparison report."""
        
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
        
        /* Exact match - green */
        .match-row {{
            background-color: #d4edda;
        }}
        
        .match-row td {{
            color: #155724;
        }}
        
        /* Similar (normalized match) - light yellow */
        .similar-row {{
            background-color: #fff9e6;
        }}
        
        .similar-row td {{
            color: #664d00;
        }}
        
        /* PDF only - light red across entire row */
        .pdf-only-row {{
            background-color: #fee;
        }}
        
        .pdf-only-row td:first-child {{
            color: #c00;
        }}
        
        .pdf-only-row td:last-child {{
            color: #c00;
            text-align: center;
            font-style: italic;
            opacity: 0.5;
        }}
        
        /* HTML only - light blue across entire row */
        .html-only-row {{
            background-color: #e6f3ff;
        }}
        
        .html-only-row td:first-child {{
            color: #004085;
            text-align: center;
            font-style: italic;
            opacity: 0.5;
        }}
        
        .html-only-row td:last-child {{
            color: #004085;
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
            background-color: #d4edda;
            color: #155724;
        }}
        
        .legend-similar {{
            background-color: #fff9e6;
            color: #664d00;
        }}
        
        .legend-pdf-only {{
            background-color: #fee;
            color: #c00;
        }}
        
        .legend-html-only {{
            background-color: #e6f3ff;
            color: #004085;
        }}
        
        .alignment-table tr:hover {{
            filter: brightness(0.95);
        }}
        
        .info-box {{
            background: #e8f4f8;
            border-left: 4px solid #17a2b8;
            padding: 15px 20px;
            margin: 20px 30px;
            font-size: 14px;
            color: #004085;
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
        
        <div class="info-box">
            <strong>Note:</strong> This comparison uses minimal normalization. 
            Only truly equivalent representations are considered the same (e.g., "5%" = "5 percent"). 
            Numbers spelled out differently, abbreviations, and other variations are shown as differences.
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
                    <td class="empty-cell">[not in HTML]</td>
                </tr>
                """
            
            elif segment.alignment_type == AlignmentType.PDF_GAP:
                html_content += f"""
                <tr class="html-only-row">
                    <td class="empty-cell">[not in PDF]</td>
                    <td>{html_module.escape(segment.html_text)}</td>
                </tr>
                """
        
        html_content += """
            </table>
        </div>
        
        <div class="legend">
            <h3>Legend</h3>
            <span class="legend-item legend-match">Exact Match</span>
            <span class="legend-item legend-similar">Similar (e.g., % vs percent)</span>
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
        """Main comparison method."""
        # Extract text
        pdf_text = self.extract_pdf_text(pdf_path)
        html_text = self.extract_html_text(html_path)
        
        # Tokenize
        pdf_tokens = self.tokenize(pdf_text)
        html_tokens = self.tokenize(html_text)
        
        # Align sequences
        segments = self.align_sequences(pdf_tokens, html_tokens)
        
        # Merge adjacent segments
        segments = self.merge_adjacent_segments(segments)
        
        # Calculate statistics
        total_pdf_tokens = len(pdf_tokens)
        total_html_tokens = len(html_tokens)
        
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


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python transcript_compare_realistic.py <pdf/txt file> <html file> [output.html]")
        print("\nFor Canadian transcripts where $ means CAD:")
        print("  python transcript_compare_realistic.py --cad earnings.pdf earnings.html")
        sys.exit(1)
    
    # Check for --cad flag
    args = sys.argv[1:]
    use_cad = False
    if '--cad' in args:
        use_cad = True
        args.remove('--cad')
    
    if len(args) < 2:
        print("Error: Not enough arguments")
        sys.exit(1)
    
    pdf_path = args[0]
    html_path = args[1]
    output_path = args[2] if len(args) > 2 else "comparison_report.html"
    
    # Validate files
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not Path(html_path).exists():
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    # Set up custom equivalents based on flags
    custom_equivalents = {}
    if use_cad:
        # For Canadian transcripts, $ and CAD are equivalent
        custom_equivalents = {
            '$': 'cad',
            'usd': 'cad',  # Sometimes USD is explicitly mentioned but means CAD
            'dollars': 'cad'
        }
        print("Using Canadian dollar equivalents ($ = CAD)")
    
    # Run comparison
    print(f"Comparing transcripts (realistic mode)...")
    comparer = RealisticTranscriptComparer(custom_equivalents)
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