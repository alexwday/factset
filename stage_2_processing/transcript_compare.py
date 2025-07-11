#!/usr/bin/env python3
"""
All-in-one transcript comparison tool.
Compares PDF/text transcripts with HTML transcripts and generates a visual HTML report.

Usage: python transcript_compare.py <pdf/txt file> <html file> [output.html]
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
    SUBSTITUTION = "substitution"
    PDF_GAP = "pdf_gap"      # Content only in HTML
    HTML_GAP = "html_gap"    # Content only in PDF


@dataclass
class AlignedSegment:
    """Represents an aligned segment between transcripts."""
    alignment_type: AlignmentType
    pdf_text: str
    html_text: str
    score: float = 1.0


class TranscriptComparer:
    """Complete transcript comparison tool."""
    
    def __init__(self):
        # Common number substitutions
        self.number_map = {
            '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
            '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten',
            '11': 'eleven', '12': 'twelve', '13': 'thirteen', '14': 'fourteen',
            '15': 'fifteen', '16': 'sixteen', '17': 'seventeen', '18': 'eighteen',
            '19': 'nineteen', '20': 'twenty', '21': 'twenty one', '22': 'twenty two',
            '23': 'twenty three', '24': 'twenty four', '25': 'twenty five',
            '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty',
            '70': 'seventy', '80': 'eighty', '90': 'ninety', '100': 'one hundred',
            '125': 'one hundred twenty five', '150': 'one hundred fifty',
            '200': 'two hundred', '250': 'two hundred fifty',
            '300': 'three hundred', '400': 'four hundred',
            '425': 'four hundred twenty five', '500': 'five hundred',
            '1000': 'one thousand', '2000': 'two thousand',
            '2023': 'twenty twenty three', '2024': 'twenty twenty four'
        }
        
        # Common abbreviations
        self.abbrev_map = {
            'ceo': 'chief executive officer',
            'cfo': 'chief financial officer',
            'coo': 'chief operating officer',
            'cto': 'chief technology officer',
            'evp': 'executive vice president',
            'svp': 'senior vice president',
            'vp': 'vice president',
            'q1': 'first quarter',
            'q2': 'second quarter',
            'q3': 'third quarter',
            'q4': 'fourth quarter',
            'yoy': 'year over year',
            'ytd': 'year to date',
            'qoq': 'quarter over quarter',
            'b': 'billion',
            'm': 'million',
            'k': 'thousand'
        }
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF or text file."""
        if pdf_path.endswith('.txt'):
            with open(pdf_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber required for PDF files. Install with: pip install pdfplumber")
        
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
    
    def normalize_token(self, token: str) -> str:
        """
        Normalize a token for comparison.
        This is where the magic happens for recognizing equivalents.
        """
        # Remove punctuation
        clean = re.sub(r'[.,;:!?\'"\-()]', '', token).lower()
        
        # Check if it's a number that should be spelled out
        if clean in self.number_map:
            return self.number_map[clean]
        
        # Handle percentages: "15%" -> "fifteen percent"
        if clean.endswith('%'):
            number = clean[:-1]
            if number in self.number_map:
                return self.number_map[number] + ' percent'
            return number + ' percent'
        
        # Handle currency: "$2.5" -> "2.5 dollars"
        if clean.startswith('$'):
            amount = clean[1:]
            # Handle decimals
            if '.' in amount:
                parts = amount.split('.')
                if parts[0] in self.number_map:
                    result = self.number_map[parts[0]] + ' point ' + parts[1]
                else:
                    result = amount
            else:
                if amount in self.number_map:
                    result = self.number_map[amount]
                else:
                    result = amount
            return result + ' dollars'
        
        # Handle common abbreviations
        if clean in self.abbrev_map:
            return self.abbrev_map[clean]
        
        # Handle compound numbers like "2.5" -> "two point five"
        if '.' in clean and clean.replace('.', '').isdigit():
            parts = clean.split('.')
            normalized_parts = []
            for part in parts:
                if part in self.number_map:
                    normalized_parts.append(self.number_map[part])
                else:
                    normalized_parts.append(part)
            return ' point '.join(normalized_parts)
        
        return clean
    
    def tokenize_and_normalize(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Tokenize text and return both original and normalized versions.
        """
        # Split on whitespace but preserve original tokens
        original_tokens = re.findall(r'\S+', text)
        normalized_tokens = [self.normalize_token(token) for token in original_tokens]
        
        return original_tokens, normalized_tokens
    
    def align_sequences(self, pdf_orig: List[str], pdf_norm: List[str],
                       html_orig: List[str], html_norm: List[str]) -> List[AlignedSegment]:
        """
        Align sequences using SequenceMatcher on normalized tokens.
        """
        matcher = SequenceMatcher(None, pdf_norm, html_norm)
        segments = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Perfect match
                pdf_text = ' '.join(pdf_orig[i1:i2])
                html_text = ' '.join(html_orig[j1:j2])
                
                segments.append(AlignedSegment(
                    alignment_type=AlignmentType.MATCH,
                    pdf_text=pdf_text,
                    html_text=html_text,
                    score=1.0
                ))
                
            elif tag == 'replace':
                # Substitution - check if it's a small variation
                pdf_text = ' '.join(pdf_orig[i1:i2])
                html_text = ' '.join(html_orig[j1:j2])
                
                # For small replacements, show as substitution
                if (i2 - i1) <= 3 and (j2 - j1) <= 3:
                    segments.append(AlignedSegment(
                        alignment_type=AlignmentType.SUBSTITUTION,
                        pdf_text=pdf_text,
                        html_text=html_text,
                        score=0.5
                    ))
                else:
                    # Large replacement - show as separate gaps
                    if i2 > i1:
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.HTML_GAP,
                            pdf_text=pdf_text,
                            html_text='',
                            score=0
                        ))
                    if j2 > j1:
                        segments.append(AlignedSegment(
                            alignment_type=AlignmentType.PDF_GAP,
                            pdf_text='',
                            html_text=html_text,
                            score=0
                        ))
                
            elif tag == 'delete':
                # PDF only
                pdf_text = ' '.join(pdf_orig[i1:i2])
                segments.append(AlignedSegment(
                    alignment_type=AlignmentType.HTML_GAP,
                    pdf_text=pdf_text,
                    html_text='',
                    score=0
                ))
                
            elif tag == 'insert':
                # HTML only
                html_text = ' '.join(html_orig[j1:j2])
                segments.append(AlignedSegment(
                    alignment_type=AlignmentType.PDF_GAP,
                    pdf_text='',
                    html_text=html_text,
                    score=0
                ))
        
        return segments
    
    def merge_small_segments(self, segments: List[AlignedSegment]) -> List[AlignedSegment]:
        """Merge small adjacent segments of the same type for cleaner output."""
        if not segments:
            return segments
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # Merge if same type and both are small
            if (current.alignment_type == next_seg.alignment_type and
                len(current.pdf_text.split()) < 5 and
                len(next_seg.pdf_text.split()) < 5):
                
                # Merge texts
                if current.pdf_text and next_seg.pdf_text:
                    current.pdf_text += ' ' + next_seg.pdf_text
                else:
                    current.pdf_text += next_seg.pdf_text
                    
                if current.html_text and next_seg.html_text:
                    current.html_text += ' ' + next_seg.html_text
                else:
                    current.html_text += next_seg.html_text
                    
            else:
                # Can't merge, save current and move to next
                merged.append(current)
                current = next_seg
        
        # Don't forget the last segment
        merged.append(current)
        
        return merged
    
    def generate_html_report(self, results: Dict[str, any], output_path: str):
        """Generate the two-column HTML comparison report."""
        
        # HTML template with improved styling
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
            background: #2c3e50;
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
            background: #ecf0f1;
            padding: 20px 30px;
            display: flex;
            justify-content: space-around;
            border-bottom: 1px solid #bdc3c7;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
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
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .alignment-table td:first-child {{
            width: 50%;
            border-right: 2px solid #ecf0f1;
        }}
        
        /* Match - both columns green */
        .match-row {{
            background-color: #d4edda;
        }}
        
        .match-row td {{
            color: #155724;
        }}
        
        /* Substitution - both columns yellow */
        .substitution-row {{
            background-color: #fff3cd;
        }}
        
        .substitution-row td {{
            color: #856404;
            font-weight: 500;
        }}
        
        /* PDF only - entire row light red */
        .pdf-only-row {{
            background-color: #f8d7da;
        }}
        
        .pdf-only-row td:first-child {{
            color: #721c24;
        }}
        
        .pdf-only-row td:last-child {{
            color: #721c24;
            text-align: center;
            font-style: italic;
            opacity: 0.6;
        }}
        
        /* HTML only - entire row light blue */
        .html-only-row {{
            background-color: #cfe2ff;
        }}
        
        .html-only-row td:first-child {{
            color: #084298;
            text-align: center;
            font-style: italic;
            opacity: 0.6;
        }}
        
        .html-only-row td:last-child {{
            color: #084298;
        }}
        
        .column-header {{
            background: #34495e;
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
        
        .legend-substitution {{
            background-color: #fff3cd;
            color: #856404;
        }}
        
        .legend-pdf-only {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        
        .legend-html-only {{
            background-color: #cfe2ff;
            color: #084298;
        }}
        
        .empty-cell {{
            opacity: 0.6;
            font-style: italic;
        }}
        
        /* Improve readability */
        .alignment-table tr:hover {{
            filter: brightness(0.95);
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
                <div class="stat-value">{results['alignment_rate']:.1f}%</div>
                <div class="stat-label">Alignment Rate</div>
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
                <div class="stat-value">{results['aligned_tokens']:,}</div>
                <div class="stat-label">Aligned Tokens</div>
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
            
            elif segment.alignment_type == AlignmentType.SUBSTITUTION:
                html_content += f"""
                <tr class="substitution-row">
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
        
        # Close HTML
        html_content += """
            </table>
        </div>
        
        <div class="legend">
            <h3>Legend</h3>
            <span class="legend-item legend-match">✓ Exact Match</span>
            <span class="legend-item legend-substitution">≈ Similar Content</span>
            <span class="legend-item legend-pdf-only">− PDF Only</span>
            <span class="legend-item legend-html-only">+ HTML Only</span>
        </div>
    </div>
</body>
</html>
"""
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def compare(self, pdf_path: str, html_path: str) -> Dict[str, any]:
        """
        Main comparison method.
        """
        # Extract text
        pdf_text = self.extract_pdf_text(pdf_path)
        html_text = self.extract_html_text(html_path)
        
        # Tokenize and normalize
        pdf_orig, pdf_norm = self.tokenize_and_normalize(pdf_text)
        html_orig, html_norm = self.tokenize_and_normalize(html_text)
        
        # Align sequences
        segments = self.align_sequences(pdf_orig, pdf_norm, html_orig, html_norm)
        
        # Merge small segments for cleaner output
        segments = self.merge_small_segments(segments)
        
        # Calculate statistics
        total_pdf_tokens = len(pdf_orig)
        total_html_tokens = len(html_orig)
        
        matched_tokens = sum(
            len(seg.pdf_text.split()) 
            for seg in segments 
            if seg.alignment_type in [AlignmentType.MATCH, AlignmentType.SUBSTITUTION]
        )
        
        return {
            'pdf_path': pdf_path,
            'html_path': html_path,
            'total_pdf_tokens': total_pdf_tokens,
            'total_html_tokens': total_html_tokens,
            'aligned_tokens': matched_tokens,
            'alignment_rate': matched_tokens / max(total_pdf_tokens, total_html_tokens) * 100,
            'segments': segments
        }


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python transcript_compare.py <pdf/txt file> <html file> [output.html]")
        print("\nExample:")
        print("  python transcript_compare.py earnings.pdf earnings.html")
        print("  python transcript_compare.py earnings.txt earnings.html report.html")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    html_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "comparison_report.html"
    
    # Validate input files
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not Path(html_path).exists():
        print(f"Error: File not found: {html_path}")
        sys.exit(1)
    
    # Run comparison
    print(f"Comparing transcripts...")
    print(f"  PDF/Text: {pdf_path}")
    print(f"  HTML: {html_path}")
    
    comparer = TranscriptComparer()
    results = comparer.compare(pdf_path, html_path)
    
    # Generate HTML report
    comparer.generate_html_report(results, output_path)
    
    # Print summary
    print(f"\n✅ Comparison complete!")
    print(f"  Alignment rate: {results['alignment_rate']:.1f}%")
    print(f"  Total segments: {len(results['segments'])}")
    print(f"  Report saved to: {output_path}")
    
    # Show breakdown
    match_count = sum(1 for s in results['segments'] if s.alignment_type == AlignmentType.MATCH)
    sub_count = sum(1 for s in results['segments'] if s.alignment_type == AlignmentType.SUBSTITUTION)
    pdf_only = sum(1 for s in results['segments'] if s.alignment_type == AlignmentType.HTML_GAP)
    html_only = sum(1 for s in results['segments'] if s.alignment_type == AlignmentType.PDF_GAP)
    
    print(f"\n  Segment breakdown:")
    print(f"    - Exact matches: {match_count}")
    print(f"    - Substitutions: {sub_count}")
    print(f"    - PDF only: {pdf_only}")
    print(f"    - HTML only: {html_only}")


if __name__ == "__main__":
    main()