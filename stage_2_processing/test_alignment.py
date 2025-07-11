#!/usr/bin/env python3
"""
Test the alignment to show how it breaks down large yellow blocks.
"""

# Mock the imports for testing
class AlignmentType:
    MATCH = "match"
    SIMILAR = "similar"
    PDF_GAP = "pdf_gap"
    HTML_GAP = "html_gap"

# Example text
pdf_text = "to our retail capital parameters. Moving to Slide 11. All bank net interest income was up 22% year-over-year, or up 14% excluding trading revenue and."
html_text = "to our retail capital parameters. Moving to slide 11, all-bank net interest income was up 22% year-over-year or up 14% excluding trading revenue in."

# Tokenize
pdf_tokens = pdf_text.split()
html_tokens = html_text.split()

print("PDF tokens:", pdf_tokens)
print("\nHTML tokens:", html_tokens)
print("\n" + "="*80)
print("TOKEN-BY-TOKEN COMPARISON:")
print("="*80)

# Compare each token
for i in range(max(len(pdf_tokens), len(html_tokens))):
    if i < len(pdf_tokens) and i < len(html_tokens):
        pdf_tok = pdf_tokens[i]
        html_tok = html_tokens[i]
        
        # Check exact match
        if pdf_tok == html_tok:
            print(f"{i:2d}: EXACT MATCH - '{pdf_tok}' == '{html_tok}'")
        else:
            # Check normalized match
            pdf_norm = pdf_tok.lower().replace('-', '').replace(',', '').replace('.', '')
            html_norm = html_tok.lower().replace('-', '').replace(',', '').replace('.', '')
            
            if pdf_norm == html_norm:
                print(f"{i:2d}: SIMILAR     - '{pdf_tok}' ≈ '{html_tok}' (normalized to '{pdf_norm}')")
            else:
                print(f"{i:2d}: DIFFERENT   - '{pdf_tok}' ≠ '{html_tok}'")
    elif i < len(pdf_tokens):
        print(f"{i:2d}: PDF ONLY    - '{pdf_tokens[i]}'")
    else:
        print(f"{i:2d}: HTML ONLY   - '{html_tokens[i]}'")

# Summary
exact_matches = sum(1 for i in range(min(len(pdf_tokens), len(html_tokens))) 
                   if pdf_tokens[i] == html_tokens[i])
similar_matches = sum(1 for i in range(min(len(pdf_tokens), len(html_tokens))) 
                     if pdf_tokens[i] != html_tokens[i] and 
                     pdf_tokens[i].lower().replace('-', '').replace(',', '').replace('.', '') == 
                     html_tokens[i].lower().replace('-', '').replace(',', '').replace('.', ''))

print(f"\n{'='*80}")
print(f"SUMMARY:")
print(f"Total tokens: PDF={len(pdf_tokens)}, HTML={len(html_tokens)}")
print(f"Exact matches: {exact_matches}")
print(f"Similar (normalized) matches: {similar_matches}")
print(f"Total matches: {exact_matches + similar_matches}")
print(f"Match rate: {(exact_matches + similar_matches) / max(len(pdf_tokens), len(html_tokens)) * 100:.1f}%")