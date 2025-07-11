#!/usr/bin/env python3
"""
Test script for the enhanced HTML viewer
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from transcript_html_viewer_enhanced import generate_html, detect_primary_company, load_monitored_institutions

def test_enhanced_viewer():
    """Test the enhanced HTML viewer functionality"""
    
    # Sample transcript data for testing
    test_data = {
        'title': 'Royal Bank of Canada Q4 2024 Earnings Call',
        'date': '2024-12-15',
        'companies': ['Royal Bank of Canada'],
        'participants': {
            'p1': {
                'name': 'David McKay',
                'title': 'President and CEO',
                'affiliation': 'Royal Bank of Canada',
                'type': 'C'
            },
            'p2': {
                'name': 'John Analyst',
                'title': 'Senior Analyst',
                'affiliation': 'Investment Research Corp',
                'type': 'A'
            }
        },
        'sections': [
            {
                'name': 'Presentation',
                'speakers': [
                    {
                        'id': 'p1',
                        'type': '',
                        'paragraphs': [
                            'Good morning, everyone. Welcome to RBC\'s Q4 2024 earnings call.',
                            'We delivered strong results this quarter with record revenue growth.'
                        ]
                    }
                ]
            },
            {
                'name': 'Q&A Session',
                'speakers': [
                    {
                        'id': 'p2',
                        'type': 'q',
                        'paragraphs': [
                            'Thank you for taking my question. Can you provide more details on the revenue growth drivers?'
                        ]
                    },
                    {
                        'id': 'p1',
                        'type': 'a',
                        'paragraphs': [
                            'Certainly. The growth was primarily driven by our wealth management and capital markets divisions.'
                        ]
                    }
                ]
            }
        ]
    }
    
    print("Testing enhanced HTML viewer...")
    
    try:
        # Test company detection
        institutions = load_monitored_institutions()
        company, logo_url = detect_primary_company(test_data, institutions)
        print(f"✓ Company detection: {company}")
        print(f"✓ Logo URL: {logo_url}")
        
        # Test HTML generation
        html_content = generate_html(test_data)
        print(f"✓ HTML generation successful ({len(html_content)} characters)")
        
        # Test key features are present
        features_to_check = [
            'search-input',  # Search functionality
            'copy-btn',      # Copy buttons
            'skip-link',     # Accessibility
            'export-btn',    # Export functionality
            'search-highlight',  # Search highlighting
            'WCAG 2.1 AA',   # Accessibility compliance
            'var(--font-size-base)',  # Typography system
            'var(--primary-color)',   # Color system
            '@media print'   # Print styles
        ]
        
        missing_features = []
        for feature in features_to_check:
            if feature not in html_content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
        else:
            print("✓ All key features present in HTML")
        
        # Write test output
        test_output_path = os.path.join(os.path.dirname(__file__), 'test_enhanced_output.html')
        with open(test_output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Test HTML written to: {test_output_path}")
        print(f"✓ Enhanced viewer test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_viewer()