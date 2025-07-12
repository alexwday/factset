#!/usr/bin/env python3
"""
Test script for enhanced folder structure and quarter/year parsing.
Tests the regex patterns and folder creation logic without requiring NAS connection.
"""

import re
import xml.etree.ElementTree as ET
import io
from typing import Optional, Tuple

def parse_quarter_and_year_from_xml(xml_content: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Parse quarter and fiscal year from transcript XML title."""
    try:
        # Parse only until we find the title
        root = ET.parse(io.BytesIO(xml_content)).getroot()
        namespace = ""
        if root.tag.startswith('{'):
            namespace = root.tag.split('}')[0] + '}'
        
        meta = root.find(f"{namespace}meta" if namespace else "meta")
        if meta is None:
            return None, None
            
        title_elem = meta.find(f"{namespace}title" if namespace else "title")
        if title_elem is None or not title_elem.text:
            return None, None
            
        title = title_elem.text.strip()
        
        # Try multiple patterns in order of likelihood
        patterns = [
            r"Q([1-4])\s+(20\d{2})\s+Earnings\s+Call",  # "Q1 2024 Earnings Call"
            r".*Q([1-4])\s+(20\d{2})\s+Earnings\s+Call.*",  # Anywhere in title
            r"(First|Second|Third|Fourth)\s+Quarter\s+(20\d{2})",  # "First Quarter 2024"
            r"(20\d{2})\s+Q([1-4])",  # "2024 Q1"
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                if i == 2:  # "(First|Second|Third|Fourth)\\s+Quarter\\s+(20\\d{2})"
                    quarter_map = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}
                    quarter = quarter_map.get(match.group(1).lower())
                    year = match.group(2)
                    return quarter, year
                elif i == 3:  # "(20\\d{2})\\s+Q([1-4])"
                    year = match.group(1)
                    quarter = f"Q{match.group(2)}"
                    return quarter, year
                else:  # Standard Q([1-4])\\s+(20\\d{2}) patterns
                    quarter = f"Q{match.group(1)}"
                    year = match.group(2)
                    return quarter, year
        
        # Final fallback: Find Q and year separately
        quarter_match = re.search(r"Q([1-4])", title)
        year_match = re.search(r"(20\d{2})", title)
        if quarter_match and year_match:
            return f"Q{quarter_match.group(1)}", year_match.group(2)
            
        return None, None
        
    except Exception as e:
        print(f"Error parsing XML for quarter/year: {e}")
        return None, None

def test_title_parsing():
    """Test quarter/year parsing with various title formats."""
    
    # Test cases: (title, expected_quarter, expected_year)
    test_cases = [
        # Standard format
        ("Q1 2024 Earnings Call", "Q1", "2024"),
        ("Q2 2023 Earnings Call", "Q2", "2023"), 
        ("Q3 2024 Earnings Call", "Q3", "2024"),
        ("Q4 2023 Earnings Call", "Q4", "2023"),
        
        # With company name prefix
        ("Royal Bank of Canada Q1 2024 Earnings Call", "Q1", "2024"),
        ("TD Bank Q2 2024 Earnings Call", "Q2", "2024"),
        
        # Alternative format
        ("First Quarter 2024", "Q1", "2024"),
        ("Second Quarter 2023", "Q2", "2023"),
        ("Third Quarter 2024", "Q3", "2024"),
        ("Fourth Quarter 2023", "Q4", "2023"),
        
        # Reversed format
        ("2024 Q1 Results", "Q1", "2024"),
        ("2023 Q4 Earnings", "Q4", "2023"),
        
        # Complex titles
        ("Royal Bank of Canada Reports Q1 2024 Earnings Call Transcript", "Q1", "2024"),
        ("Bank of Montreal Special Q4 2023 Earnings Call", "Q4", "2023"),
        
        # Edge cases that should fail
        ("Annual Report 2024", None, None),
        ("Investor Day 2024", None, None),
        ("Q5 2024 Earnings Call", None, None),  # Invalid quarter
        ("Q1 1999 Earnings Call", None, None),  # Old year
    ]
    
    print("Testing Quarter/Year Parsing:")
    print("=" * 60)
    
    passed = 0
    total = len(test_cases)
    
    for i, (title, expected_quarter, expected_year) in enumerate(test_cases, 1):
        # Create mock XML
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<transcript>
    <meta>
        <title>{title}</title>
        <date>2024-01-01</date>
    </meta>
    <body>
        <section name="Presentation">
        </section>
    </body>
</transcript>""".encode('utf-8')
        
        quarter, year = parse_quarter_and_year_from_xml(xml_content)
        
        success = quarter == expected_quarter and year == expected_year
        status = "✅ PASS" if success else "❌ FAIL"
        
        print(f"{i:2d}. {status} | '{title}' -> Q={quarter}, Y={year}")
        if not success:
            print(f"     Expected: Q={expected_quarter}, Y={expected_year}")
        
        if success:
            passed += 1
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total-passed} tests failed")
    
    return passed == total

def test_path_creation():
    """Test enhanced path creation logic."""
    
    print("\nTesting Path Creation:")
    print("=" * 60)
    
    # Mock data
    test_cases = [
        ("RY-CA", "Canadian", "Corrected", "Q1", "2024"),
        ("JPM-US", "US", "Raw", "Q4", "2023"),
        ("MFC-CA", "Insurance", "NearRealTime", "Q2", "2024"),
        ("TD-CA", "Canadian", "Corrected", "Unknown", "Unknown"),
    ]
    
    base_path = "FactSet/Outputs/Data"
    
    for ticker, institution_type, transcript_type, quarter, year in test_cases:
        # Simulate path creation
        path_components = [
            base_path, year, quarter, institution_type,
            f"{ticker}_Company_Name", transcript_type
        ]
        
        full_path = '/'.join(path_components)
        
        # Check path length (Windows limit is 260 characters)
        path_length = len(full_path)
        length_ok = path_length <= 250
        length_status = "✅" if length_ok else "❌"
        
        print(f"{length_status} {ticker} | Length: {path_length:3d} | {full_path}")
    
    print("=" * 60)

def test_error_scenarios():
    """Test error handling scenarios."""
    
    print("\nTesting Error Scenarios:")
    print("=" * 60)
    
    # Test malformed XML
    malformed_xml = b"<invalid>xml content"
    quarter, year = parse_quarter_and_year_from_xml(malformed_xml)
    print(f"✅ Malformed XML handled: Q={quarter}, Y={year}")
    
    # Test empty XML
    empty_xml = b"<?xml version='1.0'?><transcript></transcript>"
    quarter, year = parse_quarter_and_year_from_xml(empty_xml)
    print(f"✅ Empty XML handled: Q={quarter}, Y={year}")
    
    # Test XML without title
    no_title_xml = b"""<?xml version="1.0"?>
<transcript>
    <meta>
        <date>2024-01-01</date>
    </meta>
</transcript>"""
    quarter, year = parse_quarter_and_year_from_xml(no_title_xml)
    print(f"✅ No title XML handled: Q={quarter}, Y={year}")
    
    print("=" * 60)

if __name__ == "__main__":
    print("Enhanced Folder Structure - Test Suite")
    print("=" * 60)
    
    # Run all tests
    parsing_success = test_title_parsing()
    test_path_creation()
    test_error_scenarios()
    
    print(f"\n🏁 Test Suite Complete")
    if parsing_success:
        print("✅ Ready for implementation!")
    else:
        print("❌ Some tests failed - review before implementation")