#!/usr/bin/env python3
"""Extract and consolidate financial categories from researched YAML files."""

import yaml
import json
from pathlib import Path

def extract_comprehensive_description(category_data):
    """Merge all relevant fields into a single comprehensive description."""
    
    parts = []
    
    # Main description
    if 'description' in category_data:
        parts.append(category_data['description'].strip())
    
    # Add detailed scope if available
    if 'detailed_scope' in category_data:
        scope = category_data['detailed_scope']
        if 'includes' in scope:
            includes = ', '.join(scope['includes'][:10])  # Limit to first 10 items
            parts.append(f"Includes: {includes}")
        if 'excludes' in scope:
            excludes = ', '.join(scope['excludes'][:5])  # Limit to first 5 items
            parts.append(f"Excludes: {excludes}")
    
    # Add key indicators
    if 'key_indicators' in category_data:
        indicators = category_data['key_indicators']
        if 'primary' in indicators:
            primary_indicators = ', '.join(indicators['primary'][:8])
            parts.append(f"Key indicators: {primary_indicators}")
    
    # Add common phrases if available
    if 'common_phrases' in category_data:
        phrases = category_data['common_phrases']
        if 'positive_signals' in phrases:
            example_phrases = ', '.join(phrases['positive_signals'][:3])
            parts.append(f"Example phrases: {example_phrases}")
    
    # Add prompt guidance if available
    if 'prompt_guidance_for_llm' in category_data:
        guidance = category_data['prompt_guidance_for_llm'].strip()
        # Take first 200 characters of guidance
        if len(guidance) > 200:
            guidance = guidance[:200] + "..."
        parts.append(f"Guidance: {guidance}")
    
    return ' '.join(parts)

def main():
    # Path to researched categories
    categories_dir = Path("/Users/alexwday/Projects/factset/database_refresh/06_llm_classification/researched_categories")
    
    # Initialize with ID 0 for non-relevant content
    financial_categories = [
        {
            "id": 0,
            "name": "Non-Relevant",
            "description": "Content with no financial or business substance. Includes operator statements, pure pleasantries, acknowledgments, transitions, and technical announcements. Use for 'Thank you', 'Good morning', 'Next question please', 'Can you hear me?', and similar non-substantive content. This category ensures every paragraph has a classification even when no financial topic is discussed."
        }
    ]
    
    # Process each researched category file
    yaml_files = sorted(categories_dir.glob("*.yaml"))
    
    for idx, yaml_file in enumerate(yaml_files, start=1):
        print(f"Processing {yaml_file.name}...")
        
        with open(yaml_file, 'r') as f:
            content = yaml.safe_load(f)
            
            if isinstance(content, list) and len(content) > 0:
                category_data = content[0]
            elif isinstance(content, dict):
                category_data = content
            else:
                print(f"  Skipping - unexpected format")
                continue
            
            # Extract name and create comprehensive description
            name = category_data.get('name', yaml_file.stem.replace('_', ' ').title())
            description = extract_comprehensive_description(category_data)
            
            financial_categories.append({
                "id": idx,
                "name": name,
                "description": description
            })
            
            print(f"  Added: {name} (ID: {idx})")
    
    # Create the new financial_categories.yaml
    output_file = Path("/Users/alexwday/Projects/factset/database_refresh/06_llm_classification/financial_categories.yaml")
    
    with open(output_file, 'w') as f:
        yaml.dump(financial_categories, f, default_flow_style=False, width=120, sort_keys=False)
    
    print(f"\nCreated {output_file} with {len(financial_categories)} categories")
    
    # Also create a JSON version for easy lookup
    json_file = output_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(financial_categories, f, indent=2)
    
    print(f"Also created {json_file} for reference")

if __name__ == "__main__":
    main()