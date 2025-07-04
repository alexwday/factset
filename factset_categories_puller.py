"""
FactSet Categories Puller
This script pulls all available categories from the FactSet Events and Transcripts API
and saves them to a CSV file for reference.
"""

import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
import os
from urllib.parse import quote
from datetime import datetime

# =============================================================================
# CONFIGURATION VARIABLES (HARDCODED)
# =============================================================================

# SSL and Proxy Configuration
SSL_CERT_PATH = "/Users/alexwday/path/to/ssl/certificate.cer"
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

# API Configuration
API_USERNAME = "x"
API_PASSWORD = "x"

# Output Configuration
OUTPUT_FILE = "factset_categories.csv"

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set up SSL certificate environment variables
os.environ["REQUESTS_CA_BUNDLE"] = SSL_CERT_PATH
os.environ["SSL_CERT_FILE"] = SSL_CERT_PATH

# Set up proxy authentication
user = PROXY_USER
password = quote(PROXY_PASSWORD)

# Configure FactSet API client
configuration = fds.sdk.EventsandTranscripts.Configuration(
    username=API_USERNAME,
    password=API_PASSWORD,
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=SSL_CERT_PATH
)
configuration.get_basic_auth_token()

# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def get_all_categories():
    """
    Fetch all available categories from the FactSet API
    
    Returns:
        pd.DataFrame: DataFrame with all categories
    """
    try:
        print("Fetching all available categories from FactSet API...")
        
        with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
            api_instance = transcripts_api.TranscriptsApi(api_client)
            
            # Get all categories
            response = api_instance.getcategories()
            
            if not response or not hasattr(response, 'data') or not response.data:
                print("No categories found")
                return pd.DataFrame()
            
            # Convert response to DataFrame
            df = pd.DataFrame(response.to_dict()['data'])
            print(f"Found {len(df)} categories")
            
            return df
            
    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        return pd.DataFrame()

def analyze_categories(df):
    """
    Analyze and categorize the retrieved categories
    
    Args:
        df (pd.DataFrame): DataFrame with categories
    
    Returns:
        dict: Analysis results
    """
    if df.empty:
        return {}
    
    analysis = {
        'total_count': len(df),
        'country_codes': [],
        'industry_codes': [],
        'subject_codes': [],
        'other_codes': []
    }
    
    # Analyze each category
    for _, row in df.iterrows():
        category = str(row.get('category', ''))
        
        if category.startswith('CN:'):
            analysis['country_codes'].append(category)
        elif category.startswith('IN:'):
            analysis['industry_codes'].append(category)
        elif category.startswith('SU:') or category.startswith('SUB:'):
            analysis['subject_codes'].append(category)
        else:
            analysis['other_codes'].append(category)
    
    # Sort categories for better readability
    for key in ['country_codes', 'industry_codes', 'subject_codes', 'other_codes']:
        analysis[key] = sorted(analysis[key])
    
    return analysis

def save_categories_to_csv(df, output_file, analysis):
    """
    Save categories DataFrame to CSV file with analysis
    
    Args:
        df (pd.DataFrame): DataFrame with categories
        output_file (str): Output CSV file path
        analysis (dict): Analysis results
    """
    if df.empty:
        print("No data to save.")
        return
    
    try:
        # Save main categories file
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} categories to: {output_file}")
        
        # Save analysis summary
        analysis_file = output_file.replace('.csv', '_analysis.txt')
        with open(analysis_file, 'w') as f:
            f.write("FactSet Categories Analysis\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total categories: {analysis['total_count']}\n\n")
            
            # Country codes
            f.write(f"Country Codes (CN:*): {len(analysis['country_codes'])}\n")
            for code in analysis['country_codes'][:20]:  # Show first 20
                f.write(f"  - {code}\n")
            if len(analysis['country_codes']) > 20:
                f.write(f"  ... and {len(analysis['country_codes']) - 20} more\n")
            f.write("\n")
            
            # Industry codes
            f.write(f"Industry Codes (IN:*): {len(analysis['industry_codes'])}\n")
            for code in analysis['industry_codes']:
                f.write(f"  - {code}\n")
            f.write("\n")
            
            # Subject codes
            f.write(f"Subject Codes: {len(analysis['subject_codes'])}\n")
            for code in analysis['subject_codes'][:20]:  # Show first 20
                f.write(f"  - {code}\n")
            if len(analysis['subject_codes']) > 20:
                f.write(f"  ... and {len(analysis['subject_codes']) - 20} more\n")
            f.write("\n")
            
            # Other codes
            if analysis['other_codes']:
                f.write(f"Other Codes: {len(analysis['other_codes'])}\n")
                for code in analysis['other_codes'][:10]:
                    f.write(f"  - {code}\n")
                if len(analysis['other_codes']) > 10:
                    f.write(f"  ... and {len(analysis['other_codes']) - 10} more\n")
        
        print(f"Analysis summary saved to: {analysis_file}")
        
        # Print summary to console
        print("\n=== CATEGORIES SUMMARY ===")
        print(f"Total categories: {analysis['total_count']}")
        print(f"Country codes (CN:*): {len(analysis['country_codes'])}")
        print(f"Industry codes (IN:*): {len(analysis['industry_codes'])}")
        print(f"Subject codes: {len(analysis['subject_codes'])}")
        print(f"Other codes: {len(analysis['other_codes'])}")
        
        if analysis['industry_codes']:
            print(f"\nIndustry codes found:")
            for code in analysis['industry_codes']:
                print(f"  - {code}")
        
    except Exception as e:
        print(f"Error saving categories: {str(e)}")

def main():
    """
    Main function to orchestrate the categories pulling process
    """
    print("=" * 60)
    print("FactSet Categories Puller")
    print("=" * 60)
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Output File: {OUTPUT_FILE}")
    print(f"  SSL Cert: {SSL_CERT_PATH}")
    print(f"  Proxy: {PROXY_URL}")
    
    # Check if SSL certificate exists
    if not os.path.exists(SSL_CERT_PATH):
        print(f"\nERROR: SSL certificate not found at {SSL_CERT_PATH}")
        print("Please update the SSL_CERT_PATH variable with the correct path.")
        return
    
    print("\nStarting categories retrieval...")
    start_time = datetime.now()
    
    try:
        # Get all categories
        categories_df = get_all_categories()
        
        if categories_df.empty:
            print("No categories retrieved.")
            return
        
        # Analyze categories
        analysis = analyze_categories(categories_df)
        
        # Save to CSV with analysis
        save_categories_to_csv(categories_df, OUTPUT_FILE, analysis)
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        print(f"\nExecution completed in: {execution_time}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()