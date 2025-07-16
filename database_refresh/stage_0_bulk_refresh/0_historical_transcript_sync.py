"""
Stage 0: Historical Transcript Sync
Downloads earnings transcripts from FactSet API with comprehensive functionality.
Handles authentication, configuration, directory management, and API integration.
"""

import os
import tempfile
import logging
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List
import io
import re

import yaml
import fds.sdk.EventsandTranscripts
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None

def setup_logging() -> logging.Logger:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def validate_environment_variables() -> None:
    """Validate all required environment variables are present."""
    global logger
    
    required_env_vars = [
        'API_USERNAME', 'API_PASSWORD', 
        'PROXY_USER', 'PROXY_PASSWORD', 'PROXY_URL',
        'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
        'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"All {len(required_env_vars)} required environment variables validated")

def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    global logger
    
    try:
        conn = SMBConnection(
            username=os.getenv('NAS_USERNAME'),
            password=os.getenv('NAS_PASSWORD'),
            my_name=os.getenv('CLIENT_MACHINE_NAME'),
            remote_name=os.getenv('NAS_SERVER_NAME'),
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        nas_port = int(os.getenv('NAS_PORT', 445))
        if conn.connect(os.getenv('NAS_SERVER_IP'), nas_port):
            logger.info("Successfully connected to NAS")
            return conn
        else:
            logger.error("Failed to connect to NAS")
            return None
            
    except Exception as e:
        logger.error(f"Error connecting to NAS: {e}")
        return None

def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    global logger
    
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv('NAS_SHARE_NAME'), nas_file_path, file_obj)
        file_obj.seek(0)
        content = file_obj.read()
        logger.info(f"Successfully downloaded file from NAS: {nas_file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to download file from NAS {nas_file_path}: {e}")
        return None

def validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate that configuration contains required sections and fields."""
    global logger
    
    # Required top-level sections
    required_sections = ['api_settings', 'monitored_institutions', 'ssl_cert_path', 'sync_start_date']
    
    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Validate api_settings structure
    required_api_settings = [
        'industry_categories', 'transcript_types', 'sort_order', 'pagination_limit',
        'pagination_offset', 'request_delay', 'max_retries', 'retry_delay',
        'use_exponential_backoff', 'max_backoff_delay'
    ]
    for setting in required_api_settings:
        if setting not in config['api_settings']:
            error_msg = f"Missing required API setting: {setting}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Validate monitored_institutions is not empty
    if not config['monitored_institutions']:
        error_msg = "monitored_institutions cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate ssl_cert_path is not empty
    if not config['ssl_cert_path'] or not config['ssl_cert_path'].strip():
        error_msg = "ssl_cert_path cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Validate sync_start_date format
    try:
        from datetime import datetime
        datetime.strptime(config['sync_start_date'], "%Y-%m-%d")
    except ValueError as e:
        error_msg = f"Invalid sync_start_date format: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration structure validation passed")

def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate YAML configuration from NAS."""
    global logger
    
    try:
        config_path = os.getenv('CONFIG_PATH')
        logger.info(f"Loading YAML configuration from NAS: {config_path}")
        
        config_data = nas_download_file(nas_conn, config_path)
        if not config_data:
            error_msg = f"Failed to download configuration file from NAS: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Parse YAML configuration
        try:
            config = yaml.safe_load(config_data.decode('utf-8'))
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in configuration file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate configuration structure
        validate_config_structure(config)
        
        logger.info(f"Successfully loaded YAML configuration with {len(config['monitored_institutions'])} institutions")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from NAS: {e}")
        raise

def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for API use."""
    global logger, config
    
    try:
        cert_path = config['ssl_cert_path']
        logger.info(f"Downloading SSL certificate from NAS: {cert_path}")
        
        cert_data = nas_download_file(nas_conn, cert_path)
        if not cert_data:
            error_msg = f"Failed to download SSL certificate from NAS: {cert_path}"
            logger.error(error_msg)
            return None
        
        # Create temporary certificate file
        temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
        temp_cert.write(cert_data)
        temp_cert.close()
        
        # Set environment variables for SSL
        os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
        os.environ["SSL_CERT_FILE"] = temp_cert.name
        
        logger.info(f"SSL certificate configured successfully: {temp_cert.name}")
        return temp_cert.name
        
    except Exception as e:
        logger.error(f"Error setting up SSL certificate: {e}")
        return None

def setup_proxy_configuration() -> str:
    """Configure proxy URL for API authentication."""
    global logger
    
    try:
        proxy_user = os.getenv('PROXY_USER')
        proxy_password = os.getenv('PROXY_PASSWORD')
        proxy_url = os.getenv('PROXY_URL')
        proxy_domain = os.getenv('PROXY_DOMAIN', 'MAPLE')
        
        # Escape domain and user for NTLM authentication
        escaped_domain = quote(proxy_domain + '\\' + proxy_user)
        quoted_password = quote(proxy_password)
        
        # Construct proxy URL
        proxy_url_formatted = f"http://{escaped_domain}:{quoted_password}@{proxy_url}"
        
        logger.info("Proxy configuration completed successfully")
        return proxy_url_formatted
        
    except Exception as e:
        logger.error(f"Error configuring proxy: {e}")
        raise

def setup_factset_api_client(proxy_url: str, ssl_cert_path: str):
    """Configure FactSet API client with proxy and SSL settings."""
    global logger
    
    try:
        api_username = os.getenv('API_USERNAME')
        api_password = os.getenv('API_PASSWORD')
        
        # Configure FactSet API client
        configuration = fds.sdk.EventsandTranscripts.Configuration(
            username=api_username,
            password=api_password,
            proxy=proxy_url,
            ssl_ca_cert=ssl_cert_path
        )
        
        # Generate authentication token
        configuration.get_basic_auth_token()
        
        logger.info("FactSet API client configured successfully")
        return configuration
        
    except Exception as e:
        logger.error(f"Error setting up FactSet API client: {e}")
        raise

def cleanup_temporary_files(ssl_cert_path: Optional[str]) -> None:
    """Clean up temporary files."""
    global logger
    
    if ssl_cert_path:
        try:
            os.unlink(ssl_cert_path)
            logger.info("Temporary SSL certificate file cleaned up")
        except (OSError, FileNotFoundError) as e:
            logger.warning(f"Failed to clean up SSL certificate file: {e}")

def main() -> None:
    """Main function to set up authentication and API configuration."""
    global logger, config
    
    # Initialize logging
    logger = setup_logging()
    logger.info("=== STAGE 0: HISTORICAL TRANSCRIPT SYNC ===")
    
    nas_conn = None
    ssl_cert_path = None
    
    try:
        # Step 1: Validate environment variables
        logger.info("Step 1: Validating environment variables...")
        validate_environment_variables()
        
        # Step 2: Connect to NAS
        logger.info("Step 2: Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to establish NAS connection")
        
        # Step 3: Load configuration from NAS
        logger.info("Step 3: Loading configuration from NAS...")
        config = load_config_from_nas(nas_conn)
        
        # Step 4: Set up SSL certificate
        logger.info("Step 4: Setting up SSL certificate...")
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        if not ssl_cert_path:
            raise RuntimeError("Failed to set up SSL certificate")
        
        # Step 5: Configure proxy
        logger.info("Step 5: Configuring proxy authentication...")
        proxy_url = setup_proxy_configuration()
        
        # Step 6: Set up FactSet API client
        logger.info("Step 6: Setting up FactSet API client...")
        api_configuration = setup_factset_api_client(proxy_url, ssl_cert_path)
        
        logger.info("=== SETUP COMPLETE - Ready for API calls ===")
        logger.info(f"Monitoring {len(config['monitored_institutions'])} institutions")
        logger.info(f"Transcript types: {config['api_settings']['transcript_types']}")
        
        # Step 7: Create/validate Data directory structure
        logger.info("Step 7: Creating Data directory structure...")
        if not create_data_directory_structure(nas_conn):
            raise RuntimeError("Failed to create Data directory structure")
        
        # Step 8: Scan existing transcript files
        logger.info("Step 8: Scanning existing transcript inventory...")
        transcript_inventory = scan_existing_transcripts(nas_conn)
        
        logger.info("=== TRANSCRIPT INVENTORY COMPLETE ===")
        logger.info(f"Found {len(transcript_inventory)} existing transcript files")
        
        # Show sample of inventory if any files found
        if transcript_inventory:
            logger.info("Sample inventory entries:")
            for i, entry in enumerate(transcript_inventory[:3]):  # Show first 3 entries
                logger.info(f"  {i+1}. {entry['ticker']} - {entry['file_year']} {entry['file_quarter']} - {entry['transcript_type']} - {entry['filename']}")
            if len(transcript_inventory) > 3:
                logger.info(f"  ... and {len(transcript_inventory) - 3} more entries")
        else:
            logger.info("No existing transcript files found - starting with empty inventory")
        
        # TODO: Add transcript download logic here using transcript_inventory
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
            logger.info("NAS connection closed")
        
        cleanup_temporary_files(ssl_cert_path)

def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    return '/'.join(str(part) for part in parts if part)

def nas_file_exists(conn: SMBConnection, file_path: str) -> bool:
    """Check if a file or directory exists on the NAS."""
    global logger
    try:
        conn.getAttributes(os.getenv('NAS_SHARE_NAME'), file_path)
        return True
    except Exception:
        return False

def nas_create_directory(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory on NAS with safe iterative parent creation."""
    global logger
    
    normalized_path = dir_path.strip('/').rstrip('/')
    if not normalized_path:
        logger.error("Cannot create directory with empty path")
        return False
    
    path_parts = [part for part in normalized_path.split('/') if part]
    if not path_parts:
        logger.error("Cannot create directory with invalid path")
        return False
    
    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part
        
        if nas_file_exists(conn, current_path):
            continue
        
        try:
            conn.createDirectory(os.getenv('NAS_SHARE_NAME'), current_path)
            logger.debug(f"Created directory: {current_path}")
        except Exception as e:
            if not nas_file_exists(conn, current_path):
                logger.error(f"Failed to create directory {current_path}: {e}")
                return False
    
    return True

def nas_list_directories(conn: SMBConnection, directory_path: str) -> List[str]:
    """List subdirectories in a NAS directory."""
    global logger
    try:
        files = conn.listPath(os.getenv('NAS_SHARE_NAME'), directory_path)
        return [file_info.filename for file_info in files 
                if file_info.isDirectory and file_info.filename not in ['.', '..']]
    except Exception as e:
        logger.debug(f"Failed to list directories in {directory_path}: {e}")
        return []

def nas_list_files(conn: SMBConnection, directory_path: str) -> List[str]:
    """List XML files in a NAS directory."""
    global logger
    try:
        files = conn.listPath(os.getenv('NAS_SHARE_NAME'), directory_path)
        return [file_info.filename for file_info in files 
                if not file_info.isDirectory and file_info.filename.endswith('.xml')]
    except Exception as e:
        logger.debug(f"Failed to list files in {directory_path}: {e}")
        return []

def parse_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse filename format: ticker_quarter_year_transcripttype_eventid_versionid.xml"""
    global logger
    
    if not filename.endswith('.xml'):
        return None
    
    # Remove .xml extension
    basename = filename[:-4]
    
    # Split by underscores - expect 6 parts
    parts = basename.split('_')
    if len(parts) != 6:
        logger.debug(f"Filename {filename} does not match expected format (6 parts)")
        return None
    
    try:
        return {
            'ticker': parts[0],
            'quarter': parts[1],
            'year': parts[2],
            'transcript_type': parts[3],
            'event_id': parts[4],
            'version_id': parts[5]
        }
    except Exception as e:
        logger.debug(f"Error parsing filename {filename}: {e}")
        return None

def create_data_directory_structure(nas_conn: SMBConnection) -> bool:
    """Create base Data directory structure on NAS."""
    global logger
    
    base_path = os.getenv('NAS_BASE_PATH')
    data_path = nas_path_join(base_path, "Outputs", "Data")
    
    logger.info(f"Creating Data directory structure: {data_path}")
    
    if nas_create_directory(nas_conn, data_path):
        logger.info("Data directory structure created successfully")
        return True
    else:
        logger.error("Failed to create Data directory structure")
        return False

def scan_existing_transcripts(nas_conn: SMBConnection) -> List[Dict[str, str]]:
    """Scan existing transcript files and create inventory list."""
    global logger, config
    
    base_path = os.getenv('NAS_BASE_PATH')
    data_path = nas_path_join(base_path, "Outputs", "Data")
    
    transcript_inventory = []
    
    # Check if Data directory exists
    if not nas_file_exists(nas_conn, data_path):
        logger.info("Data directory does not exist - will create empty structure")
        return transcript_inventory
    
    logger.info("Scanning existing transcript files...")
    
    # Scan fiscal years
    fiscal_years = nas_list_directories(nas_conn, data_path)
    logger.info(f"Found {len(fiscal_years)} fiscal years: {fiscal_years}")
    
    for fiscal_year in fiscal_years:
        year_path = nas_path_join(data_path, fiscal_year)
        
        # Scan quarters within each year
        quarters = nas_list_directories(nas_conn, year_path)
        logger.debug(f"Year {fiscal_year} has quarters: {quarters}")
        
        for quarter in quarters:
            quarter_path = nas_path_join(year_path, quarter)
            
            # Scan company types within each quarter
            company_types = nas_list_directories(nas_conn, quarter_path)
            logger.debug(f"Year {fiscal_year} Quarter {quarter} has company types: {company_types}")
            
            for company_type in company_types:
                company_type_path = nas_path_join(quarter_path, company_type)
                
                # Scan companies within each type
                companies = nas_list_directories(nas_conn, company_type_path)
                logger.debug(f"Company type {company_type} has {len(companies)} companies")
                
                for company in companies:
                    company_path = nas_path_join(company_type_path, company)
                    
                    # Scan XML files in company directory
                    xml_files = nas_list_files(nas_conn, company_path)
                    logger.debug(f"Company {company} has {len(xml_files)} XML files")
                    
                    for xml_file in xml_files:
                        # Parse filename
                        parsed = parse_filename(xml_file)
                        if parsed:
                            transcript_record = {
                                'fiscal_year': fiscal_year,
                                'quarter': quarter,
                                'company_type': company_type,
                                'company': company,
                                'ticker': parsed['ticker'],
                                'file_quarter': parsed['quarter'],
                                'file_year': parsed['year'],
                                'transcript_type': parsed['transcript_type'],
                                'event_id': parsed['event_id'],
                                'version_id': parsed['version_id'],
                                'filename': xml_file,
                                'full_path': nas_path_join(company_path, xml_file)
                            }
                            transcript_inventory.append(transcript_record)
                        else:
                            logger.warning(f"Could not parse filename: {xml_file}")
    
    logger.info(f"Transcript inventory complete: {len(transcript_inventory)} files found")
    return transcript_inventory

if __name__ == "__main__":
    main()