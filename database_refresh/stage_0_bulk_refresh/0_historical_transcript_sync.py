"""
Stage 0: Historical Transcript Sync
Downloads earnings transcripts from FactSet API with comprehensive functionality.
Handles authentication, configuration, directory management, and API integration.
"""

import os
import tempfile
import logging
import json
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
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
execution_log = []  # Detailed execution log entries
error_log = []      # Error log entries (only if errors occur)

def setup_logging() -> logging.Logger:
    """Set up minimal console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def log_console(message: str, level: str = "INFO"):
    """Log minimal message to console."""
    global logger
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

def log_execution(message: str, details: Dict[str, Any] = None):
    """Log detailed execution information for main log file."""
    global execution_log
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'details': details or {}
    }
    execution_log.append(log_entry)

def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information for error log file."""
    global error_log
    error_entry = {
        'timestamp': datetime.now().isoformat(),
        'error_type': error_type,
        'message': message,
        'details': details or {}
    }
    error_log.append(error_entry)

def save_logs_to_nas(nas_conn: SMBConnection, stage_summary: Dict[str, Any]):
    """Save execution and error logs to NAS at completion."""
    global execution_log, error_log
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.getenv('NAS_BASE_PATH')
    logs_path = nas_path_join(base_path, "Outputs", "Logs")
    
    # Create logs directory
    nas_create_directory(nas_conn, logs_path)
    
    # Save main execution log
    main_log_content = {
        'stage': 'stage_0_historical_transcript_sync',
        'execution_start': execution_log[0]['timestamp'] if execution_log else datetime.now().isoformat(),
        'execution_end': datetime.now().isoformat(),
        'summary': stage_summary,
        'execution_log': execution_log
    }
    
    main_log_filename = f"stage_0_historical_transcript_sync_{timestamp}.json"
    main_log_path = nas_path_join(logs_path, main_log_filename)
    main_log_json = json.dumps(main_log_content, indent=2)
    main_log_obj = io.BytesIO(main_log_json.encode('utf-8'))
    
    if nas_upload_file(nas_conn, main_log_obj, main_log_path):
        log_console(f"Execution log saved: {main_log_filename}")
    
    # Save error log only if errors exist
    if error_log:
        errors_path = nas_path_join(logs_path, "Errors")
        nas_create_directory(nas_conn, errors_path)
        
        error_log_content = {
            'stage': 'stage_0_historical_transcript_sync',
            'execution_time': datetime.now().isoformat(),
            'total_errors': len(error_log),
            'error_summary': stage_summary.get('errors', {}),
            'errors': error_log
        }
        
        error_log_filename = f"stage_0_historical_transcript_sync_errors_{timestamp}.json"
        error_log_path = nas_path_join(errors_path, error_log_filename)
        error_log_json = json.dumps(error_log_content, indent=2)
        error_log_obj = io.BytesIO(error_log_json.encode('utf-8'))
        
        if nas_upload_file(nas_conn, error_log_obj, error_log_path):
            log_console(f"Error log saved: {error_log_filename}", "WARNING")

def validate_environment_variables() -> None:
    """Validate all required environment variables are present."""
    
    required_env_vars = [
        'API_USERNAME', 'API_PASSWORD', 
        'PROXY_USER', 'PROXY_PASSWORD', 'PROXY_URL',
        'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
        'NAS_SHARE_NAME', 'NAS_BASE_PATH', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
        log_error(error_msg, "environment_validation", {
            'missing_variables': missing_vars,
            'total_required': len(required_env_vars)
        })
        raise ValueError(error_msg)
    
    log_execution("Environment variables validated successfully", {
        'total_variables': len(required_env_vars),
        'variables_checked': required_env_vars
    })

def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    
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
            log_execution("NAS connection established successfully", {
                'server_name': os.getenv('NAS_SERVER_NAME'),
                'port': nas_port,
                'share_name': os.getenv('NAS_SHARE_NAME')
            })
            return conn
        else:
            log_error("Failed to connect to NAS", "nas_connection", {
                'server_name': os.getenv('NAS_SERVER_NAME'),
                'port': nas_port
            })
            return None
            
    except Exception as e:
        log_error(f"Error connecting to NAS: {e}", "nas_connection", {
            'server_name': os.getenv('NAS_SERVER_NAME'),
            'error_details': str(e)
        })
        return None

def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv('NAS_SHARE_NAME'), nas_file_path, file_obj)
        file_obj.seek(0)
        content = file_obj.read()
        log_execution(f"Successfully downloaded file from NAS: {nas_file_path}", {
            'file_path': nas_file_path,
            'file_size': len(content)
        })
        return content
    except Exception as e:
        log_error(f"Failed to download file from NAS {nas_file_path}: {e}", "nas_download", {
            'file_path': nas_file_path,
            'error_details': str(e)
        })
        return None

def nas_upload_file(conn: SMBConnection, local_file_obj: io.BytesIO, nas_file_path: str) -> bool:
    """Upload a file object to NAS."""
    try:
        # Create parent directory if needed
        parent_dir = '/'.join(nas_file_path.split('/')[:-1])
        if parent_dir:
            nas_create_directory(conn, parent_dir)
        
        # Upload file
        local_file_obj.seek(0)  # Reset file pointer
        conn.storeFile(os.getenv('NAS_SHARE_NAME'), nas_file_path, local_file_obj)
        
        log_execution(f"Successfully uploaded file to NAS: {nas_file_path}", {
            'file_path': nas_file_path,
            'file_size': len(local_file_obj.getvalue())
        })
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS {nas_file_path}: {e}", "nas_upload", {
            'file_path': nas_file_path,
            'error_details': str(e)
        })
        return False

def validate_config_structure(config: Dict[str, Any]) -> None:
    """Validate that configuration contains required sections and fields."""
    
    # Required top-level sections
    required_sections = ['api_settings', 'monitored_institutions', 'ssl_cert_path', 'stage_0']
    
    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            log_error(error_msg, "config_validation", {'missing_section': section})
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
            log_error(error_msg, "config_validation", {'missing_setting': setting})
            raise ValueError(error_msg)
    
    # Validate monitored_institutions is not empty
    if not config['monitored_institutions']:
        error_msg = "monitored_institutions cannot be empty"
        log_error(error_msg, "config_validation", {})
        raise ValueError(error_msg)
    
    # Validate ssl_cert_path is not empty
    if not config['ssl_cert_path'] or not config['ssl_cert_path'].strip():
        error_msg = "ssl_cert_path cannot be empty"
        log_error(error_msg, "config_validation", {})
        raise ValueError(error_msg)
    
    log_execution("Configuration structure validation passed", {
        'sections_validated': required_sections,
        'api_settings_validated': required_api_settings,
        'total_institutions': len(config['monitored_institutions'])
    })

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
    start_time = datetime.now()
    
    log_console("=== STAGE 0: HISTORICAL TRANSCRIPT SYNC ===")
    log_execution("Stage 0 execution started", {
        'start_time': start_time.isoformat(),
        'stage': 'stage_0_historical_transcript_sync'
    })
    
    nas_conn = None
    ssl_cert_path = None
    stage_summary = {
        'status': 'unknown',
        'total_institutions': 0,
        'transcript_files_found': 0,
        'unparseable_files': 0,
        'execution_time_seconds': 0,
        'errors': {}
    }
    
    try:
        # Step 1: Validate environment variables
        log_console("Validating environment variables...")
        validate_environment_variables()
        
        # Step 2: Connect to NAS
        log_console("Connecting to NAS...")
        nas_conn = get_nas_connection()
        if not nas_conn:
            raise RuntimeError("Failed to establish NAS connection")
        
        # Step 3: Load configuration from NAS
        log_console("Loading configuration...")
        config = load_config_from_nas(nas_conn)
        stage_summary['total_institutions'] = len(config['monitored_institutions'])
        
        # Step 4: Set up SSL certificate
        log_console("Setting up SSL certificate...")
        ssl_cert_path = setup_ssl_certificate(nas_conn)
        if not ssl_cert_path:
            raise RuntimeError("Failed to set up SSL certificate")
        
        # Step 5: Configure proxy
        log_console("Configuring proxy authentication...")
        proxy_url = setup_proxy_configuration()
        
        # Step 6: Set up FactSet API client
        log_console("Setting up FactSet API client...")
        api_configuration = setup_factset_api_client(proxy_url, ssl_cert_path)
        
        log_console("Setup complete - ready for API calls")
        log_execution("Authentication and API setup completed", {
            'monitored_institutions': len(config['monitored_institutions']),
            'transcript_types': config['api_settings']['transcript_types'],
            'sync_start_date': config['sync_start_date']
        })
        
        # Step 7: Create/validate Data directory structure
        log_console("Creating Data directory structure...")
        if not create_data_directory_structure(nas_conn):
            raise RuntimeError("Failed to create Data directory structure")
        
        # Step 8: Scan existing transcript files
        log_console("Scanning existing transcript inventory...")
        transcript_inventory = scan_existing_transcripts(nas_conn)
        stage_summary['transcript_files_found'] = len(transcript_inventory)
        
        # Extract unparseable files count from error log
        unparseable_count = sum(1 for entry in error_log if entry['error_type'] == 'unparseable_filename')
        stage_summary['unparseable_files'] = unparseable_count
        
        log_console(f"Inventory complete: {len(transcript_inventory)} files found")
        if unparseable_count > 0:
            log_console(f"Warning: {unparseable_count} files have non-conforming names", "WARNING")
        
        # TODO: Add transcript download logic here using transcript_inventory
        
        stage_summary['status'] = 'completed'
        stage_summary['execution_time_seconds'] = (datetime.now() - start_time).total_seconds()
        
    except Exception as e:
        stage_summary['status'] = 'failed'
        stage_summary['execution_time_seconds'] = (datetime.now() - start_time).total_seconds()
        stage_summary['errors']['main_execution'] = str(e)
        
        log_console(f"Stage 0 failed: {e}", "ERROR")
        log_error(f"Stage 0 execution failed: {e}", "main_execution", {
            'error_details': str(e),
            'execution_time_seconds': stage_summary['execution_time_seconds']
        })
        raise
        
    finally:
        # Save logs to NAS
        if nas_conn:
            try:
                save_logs_to_nas(nas_conn, stage_summary)
            except Exception as e:
                log_console(f"Warning: Failed to save logs to NAS: {e}", "WARNING")
        
        # Cleanup
        if nas_conn:
            nas_conn.close()
            log_console("NAS connection closed")
        
        cleanup_temporary_files(ssl_cert_path)
        log_console("=== STAGE 0 COMPLETE ===")

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
        logger.debug(f"Filename {filename} is not an XML file")
        return None
    
    # Remove .xml extension
    basename = filename[:-4]
    
    # Split by underscores - expect 6 parts
    parts = basename.split('_')
    if len(parts) != 6:
        logger.debug(f"Filename {filename} has {len(parts)} parts, expected 6 parts (ticker_quarter_year_transcripttype_eventid_versionid)")
        return None
    
    try:
        parsed = {
            'ticker': parts[0],
            'quarter': parts[1],
            'year': parts[2],
            'transcript_type': parts[3],
            'event_id': parts[4],
            'version_id': parts[5]
        }
        
        # Basic validation
        if not parsed['ticker'] or not parsed['quarter'] or not parsed['year']:
            logger.debug(f"Filename {filename} has empty required fields")
            return None
        
        return parsed
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
    unparseable_files = []
    
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
                            unparseable_files.append({
                                'filename': xml_file,
                                'full_path': nas_path_join(company_path, xml_file),
                                'location': f"{fiscal_year}/{quarter}/{company_type}/{company}",
                                'expected_format': 'ticker_quarter_year_transcripttype_eventid_versionid.xml'
                            })
                            # Log as error for tracking
                            log_error(f"Unparseable filename: {xml_file}", "unparseable_filename", {
                                'filename': xml_file,
                                'location': f"{fiscal_year}/{quarter}/{company_type}/{company}",
                                'expected_format': 'ticker_quarter_year_transcripttype_eventid_versionid.xml'
                            })
    
    # Log inventory completion
    log_execution("Transcript inventory scan completed", {
        'total_files_found': len(transcript_inventory),
        'unparseable_files': len(unparseable_files),
        'fiscal_years_scanned': fiscal_years,
        'sample_files': [entry['filename'] for entry in transcript_inventory[:5]]
    })
    
    # Report unparseable files in execution log
    if unparseable_files:
        log_execution("Found files with non-conforming filenames", {
            'total_unparseable': len(unparseable_files),
            'sample_unparseable': unparseable_files[:5],
            'expected_format': 'ticker_quarter_year_transcripttype_eventid_versionid.xml'
        })
    
    return transcript_inventory

def calculate_rolling_window() -> Tuple[datetime.date, datetime.date]:
    """Calculate 3-year rolling window from current date."""
    end_date = datetime.now().date()
    start_date = datetime(end_date.year - 3, end_date.month, end_date.day).date()
    
    log_execution("Calculated 3-year rolling window", {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'window_years': 3
    })
    
    return start_date, end_date

def get_api_transcripts_for_company(api_instance, ticker: str, institution_info: Dict[str, str], 
                                   start_date: datetime.date, end_date: datetime.date, 
                                   configuration) -> List[Dict[str, Any]]:
    """Get all transcripts for a company from the API within date range."""
    global config
    
    try:
        log_execution(f"Querying API for {ticker} transcripts", {
            'ticker': ticker,
            'institution': institution_info['name'],
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat()
        })
        
        api_params = {
            'ids': [ticker],
            'start_date': start_date,
            'end_date': end_date,
            'categories': config['api_settings']['industry_categories'],
            'sort': config['api_settings']['sort_order'],
            'pagination_limit': config['api_settings']['pagination_limit'],
            'pagination_offset': config['api_settings']['pagination_offset']
        }
        
        response = api_instance.get_transcripts_ids(**api_params)
        
        if not response or not hasattr(response, 'data') or not response.data:
            log_execution(f"No transcripts found for {ticker}", {'ticker': ticker})
            return []
        
        all_transcripts = [transcript.to_dict() for transcript in response.data]
        
        # Filter to only transcripts where our ticker is the SOLE primary ID (anti-contamination)
        filtered_transcripts = []
        for transcript in all_transcripts:
            primary_ids = transcript.get('primary_ids', [])
            if isinstance(primary_ids, list) and primary_ids == [ticker]:
                filtered_transcripts.append(transcript)
        
        # Filter for earnings transcripts only
        earnings_transcripts = [
            transcript for transcript in filtered_transcripts
            if transcript.get('event_type', '') and 'Earnings' in str(transcript.get('event_type', ''))
        ]
        
        log_execution(f"API query completed for {ticker}", {
            'ticker': ticker,
            'total_transcripts': len(all_transcripts),
            'filtered_transcripts': len(filtered_transcripts),
            'earnings_transcripts': len(earnings_transcripts)
        })
        
        return earnings_transcripts
        
    except Exception as e:
        log_error(f"Error querying API for {ticker}: {e}", "api_query", {
            'ticker': ticker,
            'error_details': str(e)
        })
        return []

def create_api_transcript_list(api_transcripts: List[Dict[str, Any]], ticker: str, 
                              institution_info: Dict[str, str]) -> List[Dict[str, str]]:
    """Convert API transcripts to standardized format for comparison."""
    api_list = []
    
    for transcript in api_transcripts:
        for transcript_type in config['api_settings']['transcript_types']:
            if transcript.get('transcript_type') == transcript_type:
                api_record = {
                    'company_type': institution_info['type'],
                    'company': ticker,
                    'ticker': ticker,
                    'transcript_type': transcript_type,
                    'event_id': str(transcript.get('event_id', '')),
                    'version_id': str(transcript.get('version_id', '')),
                    'event_date': transcript.get('event_date', ''),
                    'transcript_link': transcript.get('transcripts_link', '')
                }
                api_list.append(api_record)
    
    return api_list

def compare_transcripts(api_transcripts: List[Dict[str, str]], 
                       nas_transcripts: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Compare API vs NAS transcripts and determine what to download/remove."""
    
    # Create lookup dictionaries by event_id
    api_by_event_id = {}
    for transcript in api_transcripts:
        event_id = transcript['event_id']
        if event_id not in api_by_event_id:
            api_by_event_id[event_id] = []
        api_by_event_id[event_id].append(transcript)
    
    nas_by_event_id = {}
    for transcript in nas_transcripts:
        event_id = transcript['event_id']
        if event_id not in nas_by_event_id:
            nas_by_event_id[event_id] = []
        nas_by_event_id[event_id].append(transcript)
    
    to_download = []
    to_remove = []
    
    # Process each event_id in API
    for event_id, api_versions in api_by_event_id.items():
        if event_id in nas_by_event_id:
            # Event exists in both API and NAS - compare versions
            nas_versions = nas_by_event_id[event_id]
            
            for api_transcript in api_versions:
                # Find matching NAS transcript by transcript_type
                matching_nas = None
                for nas_transcript in nas_versions:
                    if (nas_transcript['transcript_type'] == api_transcript['transcript_type'] and
                        nas_transcript['ticker'] == api_transcript['ticker']):
                        matching_nas = nas_transcript
                        break
                
                if matching_nas:
                    # Compare versions - API version is always considered latest
                    if api_transcript['version_id'] != matching_nas['version_id']:
                        to_download.append(api_transcript)
                        to_remove.append(matching_nas)
                    # If versions are same, no action needed
                else:
                    # New transcript type for this event_id
                    to_download.append(api_transcript)
        else:
            # New event_id - download all versions
            to_download.extend(api_versions)
    
    # Process each event_id in NAS that's not in API (outside 3-year window)
    for event_id, nas_versions in nas_by_event_id.items():
        if event_id not in api_by_event_id:
            to_remove.extend(nas_versions)
    
    return to_download, to_remove

if __name__ == "__main__":
    main()