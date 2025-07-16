"""
Minimal Stage 0: Transcript Sync - Foundation Setup
Downloads earnings transcripts from FactSet API with minimal required functionality.
Focuses on core authentication, configuration, and API setup.
"""

import os
import json
import tempfile
import logging
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional
import io

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
    required_sections = ['api_settings', 'monitored_institutions', 'ssl_cert_nas_path']
    
    for section in required_sections:
        if section not in config:
            error_msg = f"Missing required configuration section: {section}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # Validate api_settings structure
    required_api_settings = ['request_delay', 'max_retries', 'retry_delay', 'transcript_types']
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
    
    # Validate ssl_cert_nas_path is not empty
    if not config['ssl_cert_nas_path'] or not config['ssl_cert_nas_path'].strip():
        error_msg = "ssl_cert_nas_path cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration structure validation passed")

def load_config_from_nas(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate configuration from NAS."""
    global logger
    
    try:
        config_path = os.getenv('CONFIG_PATH')
        logger.info(f"Loading configuration from NAS: {config_path}")
        
        config_data = nas_download_file(nas_conn, config_path)
        if not config_data:
            error_msg = f"Failed to download configuration file from NAS: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Parse JSON configuration
        try:
            config = json.loads(config_data.decode('utf-8'))
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in configuration file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Validate configuration structure
        validate_config_structure(config)
        
        logger.info(f"Successfully loaded configuration with {len(config['monitored_institutions'])} institutions")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration from NAS: {e}")
        raise

def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for API use."""
    global logger, config
    
    try:
        cert_path = config['ssl_cert_nas_path']
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
    logger.info("=== MINIMAL STAGE 0: TRANSCRIPT SYNC SETUP ===")
    
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
        
        # TODO: Add transcript download logic here
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise
        
    finally:
        # Cleanup
        if nas_conn:
            nas_conn.close()
            logger.info("NAS connection closed")
        
        cleanup_temporary_files(ssl_cert_path)

if __name__ == "__main__":
    main()