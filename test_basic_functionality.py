"""
Basic functionality tests for earnings transcript sync
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile
import io
from datetime import datetime

# Import the main module
import earnings_transcript_repository_sync_v2 as sync_module

class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality without external dependencies"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.env_vars = {
            'API_USERNAME': 'test_user',
            'API_PASSWORD': 'test_pass',
            'PROXY_USER': 'proxy_user',
            'PROXY_PASSWORD': 'proxy_pass',
            'NAS_USERNAME': 'nas_user',
            'NAS_PASSWORD': 'nas_pass',
            'NAS_SERVER_IP': '192.168.1.100',
            'DEBUG_MODE': 'true'
        }
        
        self.env_patcher = patch.dict(os.environ, self.env_vars)
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test environment"""
        self.env_patcher.stop()
    
    def test_sanitize_for_filename(self):
        """Test filename sanitization"""
        test_cases = [
            ("Normal Text", "Normal_Text"),
            ("Text/With\\Bad:Chars", "Text_With_Bad_Chars"),
            ("", "unknown"),
            (None, "unknown"),
            ("Test   Multiple    Spaces", "Test_Multiple_Spaces")
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = sync_module.sanitize_for_filename(input_text)
                self.assertEqual(result, expected)
    
    def test_nas_path_join(self):
        """Test NAS path joining"""
        result = sync_module.nas_path_join("folder1", "folder2", "file.xml")
        self.assertEqual(result, "folder1/folder2/file.xml")
        
        # Test with empty parts
        result = sync_module.nas_path_join("folder1", "", "file.xml")
        self.assertEqual(result, "folder1/file.xml")
    
    def test_validate_transcript_data(self):
        """Test transcript data validation"""
        # Valid transcript
        valid_transcript = {
            'event_type': 'Earnings',
            'transcript_type': 'Corrected',
            'event_id': 'E123456',
            'report_id': 'R789012',
            'transcripts_link': 'https://api.factset.com/transcript/123'
        }
        self.assertTrue(sync_module.validate_transcript_data(valid_transcript))
        
        # Missing required field
        invalid_transcript = {
            'event_type': 'Earnings',
            'transcript_type': 'Corrected',
            # Missing event_id
            'report_id': 'R789012',
            'transcripts_link': 'https://api.factset.com/transcript/123'
        }
        self.assertFalse(sync_module.validate_transcript_data(invalid_transcript))
        
        # Invalid link format
        invalid_link_transcript = {
            'event_type': 'Earnings',
            'transcript_type': 'Corrected',
            'event_id': 'E123456',
            'report_id': 'R789012',
            'transcripts_link': 'invalid_link'
        }
        self.assertFalse(sync_module.validate_transcript_data(invalid_link_transcript))
    
    def test_create_filename(self):
        """Test filename creation"""
        transcript_data = {
            'event_date': '2024-01-15',
            'event_type': 'Earnings Call',
            'transcript_type': 'Corrected',
            'event_id': 'E123456',
            'report_id': 'R789012',
            'version_id': 'V1'
        }
        
        filename = sync_module.create_filename(transcript_data, target_ticker="RY-CA")
        expected_parts = ["RY-CA", "2024-01-15", "Earnings_Call", "Corrected", "E123456", "R789012", "V1.xml"]
        expected = "_".join(expected_parts)
        
        self.assertEqual(filename, expected)
    
    @patch('earnings_transcript_repository_sync_v2.logger')
    def test_download_failure_tracking(self, mock_logger):
        """Test download failure tracking"""
        failure = sync_module.DownloadFailure(
            ticker="RY-CA",
            transcript_type="Corrected",
            filename="test_file.xml",
            transcript_link="https://test.com/transcript",
            error="Connection timeout",
            attempts=1
        )
        
        self.assertEqual(failure.ticker, "RY-CA")
        self.assertEqual(failure.error, "Connection timeout")
        self.assertEqual(failure.attempts, 1)
    
    @patch('earnings_transcript_repository_sync_v2.SMBConnection')
    def test_nas_connection_creation(self, mock_smb):
        """Test NAS connection creation"""
        mock_conn = Mock()
        mock_conn.connect.return_value = True
        mock_smb.return_value = mock_conn
        
        with patch('earnings_transcript_repository_sync_v2.logger'):
            result = sync_module.get_nas_connection()
        
        self.assertIsNotNone(result)
        mock_smb.assert_called_once()
        mock_conn.connect.assert_called_once()

class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_missing_required_env_vars(self):
        """Test that missing environment variables raise error"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                # Force re-import to trigger validation
                import importlib
                importlib.reload(sync_module)
            
            self.assertIn("Missing required environment variables", str(context.exception))

if __name__ == '__main__':
    unittest.main()