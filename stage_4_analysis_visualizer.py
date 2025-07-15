#!/usr/bin/env python3
"""
FactSet Stage 4 Analysis Visualizer
===================================

Comprehensive visualization tool for analyzing Stage 4 LLM classification outputs.
Provides detailed insights into token distributions, section classification success,
speaker block patterns, and paragraph-level chunking optimization.

Author: Claude Code
Date: 2024-07-15
"""

import json
import os
import tempfile
import logging
import io
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from smb.SMBConnection import SMBConnection
from urllib.parse import quote
from dotenv import load_dotenv
import tiktoken
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Authentication and connection settings from environment (same as Stage 4)
API_USERNAME = os.getenv("API_USERNAME")
API_PASSWORD = os.getenv("API_PASSWORD")
PROXY_USER = os.getenv("PROXY_USER")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
PROXY_URL = os.getenv("PROXY_URL")
NAS_USERNAME = os.getenv("NAS_USERNAME")
NAS_PASSWORD = os.getenv("NAS_PASSWORD")
NAS_SERVER_IP = os.getenv("NAS_SERVER_IP")
NAS_SERVER_NAME = os.getenv("NAS_SERVER_NAME")
NAS_SHARE_NAME = os.getenv("NAS_SHARE_NAME")
NAS_BASE_PATH = os.getenv("NAS_BASE_PATH")
NAS_PORT = int(os.getenv("NAS_PORT", 445))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CLIENT_MACHINE_NAME = os.getenv("CLIENT_MACHINE_NAME")
PROXY_DOMAIN = os.getenv("PROXY_DOMAIN", "MAPLE")

# Global variables for configuration
config = {}
logger = None

@dataclass
class AnalysisConfig:
    """Configuration for Stage 4 analysis visualization."""
    stage_4_output_path: str = "Outputs/Stage4/"
    stage_4_output_file: str = "stage_4_classified_transcripts.json"
    ssl_cert_nas_path: str = "Inputs/certificate/certificate.cer"
    max_token_length: int = 750  # Character limit from Stage 4
    tiktoken_model: str = "gpt-3.5-turbo"
    output_directory: str = "Outputs/Analysis/"
    chart_width: int = 1200
    chart_height: int = 800

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    temp_log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        suffix=".log",
        prefix=f'stage_4_analysis_log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_',
        delete=False,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(temp_log_file.name), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.temp_log_file = temp_log_file.name
    return logger


def validate_file_path(path: str) -> bool:
    """Validate file path for security."""
    if not path or not isinstance(path, str):
        return False
    
    # Check for directory traversal attempts
    if '..' in path or path.startswith('/'):
        return False
    
    # Check for invalid characters
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in path for char in invalid_chars):
        return False
    
    return True


def validate_nas_path(path: str) -> bool:
    """Validate NAS path structure."""
    if not path or not isinstance(path, str):
        return False
    
    # Ensure path is relative and safe
    normalized = path.strip('/')
    parts = normalized.split('/')
    
    for part in parts:
        if not part or part in ['.', '..']:
            return False
        if not validate_file_path(part):
            return False
    
    return True


def get_nas_connection() -> Optional[SMBConnection]:
    """Create and return an SMB connection to the NAS."""
    global logger
    try:
        conn = SMBConnection(
            username=NAS_USERNAME,
            password=NAS_PASSWORD,
            my_name=CLIENT_MACHINE_NAME,
            remote_name=NAS_SERVER_NAME,
            use_ntlm_v2=True,
            is_direct_tcp=True,
        )
        if conn.connect(NAS_SERVER_IP, NAS_PORT):
            logger.info("Connected to NAS successfully")
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
    
    # Validate path for security
    if not validate_nas_path(nas_file_path):
        logger.error(f"Invalid NAS file path: {nas_file_path}")
        return None
    
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(NAS_SHARE_NAME, nas_file_path, file_obj)
        file_obj.seek(0)
        return file_obj.read()
    except Exception as e:
        logger.error(f"Failed to download file from NAS {nas_file_path}: {e}")
        return None


def nas_path_join(*parts: str) -> str:
    """Join path parts for NAS paths using forward slashes."""
    clean_parts = []
    for part in parts:
        if part:
            clean_part = str(part).strip("/")
            if clean_part:
                clean_parts.append(clean_part)
    return "/".join(clean_parts)


def load_stage_config(nas_conn: SMBConnection) -> Dict[str, Any]:
    """Load and validate shared configuration from NAS."""
    global logger
    try:
        logger.info("Loading shared configuration from NAS...")
        config_data = nas_download_file(nas_conn, CONFIG_PATH)
        if config_data:
            stage_config = json.loads(config_data.decode("utf-8"))
            logger.info("Successfully loaded shared configuration from NAS")
            return stage_config
        else:
            logger.error("Config file not found on NAS - script cannot proceed")
            raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading config from NAS: {e}")
        raise


def setup_ssl_certificate(nas_conn: SMBConnection) -> Optional[str]:
    """Download SSL certificate from NAS and set up for use."""
    global logger, config
    try:
        logger.info("Downloading SSL certificate from NAS...")
        cert_data = nas_download_file(nas_conn, config['ssl_cert_nas_path'])
        if cert_data:
            temp_cert = tempfile.NamedTemporaryFile(mode='wb', suffix='.cer', delete=False)
            temp_cert.write(cert_data)
            temp_cert.close()
            
            os.environ["REQUESTS_CA_BUNDLE"] = temp_cert.name
            os.environ["SSL_CERT_FILE"] = temp_cert.name
            
            logger.info("SSL certificate downloaded from NAS")
            return temp_cert.name
        else:
            logger.error("Failed to download SSL certificate from NAS")
            return None
    except Exception as e:
        logger.error(f"Error downloading SSL certificate from NAS: {e}")
        return None

class TokenAnalyzer:
    """Enhanced token analysis with tiktoken integration."""
    
    def __init__(self, analysis_config: AnalysisConfig):
        self.analysis_config = analysis_config
        self.encoding = None
        self.ssl_cert_file = None
        
    def initialize_tiktoken(self, nas_conn: SMBConnection) -> bool:
        """Initialize tiktoken with corporate SSL/proxy setup."""
        global logger, config
        try:
            # Setup SSL certificate
            self.ssl_cert_file = setup_ssl_certificate(nas_conn)
            if not self.ssl_cert_file:
                logger.error("Failed to setup SSL certificate")
                return False
            
            # Setup corporate proxy
            self._setup_corporate_proxy()
            
            # Initialize tiktoken
            self.encoding = tiktoken.encoding_for_model(self.analysis_config.tiktoken_model)
            logger.info(f"tiktoken initialized successfully with model: {self.analysis_config.tiktoken_model}")
            return True
            
        except Exception as e:
            logger.error(f"tiktoken initialization failed: {e}")
            return False
    
    def _setup_corporate_proxy(self):
        """Setup corporate proxy for HTTP requests."""
        try:
            if not all([PROXY_USER, PROXY_PASSWORD, PROXY_URL]):
                logger.warning("Proxy configuration incomplete - continuing without proxy")
                return None
            
            user = PROXY_USER
            password = quote(PROXY_PASSWORD)
            escaped_domain = quote(PROXY_DOMAIN + '\\' + user)
            full_proxy_url = f"http://{escaped_domain}:{password}@{PROXY_URL}"
            
            # Set proxy environment variables
            os.environ["HTTP_PROXY"] = full_proxy_url
            os.environ["HTTPS_PROXY"] = full_proxy_url
            
            logger.info("Corporate proxy setup complete")
            
        except Exception as e:
            logger.error(f"Corporate proxy setup failed: {e}")
            return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        if not self.encoding:
            # Fallback to character count / 4 (rough approximation)
            return len(text) // 4
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            logger.warning(f"Token counting failed, using character approximation: {e}")
            return len(text) // 4
    
    def analyze_truncation_impact(self, text: str) -> Dict[str, Any]:
        """Analyze the impact of character truncation on token counts."""
        if not text:
            return {"original_tokens": 0, "truncated_tokens": 0, "truncation_ratio": 0.0}
        
        original_tokens = self.count_tokens(text)
        truncated_text = text[:self.analysis_config.max_token_length]
        truncated_tokens = self.count_tokens(truncated_text)
        
        truncation_ratio = truncated_tokens / original_tokens if original_tokens > 0 else 1.0
        
        return {
            "original_tokens": original_tokens,
            "truncated_tokens": truncated_tokens,
            "truncation_ratio": truncation_ratio,
            "was_truncated": len(text) > self.analysis_config.max_token_length
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.ssl_cert_file and os.path.exists(self.ssl_cert_file):
            os.unlink(self.ssl_cert_file)

class Stage4Analyzer:
    """Main analyzer for Stage 4 outputs."""
    
    def __init__(self, analysis_config: AnalysisConfig):
        self.analysis_config = analysis_config
        self.token_analyzer = TokenAnalyzer(analysis_config)
        self.data = None
        self.df = None
        self.nas_conn = None
        
    def load_data(self) -> bool:
        """Load Stage 4 output data from NAS."""
        global logger, config
        try:
            # Connect to NAS
            self.nas_conn = get_nas_connection()
            if not self.nas_conn:
                logger.error("Failed to connect to NAS")
                return False
            
            # Load config from NAS
            config = load_stage_config(self.nas_conn)
            
            # Construct file path using config
            file_path = nas_path_join(
                NAS_BASE_PATH,
                config["stage_4"]["output_path"],
                config["stage_4"]["output_file"]
            )
            
            # Download Stage 4 output file
            file_data = nas_download_file(self.nas_conn, file_path)
            
            if not file_data:
                logger.error(f"Failed to download Stage 4 output file: {file_path}")
                return False
            
            # Parse JSON data
            self.data = json.loads(file_data.decode('utf-8'))
            
            # Convert to DataFrame for analysis
            self.df = pd.DataFrame(self.data['records'])
            
            logger.info(f"Loaded {len(self.df)} records from Stage 4 output")
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return False
    
    def initialize_token_analysis(self) -> bool:
        """Initialize token analysis with tiktoken."""
        if not self.nas_conn:
            logger.error("NAS connection not established")
            return False
        return self.token_analyzer.initialize_tiktoken(self.nas_conn)
    
    def analyze_token_distributions(self) -> Dict[str, Any]:
        """Analyze token distributions across transcripts."""
        if self.df is None:
            return {}
        
        logger.info("Analyzing token distributions...")
        
        # Add token counts to DataFrame
        self.df['token_count'] = self.df['paragraph_content'].apply(
            lambda x: self.token_analyzer.count_tokens(x) if pd.notna(x) else 0
        )
        
        # Add truncation analysis
        truncation_analysis = self.df['paragraph_content'].apply(
            lambda x: self.token_analyzer.analyze_truncation_impact(x) if pd.notna(x) else {}
        )
        
        self.df['original_tokens'] = truncation_analysis.apply(
            lambda x: x.get('original_tokens', 0)
        )
        self.df['truncated_tokens'] = truncation_analysis.apply(
            lambda x: x.get('truncated_tokens', 0)
        )
        self.df['was_truncated'] = truncation_analysis.apply(
            lambda x: x.get('was_truncated', False)
        )
        
        # Calculate transcript-level statistics
        transcript_stats = self.df.groupby('filename').agg({
            'token_count': ['sum', 'count', 'mean', 'std'],
            'original_tokens': 'sum',
            'was_truncated': 'sum'
        }).round(2)
        
        # Overall statistics
        total_tokens = self.df['token_count'].sum()
        total_original_tokens = self.df['original_tokens'].sum()
        truncation_rate = self.df['was_truncated'].mean()
        
        return {
            'total_tokens': int(total_tokens),
            'total_original_tokens': int(total_original_tokens),
            'truncation_rate': float(truncation_rate),
            'transcript_stats': transcript_stats,
            'paragraph_token_distribution': self.df['token_count'].describe(),
            'tokens_lost_to_truncation': int(total_original_tokens - total_tokens)
        }
    
    def analyze_section_classification(self) -> Dict[str, Any]:
        """Analyze section classification success and patterns."""
        if self.df is None:
            return {}
        
        logger.info("Analyzing section classification...")
        
        # Overall classification distribution
        section_distribution = self.df['section_type'].value_counts()
        
        # Confidence score analysis
        confidence_stats = self.df.groupby('section_type')['section_type_confidence'].describe()
        
        # Classification method analysis
        method_distribution = self.df['section_type_method'].value_counts()
        
        # Success rate by institution type
        institution_success = self.df.groupby('institution_type').agg({
            'section_type_confidence': ['mean', 'count'],
            'section_type': lambda x: x.value_counts().to_dict()
        })
        
        # High confidence classifications (>0.8)
        high_confidence_rate = (self.df['section_type_confidence'] > 0.8).mean()
        
        return {
            'section_distribution': section_distribution.to_dict(),
            'confidence_stats': confidence_stats.to_dict(),
            'method_distribution': method_distribution.to_dict(),
            'institution_success': institution_success.to_dict(),
            'high_confidence_rate': float(high_confidence_rate),
            'total_classifications': len(self.df)
        }
    
    def analyze_speaker_blocks(self) -> Dict[str, Any]:
        """Analyze speaker block patterns and distributions."""
        if self.df is None:
            return {}
        
        logger.info("Analyzing speaker block patterns...")
        
        # Speaker blocks per transcript
        speaker_blocks_per_transcript = self.df.groupby('filename')['speaker_block_id'].nunique()
        
        # Speaker blocks by section type
        speaker_blocks_by_section = self.df.groupby(['section_type', 'filename'])['speaker_block_id'].nunique()
        
        # Speaker role analysis (extract role from speaker field)
        self.df['speaker_role'] = self.df['speaker'].apply(self._extract_speaker_role)
        speaker_role_distribution = self.df['speaker_role'].value_counts()
        
        # Paragraph count per speaker block
        paragraphs_per_speaker_block = self.df.groupby(['filename', 'speaker_block_id']).size()
        
        # Speaker interaction patterns
        speaker_transitions = self._analyze_speaker_transitions()
        
        return {
            'speaker_blocks_per_transcript': speaker_blocks_per_transcript.describe().to_dict(),
            'speaker_blocks_by_section': speaker_blocks_by_section.describe().to_dict(),
            'speaker_role_distribution': speaker_role_distribution.to_dict(),
            'paragraphs_per_speaker_block': paragraphs_per_speaker_block.describe().to_dict(),
            'speaker_transitions': speaker_transitions
        }
    
    def _extract_speaker_role(self, speaker: str) -> str:
        """Extract speaker role from speaker field."""
        if pd.isna(speaker):
            return 'Unknown'
        
        speaker_lower = speaker.lower()
        if 'ceo' in speaker_lower or 'chief executive' in speaker_lower:
            return 'CEO'
        elif 'cfo' in speaker_lower or 'chief financial' in speaker_lower:
            return 'CFO'
        elif 'analyst' in speaker_lower:
            return 'Analyst'
        elif 'operator' in speaker_lower:
            return 'Operator'
        elif any(title in speaker_lower for title in ['president', 'chairman', 'chief', 'head']):
            return 'Executive'
        else:
            return 'Other'
    
    def _analyze_speaker_transitions(self) -> Dict[str, Any]:
        """Analyze speaker transition patterns."""
        transitions = defaultdict(int)
        
        for filename in self.df['filename'].unique():
            transcript_df = self.df[self.df['filename'] == filename].sort_values('paragraph_id')
            
            prev_role = None
            for role in transcript_df['speaker_role']:
                if prev_role and role != prev_role:
                    transitions[f"{prev_role} -> {role}"] += 1
                prev_role = role
        
        return dict(transitions)
    
    def generate_visualizations(self) -> Dict[str, str]:
        """Generate comprehensive visualizations."""
        if self.df is None:
            logger.error("No data loaded for visualization")
            return {}
        
        logger.info("Generating visualizations...")
        
        # Create output directory
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        visualization_files = {}
        
        # 1. Token Distribution Analysis
        visualization_files['token_distributions'] = self._create_token_distribution_charts(output_dir)
        
        # 2. Section Classification Analysis
        visualization_files['section_classification'] = self._create_section_classification_charts(output_dir)
        
        # 3. Speaker Block Analysis
        visualization_files['speaker_blocks'] = self._create_speaker_block_charts(output_dir)
        
        # 4. Chunking Optimization Analysis
        visualization_files['chunking_analysis'] = self._create_chunking_analysis_charts(output_dir)
        
        # 5. Overall Summary Dashboard
        visualization_files['summary_dashboard'] = self._create_summary_dashboard(output_dir)
        
        return visualization_files
    
    def _create_token_distribution_charts(self, output_dir: Path) -> str:
        """Create token distribution visualization charts."""
        
        # Create subplot figure with multiple panels
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Token Count Distribution',
                'Tokens by Institution Type',
                'Tokens by Section Type',
                'Truncation Impact Analysis'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Token count histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['token_count'],
                name='Token Count',
                nbinsx=50,
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. Tokens by institution type (box plot)
        for i, inst_type in enumerate(self.df['institution_type'].unique()):
            data = self.df[self.df['institution_type'] == inst_type]['token_count']
            fig.add_trace(
                go.Box(
                    y=data,
                    name=inst_type,
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # 3. Tokens by section type
        for section_type in self.df['section_type'].unique():
            data = self.df[self.df['section_type'] == section_type]['token_count']
            fig.add_trace(
                go.Box(
                    y=data,
                    name=section_type,
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        # 4. Truncation impact
        truncation_data = self.df.groupby('was_truncated')['token_count'].sum()
        fig.add_trace(
            go.Bar(
                x=['Not Truncated', 'Truncated'],
                y=truncation_data.values,
                name='Token Impact',
                marker_color=['green', 'red']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Token Distribution Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        output_file = output_dir / "token_distributions.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def _create_section_classification_charts(self, output_dir: Path) -> str:
        """Create section classification analysis charts."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Section Type Distribution',
                'Confidence Score Distribution',
                'Classification Method Usage',
                'Success Rate by Institution'
            ),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "bar"}]]
        )
        
        # 1. Section type pie chart
        section_counts = self.df['section_type'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=section_counts.index,
                values=section_counts.values,
                name="Section Types"
            ),
            row=1, col=1
        )
        
        # 2. Confidence score distribution
        for section_type in self.df['section_type'].unique():
            data = self.df[self.df['section_type'] == section_type]['section_type_confidence']
            fig.add_trace(
                go.Histogram(
                    x=data,
                    name=f'{section_type} Confidence',
                    opacity=0.7,
                    nbinsx=20
                ),
                row=1, col=2
            )
        
        # 3. Classification method usage
        method_counts = self.df['section_type_method'].value_counts()
        fig.add_trace(
            go.Bar(
                x=method_counts.index,
                y=method_counts.values,
                name="Methods Used"
            ),
            row=2, col=1
        )
        
        # 4. Success rate by institution
        inst_confidence = self.df.groupby('institution_type')['section_type_confidence'].mean()
        fig.add_trace(
            go.Bar(
                x=inst_confidence.index,
                y=inst_confidence.values,
                name="Avg Confidence"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Section Classification Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        output_file = output_dir / "section_classification.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def _create_speaker_block_charts(self, output_dir: Path) -> str:
        """Create speaker block analysis charts."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Speaker Role Distribution',
                'Speaker Blocks per Transcript',
                'Paragraphs per Speaker Block',
                'Speaker Transitions'
            )
        )
        
        # 1. Speaker role distribution
        role_counts = self.df['speaker_role'].value_counts()
        fig.add_trace(
            go.Bar(
                x=role_counts.index,
                y=role_counts.values,
                name="Speaker Roles"
            ),
            row=1, col=1
        )
        
        # 2. Speaker blocks per transcript
        blocks_per_transcript = self.df.groupby('filename')['speaker_block_id'].nunique()
        fig.add_trace(
            go.Histogram(
                x=blocks_per_transcript,
                name='Blocks per Transcript',
                nbinsx=30
            ),
            row=1, col=2
        )
        
        # 3. Paragraphs per speaker block
        paragraphs_per_block = self.df.groupby(['filename', 'speaker_block_id']).size()
        fig.add_trace(
            go.Histogram(
                x=paragraphs_per_block,
                name='Paragraphs per Block',
                nbinsx=20
            ),
            row=2, col=1
        )
        
        # 4. Top speaker transitions
        transitions = self._analyze_speaker_transitions()
        top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_transitions:
            fig.add_trace(
                go.Bar(
                    x=[t[1] for t in top_transitions],
                    y=[t[0] for t in top_transitions],
                    orientation='h',
                    name="Top Transitions"
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Speaker Block Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        output_file = output_dir / "speaker_blocks.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def _create_chunking_analysis_charts(self, output_dir: Path) -> str:
        """Create chunking optimization analysis charts."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Character Length Distribution',
                'Truncation Frequency',
                'Token Density by Speaker Role',
                'Optimal Chunking Analysis'
            )
        )
        
        # 1. Character length distribution
        char_lengths = self.df['paragraph_content'].apply(len)
        fig.add_trace(
            go.Histogram(
                x=char_lengths,
                name='Character Length',
                nbinsx=50
            ),
            row=1, col=1
        )
        
        # Add vertical line at 750 (truncation limit)
        fig.add_vline(
            x=750, line_dash="dash", line_color="red",
            annotation_text="Truncation Limit",
            row=1, col=1
        )
        
        # 2. Truncation frequency
        truncation_by_role = self.df.groupby('speaker_role')['was_truncated'].mean()
        fig.add_trace(
            go.Bar(
                x=truncation_by_role.index,
                y=truncation_by_role.values,
                name="Truncation Rate"
            ),
            row=1, col=2
        )
        
        # 3. Token density by speaker role
        token_density = self.df.groupby('speaker_role')['token_count'].mean()
        fig.add_trace(
            go.Bar(
                x=token_density.index,
                y=token_density.values,
                name="Avg Tokens"
            ),
            row=2, col=1
        )
        
        # 4. Cumulative distribution for chunking
        sorted_lengths = np.sort(char_lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        
        fig.add_trace(
            go.Scatter(
                x=sorted_lengths,
                y=cumulative,
                mode='lines',
                name="Cumulative Distribution"
            ),
            row=2, col=2
        )
        
        # Add chunking reference lines
        for chunk_size in [500, 750, 1000, 1500]:
            percentage = (char_lengths <= chunk_size).mean() * 100
            fig.add_vline(
                x=chunk_size, line_dash="dot",
                annotation_text=f"{chunk_size}ch ({percentage:.1f}%)",
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Chunking Optimization Analysis",
            showlegend=True,
            height=800,
            width=1200
        )
        
        output_file = output_dir / "chunking_analysis.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def _create_summary_dashboard(self, output_dir: Path) -> str:
        """Create overall summary dashboard."""
        
        # Calculate key metrics
        total_transcripts = self.df['filename'].nunique()
        total_paragraphs = len(self.df)
        avg_confidence = self.df['section_type_confidence'].mean()
        truncation_rate = self.df['was_truncated'].mean()
        
        # Create summary metrics
        fig = go.Figure()
        
        # Add gauge charts for key metrics
        fig.add_trace(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_confidence,
            domain = {'x': [0, 0.5], 'y': [0.5, 1]},
            title = {'text': "Average Classification Confidence"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "gray"},
                    {'range': [0.8, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.85
                }
            }
        ))
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = truncation_rate * 100,
            domain = {'x': [0.5, 1], 'y': [0.5, 1]},
            title = {'text': "Truncation Rate (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 50], 'color': "yellow"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ]
            }
        ))
        
        # Add text summary
        summary_text = f"""
        <b>Processing Summary:</b><br>
        • Total Transcripts: {total_transcripts:,}<br>
        • Total Paragraphs: {total_paragraphs:,}<br>
        • Average Confidence: {avg_confidence:.3f}<br>
        • Truncation Rate: {truncation_rate:.1%}<br>
        • High Confidence Rate: {(self.df['section_type_confidence'] > 0.8).mean():.1%}
        """
        
        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.25, y=0.4,
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title_text="Stage 4 Analysis Summary Dashboard",
            height=600,
            width=1200
        )
        
        output_file = output_dir / "summary_dashboard.html"
        fig.write_html(str(output_file))
        
        return str(output_file)
    
    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        
        # Analyze all aspects
        token_analysis = self.analyze_token_distributions()
        section_analysis = self.analyze_section_classification()
        speaker_analysis = self.analyze_speaker_blocks()
        
        # Generate visualizations
        viz_files = self.generate_visualizations()
        
        # Create report
        report = f"""
# FactSet Stage 4 Analysis Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Transcripts**: {self.df['filename'].nunique():,}
- **Total Paragraphs**: {len(self.df):,}
- **Total Tokens**: {token_analysis.get('total_tokens', 0):,}
- **Average Classification Confidence**: {section_analysis.get('high_confidence_rate', 0):.1%}

## Token Analysis
- **Tokens Lost to Truncation**: {token_analysis.get('tokens_lost_to_truncation', 0):,}
- **Truncation Rate**: {token_analysis.get('truncation_rate', 0):.1%}
- **Token Efficiency**: {(token_analysis.get('total_tokens', 0) / token_analysis.get('total_original_tokens', 1)):.1%}

## Section Classification
- **Management Discussion**: {section_analysis.get('section_distribution', {}).get('Management Discussion', 0):,}
- **Investor Q&A**: {section_analysis.get('section_distribution', {}).get('Investor Q&A', 0):,}
- **High Confidence Rate**: {section_analysis.get('high_confidence_rate', 0):.1%}

## Speaker Analysis
- **Unique Speaker Roles**: {len(speaker_analysis.get('speaker_role_distribution', {})):,}
- **Average Paragraphs per Speaker Block**: {speaker_analysis.get('paragraphs_per_speaker_block', {}).get('mean', 0):.1f}

## Visualization Files Generated
"""
        
        for viz_name, viz_file in viz_files.items():
            report += f"- **{viz_name.replace('_', ' ').title()}**: {viz_file}\n"
        
        report += f"""
## Recommendations
1. **Chunking Strategy**: {'Paragraph-level chunking appears optimal' if token_analysis.get('truncation_rate', 0) < 0.1 else 'Consider larger chunk sizes or sub-paragraph splitting'}
2. **Classification Quality**: {'High confidence classifications indicate good LLM performance' if section_analysis.get('high_confidence_rate', 0) > 0.8 else 'Consider refining classification prompts or methods'}
3. **Token Efficiency**: {'Current truncation settings are appropriate' if token_analysis.get('truncation_rate', 0) < 0.2 else 'Consider increasing character limits or improving truncation logic'}

---
*Generated by FactSet Stage 4 Analysis Visualizer*
"""
        
        # Save report
        report_file = Path(self.config.output_directory) / "analysis_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return str(report_file)
    
    def cleanup(self):
        """Clean up resources."""
        if self.token_analyzer:
            self.token_analyzer.cleanup()
        if self.nas_conn:
            self.nas_conn.close()

def main():
    """Main execution function."""
    global logger
    
    # Setup logging
    logger = setup_logging()
    
    # Configuration
    analysis_config = AnalysisConfig()
    
    # Initialize analyzer
    analyzer = Stage4Analyzer(analysis_config)
    
    try:
        # Load data
        logger.info("Loading Stage 4 data...")
        if not analyzer.load_data():
            logger.error("Failed to load Stage 4 data")
            return
        
        # Initialize token analysis
        logger.info("Initializing token analysis...")
        if not analyzer.initialize_token_analysis():
            logger.warning("Token analysis initialization failed - using fallback methods")
        
        # Generate comprehensive report
        logger.info("Generating comprehensive analysis report...")
        report_file = analyzer.generate_report()
        
        logger.info(f"Analysis complete! Report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("FACTSET STAGE 4 ANALYSIS COMPLETE")
        print("="*60)
        print(f"Report Location: {report_file}")
        print(f"Output Directory: {analysis_config.output_directory}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
    
    finally:
        # Cleanup
        analyzer.cleanup()

if __name__ == "__main__":
    main()