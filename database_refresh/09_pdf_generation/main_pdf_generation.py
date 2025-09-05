"""
Stage 9: PDF Generation
Processes Stage 8 embeddings content to create formatted PDF documents for each transcript.
Self-contained standalone script that loads config from NAS at runtime.

Architecture based on Stage 8 pattern with PDF generation logic.
Creates professional PDFs with smart paragraph placement and section formatting.
"""

import os
import tempfile
import logging
import json
import time
from datetime import datetime
from urllib.parse import quote
from typing import Dict, Any, Optional, List, Tuple
import io
import re
import csv
from collections import defaultdict, OrderedDict

import yaml
from smb.SMBConnection import SMBConnection
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Flowable
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import PageTemplate, Frame, BaseDocTemplate, NextPageTemplate, FrameBreak
from reportlab.platypus.paragraph import Paragraph as PlatypusParagraph
from reportlab.pdfgen import canvas
from reportlab.platypus.flowables import KeepTogether
from reportlab.platypus.doctemplate import ActionFlowable

# Load environment variables
load_dotenv()

# Global variables
config = {}
logger = None
execution_log = []  # Detailed execution log entries
error_log = []  # Error log entries (only if errors occur)


def setup_logging() -> logging.Logger:
    """Set up minimal console logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
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
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "details": details or {},
    }
    execution_log.append(log_entry)


def log_error(message: str, error_type: str, details: Dict[str, Any] = None):
    """Log error information for error log file."""
    global error_log
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "error_type": error_type,
        "message": message,
        "details": details or {},
    }
    error_log.append(error_entry)


class EnhancedErrorLogger:
    """Handles separate error logging for different failure types."""
    
    def __init__(self):
        self.pdf_errors = []
        self.formatting_errors = []
        self.processing_errors = []
        self.validation_errors = []
        self.total_pdfs = 0
        self.total_transcripts = 0
    
    def log_pdf_error(self, ticker: str, error: str, details: Dict = None):
        """Log PDF generation error."""
        self.pdf_errors.append({
            "ticker": ticker,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
        log_error(f"PDF generation error for {ticker}: {error}", "pdf_generation", details)
    
    def log_formatting_error(self, ticker: str, error: str, details: Dict = None):
        """Log formatting error."""
        self.formatting_errors.append({
            "ticker": ticker,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
        log_error(f"Formatting error for {ticker}: {error}", "formatting", details)
    
    def log_processing_error(self, ticker: str, error: str, details: Dict = None):
        """Log processing error."""
        self.processing_errors.append({
            "ticker": ticker,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
        log_error(f"Processing error for {ticker}: {error}", "processing", details)
    
    def log_validation_error(self, ticker: str, error: str, details: Dict = None):
        """Log validation error."""
        self.validation_errors.append({
            "ticker": ticker,
            "error": error,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
        log_error(f"Validation error for {ticker}: {error}", "validation", details)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "total_pdfs": self.total_pdfs,
            "total_transcripts": self.total_transcripts,
            "pdf_errors": len(self.pdf_errors),
            "formatting_errors": len(self.formatting_errors),
            "processing_errors": len(self.processing_errors),
            "validation_errors": len(self.validation_errors),
            "total_errors": len(self.pdf_errors) + len(self.formatting_errors) + 
                          len(self.processing_errors) + len(self.validation_errors)
        }


# NAS Connection Functions
def validate_nas_path(path: str) -> bool:
    """Validate NAS path to prevent directory traversal."""
    if not path:
        return False
    
    # Check for directory traversal attempts
    if ".." in path or path.startswith("/") or path.startswith("\\"):
        return False
    
    # Check for invalid characters
    invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\x00"]
    if any(char in path for char in invalid_chars):
        return False
    
    return True


def nas_create_connection() -> Optional[SMBConnection]:
    """Create and return SMB connection to NAS."""
    try:
        log_console("Connecting to NAS...")
        
        # Get credentials from environment (matching Stage 8 pattern)
        nas_username = os.getenv("NAS_USERNAME")
        nas_password = os.getenv("NAS_PASSWORD")
        nas_server_name = os.getenv("NAS_SERVER_NAME")
        nas_server_ip = os.getenv("NAS_SERVER_IP")
        client_machine_name = os.getenv("CLIENT_MACHINE_NAME", "python_script")
        
        if not nas_username or not nas_password:
            log_error("NAS credentials not found in environment", "authentication")
            return None
        
        # Create SMB connection (matching Stage 8 exactly)
        conn = SMBConnection(
            username=nas_username,
            password=nas_password,
            my_name=client_machine_name,
            remote_name=nas_server_name,
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        # Connect to NAS
        nas_port = int(os.getenv("NAS_PORT", 445))
        if conn.connect(nas_server_ip, nas_port):
            log_console("Connected to NAS successfully")
            log_execution("NAS connection established successfully", {
                "connection_type": "SMB/CIFS",
                "server": nas_server_name,
                "port": nas_port
            })
            return conn
        else:
            log_error("Failed to connect to NAS", "connection", {
                "server": nas_server_name,
                "port": nas_port
            })
            return None
        
    except Exception as e:
        log_error(f"Error connecting to NAS: {str(e)}", "connection", {"error": str(e)})
        return None


def nas_download_file(conn: SMBConnection, nas_file_path: str) -> Optional[bytes]:
    """Download a file from NAS and return as bytes."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, temp_file)
            temp_file.seek(0)
            content = temp_file.read()
            os.unlink(temp_file.name)
            return content
    except Exception as e:
        log_error(f"Failed to download file from NAS: {nas_file_path}", "nas_download", {"error": str(e)})
        return None


def nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    """Create directory recursively on NAS."""
    try:
        path_parts = dir_path.split("/")
        current_path = ""
        
        for part in path_parts:
            if not part:
                continue
                
            current_path = f"{current_path}/{part}" if current_path else part
            
            try:
                conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
            except Exception:
                # Directory might already exist, continue
                pass
        
        return True
    except Exception as e:
        log_error(f"Failed to create directory: {dir_path}", "nas_directory", {"error": str(e)})
        return False


def nas_upload_file(conn: SMBConnection, local_file_path: str, nas_file_path: str) -> bool:
    """Upload a file to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return False
    
    try:
        # Create directory if needed
        dir_path = "/".join(nas_file_path.split("/")[:-1])
        if dir_path:
            nas_create_directory_recursive(conn, dir_path)
        
        # Upload file
        with open(local_file_path, 'rb') as local_file:
            conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, local_file)
        
        return True
    except Exception as e:
        log_error(f"Failed to upload file to NAS: {nas_file_path}", "nas_upload", {"error": str(e)})
        return False


def nas_upload_content(conn: SMBConnection, content: bytes, nas_file_path: str) -> bool:
    """Upload content directly to NAS."""
    if not validate_nas_path(nas_file_path):
        log_error(f"Invalid NAS path: {nas_file_path}", "path_validation")
        return False
    
    try:
        # Create directory if needed
        dir_path = "/".join(nas_file_path.split("/")[:-1])
        if dir_path:
            nas_create_directory_recursive(conn, dir_path)
        
        # Upload content
        file_obj = io.BytesIO(content)
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), nas_file_path, file_obj)
        
        return True
    except Exception as e:
        log_error(f"Failed to upload content to NAS: {nas_file_path}", "nas_upload", {"error": str(e)})
        return False


def nas_load_config(conn: SMBConnection) -> Dict[str, Any]:
    """Load configuration from NAS."""
    global config
    
    # Use CONFIG_PATH environment variable like Stage 8
    config_path = os.getenv("CONFIG_PATH", "Finance Data and Analytics/DSA/Earnings Call Transcripts/config.yaml")
    log_console("Loading configuration from NAS...")
    
    try:
        config_content = nas_download_file(conn, config_path)
        if not config_content:
            raise Exception("Failed to download config file")
        
        config_data = yaml.safe_load(config_content.decode('utf-8'))
        
        # Extract Stage 9 specific config
        stage_config = config_data.get("stage_09_pdf_generation", {})
        
        # Store the full config (following Stage 8 pattern)
        config.update(config_data)
        
        log_console("Configuration loaded successfully")
        log_execution("Configuration loaded", {
            "config_path": config_path,
            "stage_config": stage_config
        })
        
        return config
        
    except Exception as e:
        log_error(f"Failed to load configuration: {str(e)}", "config_load", {"error": str(e)})
        raise


# PDF Generation Classes
class TranscriptPDF:
    """Handles PDF generation for earnings transcripts with smart layout."""
    
    def __init__(self, output_path: str, transcript_data: Dict[str, Any]):
        """Initialize PDF generator with transcript data."""
        self.output_path = output_path
        self.transcript_data = transcript_data
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Extract metadata from first record
        first_record = transcript_data['records'][0]
        self.title = first_record.get('title', 'Earnings Call Transcript')
        self.ticker = first_record.get('ticker', '')
        self.fiscal_year = first_record.get('fiscal_year', '')
        self.fiscal_quarter = first_record.get('fiscal_quarter', '')
        self.bank = first_record.get('bank', self.ticker)  # Use ticker as fallback
        
        # Page tracking
        self.current_section = "COVER PAGE"
        self.page_num = 1  # Start at 1, not 0
        self.current_speaker = None
        self.current_qa_conversation = None
        self.continuation_info = {}  # Track what needs continuation on each page
        
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#4a4a4a'),
            spaceBefore=4,
            spaceAfter=4,
            alignment=TA_CENTER
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=12,
            spaceAfter=12,
            keepWithNext=True,
            alignment=TA_LEFT
        ))
        
        # Q&A block header
        self.styles.add(ParagraphStyle(
            name='QAHeader',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=8,
            spaceAfter=6,
            keepWithNext=True,
            leftIndent=0
        ))
        
        # Speaker style
        self.styles.add(ParagraphStyle(
            name='Speaker',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold',
            spaceBefore=6,
            spaceAfter=4,
            keepWithNext=True,
            leftIndent=20
        ))
        
        # Paragraph content style
        self.styles.add(ParagraphStyle(
            name='Content',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#333333'),
            spaceBefore=2,
            spaceAfter=2,
            alignment=TA_JUSTIFY,
            leftIndent=40,
            rightIndent=20,
            leading=12
        ))
        
        # Header/Footer style
        self.styles.add(ParagraphStyle(
            name='HeaderFooter',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER
        ))
    
    def _create_header_footer(self, canvas_obj, doc):
        """Create headers and footers for each page."""
        canvas_obj.saveState()
        
        # Check if this page needs continuation markers
        page_info = self.continuation_info.get(self.page_num, {})
        
        # Add continuation info if this is not the first page of a section
        if self.page_num > 1 and page_info:
            y_position = letter[1] - 0.8*inch
            
            # Add Q&A conversation continuation if applicable
            if 'qa_id' in page_info and self.current_section == "Q&A":
                canvas_obj.setFont("Helvetica-Bold", 10)
                canvas_obj.setFillColor(colors.HexColor('#34495e'))
                canvas_obj.drawString(inch + 0.2*inch, y_position, 
                                    f"Q&A Conversation {page_info['qa_id']} (Continued...)")
                y_position -= 0.2*inch
            
            # Add speaker continuation
            if 'speaker' in page_info:
                canvas_obj.setFont("Helvetica-Bold", 9)
                canvas_obj.setFillColor(colors.HexColor('#2c3e50'))
                canvas_obj.drawString(inch + 0.4*inch, y_position,
                                    f"{page_info['speaker']} (Continued...)")
        
        # Header
        header_text = f"{self.title[:50]}... | {self.current_section}"
        canvas_obj.setFont("Helvetica", 9)
        canvas_obj.setFillColor(colors.HexColor('#666666'))
        canvas_obj.drawString(inch, letter[1] - 0.5*inch, header_text)
        canvas_obj.drawRightString(letter[0] - inch, letter[1] - 0.5*inch, f"Page {self.page_num}")
        
        # Header line
        canvas_obj.setStrokeColor(colors.HexColor('#cccccc'))
        canvas_obj.line(inch, letter[1] - 0.6*inch, letter[0] - inch, letter[1] - 0.6*inch)
        
        # Footer - fix quarter formatting
        quarter_text = f"Q{self.fiscal_quarter}" if self.fiscal_quarter else ""
        footer_text = f"{self.bank} | FY{self.fiscal_year} {quarter_text}"
        canvas_obj.drawString(inch, 0.5*inch, footer_text)
        canvas_obj.drawRightString(letter[0] - inch, 0.5*inch, f"Page {self.page_num}")
        
        # Footer line
        canvas_obj.line(inch, 0.6*inch, letter[0] - inch, 0.6*inch)
        
        canvas_obj.restoreState()
    
    def _create_title_page(self) -> List:
        """Create the title page elements."""
        elements = []
        
        # Main title
        elements.append(Paragraph(self.title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Metadata
        elements.append(Paragraph(f"<b>{self.bank}</b>", self.styles['Subtitle']))
        elements.append(Paragraph(f"Ticker: {self.ticker}", self.styles['Subtitle']))
        elements.append(Paragraph(f"Fiscal Year {self.fiscal_year}, Quarter {self.fiscal_quarter}", 
                                self.styles['Subtitle']))
        
        # No page break here - let content flow naturally
        elements.append(Spacer(1, 0.3*inch))
        return elements
    
    def _group_content_by_structure(self) -> Dict[str, Any]:
        """Group content by sections, speakers, and Q&A blocks."""
        grouped = {
            'Management Discussion': OrderedDict(),
            'Q&A': OrderedDict()
        }
        
        for record in self.transcript_data['records']:
            section = record.get('section_name', '')
            
            # Match exact section names from Stage 7
            if section == 'MANAGEMENT DISCUSSION SECTION':
                # Group by speaker_block_id to keep blocks together
                block_id = record.get('speaker_block_id', 0)
                if block_id not in grouped['Management Discussion']:
                    grouped['Management Discussion'][block_id] = []
                grouped['Management Discussion'][block_id].append(record)
            
            elif section == 'Q&A':
                # Use qa_group_id from Stage 5/7
                qa_group_id = record.get('qa_group_id', 0)
                if qa_group_id not in grouped['Q&A']:
                    grouped['Q&A'][qa_group_id] = OrderedDict()
                
                # Within each Q&A group, organize by speaker block
                speaker_block_id = record.get('speaker_block_id', 0)
                if speaker_block_id not in grouped['Q&A'][qa_group_id]:
                    grouped['Q&A'][qa_group_id][speaker_block_id] = []
                grouped['Q&A'][qa_group_id][speaker_block_id].append(record)
        
        return grouped
    
    def _create_speaker_block(self, block_id: int, paragraphs: List[Dict], qa_group_id: int = None) -> List:
        """Create a speaker block with paragraphs."""
        elements = []
        
        # Get actual speaker name from first record
        if paragraphs:
            speaker = paragraphs[0].get('speaker', f'Speaker {block_id}')
        else:
            return elements
        
        # Create a marker flowable to track current speaker/QA
        class SpeakerMarker(ActionFlowable):
            def __init__(self, pdf_doc, speaker_name, qa_id=None):
                ActionFlowable.__init__(self)
                self.pdf_doc = pdf_doc
                self.speaker_name = speaker_name
                self.qa_id = qa_id
                
            def apply(self, doc):
                # Store current speaker and QA info for page continuations
                page_num = doc.page
                if page_num not in self.pdf_doc.continuation_info:
                    self.pdf_doc.continuation_info[page_num] = {}
                self.pdf_doc.continuation_info[page_num]['speaker'] = self.speaker_name
                if self.qa_id is not None:
                    self.pdf_doc.continuation_info[page_num]['qa_id'] = self.qa_id
        
        # Add marker to track this speaker block
        elements.append(SpeakerMarker(self, speaker, qa_group_id))
        
        # Speaker name
        speaker_para = Paragraph(speaker, self.styles['Speaker'])
        
        # Collect all paragraph content for this speaker block
        content_paras = []
        for para_data in paragraphs:
            # Use paragraph_content from Stage 7
            content = para_data.get('paragraph_content', '').strip()
            if content:
                # Clean and escape content for reportlab
                content = content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Each paragraph should be kept together (not split)
                content_paras.append(KeepTogether([Paragraph(content, self.styles['Content'])]))
        
        # Keep speaker with first paragraph
        if content_paras:
            # Keep speaker name with at least first paragraph
            elements.append(KeepTogether([speaker_para, content_paras[0]]))
            # Rest of paragraphs can flow but each kept whole
            elements.extend(content_paras[1:])
        else:
            elements.append(speaker_para)
        
        return elements
    
    def _create_md_section(self, content: Dict[int, List]) -> List:
        """Create Management Discussion section."""
        elements = []
        
        # Create custom flowable to update section
        class SectionUpdater(Flowable):
            def __init__(self, pdf_doc, section_name):
                Flowable.__init__(self)
                self.pdf_doc = pdf_doc
                self.section_name = section_name
                self.width = 0
                self.height = 0
                
            def draw(self):
                self.pdf_doc.current_section = self.section_name
        
        elements.append(SectionUpdater(self, "MANAGEMENT DISCUSSION SECTION"))
        elements.append(Paragraph("Management Discussion", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Process each speaker block in order (sorted by block_id)
        for block_id in sorted(content.keys()):
            paragraphs = content[block_id]
            if paragraphs:
                speaker_elements = self._create_speaker_block(block_id, paragraphs)
                elements.extend(speaker_elements)
                elements.append(Spacer(1, 0.05*inch))
        
        # Only page break here - between MD and Q&A sections
        elements.append(PageBreak())
        return elements
    
    def _create_qa_section(self, content: Dict[int, Dict[int, List]]) -> List:
        """Create Q&A section."""
        elements = []
        
        # No page break here - already added at end of MD section
        
        # Create custom flowable to update section
        class SectionUpdater(Flowable):
            def __init__(self, pdf_doc, section_name):
                Flowable.__init__(self)
                self.pdf_doc = pdf_doc
                self.section_name = section_name
                self.width = 0
                self.height = 0
                
            def draw(self):
                self.pdf_doc.current_section = self.section_name
        
        elements.append(SectionUpdater(self, "Q&A"))
        elements.append(Paragraph("Questions & Answers", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Process each Q&A group in order
        for qa_idx, qa_group_id in enumerate(sorted(content.keys())):
            qa_group = content[qa_group_id]
            
            # Q&A group header (conversation number)
            qa_header_text = f"Q&A Conversation {qa_group_id}"
            qa_header = Paragraph(qa_header_text, self.styles['QAHeader'])
            
            # Keep Q&A header with first content
            qa_elements = [qa_header]
            
            # Process speaker blocks within Q&A group
            first_speaker = True
            for block_id in sorted(qa_group.keys()):
                paragraphs = qa_group[block_id]
                if paragraphs:
                    speaker_elements = self._create_speaker_block(block_id, paragraphs, qa_group_id)
                    if first_speaker:
                        # Keep Q&A header with first speaker
                        qa_elements.append(KeepTogether([qa_header] + speaker_elements[:1]))
                        qa_elements.extend(speaker_elements[1:])
                        first_speaker = False
                    else:
                        qa_elements.extend(speaker_elements)
            
            elements.extend(qa_elements[1:])  # Skip duplicate header
            
            # Add spacing between Q&A conversations (but not a page break)
            if qa_idx < len(content) - 1:
                elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def generate(self) -> bool:
        """Generate the PDF document."""
        try:
            # Create the document
            doc = SimpleDocTemplate(
                self.output_path,
                pagesize=letter,
                rightMargin=inch,
                leftMargin=inch,
                topMargin=inch,
                bottomMargin=inch
            )
            
            # Build content
            elements = []
            
            # Add title page
            elements.extend(self._create_title_page())
            
            # Group content by structure
            grouped_content = self._group_content_by_structure()
            
            # Add Management Discussion section
            if grouped_content['Management Discussion']:
                elements.extend(self._create_md_section(grouped_content['Management Discussion']))
            
            # Add Q&A section
            if grouped_content['Q&A']:
                elements.extend(self._create_qa_section(grouped_content['Q&A']))
            
            # Build PDF with header/footer
            def on_page(canvas_obj, doc):
                self._create_header_footer(canvas_obj, doc)
                self.page_num += 1
            
            doc.build(elements, onFirstPage=on_page, onLaterPages=on_page)
            
            return True
            
        except Exception as e:
            log_error(f"Failed to generate PDF: {str(e)}", "pdf_generation", {"error": str(e)})
            return False


def process_transcript_to_pdf(conn: SMBConnection, transcript_file: str, 
                             records: List[Dict], error_logger: EnhancedErrorLogger) -> bool:
    """Process a single transcript into PDF format."""
    try:
        # Extract ticker from filename
        ticker = transcript_file.split('_')[0]
        
        log_console(f"  Generating PDF for {ticker}...")
        
        # Prepare transcript data
        transcript_data = {
            'filename': transcript_file,
            'records': records
        }
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
        
        # Generate PDF
        pdf_generator = TranscriptPDF(temp_pdf_path, transcript_data)
        if not pdf_generator.generate():
            error_logger.log_pdf_error(ticker, "PDF generation failed")
            return False
        
        # Upload to NAS
        output_dir = config.get('output_data_path', 'Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh')
        pdf_filename = transcript_file.replace('.xml', '.pdf')
        nas_pdf_path = f"{output_dir}/stage_09_pdfs/{pdf_filename}"
        
        with open(temp_pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
        
        if not nas_upload_content(conn, pdf_content, nas_pdf_path):
            error_logger.log_pdf_error(ticker, "Failed to upload PDF to NAS", {"path": nas_pdf_path})
            return False
        
        # Clean up temp file
        os.unlink(temp_pdf_path)
        
        error_logger.total_pdfs += 1
        log_console(f"    ✓ PDF generated: {pdf_filename}")
        log_execution(f"PDF generated for {ticker}", {"filename": pdf_filename, "size": len(pdf_content)})
        
        return True
        
    except Exception as e:
        error_logger.log_processing_error(
            ticker if 'ticker' in locals() else 'unknown',
            f"Error processing transcript: {str(e)}",
            {"error": str(e), "file": transcript_file}
        )
        return False


def process_stage_07_output(conn: SMBConnection, error_logger: EnhancedErrorLogger):
    """Process Stage 7 output to generate PDFs."""
    global config
    
    try:
        # Download Stage 7 output (JSON format with paragraph-level data)
        input_path = config.get('input_data_path', 
                               'Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh/stage_07_summarized_content.json')
        
        log_console("Downloading Stage 7 output from NAS...")
        stage_07_content = nas_download_file(conn, input_path)
        
        if not stage_07_content:
            raise Exception("Failed to download Stage 7 output")
        
        # Parse JSON data
        try:
            all_records = json.loads(stage_07_content.decode('utf-8'))
        except json.JSONDecodeError:
            # Handle potentially unclosed JSON array from Stage 7
            content_str = stage_07_content.decode('utf-8').strip()
            if content_str.endswith(','):
                content_str = content_str[:-1] + ']'
            elif not content_str.endswith(']'):
                content_str = content_str + ']'
            all_records = json.loads(content_str)
        
        log_console(f"Loaded {len(all_records)} records from Stage 7")
        
        # Group records by transcript
        transcripts = defaultdict(list)
        for record in all_records:
            filename = record.get('filename', '')
            if filename:
                transcripts[filename].append(record)
        
        log_console(f"Found {len(transcripts)} unique transcripts to process")
        error_logger.total_transcripts = len(transcripts)
        
        # Apply dev mode limit if configured
        if config.get('dev_mode', False):
            max_transcripts = config.get('dev_max_transcripts', 5)
            transcript_items = list(transcripts.items())[:max_transcripts]
            log_console(f"Dev mode: Processing only {len(transcript_items)} transcripts")
        else:
            transcript_items = list(transcripts.items())
        
        # Process each transcript
        successful = 0
        failed = 0
        
        for idx, (transcript_file, records) in enumerate(transcript_items, 1):
            log_console(f"Processing transcript {idx}/{len(transcript_items)}: {transcript_file}")
            
            if process_transcript_to_pdf(conn, transcript_file, records, error_logger):
                successful += 1
            else:
                failed += 1
            
            # Add a small delay to avoid overwhelming the system
            if idx % 10 == 0:
                time.sleep(0.5)
        
        log_console(f"\nPDF generation complete: {successful} successful, {failed} failed")
        log_execution("PDF generation completed", {
            "total_transcripts": len(transcript_items),
            "successful": successful,
            "failed": failed
        })
        
    except Exception as e:
        log_error(f"Error processing Stage 8 output: {str(e)}", "processing", {"error": str(e)})
        raise


def save_logs_to_nas(conn: SMBConnection, error_logger: EnhancedErrorLogger):
    """Save execution and error logs to NAS."""
    global execution_log, error_log
    
    try:
        log_dir = config.get('output_logs_path', 
                            'Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Logs')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save execution log
        if execution_log:
            exec_log_path = f"{log_dir}/stage_09_pdf_generation_{timestamp}.log"
            exec_content = json.dumps(execution_log, indent=2).encode('utf-8')
            nas_upload_content(conn, exec_content, exec_log_path)
            log_console(f"Execution log saved: {exec_log_path}")
        
        # Save error log if errors occurred
        if error_log or error_logger.get_summary()['total_errors'] > 0:
            error_log_path = f"{log_dir}/stage_09_pdf_generation_errors_{timestamp}.log"
            
            # Combine global errors with error logger details
            error_report = {
                "summary": error_logger.get_summary(),
                "pdf_errors": error_logger.pdf_errors,
                "formatting_errors": error_logger.formatting_errors,
                "processing_errors": error_logger.processing_errors,
                "validation_errors": error_logger.validation_errors,
                "global_errors": error_log
            }
            
            error_content = json.dumps(error_report, indent=2).encode('utf-8')
            nas_upload_content(conn, error_content, error_log_path)
            log_console(f"Error log saved: {error_log_path}")
        
    except Exception as e:
        log_console(f"Warning: Failed to save logs to NAS: {str(e)}", "WARNING")


def main():
    """Main entry point for Stage 9 PDF generation."""
    global logger, config
    
    # Initialize logging
    logger = setup_logging()
    log_console("=" * 80)
    log_console("Stage 9: PDF Generation - Starting")
    log_console("=" * 80)
    
    # Initialize error logger
    error_logger = EnhancedErrorLogger()
    
    # Create NAS connection
    conn = nas_create_connection()
    if not conn:
        log_console("Failed to connect to NAS", "ERROR")
        return
    
    try:
        # Load configuration from NAS
        nas_load_config(conn)
        
        # Process Stage 7 output to generate PDFs
        process_stage_07_output(conn, error_logger)
        
        # Print summary
        summary = error_logger.get_summary()
        log_console("\n" + "=" * 80)
        log_console("Stage 9: PDF Generation - Summary")
        log_console("=" * 80)
        log_console(f"Total transcripts processed: {summary['total_transcripts']}")
        log_console(f"Total PDFs generated: {summary['total_pdfs']}")
        log_console(f"PDF generation errors: {summary['pdf_errors']}")
        log_console(f"Formatting errors: {summary['formatting_errors']}")
        log_console(f"Processing errors: {summary['processing_errors']}")
        log_console(f"Validation errors: {summary['validation_errors']}")
        log_console(f"Total errors: {summary['total_errors']}")
        
        # Save logs
        save_logs_to_nas(conn, error_logger)
        
        log_console("=" * 80)
        log_console("Stage 9: PDF Generation - Complete")
        log_console("=" * 80)
        
    except Exception as e:
        log_console(f"Fatal error: {str(e)}", "ERROR")
        log_error(f"Fatal error in main: {str(e)}", "fatal", {"error": str(e)})
        
        # Try to save logs even on fatal error
        try:
            save_logs_to_nas(conn, error_logger)
        except:
            pass
    
    finally:
        # Close NAS connection
        if conn:
            conn.close()


if __name__ == "__main__":
    main()