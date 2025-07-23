#!/usr/bin/env python3
"""
Stage 5: Q&A Boundary Detection & Conversation Pairing - Clean Implementation
A completely rewritten, simplified version focusing on core functionality.
"""

import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
import io

# Core dependencies
import yaml
import requests
from dotenv import load_dotenv
from openai import OpenAI
from smb.SMBConnection import SMBConnection

# Load environment variables
load_dotenv()

# Global state - keep it simple
config = None
nas_conn = None
llm_client = None
ssl_cert_path = None

def setup_logging():
    """Simple logging setup."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def validate_environment():
    """Check required environment variables."""
    required_vars = [
        'NAS_USERNAME', 'NAS_PASSWORD', 'NAS_SERVER_IP', 'NAS_SERVER_NAME',
        'NAS_SHARE_NAME', 'CONFIG_PATH', 'CLIENT_MACHINE_NAME',
        'LLM_CLIENT_ID', 'LLM_CLIENT_SECRET'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
    
    logger.info("Environment variables validated")

def connect_to_nas():
    """Create NAS connection."""
    global nas_conn
    
    try:
        nas_conn = SMBConnection(
            username=os.getenv('NAS_USERNAME'),
            password=os.getenv('NAS_PASSWORD'),
            my_name=os.getenv('CLIENT_MACHINE_NAME'),
            remote_name=os.getenv('NAS_SERVER_NAME'),
            use_ntlm_v2=True,
            is_direct_tcp=True
        )
        
        if nas_conn.connect(os.getenv('NAS_SERVER_IP'), 445):
            logger.info("NAS connection established")
            return True
        else:
            logger.error("Failed to connect to NAS")
            return False
            
    except Exception as e:
        logger.error(f"NAS connection error: {e}")
        return False

def load_config():
    """Load configuration from NAS."""
    global config
    
    try:
        # Download config file from NAS
        config_path = os.getenv('CONFIG_PATH')
        file_obj = io.BytesIO()
        nas_conn.retrieveFile(os.getenv('NAS_SHARE_NAME'), config_path, file_obj)
        file_obj.seek(0)
        
        # Parse YAML
        config = yaml.safe_load(file_obj.read().decode('utf-8'))
        logger.info("Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return False

def setup_ssl():
    """Download and setup SSL certificate."""
    global ssl_cert_path
    
    try:
        # Download SSL cert from NAS
        cert_path = config['ssl_cert_path']
        file_obj = io.BytesIO()
        nas_conn.retrieveFile(os.getenv('NAS_SHARE_NAME'), cert_path, file_obj)
        file_obj.seek(0)
        
        # Save to temp file
        temp_cert = tempfile.NamedTemporaryFile(suffix='.cer', delete=False)
        temp_cert.write(file_obj.read())
        temp_cert.close()
        
        ssl_cert_path = temp_cert.name
        os.environ['REQUESTS_CA_BUNDLE'] = ssl_cert_path
        
        logger.info("SSL certificate configured")
        return True
        
    except Exception as e:
        logger.error(f"SSL setup failed: {e}")
        return False

def get_oauth_token():
    """Get OAuth token for LLM API."""
    try:
        llm_config = config['stage_05_qa_pairing']['llm_config']
        
        auth_data = {
            'grant_type': 'client_credentials',
            'client_id': os.getenv('LLM_CLIENT_ID'),
            'client_secret': os.getenv('LLM_CLIENT_SECRET')
        }
        
        response = requests.post(
            llm_config['token_endpoint'],
            data=auth_data,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            verify=ssl_cert_path if ssl_cert_path else True,
            timeout=30
        )
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get('access_token')
            if token:
                logger.info("OAuth token obtained successfully")
                return token
        
        logger.error(f"OAuth failed: {response.status_code}")
        return None
        
    except Exception as e:
        logger.error(f"OAuth token error: {e}")
        return None

def setup_llm_client():
    """Setup OpenAI client."""
    global llm_client
    
    try:
        # Get OAuth token
        token = get_oauth_token()
        if not token:
            logger.error("No OAuth token - cannot setup LLM client")
            return False
        
        # Create client
        llm_config = config['stage_05_qa_pairing']['llm_config']
        llm_client = OpenAI(
            api_key=token,
            base_url=llm_config['base_url'],
            timeout=llm_config.get('timeout', 60)
        )
        
        logger.info("LLM client configured successfully")
        return True
        
    except Exception as e:
        logger.error(f"LLM client setup failed: {e}")
        return False

def load_stage4_data():
    """Load Stage 4 data from NAS."""
    try:
        input_path = config['stage_05_qa_pairing']['input_data_path']
        file_obj = io.BytesIO()
        nas_conn.retrieveFile(os.getenv('NAS_SHARE_NAME'), input_path, file_obj)
        file_obj.seek(0)
        
        data = json.loads(file_obj.read().decode('utf-8'))
        
        # Handle both formats
        if isinstance(data, list):
            records = data
        else:
            records = data.get('records', [])
        
        logger.info(f"Loaded {len(records)} records from Stage 4")
        return records
        
    except Exception as e:
        logger.error(f"Failed to load Stage 4 data: {e}")
        return []

def group_by_transcript(records):
    """Group records by transcript."""
    transcripts = {}
    for record in records:
        transcript_key = record.get('filename', f"{record.get('ticker', 'unknown')}_{record.get('event_id', 'unknown')}")
        if transcript_key not in transcripts:
            transcripts[transcript_key] = []
        transcripts[transcript_key].append(record)
    
    logger.info(f"Grouped records into {len(transcripts)} transcripts")
    return transcripts

def group_by_speaker_blocks(records):
    """Group records by speaker blocks."""
    speaker_blocks = {}
    
    for record in records:
        block_id = record.get('speaker_block_id')
        if not block_id:
            continue
            
        if block_id not in speaker_blocks:
            speaker_blocks[block_id] = {
                'speaker_block_id': block_id,
                'speaker': record.get('speaker'),
                'section_name': record.get('section_name'),
                'paragraphs': []
            }
        
        speaker_blocks[block_id]['paragraphs'].append(record)
    
    # Convert to sorted list
    blocks = [speaker_blocks[bid] for bid in sorted(speaker_blocks.keys())]
    return blocks

def create_llm_prompt(speaker_blocks, current_index, current_qa_id):
    """Create prompt for LLM boundary detection."""
    context_blocks = []
    
    for i, block in enumerate(speaker_blocks[current_index:current_index+5], 1):
        content = ' '.join(p['paragraph_content'] for p in block['paragraphs'])
        qa_marker = f" (QA ID: {current_qa_id})" if i == 1 else ""
        
        context_blocks.append(f"""Speaker Block {i}{qa_marker}:
Speaker: {block['speaker']}
Content: {content}
""")
    
    prompt = f"""Analyze this earnings call Q&A transcript to find where the current analyst's session ends.

You are looking at numbered speaker blocks. Block 1 is assigned to QA ID {current_qa_id}.

Find the block number where the NEXT analyst session should begin (meaning the current session ends at the previous block).

{"".join(context_blocks)}

Return your decision using the provided function."""

    return prompt

def make_llm_decision(speaker_blocks, current_index, current_qa_id):
    """Make LLM boundary decision."""
    try:
        prompt = create_llm_prompt(speaker_blocks, current_index, current_qa_id)
        
        tools = [{
            "type": "function",
            "function": {
                "name": "boundary_decision",
                "description": "Identify where the next analyst session begins",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "next_analyst_index": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Speaker block index where next analyst begins"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief explanation of the decision"
                        }
                    },
                    "required": ["next_analyst_index", "reasoning"]
                }
            }
        }]
        
        response = llm_client.chat.completions.create(
            model=config['stage_05_qa_pairing']['llm_config']['model'],
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="required",
            temperature=0.1,
            max_tokens=300
        )
        
        if response.choices and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            
            return {
                'next_index': result['next_analyst_index'],
                'reasoning': result['reasoning']
            }
        
        return None
        
    except Exception as e:
        logger.error(f"LLM decision failed: {e}")
        return None

def process_qa_boundaries(transcript_records, transcript_id):
    """Process Q&A boundaries for a transcript."""
    try:
        # Group by speaker blocks
        speaker_blocks = group_by_speaker_blocks(transcript_records)
        
        # Filter to Q&A sections only
        qa_blocks = [b for b in speaker_blocks if b.get('section_name') == 'Q&A']
        
        if not qa_blocks:
            logger.info(f"No Q&A blocks found in {transcript_id}")
            return transcript_records
        
        logger.info(f"Processing {len(qa_blocks)} Q&A blocks for {transcript_id}")
        
        # Process boundaries
        current_index = 0
        current_qa_id = 1
        qa_assignments = {}
        
        while current_index < len(qa_blocks):
            # Get LLM decision
            decision = make_llm_decision(qa_blocks, current_index, current_qa_id)
            
            if decision and decision['next_index'] > 1:
                # Assign blocks to current QA ID
                next_abs_index = current_index + decision['next_index'] - 1
                for i in range(current_index, min(next_abs_index, len(qa_blocks))):
                    block_id = qa_blocks[i]['speaker_block_id']
                    qa_assignments[block_id] = current_qa_id
                
                logger.info(f"QA ID {current_qa_id}: blocks {current_index+1}-{next_abs_index}, reason: {decision['reasoning']}")
                current_index = next_abs_index
                current_qa_id += 1
            else:
                # Fallback: assign current block
                block_id = qa_blocks[current_index]['speaker_block_id']
                qa_assignments[block_id] = current_qa_id
                current_index += 1
                current_qa_id += 1
        
        # Apply assignments to records
        enhanced_records = []
        for record in transcript_records:
            enhanced_record = record.copy()
            
            if record.get('section_name') == 'Q&A':
                block_id = record.get('speaker_block_id')
                if block_id in qa_assignments:
                    enhanced_record['qa_group_id'] = qa_assignments[block_id]
                    enhanced_record['qa_group_confidence'] = 0.8
                    enhanced_record['qa_group_method'] = 'llm_boundary_detection'
                else:
                    enhanced_record['qa_group_id'] = None
                    enhanced_record['qa_group_confidence'] = None
                    enhanced_record['qa_group_method'] = None
            else:
                enhanced_record['qa_group_id'] = None
                enhanced_record['qa_group_confidence'] = None
                enhanced_record['qa_group_method'] = None
            
            enhanced_records.append(enhanced_record)
        
        logger.info(f"Assigned {len(set(qa_assignments.values()))} Q&A groups to {transcript_id}")
        return enhanced_records
        
    except Exception as e:
        logger.error(f"Q&A processing failed for {transcript_id}: {e}")
        return transcript_records

def save_results(all_records):
    """Save results to NAS."""
    try:
        output_data = {
            'schema_version': '2.0',
            'processing_timestamp': datetime.now().isoformat(),
            'processing_method': 'clean_implementation',
            'total_records': len(all_records),
            'records': all_records
        }
        
        # Save to NAS
        output_path = config['stage_05_qa_pairing']['output_data_path']
        filename = 'stage_05_qa_paired_content_clean.json'
        nas_path = f"{output_path}/{filename}"
        
        # Create directory if needed
        try:
            nas_conn.createDirectory(os.getenv('NAS_SHARE_NAME'), output_path)
        except:
            pass  # Directory might already exist
        
        # Upload file
        json_data = json.dumps(output_data, indent=2)
        file_obj = io.BytesIO(json_data.encode('utf-8'))
        nas_conn.storeFile(os.getenv('NAS_SHARE_NAME'), nas_path, file_obj)
        
        logger.info(f"Results saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return False

def cleanup():
    """Cleanup resources."""
    if ssl_cert_path and os.path.exists(ssl_cert_path):
        try:
            os.unlink(ssl_cert_path)
        except:
            pass
    
    if nas_conn:
        try:
            nas_conn.close()
        except:
            pass

def main():
    """Main execution function."""
    try:
        logger.info("=== STAGE 5 Q&A PAIRING - CLEAN IMPLEMENTATION ===")
        
        # Step 1: Environment validation
        logger.info("Step 1: Validating environment...")
        validate_environment()
        
        # Step 2: Connect to NAS
        logger.info("Step 2: Connecting to NAS...")
        if not connect_to_nas():
            logger.error("Failed to connect to NAS")
            return
        
        # Step 3: Load configuration
        logger.info("Step 3: Loading configuration...")
        if not load_config():
            logger.error("Failed to load configuration")
            return
        
        # Step 4: Setup SSL
        logger.info("Step 4: Setting up SSL...")
        if not setup_ssl():
            logger.error("Failed to setup SSL")
            return
        
        # Step 5: Setup LLM client
        logger.info("Step 5: Setting up LLM client...")
        if not setup_llm_client():
            logger.error("Failed to setup LLM client")
            return
        
        # Step 6: Load Stage 4 data
        logger.info("Step 6: Loading Stage 4 data...")
        records = load_stage4_data()
        if not records:
            logger.error("No records to process")
            return
        
        # Step 7: Group by transcript
        logger.info("Step 7: Grouping by transcript...")
        transcripts = group_by_transcript(records)
        
        # Step 8: Process Q&A boundaries
        logger.info("Step 8: Processing Q&A boundaries...")
        all_enhanced_records = []
        
        # Development mode - limit transcripts
        dev_mode = config['stage_05_qa_pairing'].get('dev_mode', False)
        if dev_mode:
            max_transcripts = config['stage_05_qa_pairing'].get('dev_max_transcripts', 2)
            transcript_items = list(transcripts.items())[:max_transcripts]
            transcripts = dict(transcript_items)
            logger.info(f"Development mode: Processing {len(transcripts)} transcripts")
        
        for i, (transcript_id, transcript_records) in enumerate(transcripts.items(), 1):
            logger.info(f"Processing transcript {i}/{len(transcripts)}: {transcript_id}")
            enhanced_records = process_qa_boundaries(transcript_records, transcript_id)
            all_enhanced_records.extend(enhanced_records)
        
        # Step 9: Save results
        logger.info("Step 9: Saving results...")
        if save_results(all_enhanced_records):
            logger.info("✅ Stage 5 Q&A pairing completed successfully")
        else:
            logger.error("Failed to save results")
        
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        
    finally:
        cleanup()

if __name__ == "__main__":
    main()