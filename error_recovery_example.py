"""
Example of error recovery mechanism
"""
import json
from datetime import datetime

def save_failed_downloads(nas_conn, failed_downloads):
    """Save list of failed downloads for retry in next run"""
    failed_file = {
        'timestamp': datetime.now().isoformat(),
        'failed_downloads': failed_downloads
    }
    
    failed_json = json.dumps(failed_file, indent=2)
    failed_path = nas_path_join(NAS_BASE_PATH, "Outputs", "failed_downloads.json")
    
    file_obj = io.BytesIO(failed_json.encode('utf-8'))
    nas_upload_file(nas_conn, file_obj, failed_path)

def load_previous_failures(nas_conn):
    """Load previously failed downloads for retry"""
    failed_path = nas_path_join(NAS_BASE_PATH, "Outputs", "failed_downloads.json")
    
    try:
        data = nas_download_file(nas_conn, failed_path)
        if data:
            return json.loads(data.decode('utf-8')).get('failed_downloads', [])
    except:
        pass
    return []