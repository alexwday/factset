import pandas as pd
import fds.sdk.EventsandTranscripts
from fds.sdk.EventsandTranscripts.api import transcripts_api
from dateutil.parser import parse as dateutil_parser
import os
from urllib.parse import quote

CONFIG = {"SSL_CERT": "/Users/alexwday/path/to/ssl/certificate.cer"}
PROXY_USER = "XXXXXXX"
PROXY_PASSWORD = "XXXXXXX"
PROXY_URL = "oproxy.fg.rbc.com:8080"

ssl_cert_path = CONFIG["SSL_CERT"]
os.environ["REQUESTS_CA_BUNDLE"] = ssl_cert_path
os.environ["SSL_CERT_FILE"] = ssl_cert_path

user = PROXY_USER
password = quote(PROXY_PASSWORD)

configuration = fds.sdk.EventsandTranscripts.Configuration(
    username='x',
    password='x',
    proxy="http://%s:%s@%s" % ("MAPLE%5C" + user, password, PROXY_URL),
    ssl_ca_cert=ssl_cert_path
)
configuration.get_basic_auth_token()

def get_transcripts_for_date(target_date_str):
    with fds.sdk.EventsandTranscripts.ApiClient(configuration) as api_client:
        api_instance = transcripts_api.TranscriptsApi(api_client)
        target_date = dateutil_parser(target_date_str).date()
        
        response = api_instance.get_transcripts_dates(
            start_date=target_date,
            end_date=target_date,
            time_zone="America/New_York",
            sort=["-storyDateTime"],
            pagination_limit=1000,
            pagination_offset=0
        )
        
        if not response or not hasattr(response, 'data') or not response.data:
            print(f"No transcripts found for {target_date}")
            return None
        
        df = pd.DataFrame(response.to_dict()['data'])
        print(f"Found {len(df)} transcripts on {target_date}")
        
        output_file = f"transcripts_{target_date_str.replace('-', '_')}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}")
        
        return df

if __name__ == "__main__":
    get_transcripts_for_date('2024-05-29')