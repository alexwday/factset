import os
import urllib.parse

def get_proxy_config():
    """Get proxy configuration from environment variables"""
    http_proxy = os.getenv('HTTP_PROXY') or os.getenv('http_proxy')
    https_proxy = os.getenv('HTTPS_PROXY') or os.getenv('https_proxy')
    no_proxy = os.getenv('NO_PROXY') or os.getenv('no_proxy', '')
    
    proxies = {}
    if http_proxy:
        proxies['http'] = http_proxy
        print(f"HTTP Proxy configured: {http_proxy}")
    if https_proxy:
        proxies['https'] = https_proxy
        print(f"HTTPS Proxy configured: {https_proxy}")
    
    # Parse no_proxy list
    no_proxy_list = [host.strip() for host in no_proxy.split(',') if host.strip()]
    if no_proxy_list:
        print(f"No proxy for: {no_proxy_list}")
    
    return proxies, no_proxy_list

def should_bypass_proxy(url, no_proxy_list):
    """Check if URL should bypass proxy based on no_proxy list"""
    if not no_proxy_list:
        return False
    
    parsed_url = urllib.parse.urlparse(url)
    hostname = parsed_url.hostname
    
    if not hostname:
        return False
    
    # Check if hostname matches any no_proxy entry
    for no_proxy_host in no_proxy_list:
        if no_proxy_host == hostname or hostname.endswith('.' + no_proxy_host):
            return True
    
    return False