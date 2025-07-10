"""
Example of concurrent processing with rate limiting
"""
import asyncio
from datetime import datetime
import time

class RateLimiter:
    """Rate limiter to ensure we stay within API limits"""
    def __init__(self, max_requests_per_second=10):
        self.max_requests_per_second = max_requests_per_second
        self.semaphore = asyncio.Semaphore(max_requests_per_second)
        self.request_times = []
    
    async def acquire(self):
        async with self.semaphore:
            now = time.time()
            # Remove requests older than 1 second
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # If we've hit the rate limit, wait
            if len(self.request_times) >= self.max_requests_per_second:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(time.time())

# Example usage:
async def process_institution_async(ticker, rate_limiter):
    """Process a single institution with rate limiting"""
    await rate_limiter.acquire()
    # Your API call here
    print(f"Processing {ticker} at {datetime.now()}")

async def process_all_institutions(institutions):
    """Process all institutions concurrently with rate limiting"""
    rate_limiter = RateLimiter(max_requests_per_second=10)
    tasks = [process_institution_async(ticker, rate_limiter) for ticker in institutions]
    await asyncio.gather(*tasks)

# Run it:
# asyncio.run(process_all_institutions(['RY-CA', 'TD-CA', 'BMO-CA']))