# Earnings Transcript Repository Sync v2

A production-ready script that maintains a NAS-based repository of earnings transcripts for monitored financial institutions using the FactSet Events & Transcripts API.

## Features

- **Environment-based Configuration**: Secure credential management via .env files
- **Concurrent Downloads**: Multi-threaded downloading with configurable rate limiting
- **Comprehensive Error Handling**: Specific exception handling with retry mechanisms
- **Input Validation**: API response validation and data integrity checks
- **Failed Download Tracking**: Tracks and reports failed downloads for analysis
- **NAS Integration**: Directly stores transcripts on network-attached storage
- **Production Logging**: Structured logging with debug mode support
- **Notebook Compatible**: Works from both Jupyter notebooks and command line

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# FactSet API Configuration
API_USERNAME=your_factset_username
API_PASSWORD=your_factset_api_password

# Proxy Configuration
PROXY_USER=your_proxy_username
PROXY_PASSWORD=your_proxy_password

# NAS Configuration
NAS_USERNAME=your_nas_username
NAS_PASSWORD=your_nas_password
NAS_SERVER_IP=192.168.1.100
```

### 3. Run the Script

**From Command Line:**
```bash
python earnings_transcript_repository_sync_v2.py
```

**From Jupyter Notebook:**
```python
exec(open('earnings_transcript_repository_sync_v2.py').read())
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONCURRENT_DOWNLOADS` | 5 | Number of concurrent download threads |
| `RATE_LIMIT_PER_SECOND` | 10 | Maximum API requests per second |
| `DEBUG_MODE` | false | Enable debug logging |

## Directory Structure

The script creates the following structure on your NAS:

```
transcript_repository/
├── Inputs/
│   └── certificate/
│       └── certificate.cer
└── Outputs/
    ├── data/
    │   ├── corrected/
    │   │   ├── RY-CA/
    │   │   ├── TD-CA/
    │   │   └── ...
    │   ├── raw/
    │   └── nearrealtime/
    ├── logs/
    │   ├── sync_log_*.log
    │   └── failed_downloads_*.json
    └── listing/
        ├── corrected_listing.json
        ├── raw_listing.json
        ├── nearrealtime_listing.json
        └── complete_inventory_*.json
```

## Monitoring and Troubleshooting

### Log Files
- **Console Output**: Real-time progress and status
- **Log Files**: Detailed logs stored on NAS in `Outputs/logs/`
- **Failed Downloads**: JSON reports of any download failures

### Debug Mode
Enable debug mode for detailed logging:
```env
DEBUG_MODE=true
```

### Performance Tuning
Adjust concurrent downloads and rate limiting based on your environment:
```env
MAX_CONCURRENT_DOWNLOADS=3  # Reduce for slower networks
RATE_LIMIT_PER_SECOND=5     # Reduce to stay well within API limits
```

## Monitored Institutions

The script monitors earnings transcripts for:

### Canadian Banks
- Royal Bank of Canada (RY-CA)
- Toronto-Dominion Bank (TD-CA)
- Bank of Nova Scotia (BNS-CA)
- Bank of Montreal (BMO-CA)
- Canadian Imperial Bank of Commerce (CM-CA)
- National Bank of Canada (NA-CA)

### US Banks
- JPMorgan Chase & Co. (JPM-US)
- Bank of America Corporation (BAC-US)
- Wells Fargo & Company (WFC-US)
- Citigroup Inc. (C-US)
- Goldman Sachs Group Inc. (GS-US)
- Morgan Stanley (MS-US)

### Insurance Companies
- Manulife Financial Corporation (MFC-CA)
- Sun Life Financial Inc. (SLF-CA)
- UnitedHealth Group Incorporated (UNH-US)

## Testing

Run basic functionality tests:
```bash
python test_basic_functionality.py
```

## Security

- All credentials are stored in environment variables
- The `.env` file is automatically ignored by git
- SSL certificates are validated before use
- API authentication uses FactSet's standard methods

## Error Recovery

The script includes comprehensive error recovery:
- **Retry Logic**: Failed downloads are retried with exponential backoff
- **Failure Tracking**: All failures are logged with detailed error messages
- **Resume Capability**: Re-running the script will only download new transcripts
- **Graceful Degradation**: Individual failures don't stop the entire sync

## Production Deployment

For production use:
1. Set up the script on a scheduled task (cron, Windows Task Scheduler, etc.)
2. Monitor the log files for any issues
3. Set up alerts based on failed download reports
4. Ensure NAS storage has sufficient space for transcript growth

## API Rate Limits

The script respects FactSet's API rate limits:
- Default: 10 requests per second
- Configurable via `RATE_LIMIT_PER_SECOND`
- Built-in rate limiter prevents exceeding limits

## Support

For issues or questions:
1. Check the log files for detailed error messages
2. Review the failed downloads JSON for specific failures
3. Ensure all environment variables are correctly set
4. Verify network connectivity to NAS and FactSet API