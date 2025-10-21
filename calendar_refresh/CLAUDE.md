# 🚀 Calendar Events Refresh - AI Development Configuration

## 🎯 PROJECT CONTEXT
- **Type**: Simple ETL for Calendar Events Data
- **Script**: Single-file `main_calendar_refresh.py`
- **Stack**: Python 3.11+, FactSet Calendar Events API, SMB/NAS, CSV
- **Architecture**: Replace strategy - fresh data each run, no incremental updates
- **Focus**: Maintain current snapshot of calendar events (all types) for monitored institutions

## 🚨 CRITICAL RULES
- ALWAYS replace master CSV with fresh data (no delta detection)
- Query date range: Past 6 months + Future 6 months from current date
- NEVER commit API keys, OAuth tokens, or credentials
- Validate environment variables before execution
- Handle timezone conversion (UTC → Toronto time EST/EDT)
- No archiving (Option A) - master CSV only
- 17 fields in CSV - removed calculated fields for simplicity

## 🛠️ DEVELOPMENT WORKFLOW

### Environment Requirements
```bash
# Required environment variables (same as database_refresh)
API_USERNAME=<factset_api_username>
API_PASSWORD=<factset_api_password>
PROXY_USER=<corporate_proxy_user>
PROXY_PASSWORD=<corporate_proxy_password>
PROXY_URL=<proxy_server_url>
PROXY_DOMAIN=MAPLE  # Default value
NAS_USERNAME=<nas_server_username>
NAS_PASSWORD=<nas_server_password>
NAS_SERVER_IP=<nas_server_ip>
NAS_SERVER_NAME=<nas_server_name>
NAS_SHARE_NAME=<nas_share_name>
NAS_PORT=445  # Default
CONFIG_PATH=<config_yaml_path_on_nas>
CLIENT_MACHINE_NAME=<local_machine_name>
```

### Git Integration
- Branch pattern: feature/calendar-*, bugfix/calendar-*, hotfix/calendar-*
- Commit format: "Calendar: [description]"
- Focus: Simplicity and reliability

## 🤖 CALENDAR REFRESH CONFIGURATION

### Core Functionality
- **Single API Call**: Batch query for all monitored institutions
- **Date Window**: Configurable months backward and forward
- **Replace Strategy**: Overwrites master CSV each run
- **No Staging**: Direct API → CSV transformation
- **Fast Execution**: Expected runtime 30-60 seconds

### Processing Flow
```
1. Connect to NAS
2. Load config.yaml
3. Setup SSL certificate
4. Configure FactSet API client
5. Calculate date range (now - 6mo to now + 6mo)
6. Query Calendar Events API (single batch call)
7. Transform events to CSV rows with enrichment
8. Save master CSV (replace existing)
9. Save execution logs
10. Done!
```

## 📁 PROJECT STRUCTURE
```
calendar_refresh/
├── main_calendar_refresh.py    # Single-file ETL script
├── config.yaml                  # Configuration (on NAS)
├── postgres_schema.sql          # PostgreSQL schema for DB loading
├── CLAUDE.md                    # This file
└── README.md                    # Quick start guide

NAS Output Structure:
/Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/CalendarEvents/
├── Database/
│   └── master_calendar_events.csv    # REPLACED each run
└── Logs/
    ├── calendar_refresh_YYYY-MM-DD_HH-MM-SS.json
    └── Errors/
        └── calendar_refresh_errors_YYYY-MM-DD_HH-MM-SS.json
```

## 🔧 ESSENTIAL COMMANDS

### Basic Execution
```bash
# Navigate to calendar_refresh
cd calendar_refresh

# Run the refresh script
python main_calendar_refresh.py

# Expected output:
# Step 1: Validating environment...
# Step 2: Connecting to NAS...
# Step 3: Loading configuration...
# Step 4: Setting up SSL certificate...
# Step 5: Configuring FactSet API client...
# Step 6: Loading monitored institutions...
# Step 7: Calculating date range...
# Step 8: Querying Calendar Events API...
# Step 9: Transforming events to CSV format...
# Step 10: Saving master CSV to NAS...
# REFRESH COMPLETE
# Institutions Monitored: 91
# Events Found: 850
# Events Saved: 850
# Upcoming Events: 500
# Past Events: 350
```

### Scheduling (Daily Execution)
```bash
# Cron job example (run daily at 6 AM)
0 6 * * * cd /path/to/factset/calendar_refresh && /path/to/venv/bin/python main_calendar_refresh.py

# Windows Task Scheduler
# Program: C:\path\to\venv\Scripts\python.exe
# Arguments: C:\path\to\factset\calendar_refresh\main_calendar_refresh.py
# Start in: C:\path\to\factset\calendar_refresh
```

### Debugging Commands
```bash
# Test environment variables
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
required = ['API_USERNAME', 'API_PASSWORD', 'NAS_USERNAME', 'NAS_PASSWORD', 'CONFIG_PATH']
missing = [var for var in required if not os.getenv(var)]
print('Missing vars:', missing if missing else 'None')
"

# Test NAS connection
python -c "
import sys
sys.path.insert(0, 'calendar_refresh')
from main_calendar_refresh import get_nas_connection, setup_logging
setup_logging()
conn = get_nas_connection()
print('NAS Connected!' if conn else 'Connection Failed')
if conn:
    conn.close()
"

# Check output on NAS
# (Adjust NAS mount path for your system)
ls -lh /mnt/nas/Finance\ Data\ and\ Analytics/DSA/Earnings\ Call\ Transcripts/Outputs/CalendarEvents/Database/
```

## 🔗 INTEGRATIONS

### FactSet Calendar Events API
- **API Module**: `fds.sdk.EventsandTranscripts.api.calendar_events_api`
- **Method**: `get_company_event()`
- **Request Model**: `CompanyEventRequest`
- **Authentication**: HTTP Basic Auth with proxy support
- **Rate Limiting**: None (single batch call)

### API Request Structure
```python
company_event_request = CompanyEventRequest(
    data=CompanyEventRequestData(
        date_time=CompanyEventRequestDataDateTime(
            start=start_datetime,  # e.g., 6 months ago
            end=end_datetime,      # e.g., 6 months from now
        ),
        universe=CompanyEventRequestDataUniverse(
            symbols=monitored_tickers,  # List of all 91 tickers
            type="Tickers",
        ),
        # NOTE: No event_types filter - captures ALL event types
    ),
)
response = api_instance.get_company_event(company_event_request)
```

### NAS Storage Integration
- **Protocol**: SMB/CIFS via pysmb
- **Key Functions**:
  - `nas_download_file()`: Download config and SSL cert
  - `nas_upload_file()`: Upload master CSV and logs
  - `nas_create_directory_recursive()`: Ensure output directories exist
- **Paths**: All NAS paths use forward slashes

### Configuration Management
- **Location**: NAS at path specified in `CONFIG_PATH` env var
- **Format**: YAML with calendar_refresh and monitored_institutions sections
- **Reload**: Config loaded fresh each run (no caching)

## 📋 PROCESSING DETAILS

### Date Range Calculation
```python
def calculate_date_range(past_months=6, future_months=6):
    today = datetime.now().date()
    start_date = today - timedelta(days=past_months * 30)   # Approximate
    end_date = today + timedelta(days=future_months * 30)  # Approximate
    return start_date, end_date

# Example:
# If run on 2025-10-21:
# start_date = 2025-04-21 (6 months ago)
# end_date = 2026-04-21 (6 months ahead)
```

### Event Enrichment
Each API event is enriched with:
- **Institution Metadata**: name, ID, type from config
- **Timezone Conversion**: UTC → Toronto time (EST/EDT automatically determined)
- **Fiscal Period Parsing**: Extract Q1-Q4 and year from headline using regex
- **Audit Timestamp**: data_fetched_timestamp (when API query ran)

### CSV Field Order (17 Fields)
```
event_id, ticker, institution_name, institution_id, institution_type,
event_type, event_date_time_utc, event_date_time_local, event_date,
event_time_local, event_headline, webcast_status, webcast_url,
dial_in_info, fiscal_year, fiscal_quarter, data_fetched_timestamp
```

**Removed fields** (simplified for replace strategy):
- `last_updated_timestamp` - Always same as data_fetched_timestamp
- `is_upcoming` - Calculate in queries: `WHERE event_date_time_utc > CURRENT_TIMESTAMP`
- `days_until_event` - Calculate in queries: `EXTRACT(DAY FROM (event_date_time_utc - CURRENT_TIMESTAMP))`

## ⚠️ KNOWN ISSUES
- **Approximate Months**: Uses 30-day approximation (6 months = 180 days)
- **Timezone Hardcoded**: Always converts to America/Toronto (EST/EDT)
- **No Historical Tracking**: Each run replaces previous data
- **Event Changes**: No detection of webcast URL or time changes
- **Single Call**: If API call fails, entire refresh fails (no retry yet)

## 🚫 DO NOT MODIFY
- Environment variable names (shared with database_refresh)
- CSV field names and order (for PostgreSQL schema compatibility)
- NAS path structure conventions
- Timezone conversion logic (Toronto EST/EDT hardcoded for consistency)
- Replace strategy (no delta detection)
- 17-field CSV schema

## 💡 DEVELOPMENT NOTES

### Key Functions

**`main()`**
- Entry point with 10-step execution flow
- Comprehensive error handling with try/finally
- Summary statistics logged at completion

**`query_calendar_events()`**
- Single batch API call for all institutions
- Returns list of event dictionaries
- Error handling logs failures but returns empty list

**`enrich_event_with_institution_data()`**
- Adds institution metadata from config
- Parses fiscal year/quarter from headline
- Converts timezones and calculates derived fields

**`transform_events_to_csv_rows()`**
- Enriches all events
- Sorts by event_date_time_utc
- Returns list ready for CSV writing

**`save_master_csv()`**
- Creates CSV in memory
- Ensures directory structure exists
- Uploads to NAS, replacing existing file

### Logging System
- **Console**: Minimal step-by-step progress
- **Execution Log**: Detailed JSON saved to NAS
- **Error Log**: Separate file only if errors occur
- **Format**: ISO timestamps, structured data

### Statistics Tracked
- Institutions monitored
- Events found (from API)
- Events saved (to CSV)
- Upcoming vs past event counts
- Execution time (seconds)
- Error count

## 🔄 EXECUTION WORKFLOW

### Daily Refresh Pattern
```
Day 1: Run script → 350 events saved
Day 2: Run script → 348 events saved (2 past events removed, 0 new)
Day 3: Run script → 352 events saved (0 removed, 4 new upcoming)
...
```

**Each run:**
1. Queries API for current snapshot
2. Replaces master CSV completely
3. No comparison with previous data
4. Always reflects API state at runtime

### Re-run Capability
- **Idempotent**: Running multiple times same day produces same result
- **Safe**: No duplicate data (replace strategy)
- **Fast**: Can run hourly if needed (though daily is sufficient)

### Monitoring
- Check execution logs for success/failure status
- Monitor event counts (should be ~500-1000+ for 91 institutions, all event types)
- Alert if event count drops significantly (API issue?)

## 📊 DATA HANDLING

### Security
- **Credentials**: Never logged, URLs sanitized
- **SSL**: Corporate certificate required for API access
- **Proxy**: NTLM authentication with domain escaping

### Validation
- **Environment**: All required vars checked at startup
- **Config**: YAML schema validated on load
- **API Response**: Null/empty checks before processing
- **Dates**: Timezone-aware datetime objects throughout

### Error Handling
- **API Failures**: Logged, returns empty events list
- **NAS Failures**: Logged, raises exception to prevent data loss
- **Parse Errors**: Individual events logged but processing continues

## 🗄️ POSTGRESQL INTEGRATION

### Loading CSV to Database
```bash
# Option 1: From PostgreSQL server
psql -d calendar_events -c "
SELECT clear_calendar_events();  -- Clear old data
COPY calendar_events (event_id, ticker, ...)
FROM '/path/to/master_calendar_events.csv'
WITH (FORMAT csv, HEADER true);
"

# Option 2: From client machine
psql -d calendar_events -c "SELECT clear_calendar_events();"
psql -d calendar_events -c "\copy calendar_events (event_id, ticker, ...)
FROM 'master_calendar_events.csv' WITH (FORMAT csv, HEADER true)"
```

### Automated Daily Load
```bash
# Daily cron: Refresh CSV, then load to PostgreSQL
0 6 * * * cd /path/to/calendar_refresh && \
          python main_calendar_refresh.py && \
          psql -d calendar_events -c "SELECT clear_calendar_events();" && \
          psql -d calendar_events -c "\copy calendar_events (...) FROM 'master.csv' ..."
```

## 🔍 TROUBLESHOOTING

### Common Issues

**1. API Connection Fails**
- Check: SSL certificate downloaded from NAS?
- Check: Proxy credentials correct?
- Check: API_USERNAME and API_PASSWORD set?
- Solution: Run environment variable check script

**2. No Events Returned**
- Check: Date range calculation (past_months/future_months in config)
- Check: Monitored institutions list not empty
- Check: FactSet API status (external issue?)
- Solution: Review execution log for API response details

**3. NAS Upload Fails**
- Check: NAS credentials correct?
- Check: Network connectivity to NAS?
- Check: Output path exists and is writable?
- Solution: Test NAS connection with debug script

**4. Execution Slow**
- Expected: 30-60 seconds for 91 institutions
- If slower: Network latency to API or NAS
- Solution: Run during off-peak hours

**5. Missing Fiscal Year/Quarter**
- Expected: Some events may not have "Q1 2025" format in headline
- Impact: fiscal_year and fiscal_quarter fields will be empty
- Solution: Acceptable - not all events follow standard naming

## 📈 MONITORING AND MAINTENANCE

### Health Checks
- **Daily**: Verify execution log created
- **Weekly**: Check event counts trending correctly
- **Monthly**: Validate data quality (spot check events)

### Expected Event Counts
- **Per Institution**: 5-10+ events across all event types (earnings, dividends, conferences, etc.)
- **Total (91 institutions)**: 500-1000+ events in 12-month window
- **Deviation**: ±100 events is normal (varies by event type and season)
- **By Type**: Earnings/Dividends most frequent, conferences/meetings seasonal

### Alerts to Configure
- Execution log not created (script failed to run)
- Event count < 300 (possible API issue or data problem)
- Event count > 1500 (possible API returning unexpected volume)
- Error log created (errors occurred during run)

### Maintenance Tasks
- **Weekly**: Review error logs if present
- **Monthly**: Validate event dates are current
- **Quarterly**: Review monitored_institutions list for changes
- **Annually**: Update SSL certificate if expired

## 🎯 FUTURE ENHANCEMENTS (OPTIONAL)

### Potential Improvements
1. **Retry Logic**: Retry API call on failure with exponential backoff
2. **Diff Reporting**: Log what changed since last run (for visibility)
3. **Email Notifications**: Send summary email after each run
4. **Event Alerts**: Notify when new events appear for key institutions
5. **Data Validation**: Check for duplicate event_ids or malformed data
6. **Archive Option**: Add optional dated CSV snapshots before replace
7. **Multiple Timezones**: Support configurable timezone conversion

### PostgreSQL Enhancements
1. **Automated Loading**: Script to auto-load CSV to PostgreSQL after refresh
2. **Historical Table**: Track event changes over time
3. **Materialized Views**: Pre-compute common aggregations
4. **Partitioning**: Partition by event_date for performance

### API Enhancements
1. **Event Types**: Support additional event types beyond "Earnings"
2. **Per-Institution Queries**: Query institutions individually with rate limiting
3. **Incremental Queries**: Query only recent changes (if API supports)

## 🚀 PRODUCTION READINESS

### Pre-Deployment Checklist
- [ ] Environment variables set in production environment
- [ ] NAS connectivity tested from production server
- [ ] SSL certificate accessible on NAS
- [ ] config.yaml uploaded to NAS at CONFIG_PATH
- [ ] monitored_institutions list reviewed and accurate
- [ ] Output directory structure created on NAS
- [ ] Python dependencies installed (requirements.txt)
- [ ] Test run completed successfully
- [ ] Execution logs reviewed for errors
- [ ] Master CSV validated (row count, field integrity)
- [ ] Scheduled task/cron job configured
- [ ] Monitoring/alerting configured

### Deployment Steps
1. Clone repository to production server
2. Create Python virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure environment variables
5. Test run: `python main_calendar_refresh.py`
6. Verify output: Check master CSV on NAS
7. Schedule daily execution (cron/Task Scheduler)
8. Configure monitoring and alerts

## 📚 ADDITIONAL RESOURCES

### FactSet API Documentation
- Calendar Events API: https://developer.factset.com/api-catalog/events-and-transcripts-api
- Authentication: HTTP Basic Auth
- Python SDK: fds.sdk.EventsandTranscripts

### Dependencies
- Python 3.11+
- pyyaml: YAML configuration parsing
- pysmb: SMB/NAS connectivity
- python-dotenv: Environment variable loading
- pytz: Timezone handling
- python-dateutil: Date parsing
- fds.sdk.EventsandTranscripts: FactSet API client

### Related Projects
- **database_refresh**: Transcript processing pipeline (9 stages)
- **test_scripts/check_events_calendar.py**: Original calendar checker script
- **monitored_institutions.yaml**: Shared institution list

---

## 💡 REMEMBER
This is a **simple, fast, reliable** ETL script. Resist the urge to over-engineer!

**Key Principles:**
- Replace strategy keeps it simple
- No staging, no queues, no delta detection
- Single API call, single CSV output
- Daily refresh keeps data current
- PostgreSQL for historical tracking if needed

**Philosophy:**
The API is the source of truth. Query it, save it, done. ✅
