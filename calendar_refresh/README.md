# Calendar Events Refresh

Simple ETL to maintain a current snapshot of earnings events for monitored financial institutions using the FactSet Calendar Events API.

## Quick Start

```bash
# 1. Navigate to stage folder
cd calendar_refresh/01_calendar_query

# 2. Ensure environment variables are set (uses .env from factset/ root)
# Shares same .env as database_refresh - no copy needed!

# 3. Run the refresh script
python main_calendar_refresh.py

# Expected runtime: 30-60 seconds
```

## Structure

```
calendar_refresh/
├── 01_calendar_query/           # Main stage (single script ETL)
│   ├── main_calendar_refresh.py # Query API → Generate CSV
│   └── CLAUDE.md                # Stage documentation
├── config.yaml                  # Configuration (stored on NAS)
├── monitored_institutions.yaml  # 91 institutions list (copy from database_refresh)
├── postgres_schema.sql          # PostgreSQL schema (optional)
├── CLAUDE.md                    # Overall documentation
└── README.md                    # This file
```

**Note:** Unlike database_refresh (9 stages), this is just **1 stage** because there's no LLM processing needed - just API query → CSV transformation.

## What It Does

This script:
1. Queries the FactSet Calendar Events API for all 91 monitored institutions
2. Retrieves earnings events from the past 6 months and future 6 months
3. Enriches event data with institution metadata and timezone conversions
4. **Replaces** the master CSV file on NAS with fresh data
5. Saves execution logs for monitoring

**No incremental updates.** Each run completely replaces the master CSV with current data from the API.

## Output

### Master CSV
- **Location**: `{NAS}/Outputs/CalendarEvents/Database/master_calendar_events.csv`
- **Format**: 20 columns, sorted by event date
- **Content**: All current earnings events (past 6 months + future 6 months)
- **Update Strategy**: Complete replacement each run

### Execution Logs
- **Location**: `{NAS}/Outputs/CalendarEvents/Logs/`
- **Format**: JSON with detailed execution information
- **Errors**: Separate error log created if issues occur

## CSV Schema (17 Fields)

| Column | Description | Example |
|--------|-------------|---------|
| event_id | FactSet event identifier | ABC123XYZ |
| ticker | Institution ticker | RY-CA |
| institution_name | Full institution name | Royal Bank of Canada |
| institution_id | Sequential ID | 1 |
| institution_type | Category | Canadian_Banks |
| event_type | Type of event | Earnings |
| event_date_time_utc | Event datetime (UTC) | 2025-11-27T13:00:00Z |
| event_date_time_local | Event datetime (Toronto EST/EDT) | 2025-11-27T08:00:00-05:00 |
| event_date | Event date | 2025-11-27 |
| event_time_local | Event time with timezone | 08:00 EST |
| event_headline | Event title | Q4 2025 Earnings Call |
| webcast_status | Webcast availability | Confirmed |
| webcast_url | Webcast link | https://... |
| dial_in_info | Conference call details | Dial: 1-800-... |
| fiscal_year | Parsed fiscal year | 2025 |
| fiscal_quarter | Parsed quarter | Q4 |
| data_fetched_timestamp | When data was fetched | 2025-10-21T12:00:00Z |

## Configuration

Configuration is split across **two files on NAS** (not stored locally):
1. **config.yaml**: Calendar refresh settings
2. **monitored_institutions.yaml**: List of 91 financial institutions

The path to config.yaml is specified in your `.env` file's `CONFIG_PATH` variable. The monitored_institutions.yaml file should be in the same directory.

### What's in config.yaml (on NAS)

**Section 1: Calendar Refresh Settings**
```yaml
calendar_refresh:
  date_range:
    past_months: 6      # How far back to query (default: 6 months)
    future_months: 6    # How far ahead to query (default: 6 months)

  event_types:
    - "Earnings"        # Currently only Earnings events

  master_database_path: "Finance Data and Analytics/.../master_calendar_events.csv"
  output_logs_path: "Finance Data and Analytics/.../Logs"

  replace_on_refresh: true  # Always replace master CSV (no incremental)
```

**Section 2: SSL Certificate**
```yaml
ssl_cert_path: "Finance Data and Analytics/.../rbc-ca-bundle.cer"
```

**The SSL certificate is already on your NAS from database_refresh setup. The script downloads it at runtime - you don't need to copy it anywhere!**

### What's in monitored_institutions.yaml (on NAS)

This file contains all 91 monitored financial institutions and is **shared with database_refresh**:

```yaml
# Monitored Financial Institutions (91 Total)
RY-CA: {id: 1, name: "Royal Bank of Canada", type: "Canadian_Banks", ...}
BMO-CA: {id: 2, name: "Bank of Montreal", type: "Canadian_Banks", ...}
# ... all 91 institutions across 12 categories
```

**To set up**: Copy `database_refresh/monitored_institutions.yaml` to the same NAS directory as your `config.yaml`. The script will automatically find and load it.

## Scheduling

### Linux/Mac (cron)
```bash
# Run daily at 6 AM
0 6 * * * cd /path/to/factset/calendar_refresh/01_calendar_query && /path/to/venv/bin/python main_calendar_refresh.py
```

### Windows (Task Scheduler)
- **Program**: `C:\path\to\venv\Scripts\python.exe`
- **Arguments**: `C:\path\to\factset\calendar_refresh\01_calendar_query\main_calendar_refresh.py`
- **Start in**: `C:\path\to\factset\calendar_refresh\01_calendar_query`
- **Schedule**: Daily at 6:00 AM

## PostgreSQL Integration

Load the CSV into PostgreSQL for querying and historical tracking:

### 1. Create Database and Schema
```bash
# Create database
createdb calendar_events

# Load schema
psql -d calendar_events -f postgres_schema.sql
```

### 2. Load CSV Data
```bash
# Clear and reload (replace strategy)
psql -d calendar_events -c "SELECT clear_calendar_events();"

psql -d calendar_events -c "\copy calendar_events (
    event_id, ticker, institution_name, institution_id, institution_type,
    event_type, event_date_time_utc, event_date_time_local, event_date,
    event_time_local, event_headline, webcast_status, webcast_url,
    dial_in_info, fiscal_year, fiscal_quarter, data_fetched_timestamp
) FROM 'master_calendar_events.csv' WITH (FORMAT csv, HEADER true)"
```

### 3. Query Examples
```sql
-- Upcoming events in next 30 days
SELECT * FROM upcoming_events_30d;

-- Events by institution type
SELECT * FROM events_by_institution_type;

-- Canadian bank earnings this quarter
SELECT ticker, institution_name, event_headline, event_date, event_time_local
FROM calendar_events
WHERE institution_type = 'Canadian_Banks'
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- Calculate days until event
SELECT
    ticker,
    event_headline,
    event_date,
    EXTRACT(DAY FROM (event_date_time_utc - CURRENT_TIMESTAMP))::INTEGER as days_until
FROM calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc
LIMIT 10;
```

## Dependencies

```bash
pip install pyyaml pysmb python-dotenv pytz python-dateutil fds.sdk.EventsandTranscripts
```

Or use the shared `requirements.txt` from the main project.

## Environment Variables

**Uses the same `.env` file as `database_refresh`** - stored in `factset/` root directory.

Required variables (should already be set if database_refresh works):

```bash
# FactSet API (from .env in factset/ root)
API_USERNAME=your_factset_username
API_PASSWORD=your_factset_password

# Corporate Proxy (from .env)
PROXY_USER=your_proxy_username
PROXY_PASSWORD=your_proxy_password
PROXY_URL=proxy.company.com:port
PROXY_DOMAIN=MAPLE

# NAS Connection (from .env)
NAS_USERNAME=nas_username
NAS_PASSWORD=nas_password
NAS_SERVER_IP=192.168.x.x
NAS_SERVER_NAME=nas_server_name
NAS_SHARE_NAME=share_name
NAS_PORT=445
CLIENT_MACHINE_NAME=your_machine_name

# Configuration (from .env)
CONFIG_PATH=path/to/calendar_config.yaml  # Path on NAS
```

**Note:** The script uses `load_dotenv()` which searches up the directory tree, so it will find the `.env` file in the `factset/` root folder. **No need to copy the .env file!**

## Monitoring

### Expected Behavior
- **Execution Time**: 30-60 seconds
- **Event Count**: 300-500 events (for 91 institutions)
- **Upcoming Events**: ~200 (varies by quarter)
- **Past Events**: ~150-200 (within 6-month lookback)
- **Timezone**: All local times shown in Toronto time (EST/EDT)

### Health Checks
- ✅ Execution log created each run
- ✅ Event count within expected range
- ✅ No error log created (or minimal errors)
- ✅ Master CSV updated with fresh timestamp

### Alert Conditions
- 🚨 No execution log (script didn't run)
- 🚨 Event count < 200 (API issue?)
- 🚨 Event count > 700 (API returning unexpected data?)
- 🚨 Error log present (check for failures)

## Troubleshooting

### Script Fails to Start
- **Check**: Environment variables set? Run `python -c "import os; print(os.getenv('API_USERNAME'))"`
- **Check**: Virtual environment activated?
- **Check**: Dependencies installed? Run `pip list | grep fds`

### No Events Retrieved
- **Check**: Date range in config (past_months/future_months)
- **Check**: API credentials valid
- **Check**: Network connectivity to FactSet API
- **Review**: Execution log for API response details

### NAS Upload Fails
- **Check**: NAS credentials correct
- **Check**: Network connectivity to NAS
- **Check**: Output directory exists on NAS
- **Test**: NAS connection with test script

### Execution Slow (>2 minutes)
- **Cause**: Network latency or API throttling
- **Solution**: Run during off-peak hours
- **Note**: Some slowness is expected with 91 institutions

## File Structure

```
factset/                         # Root directory
├── .env                         # Shared environment variables
├── database_refresh/            # Transcript processing (9 stages)
│   ├── 00_download_historical/
│   ├── 01_download_daily/
│   └── ...
└── calendar_refresh/            # Calendar events (1 stage)
    ├── 01_calendar_query/       # Main stage
    │   ├── main_calendar_refresh.py  # ETL script
    │   └── CLAUDE.md            # Stage documentation
    ├── config.yaml              # Configuration (on NAS)
    ├── monitored_institutions.yaml  # 91 institutions (copy from database_refresh)
    ├── postgres_schema.sql      # PostgreSQL schema (optional)
    ├── CLAUDE.md                # Overall documentation
    └── README.md                # This file

NAS Output Structure:
/Finance Data and Analytics/DSA/Earnings Call Transcripts/
├── Outputs/Data/                # Transcript data (from database_refresh)
└── Outputs/CalendarEvents/      # Calendar events (from calendar_refresh)
    ├── Database/
    │   └── master_calendar_events.csv
    └── Logs/
        ├── calendar_refresh_YYYY-MM-DD_HH-MM-SS.json
        └── Errors/
            └── calendar_refresh_errors_*.json
```

## Key Design Decisions

### Why Replace Instead of Incremental?
- **Small Data**: Only 300-500 rows, trivial to replace
- **No Processing**: Just data transformation, very fast
- **API is Truth**: Fresh snapshot ensures accuracy
- **Simplicity**: No delta detection, no deduplication needed

### Why No Archives?
- **Option A Selected**: Master CSV only, no dated snapshots
- **Rationale**: PostgreSQL provides historical tracking if needed
- **Benefit**: Simpler, faster, less storage

### Why Single Script?
- **No Complexity**: No LLM processing, no embedding generation
- **Fast Execution**: 30-60 seconds total runtime
- **Easy Maintenance**: One file to understand and modify

## Comparison to Database Refresh

| Aspect | Database Refresh | Calendar Refresh |
|--------|------------------|------------------|
| Stages | 9 stages | 1 script |
| Processing | XML, LLM, Embeddings | API → CSV |
| Runtime | Hours | Seconds |
| Complexity | High | Low |
| Data Size | GB (transcripts) | KB (metadata) |
| Strategy | Incremental | Replace |

## Additional Resources

- **FactSet API Docs**: https://developer.factset.com/api-catalog/events-and-transcripts-api
- **CLAUDE.md**: Detailed technical documentation
- **postgres_schema.sql**: Complete database schema with examples
- **test_scripts/check_events_calendar.py**: Original prototype script

## Support

For issues or questions:
1. Check `CLAUDE.md` for detailed troubleshooting
2. Review execution logs on NAS
3. Validate environment variables
4. Test individual components (NAS connection, API access)

## License

Internal use only - RBC proprietary.
