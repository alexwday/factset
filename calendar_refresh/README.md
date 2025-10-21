# Calendar Events Refresh

Simple ETL to maintain a current snapshot of calendar events (earnings, dividends, conferences, shareholder meetings, etc.) for monitored financial institutions using the FactSet Calendar Events API.

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
├── calendar_config.yaml         # Configuration (upload to NAS)
├── postgres_schema.sql          # PostgreSQL schema (optional)
├── CLAUDE.md                    # Overall documentation
└── README.md                    # This file

Note: monitored_institutions.yaml is NOT in this folder - it's shared from database_refresh config folder
```

**Note:** Unlike database_refresh (9 stages), this is just **1 stage** because there's no LLM processing needed - just API query → CSV transformation.

## What It Does

This script:
1. Queries the FactSet Calendar Events API for all 91 monitored institutions
2. Retrieves **all event types** from the past 6 months and future 6 months
3. Enriches event data with institution metadata and timezone conversions
4. **Replaces** the master CSV file on NAS with fresh data
5. Saves execution logs for monitoring

**No incremental updates.** Each run completely replaces the master CSV with current data from the API.

## Available Event Types

The FactSet Calendar Events API provides the following event types for monitored institutions:

| Event Type | Description | Example |
|------------|-------------|---------|
| **Earnings** | Earnings calls and releases (confirmed dates) | Q4 2025 Earnings Conference Call |
| **ConfirmedEarningsRelease** | Confirmed earnings release with specific date/time | Q3 2025 Earnings Release - 8:00 AM EST |
| **ProjectedEarningsRelease** | Projected/estimated earnings release dates | Q2 2025 Earnings (Estimated) |
| **Dividend** | Dividend announcements and declarations | Quarterly Dividend Declaration |
| **Conference** | Company presentations at industry conferences | Morgan Stanley Financials Conference |
| **ShareholdersMeeting** | Annual or special shareholder meetings | 2026 Annual Shareholders Meeting |
| **SalesRevenueCall** | Sales and revenue-specific conference calls | Monthly Sales Results Call |
| **SalesRevenueMeeting** | Sales/revenue investor meetings | Quarterly Sales Review with Analysts |
| **SalesRevenueRelease** | Sales and revenue data releases | Monthly Sales Report Release |
| **AnalystsInvestorsMeeting** | Analyst and investor day events | 2025 Investor Day Presentation |
| **SpecialSituation** | Special corporate events (M&A, restructuring, etc.) | Strategic Update Call |

**The script captures ALL event types by default.** No filtering is applied unless you configure it in `calendar_config.yaml`.

### Filtering Event Types (Optional)

If you only want specific event types, you can add filtering to your `calendar_config.yaml`:

```yaml
calendar_refresh:
  date_range:
    past_months: 6
    future_months: 6

  # Optional: Specify which event types to capture
  # If omitted or empty, ALL event types are captured (recommended)
  event_types:
    - "Earnings"
    - "ConfirmedEarningsRelease"
    - "Dividend"

  master_database_path: "..."
  output_logs_path: "..."
```

**Note:** Filtering is NOT implemented in the current version. The script captures all events. To add filtering support, you would need to modify `main_calendar_refresh.py` to read the `event_types` config and apply it to the API request.

## Output

### Master CSV
- **Location**: `{NAS}/Outputs/CalendarEvents/Database/master_calendar_events.csv`
- **Format**: 16 columns, sorted by event date
- **Content**: All calendar events across all types (past 6 months + future 6 months)
- **Update Strategy**: Complete replacement each run

### Execution Logs
- **Location**: `{NAS}/Outputs/CalendarEvents/Logs/`
- **Format**: JSON with detailed execution information
- **Errors**: Separate error log created if issues occur

## CSV Schema (16 Fields)

**Updated to match FactSet API field names**

| Column | Description | Example | API Source |
|--------|-------------|---------|------------|
| event_id | FactSet event identifier | ABC123XYZ | event_id |
| ticker | Institution ticker | RY-CA | ticker |
| institution_name | Full institution name | Royal Bank of Canada | config |
| institution_id | Sequential ID | 1 | config |
| institution_type | Category | Canadian_Banks | config |
| event_type | Type of event | Earnings | event_type |
| event_date_time_utc | Event datetime (UTC) | 2025-11-27T13:00:00Z | event_date_time |
| event_date_time_local | Event datetime (Toronto EST/EDT) | 2025-11-27T08:00:00-05:00 | calculated |
| event_date | Event date | 2025-11-27 | calculated |
| event_time_local | Event time with timezone | 08:00 EST | calculated |
| event_headline | Event description | Q4 2025 Earnings Call | **description** |
| webcast_link | Webcast URL | https://... | **webcast_link** |
| contact_info | Contact details | Contact: John \| Phone: 1-800... | **contact_name, contact_phone, contact_email** |
| fiscal_year | Fiscal year | 2025 | **fiscal_year** |
| fiscal_period | Fiscal period/quarter | Q4 | **fiscal_period** |
| data_fetched_timestamp | When data was fetched | 2025-10-21T12:00:00Z | calculated |

## Configuration

Configuration uses **two files on NAS** (not stored locally):
1. **calendar_config.yaml**: Calendar refresh settings (calendar_refresh only)
2. **monitored_institutions.yaml**: List of 91 financial institutions (**SHARED** with database_refresh)

**Important**: The calendar_refresh script uses a **different config file** than database_refresh:
- database_refresh: Uses `config.yaml` (existing)
- calendar_refresh: Uses `calendar_config.yaml` (new file)
- **Both scripts share the same `monitored_institutions.yaml`** (already exists from database_refresh)

**You only need to upload `calendar_config.yaml` - the script will automatically use the existing `monitored_institutions.yaml` from the database_refresh config folder!**

### What's in calendar_config.yaml (on NAS)

**Section 1: Calendar Refresh Settings**
```yaml
calendar_refresh:
  date_range:
    past_months: 6      # How far back to query (default: 6 months)
    future_months: 6    # How far ahead to query (default: 6 months)

  # event_types: []     # Optional: Filter specific types (see Available Event Types section)
                        # If omitted, ALL event types are captured (default behavior)

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

**No setup needed!** The calendar_refresh script automatically uses the existing `monitored_institutions.yaml` file from the database_refresh config folder (specified by `CONFIG_PATH` in your `.env`). Both scripts share this file.

### Setting Config Paths

Both scripts use separate environment variables for their config files:

**database_refresh**: Uses `CONFIG_PATH`
```bash
CONFIG_PATH="Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/config.yaml"
```

**calendar_refresh**: Uses `CALENDAR_CONFIG_PATH`
```bash
CALENDAR_CONFIG_PATH="Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/calendar_config.yaml"
```

**Add both to your `.env` file** in the `factset/` root directory:
```bash
# database_refresh config
CONFIG_PATH=Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/config.yaml

# calendar_refresh config
CALENDAR_CONFIG_PATH=Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/calendar_config.yaml
```

This way both can run without changing environment variables!

**NAS folder structure** (after uploading calendar_config.yaml):
```
/Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/
├── config.yaml                    # database_refresh config (existing)
├── calendar_config.yaml           # calendar_refresh config (NEW - upload this)
└── monitored_institutions.yaml    # Shared by both (already exists from database_refresh)
```

**What to upload**: Only `calendar_config.yaml` - everything else already exists!

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

Load the CSV into PostgreSQL for querying and historical tracking.

**Note**: The CSV is already deduplicated during the refresh process, so you can query the table directly without any views.

### 1. Create Database and Schema
```bash
# Create database
createdb aegis_calendar

# Load schema
psql -d aegis_calendar -f postgres_schema.sql
```

### 2. Load CSV Data
```bash
# Clear and reload (replace strategy)
psql -d aegis_calendar -c "DELETE FROM aegis_calendar_events;"

psql -d aegis_calendar -c "\copy aegis_calendar_events (
    event_id, ticker, institution_name, institution_id, institution_type,
    event_type, event_date_time_utc, event_date_time_local, event_date,
    event_time_local, event_headline, webcast_link, contact_info,
    fiscal_year, fiscal_period, data_fetched_timestamp
) FROM 'master_calendar_events.csv' WITH (FORMAT csv, HEADER true)"
```

### 3. Query Examples
```sql
-- Upcoming events in next 30 days
SELECT ticker, institution_name, event_type, event_headline, event_date, event_time_local
FROM aegis_calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
  AND event_date_time_utc <= CURRENT_TIMESTAMP + INTERVAL '30 days'
ORDER BY event_date_time_utc;

-- Canadian bank earnings this quarter
SELECT ticker, institution_name, event_type, event_headline, event_date, event_time_local
FROM aegis_calendar_events
WHERE institution_type = 'Canadian_Banks'
  AND event_type IN ('Earnings', 'ConfirmedEarningsRelease')
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- All event types for a specific institution
SELECT event_type, event_headline, event_date, event_time_local
FROM aegis_calendar_events
WHERE ticker = 'RY-CA'
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- Count events by type
SELECT event_type, COUNT(*) as event_count
FROM aegis_calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
GROUP BY event_type
ORDER BY event_count DESC;

-- Upcoming dividends
SELECT ticker, institution_name, event_headline, event_date
FROM aegis_calendar_events
WHERE event_type = 'Dividend'
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;
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
CALENDAR_CONFIG_PATH=path/to/calendar_config.yaml  # Path on NAS
# Note: database_refresh uses CONFIG_PATH, calendar_refresh uses CALENDAR_CONFIG_PATH
```

**Note:** The script uses `load_dotenv()` which searches up the directory tree, so it will find the `.env` file in the `factset/` root folder. **No need to copy the .env file!**

**Example `.env` file**:
```bash
# FactSet API
API_USERNAME=your_username
API_PASSWORD=your_password

# ... (other variables)

# Config paths (both can coexist in .env)
CONFIG_PATH=Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/config.yaml
CALENDAR_CONFIG_PATH=Finance Data and Analytics/DSA/Earnings Call Transcripts/Inputs/Config/calendar_config.yaml
```

## Monitoring

### Expected Behavior
- **Execution Time**: 30-60 seconds
- **Event Count**: 500-1000+ events (for 91 institutions, all event types)
  - Previously: ~300-500 when filtering only Earnings
  - Now: Higher count due to capturing all event types (dividends, conferences, etc.)
- **Upcoming Events**: Varies by quarter and event type
- **Past Events**: Within 6-month lookback window
- **Timezone**: All local times shown in Toronto time (EST/EDT)

### Health Checks
- ✅ Execution log created each run
- ✅ Event count within expected range
- ✅ No error log created (or minimal errors)
- ✅ Master CSV updated with fresh timestamp

### Alert Conditions
- 🚨 No execution log (script didn't run)
- 🚨 Event count < 300 (API issue or data problem?)
- 🚨 Event count > 1500 (API returning unexpected volume?)
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
    ├── calendar_config.yaml     # Configuration (upload to NAS)
    ├── postgres_schema.sql      # PostgreSQL schema (optional)
    ├── CLAUDE.md                # Overall documentation
    └── README.md                # This file
    # Note: Uses monitored_institutions.yaml from database_refresh (not duplicated here)

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
