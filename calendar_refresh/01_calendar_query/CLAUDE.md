# Stage 01: Calendar Events Query

## Purpose
Query FactSet Calendar Events API for all monitored institutions and generate master CSV with current earnings events data.

## What This Stage Does

**Single-script ETL that:**
1. Connects to NAS and loads configuration
2. Queries Calendar Events API (batch call for all 91 institutions)
3. Enriches events with institution metadata
4. Converts timezones (UTC → Toronto EST/EDT)
5. Parses fiscal year/quarter from headlines
6. Saves master CSV to NAS (replaces existing)
7. Logs execution details

**Replace Strategy:** Each run completely replaces the master CSV with fresh data from the API.

## Script

**File:** `main_calendar_refresh.py`

**Execution:**
```bash
cd calendar_refresh/01_calendar_query
python main_calendar_refresh.py
```

**Expected Runtime:** 30-60 seconds

## Configuration

Uses `calendar_refresh/config.yaml` with these sections:

### Calendar Refresh Settings
```yaml
calendar_refresh:
  date_range:
    past_months: 6      # Look back 6 months
    future_months: 6    # Look ahead 6 months
  event_types:
    - "Earnings"
  master_database_path: "path/to/master_calendar_events.csv"
  output_logs_path: "path/to/logs"
```

### Environment Variables Required
```bash
# FactSet API (from .env)
API_USERNAME=your_factset_username
API_PASSWORD=your_factset_password

# Corporate Proxy (from .env)
PROXY_USER=your_proxy_user
PROXY_PASSWORD=your_proxy_password
PROXY_URL=proxy.company.com:8080
PROXY_DOMAIN=MAPLE

# NAS Connection (from .env)
NAS_USERNAME=nas_user
NAS_PASSWORD=nas_password
NAS_SERVER_IP=192.168.x.x
NAS_SERVER_NAME=nas_server
NAS_SHARE_NAME=share_name
NAS_PORT=445
CLIENT_MACHINE_NAME=your_machine

# Config Path - IMPORTANT: Use CALENDAR_CONFIG_PATH for this script!
CALENDAR_CONFIG_PATH=path/to/calendar_config.yaml  # On NAS
# Note: database_refresh uses CONFIG_PATH, this uses CALENDAR_CONFIG_PATH
# Both can be set in the same .env file
```

## Inputs

### From Environment Variables
- API credentials (username, password)
- Proxy settings (user, password, URL, domain)
- NAS connection details

### From NAS (via config)
- SSL certificate (for API calls)
- Date range configuration
- Output paths

### From NAS (monitored_institutions.yaml)
- Monitored institutions list (91 institutions)
- Institution metadata (ID, name, type, path_safe_name)

### From FactSet API
- Calendar events data (JSON response)

## Outputs

### Primary Output: Master CSV
**Location:** `{NAS}/Outputs/CalendarEvents/Database/master_calendar_events.csv`

**Fields (17 columns):**
1. event_id
2. ticker
3. institution_name
4. institution_id
5. institution_type
6. event_type
7. event_date_time_utc
8. event_date_time_local (Toronto time)
9. event_date
10. event_time_local (with EST/EDT)
11. event_headline
12. webcast_status
13. webcast_url
14. dial_in_info
15. fiscal_year
16. fiscal_quarter
17. data_fetched_timestamp

**Size:** ~300-500 rows (for 91 institutions over 12-month window)

**Update Strategy:** Complete replacement each run

### Execution Log
**Location:** `{NAS}/Outputs/CalendarEvents/Logs/calendar_refresh_YYYY-MM-DD_HH-MM-SS.json`

**Contents:**
```json
{
  "script": "calendar_refresh",
  "execution_start": "2025-10-21T12:00:00Z",
  "execution_end": "2025-10-21T12:00:45Z",
  "summary": {
    "status": "success",
    "institutions_monitored": 91,
    "events_found": 350,
    "events_saved": 350,
    "errors": 0,
    "execution_time_seconds": 45.23
  },
  "execution_log": [ /* detailed steps */ ]
}
```

### Error Log (If Errors Occur)
**Location:** `{NAS}/Outputs/CalendarEvents/Logs/Errors/calendar_refresh_errors_YYYY-MM-DD_HH-MM-SS.json`

Only created if errors occur during execution.

## Processing Logic

### Step-by-Step Flow

```
1. VALIDATE ENVIRONMENT
   - Check all required env vars present

2. CONNECT TO NAS
   - SMB connection using NAS credentials

3. LOAD CONFIG
   - Download config.yaml from NAS
   - Parse YAML

4. SETUP SSL
   - Download certificate from NAS
   - Configure for HTTPS requests

5. CONFIGURE API CLIENT
   - Build proxy URL with domain authentication
   - Create FactSet API configuration object
   - Get basic auth token

6. LOAD INSTITUTIONS
   - Try to load monitored_institutions.yaml from NAS
   - Fall back to config if separate file not found
   - Build list of 91 tickers

7. CALCULATE DATE RANGE
   - Start: today - (past_months * 30 days)
   - End: today + (future_months * 30 days)

8. QUERY API
   - Single batch call: get_company_event()
   - Request all 91 tickers in one call
   - Event type: "Earnings"
   - Returns list of events

9. TRANSFORM TO CSV
   - For each event:
     a) Enrich with institution data (name, ID, type)
     b) Convert UTC → Toronto time (EST/EDT)
     c) Parse fiscal year/quarter from headline
     d) Add data_fetched_timestamp
   - Sort by event_date_time_utc

10. SAVE MASTER CSV
    - Create CSV in memory
    - Upload to NAS (replaces existing file)

11. SAVE LOGS
    - Execution log (always)
    - Error log (only if errors)
```

### API Call Details

**Method:** `calendar_events_api.get_company_event()`

**Request Object:**
```python
CompanyEventRequest(
    data=CompanyEventRequestData(
        date_time=CompanyEventRequestDataDateTime(
            start=datetime(2025, 4, 21),  # 6 months ago
            end=datetime(2026, 4, 21)     # 6 months ahead
        ),
        universe=CompanyEventRequestDataUniverse(
            symbols=[all 91 tickers],
            type="Tickers"
        ),
        event_types=["Earnings"]
    )
)
```

**Response:** List of event objects with fields like:
- event_id
- ticker
- event_date_time
- event_headline
- webcast_status
- webcast_url
- dial_in_info

### Enrichment Process

Each raw API event is enriched:

```python
{
    # From API
    "event_id": "ABC123",
    "ticker": "RY-CA",
    "event_date_time": "2025-11-27T13:00:00Z",
    "event_headline": "Q4 2025 Earnings Call",
    ...
}

# Becomes (after enrichment):
{
    # Original API fields
    "event_id": "ABC123",
    "ticker": "RY-CA",
    "event_type": "Earnings",
    "event_headline": "Q4 2025 Earnings Call",
    "webcast_status": "Confirmed",
    "webcast_url": "https://...",
    "dial_in_info": "Dial: 1-800-...",

    # Added from config (monitored_institutions)
    "institution_name": "Royal Bank of Canada",
    "institution_id": 1,
    "institution_type": "Canadian_Banks",

    # Timezone conversions
    "event_date_time_utc": "2025-11-27T13:00:00Z",
    "event_date_time_local": "2025-11-27T08:00:00-05:00",
    "event_date": "2025-11-27",
    "event_time_local": "08:00 EST",

    # Parsed from headline (regex)
    "fiscal_year": "2025",
    "fiscal_quarter": "Q4",

    # Audit
    "data_fetched_timestamp": "2025-10-21T12:00:00Z"
}
```

## Key Functions

### `main()`
Entry point with 11-step execution flow.

### `load_monitored_institutions()`
Loads monitored institutions from separate YAML file on NAS, with fallback to config.yaml.

### `query_calendar_events()`
Queries FactSet API for all institutions in single batch call.

### `enrich_event_with_institution_data()`
Enriches raw API event with institution metadata, timezone conversion, fiscal period parsing.

### `transform_events_to_csv_rows()`
Applies enrichment to all events, sorts by date, returns CSV-ready list.

### `save_master_csv()`
Creates CSV in memory, uploads to NAS, replaces existing file.

## Error Handling

### Validation Errors
- Missing environment variables → Raise exception, exit
- Invalid config → Raise exception, exit

### Connection Errors
- NAS connection failure → Raise exception, exit
- SSL certificate download failure → Raise exception, exit

### API Errors
- API call failure → Log error, return empty events list
- Timeout → Log error, return empty list

### Save Errors
- CSV save failure → Raise exception, exit (critical)

## Monitoring

### Success Indicators
- ✅ Execution log created
- ✅ Event count 300-500 (typical range)
- ✅ No error log created
- ✅ Runtime < 90 seconds

### Alert Conditions
- 🚨 No execution log (script didn't run)
- 🚨 Event count < 200 (possible API issue)
- 🚨 Event count > 700 (unexpected data volume)
- 🚨 Error log created
- 🚨 Runtime > 120 seconds (performance issue)

## Re-run Capability

**Idempotent:** Yes
- Running multiple times produces same result
- Safe to re-run hourly if needed
- No duplicate data (replace strategy)

**Daily Schedule:** Recommended
- Cron: `0 6 * * *` (6 AM daily)
- Keeps data current
- Low cost (single API call)

## Differences from Database Refresh

| Aspect | Database Refresh | Calendar Refresh |
|--------|------------------|------------------|
| Stages | 9 stages | 1 stage |
| Strategy | Incremental | Replace |
| LLM | Yes (stages 5-8) | No |
| Runtime | Hours | Seconds |
| Complexity | High | Low |
| Data Volume | GB | KB |

## Testing

### Local Test
```bash
cd calendar_refresh/01_calendar_query
python main_calendar_refresh.py
```

### Validation Checks
1. Master CSV created on NAS
2. Row count reasonable (300-500)
3. All 17 columns present
4. Dates in expected range (±6 months)
5. Execution log shows success
6. No error log created

### Sample Output (Console)
```
============================================================
CALENDAR EVENTS REFRESH
============================================================
Step 1: Validating environment...
Step 2: Connecting to NAS...
Step 3: Loading configuration...
...
Step 10: Saving master CSV to NAS...
Master CSV saved: .../master_calendar_events.csv
============================================================
REFRESH COMPLETE
Institutions Monitored: 91
Events Found: 350
Events Saved: 350
Upcoming Events: 200
Past Events: 150
============================================================
Total execution time: 45.23 seconds
```

## Troubleshooting

### Script won't start
- Check: All env vars set? `python -c "import os; print(os.getenv('API_USERNAME'))"`
- Check: In correct directory? `pwd` should show `calendar_refresh/01_calendar_query`
- Check: Virtual env activated? `which python`

### No events returned
- Check: Date range reasonable in config?
- Check: API credentials valid?
- Review: Execution log for API response details

### NAS save fails
- Check: NAS credentials correct?
- Check: Network connectivity to NAS?
- Test: Can you access NAS manually?

### Slow execution (>2 minutes)
- Cause: Network latency to API or NAS
- Solution: Run during off-peak hours
- Normal: Some variability expected

## Next Steps After Stage 01

**Option A:** Use CSV directly
- Read from NAS for reports/analysis
- Simple, no database needed

**Option B:** Load into PostgreSQL
- Run `../postgres_schema.sql`
- Load CSV: `psql -c "\copy calendar_events (...) FROM 'master.csv' ..."`
- Query with SQL for complex analysis

**Option C:** Scheduled automation
- Set up cron/Task Scheduler
- Run daily at 6 AM
- Monitor execution logs
