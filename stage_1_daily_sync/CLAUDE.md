# Stage 1 Daily Sync - Development Documentation

## Purpose
Stage 1 performs daily incremental downloads of new/updated earnings transcripts for monitored financial institutions. This is designed for regular operational use to keep the transcript repository current.

## Architecture Overview

### Key Differences from Stage 0
- **Stage 0**: Bulk historical download (company-by-company queries)
- **Stage 1**: Daily incremental sync (date-based queries)

### Core Implementation Strategy
1. **API Endpoint**: Uses `get_transcripts_dates()` instead of `get_transcripts_ids()`
2. **Query Pattern**: Single API call per date vs 15 separate company calls
3. **Efficiency**: Retrieves ALL transcripts for target dates, then filters to monitored institutions
4. **Same Standards**: Maintains 100% of Stage 0's security, validation, and version management

## Configuration Requirements

### NAS Configuration Update
Add `stage_1` section to existing `config.json`:
```json
{
  "stage_1": {
    "sync_date_range": 1,
    "description": "Daily incremental sync of new transcripts"
  },
  // ... rest of existing config unchanged
}
```

### Configuration Parameters
- `sync_date_range`: Number of days to look back (0 = today only, 1 = today + yesterday)
- All other settings inherited from Stage 0 configuration

## Implementation Details

### Date-Based Discovery Function
```python
def get_daily_transcripts_by_date(api_instance, target_date, monitored_tickers):
    """Get all transcripts for target date, filter to monitored institutions."""
    response = api_instance.get_transcripts_dates(
        start_date=target_date,
        end_date=target_date,
        sort=["-storyDateTime"],
        pagination_limit=1000
    )
    # Filter to monitored tickers where they are SOLE primary ID
    # Maintains anti-contamination protection
```

### Processing Flow
1. Calculate date range based on `sync_date_range`
2. Query each date for ALL transcripts
3. Filter results to monitored institutions (sole primary ID)
4. Group transcripts by ticker
5. Process each institution using Stage 0's proven logic
6. Download with version management and duplicate prevention

### Maintained Features
- ✅ ALL security validation functions from Stage 0
- ✅ Version management with automatic cleanup
- ✅ Anti-contamination filtering (sole primary ID)
- ✅ Same file organization and naming
- ✅ Same error handling and retry logic
- ✅ Same audit trail and logging

## Earnings Monitor

### Overview
Optional real-time monitor for earnings season operations.

### Features
- Runs Stage 1 every 5 minutes automatically
- macOS popup notifications for each new transcript
- Tracks session activity and history
- Graceful shutdown with Ctrl+C

### Usage
```bash
# Direct execution (recommended)
python 1_transcript_daily_sync.py

# With monitor for earnings days
python earnings_monitor.py
```

### Monitor Notifications
- Individual transcript alerts with bank/date/type
- Summary notifications after each sync
- Hourly status updates during long sessions

## Execution Patterns

### Daily Production Run
```bash
# Manual daily execution
cd /Users/alexwday/Projects/factset/stage_1_daily_sync
python 1_transcript_daily_sync.py
```

### Earnings Season Monitoring
```bash
# Start monitor for continuous monitoring
python earnings_monitor.py
# Runs every 5 minutes with notifications
```

### Recovery/Catch-up Run
```json
// Temporarily adjust config for longer lookback
"sync_date_range": 7  // Look back 7 days
```

## Testing & Validation

### Pre-Deployment Checklist
- [ ] Verify config.json has stage_1 section
- [ ] Test date calculation logic
- [ ] Confirm API endpoint switch works
- [ ] Validate filtering maintains anti-contamination
- [ ] Check version management still functions
- [ ] Ensure all Stage 0 security patterns present

### Expected Behavior
1. **Efficiency**: Single API call vs 15 (per date)
2. **Coverage**: Same institutions as Stage 0
3. **Quality**: Same validation and error handling
4. **Output**: Identical file structure and naming

## Common Scenarios

### No New Transcripts
- Normal during non-earnings periods
- Script completes successfully with 0 downloads
- Monitor stays silent (no notifications)

### Version Updates
- Automatically detected and downloaded
- Old versions removed from NAS
- Monitor shows "updated version" notifications

### Multiple Transcripts Same Day
- All downloaded in single run
- Grouped by institution in logs
- Monitor shows summary notification

## Maintenance Notes

### Log Files
- Uploaded to `Outputs/Logs/stage_1_daily_sync_log_YYYY-MM-DD_HH-MM-SS.log`
- Same format as Stage 0 for consistency

### State Files (Monitor Only)
- `.monitor_state.json`: Current session tracking
- `.transcript_history.json`: Historical record
- Auto-created, can be deleted to reset

### Performance
- Typical run: 10-30 seconds (empty days)
- Earnings days: 1-3 minutes (multiple downloads)
- Monitor overhead: Minimal (subprocess isolation)

## Security Compliance

Stage 1 maintains ALL security standards from Stage 0:
- Input validation for all parameters
- Path traversal prevention
- URL sanitization in logs
- Credential protection
- Configuration validation
- SSL certificate management

## Future Enhancements

Potential improvements while maintaining current architecture:
1. Parallel date queries for multi-day ranges
2. Configurable monitor frequency
3. Email notifications option
4. Database integration for history

## Critical Reminders

1. **NEVER** modify security validation functions
2. **ALWAYS** maintain version management logic
3. **NEVER** skip anti-contamination filtering
4. **ALWAYS** validate configuration before use
5. **NEVER** log credentials or sensitive data

---

Stage 1 successfully adapts Stage 0's proven architecture for daily operations while maintaining all security, reliability, and quality standards.