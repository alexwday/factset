# Invalid Transcript Tracking System

## Overview
The Stage 00 and Stage 01 scripts have been enhanced to track and filter invalid transcripts using an Excel-based ignore list. This prevents repeated downloads of transcripts that don't match the strict earnings call format.

## Key Changes

### 1. Invalid Transcript List
- **Location**: `/Outputs/Data/InvalidTranscripts/invalid_transcripts.xlsx`
- **Format**: Excel file with the following columns:
  - `ticker`: Institution ticker symbol
  - `institution_name`: Full institution name
  - `event_id`: FactSet event identifier
  - `version_id`: Transcript version identifier
  - `title_found`: Actual title from the transcript
  - `event_date`: Date of the event
  - `transcript_type`: Type of transcript (Script, etc.)
  - `reason`: Why the transcript was rejected
  - `date_added`: When it was added to the ignore list

### 2. Processing Logic

#### First Run (Bootstrap Mode)
1. Scans existing NAS files (all valid earnings calls)
2. Creates new empty invalid transcript list
3. Downloads and checks all "new" transcripts from API
4. Valid titles (`Qx 20xx Earnings Call`) → Saved to NAS
5. Invalid titles → Added to ignore list, XML discarded
6. Builds complete ignore list for future runs

#### Subsequent Runs
1. Loads existing invalid transcript list
2. Checks new transcripts against:
   - Existing NAS files (by event_id + version_id)
   - Invalid transcript list
3. Only downloads truly new transcripts
4. Much faster as invalid transcripts are pre-filtered

### 3. Title Validation
**Strict Format Required**: `Qx 20xx Earnings Call`
- Must start with Q1, Q2, Q3, or Q4
- Must have 4-digit year starting with 20
- Must end with "Earnings Call"
- Case-insensitive matching

Examples:
- ✅ Valid: "Q1 2024 Earnings Call"
- ❌ Invalid: "Q1 2024 Earnings Conference Call"
- ❌ Invalid: "First Quarter 2024 Earnings Call"
- ❌ Invalid: "Q1 2024 Results"

## Usage

### Running the Updated Scripts

#### Stage 00 - Historical Sync
```bash
cd database_refresh/00_download_historical
python main_historical_sync_with_ignore.py
```

#### Stage 01 - Daily Sync
```bash
cd database_refresh/01_download_daily
python main_daily_sync_with_ignore.py
```

### Monitoring the Ignore List
The Excel file can be opened directly to review rejected transcripts:
- Check which titles are being rejected
- Verify no valid transcripts are incorrectly filtered
- See patterns in non-earnings transcripts

### Resetting the System
To start fresh:
1. Delete the Excel file at `/Outputs/Data/InvalidTranscripts/invalid_transcripts.xlsx`
2. The next run will rebuild the ignore list from scratch

## Benefits

1. **Efficiency**: No repeated downloads of known invalid transcripts
2. **Clean NAS**: Only valid earnings calls stored in year/quarter folders
3. **Transparency**: Excel provides clear visibility into rejected transcripts
4. **Performance**: Subsequent runs are much faster
5. **Backwards Compatible**: Works with existing NAS structure

## Migration Notes

### For Existing Installations
- First run will be slower as it builds the ignore list
- Existing valid transcripts remain untouched
- No data loss or file movement required
- System self-bootstraps on first execution

### File Organization
```
/Outputs/Data/
├── InvalidTranscripts/
│   └── invalid_transcripts.xlsx    # Ignore list
├── 2024/                           # Year folders (only valid earnings calls)
│   ├── Q1/
│   ├── Q2/
│   ├── Q3/
│   └── Q4/
└── Unknown/                        # Fallback for unparseable dates
    └── Unknown/
```

## Troubleshooting

### Issue: Excel file corrupted
**Solution**: Delete the file and let the system recreate it on next run

### Issue: Valid transcript rejected
**Solution**: Check the exact title format - must match `Qx 20xx Earnings Call` exactly

### Issue: First run taking too long
**Expected**: First run downloads many files to check titles and build the ignore list

### Issue: Can't write Excel file
**Solution**: Ensure the InvalidTranscripts directory exists and has write permissions

## Technical Details

### Dependencies
- `pandas`: DataFrame operations for ignore list
- `openpyxl`: Excel file reading/writing
- Existing dependencies: `fds.sdk.EventsandTranscripts`, `pysmb`, etc.

### Functions Added
- `load_invalid_transcript_list()`: Loads or creates the Excel ignore list
- `save_invalid_transcript_list()`: Saves updated ignore list to NAS
- `add_to_invalid_list()`: Adds a rejected transcript to the list
- `is_transcript_in_invalid_list()`: Checks if transcript should be skipped
- `is_valid_earnings_call_title()`: Validates title format
- `download_transcript_with_validation()`: Downloads and validates before saving

### Integration Points
- Both Stage 00 and 01 use the same ignore list
- Shared Excel file ensures consistency
- Compatible with existing Stage 02+ processing