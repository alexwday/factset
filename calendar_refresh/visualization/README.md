# Calendar Events Visualization

Interactive HTML calendar for visualizing financial institution events.

## Quick Start

```bash
# 1. Navigate to visualization folder
cd calendar_refresh/visualization

# 2. Generate the interactive calendar
python generate_calendar.py

# 3. Open the generated file
open output/calendar.html
# or on Windows: start output/calendar.html
```

## What It Does

This tool generates an **interactive HTML calendar** from calendar events CSV data with:

- **📅 Visual Calendar Display**: Month, week, and list views
- **🔍 Smart Filters**: Filter by institution type and event type
- **📝 Event Details**: Click any event to see full details
- **📥 Calendar Downloads**: Download .ics files to add events to Outlook/Google Calendar/Apple Calendar

## Features

### Visual Calendar
- **Monthly View**: See all events at a glance
- **Weekly View**: Detailed week-by-week breakdown
- **List View**: Chronological event listing
- **Color Coding**: Each event type has a distinct color

### Interactive Filters
- **Institution Type**: Filter by Canadian_Banks, US_Banks, etc.
- **Event Type**: Filter by Earnings, Dividend, Conference, etc.
- **Reset**: Clear all filters with one click

### Event Details
Click any event to see:
- Institution name
- Event description
- Date & time (local timezone)
- Fiscal period (if applicable)
- Webcast link (if available)
- Contact information

### Calendar Integration
- **Download .ics Files**: Click "Add to Outlook Calendar" in event details
- **One-Click Import**: Opens directly in your default calendar app
- **All Details Included**: Event time, description, location, etc.

## Usage

### Using Sample Data

The `sample_data/example_calendar_events.csv` file contains 20 example events showing different event types.

Just run:
```bash
python generate_calendar.py
```

### Using Your Own Data

1. **Copy your master CSV** from NAS to `sample_data/`:
   ```bash
   cp /path/to/master_calendar_events.csv sample_data/my_data.csv
   ```

2. **Update the script** to use your file (edit `generate_calendar.py`):
   ```python
   # Change this line:
   sample_csv = script_dir / "sample_data" / "example_calendar_events.csv"

   # To:
   sample_csv = script_dir / "sample_data" / "my_data.csv"
   ```

3. **Generate calendar**:
   ```bash
   python generate_calendar.py
   ```

### Automated Updates

To automatically regenerate the calendar when your data updates:

**Option 1: Manual Refresh**
```bash
# Copy latest data
cp /nas/path/to/master_calendar_events.csv sample_data/current.csv

# Regenerate calendar
python generate_calendar.py
```

**Option 2: Scheduled Task (Linux/Mac)**
```bash
# Add to crontab to run daily at 7 AM
0 7 * * * cd /path/to/calendar_refresh/visualization && \
          cp /nas/path/to/master_calendar_events.csv sample_data/current.csv && \
          python generate_calendar.py
```

## File Structure

```
visualization/
├── generate_calendar.py           # Main script
├── sample_data/                   # Input CSV files
│   └── example_calendar_events.csv  # Sample data (20 events)
├── output/                        # Generated HTML files
│   └── calendar.html              # Interactive calendar (generated)
└── README.md                      # This file
```

## Event Type Colors

The calendar uses color coding to distinguish event types:

| Color | Event Type |
|-------|-----------|
| 🔵 Blue | Earnings |
| 🔵 Dark Blue | Confirmed Earnings Release |
| 🔵 Light Blue | Projected Earnings Release |
| 🟢 Green | Dividend |
| 🟠 Orange | Conference |
| 🟣 Purple | Shareholders Meeting |
| 🔴 Red | Sales Revenue Call |
| 🔴 Pink-Red | Sales Revenue Meeting |
| 🔴 Pink | Sales Revenue Release |
| 🔵 Cyan | Analysts/Investors Meeting |
| 🟠 Dark Orange | Special Situation |

## CSV Format Requirements

The CSV file must have these columns:
- `event_id`: Unique event identifier
- `ticker`: Institution ticker symbol
- `institution_name`: Full institution name
- `institution_type`: Category (e.g., Canadian_Banks)
- `event_type`: Type of event
- `event_date_time_local`: Local datetime (ISO format)
- `event_date_time_utc`: UTC datetime (ISO format) - for .ics files
- `event_headline`: Event description
- `event_time_local`: Display time with timezone
- `webcast_link`: Optional webcast URL
- `contact_info`: Optional contact details
- `fiscal_year`: Optional fiscal year
- `fiscal_period`: Optional fiscal period (Q1, Q2, etc.)

## Browser Compatibility

The generated HTML calendar works in all modern browsers:
- ✅ Chrome/Edge (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Opera

## Dependencies

The Python script uses only standard library modules:
- `csv`: Reading CSV files
- `json`: Data serialization
- `datetime`: Date/time handling
- `pathlib`: File path operations
- `base64`: Encoding .ics files

**No pip installations required!**

The generated HTML uses CDN-hosted libraries (loaded from internet):
- FullCalendar.js 6.1.10 (MIT License)

## Troubleshooting

### Script fails to find CSV
**Error**: `Sample CSV not found`

**Solution**: Ensure the CSV file exists in `sample_data/` directory

### Events not showing
**Issue**: Calendar loads but no events appear

**Solution**:
1. Check CSV has valid `event_date_time_local` values
2. Ensure dates are in ISO format: `2025-11-27T08:00:00-05:00`
3. Check browser console for errors (F12)

### .ics download fails
**Issue**: "Calendar file not available"

**Solution**: Ensure CSV has `event_date_time_utc` column with valid UTC timestamps

### Filters not working
**Issue**: Selecting filters doesn't update calendar

**Solution**:
1. Check CSV has `institution_type` and `event_type` columns
2. Clear browser cache and reload

## Customization

### Changing Colors

Edit the `event_type_colors` dictionary in `generate_calendar.py`:

```python
event_type_colors = {
    'Earnings': '#2563eb',  # Change this hex color
    'Dividend': '#059669',  # Your custom color here
    # ... etc
}
```

### Changing Default View

In the HTML template, modify the `initialView` setting:

```python
initialView: 'dayGridMonth',  # Options: dayGridMonth, timeGridWeek, listMonth
```

### Adding More Filters

The script can be extended to add filters for:
- Specific tickers
- Fiscal quarters
- Date ranges
- Custom categories

## Examples

### Example 1: View All Earnings Events

1. Open `output/calendar.html` in browser
2. Select "Earnings" from Event Type filter
3. Click "All Institution Types" or select specific type
4. View filtered calendar

### Example 2: Export Canadian Bank Dividends

1. Filter: Institution Type = "Canadian_Banks"
2. Filter: Event Type = "Dividend"
3. Click each dividend event
4. Click "Add to Outlook Calendar" to download .ics
5. Import to your calendar app

### Example 3: Weekly View of All Events

1. Open calendar
2. Click "week" button in top-right
3. Navigate weeks using prev/next arrows
4. See detailed time slots for each event

## FAQ

**Q: Can I share the generated HTML with my team?**
A: Yes! The HTML file is self-contained and can be shared via email, network drive, or internal website.

**Q: Does it work offline?**
A: Partially. Event data is embedded, but FullCalendar.js loads from CDN. For fully offline use, download FullCalendar and reference it locally.

**Q: Can I embed this in SharePoint/Confluence?**
A: Yes, but you may need to adjust Content Security Policy settings to allow the embedded JavaScript.

**Q: How often should I regenerate the calendar?**
A: After each calendar refresh (daily if using scheduled refresh). You can automate this with a cron job or scheduled task.

**Q: Can I filter by multiple institution types?**
A: Not in the current version, but the script can be extended to support multi-select filters.

**Q: What's the maximum number of events?**
A: Tested with 1000+ events. Performance depends on browser, but should handle 5000+ events without issues.

## Support

For issues or questions:
1. Check this README
2. Review the sample CSV format
3. Check browser console for JavaScript errors (F12)
4. Ensure CSV columns match requirements

## License

Internal use only - RBC proprietary.
