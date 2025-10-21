#!/usr/bin/env python3
"""
Interactive Calendar Visualization Generator
Reads calendar events CSV and generates an interactive HTML calendar with:
- Visual calendar display
- Filters for institution type and event type
- Clickable events that download .ics files for Outlook/calendar apps
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import base64


def read_csv_data(csv_path):
    """Read calendar events from CSV file."""
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def deduplicate_earnings_events(events):
    """
    Deduplicate earnings-related events for the same institution on the same date.
    Priority order: Earnings > ConfirmedEarningsRelease > ProjectedEarningsRelease
    """
    # Define priority for earnings-related events (lower number = higher priority)
    earnings_priority = {
        'Earnings': 1,
        'ConfirmedEarningsRelease': 2,
        'ProjectedEarningsRelease': 3,
    }

    # Group events by ticker + date
    from collections import defaultdict
    event_groups = defaultdict(list)

    for event in events:
        ticker = event.get('ticker', '')
        event_date = event.get('event_date', '')  # Just the date part (YYYY-MM-DD)
        event_type = event.get('event_type', '')

        # Create a key for grouping: ticker + date
        key = f"{ticker}|{event_date}"
        event_groups[key].append(event)

    # Deduplicate: keep only highest priority earnings event per group
    deduplicated = []

    for key, group_events in event_groups.items():
        # Separate earnings-related from other events
        earnings_events = [e for e in group_events if e.get('event_type', '') in earnings_priority]
        other_events = [e for e in group_events if e.get('event_type', '') not in earnings_priority]

        # For earnings events, keep only the highest priority one
        if earnings_events:
            # Sort by priority (ascending - lower number is higher priority)
            earnings_events.sort(key=lambda e: earnings_priority.get(e.get('event_type', ''), 999))
            # Keep only the first (highest priority)
            deduplicated.append(earnings_events[0])

        # Add all non-earnings events (they don't need deduplication)
        deduplicated.extend(other_events)

    return deduplicated


def convert_to_fullcalendar_format(events):
    """Convert CSV events to FullCalendar format."""
    calendar_events = []

    # Define color mapping for event types
    event_type_colors = {
        'Earnings': '#2563eb',  # Blue
        'ConfirmedEarningsRelease': '#1d4ed8',  # Dark blue
        'ProjectedEarningsRelease': '#60a5fa',  # Light blue
        'Dividend': '#059669',  # Green
        'Conference': '#d97706',  # Orange
        'ShareholdersMeeting': '#7c3aed',  # Purple
        'SalesRevenueCall': '#dc2626',  # Red
        'SalesRevenueMeeting': '#e11d48',  # Pink-red
        'SalesRevenueRelease': '#be185d',  # Pink
        'AnalystsInvestorsMeeting': '#0891b2',  # Cyan
        'SpecialSituation': '#ea580c',  # Dark orange
    }

    for event in events:
        # Parse datetime
        event_datetime = event.get('event_date_time_local', '')

        calendar_event = {
            'id': event.get('event_id', ''),
            'title': f"{event.get('ticker', '')} - {event.get('event_type', '')}",
            'start': event_datetime.split('+')[0] if event_datetime else '',  # Remove timezone offset
            'description': event.get('event_headline', ''),
            'institution': event.get('institution_name', ''),
            'ticker': event.get('ticker', ''),
            'institutionType': event.get('institution_type', ''),
            'eventType': event.get('event_type', ''),
            'webcastLink': event.get('webcast_link', ''),
            'contactInfo': event.get('contact_info', ''),
            'fiscalYear': event.get('fiscal_year', ''),
            'fiscalPeriod': event.get('fiscal_period', ''),
            'eventTimeLocal': event.get('event_time_local', ''),
            'backgroundColor': event_type_colors.get(event.get('event_type', ''), '#6b7280'),
            'borderColor': event_type_colors.get(event.get('event_type', ''), '#6b7280'),
        }
        calendar_events.append(calendar_event)

    return calendar_events


def get_unique_values(events, field):
    """Get unique values for a field from events."""
    values = set()
    for event in events:
        value = event.get(field, '')
        if value:
            values.add(value)
    return sorted(list(values))


def get_grouped_event_types(events):
    """
    Get unique event types, grouping earnings-related types into one.
    Returns list of display names for the filter dropdown.
    """
    # Earnings types that should be grouped together
    earnings_types = {'Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease'}

    # Collect all unique event types from the data
    all_types = set()
    has_earnings = False

    for event in events:
        event_type = event.get('event_type', '')
        if event_type in earnings_types:
            has_earnings = True
        elif event_type:
            all_types.add(event_type)

    # Build the final list for display
    result = []
    if has_earnings:
        result.append('Earnings')  # Single option for all earnings types

    result.extend(sorted(all_types))
    return result


def generate_ics_content(event):
    """Generate iCalendar (.ics) content for an event."""
    # Parse the datetime
    event_datetime_str = event.get('event_date_time_utc', '')
    try:
        event_dt = datetime.fromisoformat(event_datetime_str.replace('Z', '+00:00'))
        dtstart = event_dt.strftime('%Y%m%dT%H%M%SZ')
        # Assume 1 hour duration
        dtend = event_dt.replace(hour=event_dt.hour + 1).strftime('%Y%m%dT%H%M%SZ')
    except:
        dtstart = '20251201T120000Z'
        dtend = '20251201T130000Z'

    # Build description
    description_parts = [event.get('event_headline', '')]
    if event.get('webcast_link'):
        description_parts.append(f"Webcast: {event['webcast_link']}")
    if event.get('contact_info'):
        description_parts.append(f"Contact: {event['contact_info']}")
    if event.get('fiscal_period'):
        description_parts.append(f"Fiscal Period: {event['fiscal_period']} {event.get('fiscal_year', '')}")

    description = '\\n'.join(description_parts)

    ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//FactSet Calendar Events//EN
BEGIN:VEVENT
UID:{event.get('event_id', 'EVENT')}@factset-calendar
DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{event.get('ticker', '')} - {event.get('event_type', '')}
DESCRIPTION:{description}
LOCATION:{event.get('institution_name', '')}
STATUS:CONFIRMED
SEQUENCE:0
END:VEVENT
END:VCALENDAR"""

    return ics_content


def generate_html(calendar_events, institution_types, event_types, csv_events):
    """Generate the HTML page with embedded calendar and filters."""

    # Create a mapping for ICS downloads
    ics_data = {}
    for csv_event in csv_events:
        event_id = csv_event.get('event_id', '')
        ics_content = generate_ics_content(csv_event)
        # Base64 encode for data URL
        ics_base64 = base64.b64encode(ics_content.encode('utf-8')).decode('utf-8')
        ics_data[event_id] = ics_base64

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Institutions Calendar Events</title>

    <!-- FullCalendar CSS -->
    <link href="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.css" rel="stylesheet">

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f9fafb;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 30px;
        }}

        header {{
            margin-bottom: 30px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 20px;
        }}

        h1 {{
            font-size: 28px;
            color: #111827;
            margin-bottom: 8px;
        }}

        .subtitle {{
            color: #6b7280;
            font-size: 14px;
        }}

        .filters {{
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .filter-group label {{
            font-size: 13px;
            font-weight: 600;
            color: #374151;
        }}

        .filter-group select {{
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            border-radius: 6px;
            font-size: 14px;
            background: white;
            cursor: pointer;
            min-width: 220px;
        }}

        .filter-group select[multiple] {{
            min-height: 120px;
        }}

        .filter-group select:focus {{
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }}

        .reset-btn {{
            padding: 8px 16px;
            background: #ef4444;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            font-weight: 500;
            margin-top: 20px;
        }}

        .reset-btn:hover {{
            background: #dc2626;
        }}

        #calendar {{
            margin-top: 20px;
        }}

        .legend {{
            margin-top: 25px;
            padding: 20px;
            background: #f9fafb;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }}

        .legend-title {{
            font-weight: 600;
            color: #111827;
            margin-bottom: 12px;
            font-size: 14px;
        }}

        .legend-items {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}

        .event-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}

        .event-modal.active {{
            display: flex;
        }}

        .modal-content {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }}

        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e5e7eb;
        }}

        .modal-title {{
            font-size: 20px;
            font-weight: 700;
            color: #111827;
        }}

        .close-btn {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #6b7280;
            padding: 0;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
        }}

        .close-btn:hover {{
            background: #f3f4f6;
            color: #111827;
        }}

        .event-detail {{
            margin-bottom: 15px;
        }}

        .event-detail-label {{
            font-weight: 600;
            color: #374151;
            font-size: 13px;
            margin-bottom: 4px;
        }}

        .event-detail-value {{
            color: #111827;
            font-size: 14px;
        }}

        .event-detail-value a {{
            color: #2563eb;
            text-decoration: none;
        }}

        .event-detail-value a:hover {{
            text-decoration: underline;
        }}

        .download-ics-btn {{
            width: 100%;
            padding: 12px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}

        .download-ics-btn:hover {{
            background: #1d4ed8;
        }}

        .fc-event {{
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📅 Financial Institutions Calendar Events</h1>
            <p class="subtitle">Interactive calendar showing earnings, dividends, conferences, and other corporate events</p>
        </header>

        <div class="filters">
            <div class="filter-group">
                <label for="institutionTypeFilter">Institution Type (Ctrl/Cmd+Click for multiple)</label>
                <select id="institutionTypeFilter" multiple>
                    {'\n'.join(f'<option value="{inst_type}">{inst_type.replace("_", " ")}</option>' for inst_type in institution_types)}
                </select>
            </div>

            <div class="filter-group">
                <label for="eventTypeFilter">Event Type (Ctrl/Cmd+Click for multiple)</label>
                <select id="eventTypeFilter" multiple>
                    {'\n'.join(f'<option value="{evt_type}">{evt_type}</option>' for evt_type in event_types)}
                </select>
            </div>

            <button class="reset-btn" onclick="resetFilters()">Reset All Filters</button>
        </div>

        <div id="calendar"></div>

        <div class="legend">
            <div class="legend-title">Event Type Legend</div>
            <div class="legend-items">
                <div class="legend-item">
                    <div class="legend-color" style="background: #2563eb;"></div>
                    <span>Earnings</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #1d4ed8;"></div>
                    <span>Confirmed Earnings Release</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #60a5fa;"></div>
                    <span>Projected Earnings Release</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #059669;"></div>
                    <span>Dividend</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #d97706;"></div>
                    <span>Conference</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #7c3aed;"></div>
                    <span>Shareholders Meeting</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #dc2626;"></div>
                    <span>Sales Revenue Call</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e11d48;"></div>
                    <span>Sales Revenue Meeting</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #be185d;"></div>
                    <span>Sales Revenue Release</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #0891b2;"></div>
                    <span>Analysts/Investors Meeting</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ea580c;"></div>
                    <span>Special Situation</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Event Detail Modal -->
    <div class="event-modal" id="eventModal">
        <div class="modal-content">
            <div class="modal-header">
                <div class="modal-title" id="modalTitle"></div>
                <button class="close-btn" onclick="closeModal()">&times;</button>
            </div>
            <div id="modalBody"></div>
        </div>
    </div>

    <!-- FullCalendar JS -->
    <script src="https://cdn.jsdelivr.net/npm/fullcalendar@6.1.10/index.global.min.js"></script>

    <script>
        // Event data
        const allEvents = {json.dumps(calendar_events, indent=8)};

        // ICS data for downloads
        const icsData = {json.dumps(ics_data, indent=8)};

        // Earnings type grouping - maps "Earnings" filter to actual event types
        const EARNINGS_TYPES = ['Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease'];

        let calendar;
        let currentEvents = [...allEvents];

        // Initialize calendar
        document.addEventListener('DOMContentLoaded', function() {{
            const calendarEl = document.getElementById('calendar');

            calendar = new FullCalendar.Calendar(calendarEl, {{
                initialView: 'dayGridMonth',
                headerToolbar: {{
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth,timeGridWeek,listMonth'
                }},
                events: currentEvents,
                eventClick: function(info) {{
                    showEventDetails(info.event);
                }},
                height: 'auto',
                eventTimeFormat: {{
                    hour: '2-digit',
                    minute: '2-digit',
                    meridiem: 'short'
                }}
            }});

            calendar.render();

            // Set default filter selections
            const institutionSelect = document.getElementById('institutionTypeFilter');
            const eventSelect = document.getElementById('eventTypeFilter');

            // Default institution types: US_Banks and Canadian_Banks
            Array.from(institutionSelect.options).forEach(option => {{
                if (option.value === 'US_Banks' || option.value === 'Canadian_Banks') {{
                    option.selected = true;
                }}
            }});

            // Default event type: Earnings (groups all earnings-related events)
            Array.from(eventSelect.options).forEach(option => {{
                if (option.value === 'Earnings') {{
                    option.selected = true;
                }}
            }});

            // Apply initial filters
            applyFilters();

            // Attach filter listeners
            institutionSelect.addEventListener('change', applyFilters);
            eventSelect.addEventListener('change', applyFilters);
        }});

        function getSelectedValues(selectElement) {{
            return Array.from(selectElement.selectedOptions).map(option => option.value);
        }}

        function applyFilters() {{
            const institutionSelect = document.getElementById('institutionTypeFilter');
            const eventSelect = document.getElementById('eventTypeFilter');

            const selectedInstitutions = getSelectedValues(institutionSelect);
            const selectedEvents = getSelectedValues(eventSelect);

            currentEvents = allEvents.filter(event => {{
                const matchesInstitution = selectedInstitutions.length === 0 ||
                                         selectedInstitutions.includes(event.institutionType);

                // For event type matching, handle "Earnings" group
                let matchesEvent;
                if (selectedEvents.length === 0) {{
                    matchesEvent = true;
                }} else {{
                    // Check if event matches any selected filter
                    matchesEvent = selectedEvents.some(selectedType => {{
                        if (selectedType === 'Earnings') {{
                            // "Earnings" filter matches any earnings-related type
                            return EARNINGS_TYPES.includes(event.eventType);
                        }} else {{
                            // Other filters match exactly
                            return event.eventType === selectedType;
                        }}
                    }});
                }}

                return matchesInstitution && matchesEvent;
            }});

            calendar.removeAllEvents();
            calendar.addEventSource(currentEvents);
        }}

        function resetFilters() {{
            const institutionSelect = document.getElementById('institutionTypeFilter');
            const eventSelect = document.getElementById('eventTypeFilter');

            // Deselect all options
            Array.from(institutionSelect.options).forEach(option => option.selected = false);
            Array.from(eventSelect.options).forEach(option => option.selected = false);

            applyFilters();
        }}

        function showEventDetails(event) {{
            const modal = document.getElementById('eventModal');
            const modalTitle = document.getElementById('modalTitle');
            const modalBody = document.getElementById('modalBody');

            modalTitle.textContent = event.title;

            let bodyHTML = `
                <div class="event-detail">
                    <div class="event-detail-label">Institution</div>
                    <div class="event-detail-value">${{event.extendedProps.institution}}</div>
                </div>
                <div class="event-detail">
                    <div class="event-detail-label">Event Description</div>
                    <div class="event-detail-value">${{event.extendedProps.description}}</div>
                </div>
                <div class="event-detail">
                    <div class="event-detail-label">Date & Time</div>
                    <div class="event-detail-value">${{event.extendedProps.eventTimeLocal}}</div>
                </div>
            `;

            if (event.extendedProps.fiscalPeriod) {{
                bodyHTML += `
                    <div class="event-detail">
                        <div class="event-detail-label">Fiscal Period</div>
                        <div class="event-detail-value">${{event.extendedProps.fiscalPeriod}} ${{event.extendedProps.fiscalYear}}</div>
                    </div>
                `;
            }}

            if (event.extendedProps.webcastLink) {{
                bodyHTML += `
                    <div class="event-detail">
                        <div class="event-detail-label">Webcast</div>
                        <div class="event-detail-value"><a href="${{event.extendedProps.webcastLink}}" target="_blank">${{event.extendedProps.webcastLink}}</a></div>
                    </div>
                `;
            }}

            if (event.extendedProps.contactInfo) {{
                bodyHTML += `
                    <div class="event-detail">
                        <div class="event-detail-label">Contact Information</div>
                        <div class="event-detail-value">${{event.extendedProps.contactInfo}}</div>
                    </div>
                `;
            }}

            bodyHTML += `
                <button class="download-ics-btn" onclick="downloadICS('${{event.id}}', '${{event.title}}')">
                    📥 Add to Outlook Calendar
                </button>
            `;

            modalBody.innerHTML = bodyHTML;
            modal.classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('eventModal').classList.remove('active');
        }}

        function downloadICS(eventId, eventTitle) {{
            const icsContent = icsData[eventId];
            if (!icsContent) {{
                alert('Calendar file not available for this event');
                return;
            }}

            // Decode base64 and create blob
            const binaryString = atob(icsContent);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            const blob = new Blob([bytes], {{ type: 'text/calendar' }});

            // Create download link
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${{eventId}}.ics`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}

        // Close modal when clicking outside
        document.getElementById('eventModal').addEventListener('click', function(e) {{
            if (e.target === this) {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>"""

    return html


def main():
    """Main execution function."""
    print("=" * 60)
    print("CALENDAR VISUALIZATION GENERATOR")
    print("=" * 60)

    # Setup paths
    script_dir = Path(__file__).parent
    sample_csv = script_dir / "sample_data" / "example_calendar_events.csv"
    output_dir = script_dir / "output"
    output_html = output_dir / "calendar.html"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Check if sample CSV exists
    if not sample_csv.exists():
        print(f"❌ Sample CSV not found: {sample_csv}")
        print("Please ensure example_calendar_events.csv exists in sample_data/")
        return

    print(f"📂 Reading CSV data from: {sample_csv.name}")
    csv_events = read_csv_data(sample_csv)
    print(f"✅ Loaded {len(csv_events)} events")

    print("🔍 Deduplicating earnings events (priority: Earnings > Confirmed > Projected)...")
    csv_events = deduplicate_earnings_events(csv_events)
    print(f"✅ After deduplication: {len(csv_events)} events")

    print("🔄 Converting to calendar format...")
    calendar_events = convert_to_fullcalendar_format(csv_events)

    print("📊 Extracting filter options...")
    institution_types = get_unique_values(csv_events, 'institution_type')
    event_types = get_grouped_event_types(csv_events)  # Groups earnings types together
    print(f"   - {len(institution_types)} institution types")
    print(f"   - {len(event_types)} event type groups (Earnings types grouped)")

    print("🎨 Generating HTML calendar...")
    html_content = generate_html(calendar_events, institution_types, event_types, csv_events)

    print(f"💾 Saving to: {output_html}")
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("=" * 60)
    print("✅ SUCCESS!")
    print(f"📄 Interactive calendar generated: {output_html}")
    print(f"🌐 Open the file in your browser to view")
    print("=" * 60)
    print("\nFeatures:")
    print("  • Interactive monthly/weekly/list views")
    print("  • Filter by institution type and event type")
    print("  • Click events to see details")
    print("  • Download .ics files for Outlook/calendar apps")


if __name__ == "__main__":
    main()
