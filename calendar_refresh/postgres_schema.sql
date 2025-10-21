-- =====================================================
-- AEGIS Calendar Events Database Schema
-- =====================================================

-- Main Table: aegis_calendar_events (16 fields from CSV)
-- =====================================================

CREATE TABLE aegis_calendar_events (
    -- Primary Key
    id SERIAL PRIMARY KEY,

    -- Unique Business Key (from FactSet API)
    event_id VARCHAR(100) NOT NULL,
    ticker VARCHAR(20) NOT NULL,

    -- Institution Information
    institution_name VARCHAR(255) NOT NULL,
    institution_id INTEGER NOT NULL,
    institution_type VARCHAR(100) NOT NULL,

    -- Event Details
    event_type VARCHAR(50) NOT NULL,
    event_headline VARCHAR(500),

    -- Event Timing (UTC and Toronto local time)
    event_date_time_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    event_date_time_local TIMESTAMP WITH TIME ZONE,
    event_date DATE NOT NULL,
    event_time_local VARCHAR(20),  -- Format: "HH:MM EST" or "HH:MM EDT"

    -- Webcast Information
    webcast_link TEXT,
    contact_info TEXT,

    -- Fiscal Period Information
    fiscal_year VARCHAR(10),       -- Can be empty string
    fiscal_period VARCHAR(10),     -- Can be empty string (Q1, Q2, Q3, Q4)

    -- Audit Field
    data_fetched_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Constraints
    CONSTRAINT unique_event UNIQUE (ticker, event_id)
);

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Date-based queries (most common)
CREATE INDEX idx_aegis_calendar_events_date ON aegis_calendar_events(event_date);
CREATE INDEX idx_aegis_calendar_events_datetime_utc ON aegis_calendar_events(event_date_time_utc);

-- Filter queries
CREATE INDEX idx_aegis_calendar_events_ticker ON aegis_calendar_events(ticker);
CREATE INDEX idx_aegis_calendar_events_institution_type ON aegis_calendar_events(institution_type);
CREATE INDEX idx_aegis_calendar_events_event_type ON aegis_calendar_events(event_type);

-- Fiscal period queries (for earnings)
CREATE INDEX idx_aegis_calendar_events_fiscal_period ON aegis_calendar_events(fiscal_year, fiscal_period)
WHERE fiscal_year IS NOT NULL AND fiscal_year != '';

-- =====================================================
-- Views for Calendar Visualization
-- =====================================================

-- View: Deduplicated events for calendar visualization
-- Matches the deduplication logic in generate_calendar.py
-- Priority: Earnings > ConfirmedEarningsRelease > ProjectedEarningsRelease
CREATE VIEW aegis_calendar_events_deduplicated AS
WITH earnings_ranked AS (
    SELECT *,
        CASE event_type
            WHEN 'Earnings' THEN 1
            WHEN 'ConfirmedEarningsRelease' THEN 2
            WHEN 'ProjectedEarningsRelease' THEN 3
            ELSE 999
        END as priority,
        ROW_NUMBER() OVER (
            PARTITION BY
                ticker,
                COALESCE(NULLIF(fiscal_year, ''), 'date_' || event_date::TEXT),
                COALESCE(NULLIF(fiscal_period, ''), 'none')
            ORDER BY
                CASE event_type
                    WHEN 'Earnings' THEN 1
                    WHEN 'ConfirmedEarningsRelease' THEN 2
                    WHEN 'ProjectedEarningsRelease' THEN 3
                    ELSE 999
                END
        ) as rn
    FROM aegis_calendar_events
    WHERE event_type IN ('Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease')
),
non_earnings AS (
    SELECT *, 999 as priority, 1 as rn
    FROM aegis_calendar_events
    WHERE event_type NOT IN ('Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease')
)
SELECT
    id, event_id, ticker, institution_name, institution_id, institution_type,
    event_type, event_headline, event_date_time_utc, event_date_time_local,
    event_date, event_time_local, webcast_link, contact_info,
    fiscal_year, fiscal_period, data_fetched_timestamp
FROM (
    SELECT * FROM earnings_ranked WHERE rn = 1
    UNION ALL
    SELECT * FROM non_earnings
) combined
ORDER BY event_date_time_utc;

-- View: Upcoming events in next 30 days
CREATE VIEW aegis_upcoming_events_30d AS
SELECT
    ticker,
    institution_name,
    institution_type,
    event_type,
    event_headline,
    event_date_time_local,
    event_date,
    event_time_local,
    EXTRACT(DAY FROM (event_date_time_utc - CURRENT_TIMESTAMP))::INTEGER as days_until_event,
    webcast_link,
    contact_info,
    fiscal_year,
    fiscal_period
FROM aegis_calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
  AND event_date_time_utc <= CURRENT_TIMESTAMP + INTERVAL '30 days'
ORDER BY event_date_time_utc;


