-- =====================================================
-- Calendar Events Database Schema
-- =====================================================
-- This schema supports loading calendar events CSV data (16 fields)
-- into PostgreSQL for querying, analytics, and calendar visualization.
--
-- Usage:
--   1. Create database: CREATE DATABASE calendar_events;
--   2. Run this schema: psql -d calendar_events -f postgres_schema.sql
--   3. Load CSV data (see COPY command examples below)
--
-- Strategy: Replace - CSV is completely replaced each run
-- =====================================================

-- Drop existing objects if they exist
DROP VIEW IF EXISTS calendar_events_deduplicated CASCADE;
DROP VIEW IF EXISTS upcoming_events_30d CASCADE;
DROP TABLE IF EXISTS calendar_events CASCADE;

-- =====================================================
-- Main Table: calendar_events (16 fields from CSV)
-- =====================================================

CREATE TABLE calendar_events (
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
    CONSTRAINT unique_event UNIQUE (ticker, event_id),
    CONSTRAINT valid_event_type CHECK (event_type IN (
        'Earnings',
        'ConfirmedEarningsRelease',
        'ProjectedEarningsRelease',
        'Dividend',
        'Conference',
        'ShareholdersMeeting',
        'SalesRevenueCall',
        'SalesRevenueMeeting',
        'SalesRevenueRelease',
        'AnalystsInvestorsMeeting',
        'SpecialSituation'
    ))
);

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Date-based queries (most common)
CREATE INDEX idx_calendar_events_date ON calendar_events(event_date);
CREATE INDEX idx_calendar_events_datetime_utc ON calendar_events(event_date_time_utc);

-- Filter queries
CREATE INDEX idx_calendar_events_ticker ON calendar_events(ticker);
CREATE INDEX idx_calendar_events_institution_type ON calendar_events(institution_type);
CREATE INDEX idx_calendar_events_event_type ON calendar_events(event_type);

-- Fiscal period queries (for earnings)
CREATE INDEX idx_calendar_events_fiscal_period ON calendar_events(fiscal_year, fiscal_period)
WHERE fiscal_year IS NOT NULL AND fiscal_year != '';

-- =====================================================
-- Views for Calendar Visualization
-- =====================================================

-- View: Deduplicated events for calendar visualization
-- Matches the deduplication logic in generate_calendar.py
-- Priority: Earnings > ConfirmedEarningsRelease > ProjectedEarningsRelease
CREATE VIEW calendar_events_deduplicated AS
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
    FROM calendar_events
    WHERE event_type IN ('Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease')
),
non_earnings AS (
    SELECT *, 999 as priority, 1 as rn
    FROM calendar_events
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
CREATE VIEW upcoming_events_30d AS
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
FROM calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
  AND event_date_time_utc <= CURRENT_TIMESTAMP + INTERVAL '30 days'
ORDER BY event_date_time_utc;

-- =====================================================
-- Common Queries for Calendar Visualization
-- =====================================================

/*
-- ===== CSV LOADING =====

-- Clear existing data and load fresh CSV (replace strategy)
DELETE FROM calendar_events;

COPY calendar_events (
    event_id, ticker, institution_name, institution_id, institution_type,
    event_type, event_date_time_utc, event_date_time_local, event_date,
    event_time_local, event_headline, webcast_link, contact_info,
    fiscal_year, fiscal_period, data_fetched_timestamp
)
FROM '/path/to/master_calendar_events.csv'
WITH (FORMAT csv, HEADER true);

-- Using psql \copy (works from client machine)
\copy calendar_events (event_id, ticker, institution_name, institution_id, institution_type, event_type, event_date_time_utc, event_date_time_local, event_date, event_time_local, event_headline, webcast_link, contact_info, fiscal_year, fiscal_period, data_fetched_timestamp) FROM 'master_calendar_events.csv' WITH (FORMAT csv, HEADER true)


-- ===== CALENDAR VISUALIZATION QUERIES =====

-- Get deduplicated events for calendar (use this for visualization)
SELECT * FROM calendar_events_deduplicated
WHERE event_date BETWEEN '2025-11-01' AND '2025-11-30'
ORDER BY event_date_time_utc;

-- Filter by institution type (e.g., Canadian Banks and US Banks)
SELECT * FROM calendar_events_deduplicated
WHERE institution_type IN ('Canadian_Banks', 'US_Banks')
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- Filter by event type (e.g., Earnings only)
SELECT * FROM calendar_events_deduplicated
WHERE event_type IN ('Earnings', 'ConfirmedEarningsRelease', 'ProjectedEarningsRelease')
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- Get all upcoming events in next 30 days
SELECT * FROM upcoming_events_30d;

-- Get events by fiscal period
SELECT
    fiscal_year,
    fiscal_period,
    institution_type,
    COUNT(*) as event_count
FROM calendar_events_deduplicated
WHERE fiscal_year IS NOT NULL AND fiscal_year != ''
  AND fiscal_period IS NOT NULL AND fiscal_period != ''
GROUP BY fiscal_year, fiscal_period, institution_type
ORDER BY fiscal_year DESC, fiscal_period, institution_type;

-- Get event counts by type
SELECT
    event_type,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as upcoming_events
FROM calendar_events_deduplicated
GROUP BY event_type
ORDER BY total_events DESC;
*/

