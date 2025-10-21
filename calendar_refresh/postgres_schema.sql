-- =====================================================
-- Calendar Events Database Schema
-- =====================================================
-- This schema supports loading calendar events CSV data
-- into PostgreSQL for querying, analytics, and historical tracking.
--
-- Usage:
--   1. Create database: CREATE DATABASE calendar_events;
--   2. Run this schema: psql -d calendar_events -f postgres_schema.sql
--   3. Load CSV data (see COPY command examples below)
-- =====================================================

-- Drop existing objects if they exist
DROP VIEW IF EXISTS upcoming_events_30d CASCADE;
DROP VIEW IF EXISTS events_by_institution_type CASCADE;
DROP VIEW IF EXISTS quarterly_earnings_calendar CASCADE;
DROP TABLE IF EXISTS calendar_events_archive CASCADE;
DROP TABLE IF EXISTS calendar_events CASCADE;
DROP FUNCTION IF EXISTS update_calendar_events_timestamp() CASCADE;
DROP FUNCTION IF EXISTS archive_old_calendar_events(INTEGER) CASCADE;

-- =====================================================
-- Main Table: calendar_events
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
    event_type VARCHAR(50) NOT NULL DEFAULT 'Earnings',
    event_headline VARCHAR(500),

    -- Event Timing
    -- Stored in both UTC and local timezone (Toronto EST/EDT)
    event_date_time_utc TIMESTAMP WITH TIME ZONE NOT NULL,
    event_date_time_local TIMESTAMP WITH TIME ZONE,
    event_date DATE NOT NULL,
    event_time_local VARCHAR(20),  -- Format: "HH:MM EST" or "HH:MM EDT"

    -- Webcast Information
    webcast_status VARCHAR(50),
    webcast_url TEXT,
    dial_in_info TEXT,

    -- Parsed Event Metadata
    fiscal_year VARCHAR(10),      -- Can be empty string in CSV
    fiscal_quarter VARCHAR(10),   -- Can be empty string in CSV

    -- Audit Field
    data_fetched_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Constraints
    CONSTRAINT unique_event UNIQUE (ticker, event_id),
    CONSTRAINT valid_event_type CHECK (event_type IN ('Earnings', 'Conference', 'Meeting', 'Other'))
);

-- Add comments for documentation
COMMENT ON TABLE calendar_events IS 'Earnings calendar events for monitored financial institutions';
COMMENT ON COLUMN calendar_events.event_id IS 'Unique event identifier from FactSet API';
COMMENT ON COLUMN calendar_events.ticker IS 'Institution ticker symbol';
COMMENT ON COLUMN calendar_events.event_date_time_utc IS 'Event datetime in UTC timezone';
COMMENT ON COLUMN calendar_events.event_date_time_local IS 'Event datetime in Toronto time (EST/EDT)';
COMMENT ON COLUMN calendar_events.data_fetched_timestamp IS 'When this data was fetched from the API';

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Primary lookup by ticker and event_id
CREATE INDEX idx_calendar_events_ticker_event ON calendar_events(ticker, event_id);

-- Filter by event date (most common query pattern)
CREATE INDEX idx_calendar_events_date ON calendar_events(event_date);
CREATE INDEX idx_calendar_events_datetime_utc ON calendar_events(event_date_time_utc);

-- Filter by institution type
CREATE INDEX idx_calendar_events_institution_type ON calendar_events(institution_type);

-- Filter by ticker
CREATE INDEX idx_calendar_events_ticker ON calendar_events(ticker);

-- Filter by fiscal period (when available)
CREATE INDEX idx_calendar_events_fiscal_period ON calendar_events(fiscal_year, fiscal_quarter)
WHERE fiscal_year IS NOT NULL AND fiscal_year != '';

-- Full text search on headlines
CREATE INDEX idx_calendar_events_headline_fts ON calendar_events
USING gin(to_tsvector('english', event_headline));

-- Note: No update trigger needed since we use replace strategy (full refresh each run)

-- =====================================================
-- Archive Table (for historical tracking)
-- =====================================================

CREATE TABLE calendar_events_archive (
    LIKE calendar_events INCLUDING ALL,
    archived_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_reason VARCHAR(100)  -- e.g., 'retention_policy', 'manual_archive'
);

CREATE INDEX idx_calendar_events_archive_archived_at ON calendar_events_archive(archived_at);
CREATE INDEX idx_calendar_events_archive_ticker_event ON calendar_events_archive(ticker, event_id);

COMMENT ON TABLE calendar_events_archive IS 'Historical archive of calendar events for audit trail';

-- =====================================================
-- Views for Common Queries
-- =====================================================

-- View: Upcoming events in next 30 days (recalculated at query time)
CREATE VIEW upcoming_events_30d AS
SELECT
    ticker,
    institution_name,
    institution_type,
    event_headline,
    event_date_time_local,
    event_date,
    event_time_local,
    EXTRACT(DAY FROM (event_date_time_utc - CURRENT_TIMESTAMP))::INTEGER as days_until_event_now,
    webcast_status,
    webcast_url,
    fiscal_year,
    fiscal_quarter
FROM calendar_events
WHERE event_date_time_utc > CURRENT_TIMESTAMP
  AND event_date_time_utc <= CURRENT_TIMESTAMP + INTERVAL '30 days'
ORDER BY event_date_time_utc;

COMMENT ON VIEW upcoming_events_30d IS 'Events scheduled in the next 30 days from current time';

-- View: Events by institution type summary
CREATE VIEW events_by_institution_type AS
SELECT
    institution_type,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as upcoming_events,
    COUNT(*) FILTER (WHERE event_date_time_utc <= CURRENT_TIMESTAMP) as past_events,
    MIN(event_date) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as next_event_date,
    MAX(event_date) FILTER (WHERE event_date_time_utc <= CURRENT_TIMESTAMP) as last_event_date
FROM calendar_events
GROUP BY institution_type
ORDER BY institution_type;

COMMENT ON VIEW events_by_institution_type IS 'Summary statistics of events grouped by institution type';

-- View: Quarterly earnings calendar
CREATE VIEW quarterly_earnings_calendar AS
SELECT
    fiscal_year,
    fiscal_quarter,
    institution_type,
    COUNT(*) as event_count,
    MIN(event_date_time_utc) as earliest_event,
    MAX(event_date_time_utc) as latest_event,
    COUNT(*) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as upcoming_count
FROM calendar_events
WHERE event_type = 'Earnings'
  AND fiscal_year IS NOT NULL
  AND fiscal_year != ''
  AND fiscal_quarter IS NOT NULL
  AND fiscal_quarter != ''
GROUP BY fiscal_year, fiscal_quarter, institution_type
ORDER BY fiscal_year DESC, fiscal_quarter DESC, institution_type;

COMMENT ON VIEW quarterly_earnings_calendar IS 'Earnings events grouped by fiscal period and institution type';

-- =====================================================
-- Functions for Data Management
-- =====================================================

-- Function: Archive old events
CREATE OR REPLACE FUNCTION archive_old_calendar_events(retention_days INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Move events older than retention period to archive
    WITH moved_events AS (
        DELETE FROM calendar_events
        WHERE event_date < CURRENT_DATE - retention_days
        RETURNING *
    )
    INSERT INTO calendar_events_archive (
        id, event_id, ticker, institution_name, institution_id, institution_type,
        event_type, event_headline, event_date_time_utc, event_date_time_local,
        event_date, event_time_local, webcast_status, webcast_url, dial_in_info,
        fiscal_year, fiscal_quarter, data_fetched_timestamp, archived_reason
    )
    SELECT
        id, event_id, ticker, institution_name, institution_id, institution_type,
        event_type, event_headline, event_date_time_utc, event_date_time_local,
        event_date, event_time_local, webcast_status, webcast_url, dial_in_info,
        fiscal_year, fiscal_quarter, data_fetched_timestamp, 'retention_policy'
    FROM moved_events;

    GET DIAGNOSTICS archived_count = ROW_COUNT;

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive_old_calendar_events(INTEGER) IS 'Archive events older than specified days (default 365)';

-- =====================================================
-- CSV Loading Helper Functions
-- =====================================================

-- Function: Clear all events (use before loading fresh CSV)
CREATE OR REPLACE FUNCTION clear_calendar_events()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM calendar_events;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION clear_calendar_events() IS 'Delete all events from calendar_events table (use before CSV reload)';

-- =====================================================
-- Sample Queries and Usage Examples
-- =====================================================

/*
-- ===== CSV LOADING EXAMPLES =====

-- Option 1: Load CSV directly (PostgreSQL server must have access to file)
COPY calendar_events (
    event_id, ticker, institution_name, institution_id, institution_type,
    event_type, event_date_time_utc, event_date_time_local, event_date,
    event_time_local, event_headline, webcast_status, webcast_url,
    dial_in_info, fiscal_year, fiscal_quarter, data_fetched_timestamp
)
FROM '/path/to/master_calendar_events.csv'
WITH (FORMAT csv, HEADER true);

-- Option 2: Clear and reload (replace strategy)
SELECT clear_calendar_events();  -- Returns count of deleted rows
COPY calendar_events (...) FROM '/path/to/master_calendar_events.csv' WITH (FORMAT csv, HEADER true);

-- Option 3: Using psql \copy (works from client machine)
-- Run from command line:
-- psql -d calendar_events -c "\copy calendar_events (event_id, ticker, ...) FROM 'master_calendar_events.csv' WITH (FORMAT csv, HEADER true)"


-- ===== SAMPLE QUERIES =====

-- Get all upcoming earnings for Canadian Banks
SELECT
    ticker,
    institution_name,
    event_headline,
    event_date,
    event_time_local,
    webcast_url
FROM calendar_events
WHERE institution_type = 'Canadian_Banks'
  AND event_type = 'Earnings'
  AND event_date_time_utc > CURRENT_TIMESTAMP
ORDER BY event_date_time_utc;

-- Find events in specific date range
SELECT
    ticker,
    institution_name,
    event_headline,
    event_date,
    days_until_event
FROM calendar_events
WHERE event_date BETWEEN '2025-11-01' AND '2025-11-30'
ORDER BY event_date_time_utc;

-- Get event count by ticker
SELECT
    ticker,
    institution_name,
    COUNT(*) as total_events,
    COUNT(*) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as upcoming_events,
    MIN(event_date) FILTER (WHERE event_date_time_utc > CURRENT_TIMESTAMP) as next_event
FROM calendar_events
GROUP BY ticker, institution_name
ORDER BY upcoming_events DESC, total_events DESC;

-- Search for specific keywords in event headlines
SELECT
    ticker,
    institution_name,
    event_headline,
    event_date
FROM calendar_events
WHERE to_tsvector('english', event_headline) @@ to_tsquery('english', 'Q4 & 2025')
ORDER BY event_date;

-- Get all events happening this week
SELECT
    ticker,
    institution_name,
    event_headline,
    event_date,
    event_time_local
FROM calendar_events
WHERE event_date >= CURRENT_DATE
  AND event_date < CURRENT_DATE + INTERVAL '7 days'
ORDER BY event_date_time_utc;

-- Archive events older than 1 year
SELECT archive_old_calendar_events(365);

-- Get quarterly distribution of events
SELECT
    fiscal_year,
    fiscal_quarter,
    COUNT(*) as event_count
FROM calendar_events
WHERE fiscal_year IS NOT NULL AND fiscal_year != ''
GROUP BY fiscal_year, fiscal_quarter
ORDER BY fiscal_year DESC, fiscal_quarter DESC;
*/

-- =====================================================
-- Grant Permissions (adjust as needed for your setup)
-- =====================================================

-- Example: Grant read-only access to reporting user
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO reporting_user;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO reporting_user;

-- Example: Grant full access to application user
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- =====================================================
-- Maintenance Recommendations
-- =====================================================

/*
-- Run VACUUM ANALYZE after large data loads
VACUUM ANALYZE calendar_events;

-- Reindex if experiencing slow queries
REINDEX TABLE calendar_events;

-- Check table statistics
SELECT
    schemaname,
    tablename,
    n_live_tup as row_count,
    n_dead_tup as dead_rows,
    last_vacuum,
    last_analyze
FROM pg_stat_user_tables
WHERE tablename = 'calendar_events';
*/
