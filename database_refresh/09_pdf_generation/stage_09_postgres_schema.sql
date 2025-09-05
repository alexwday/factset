-- Stage 09: PDF Generation PostgreSQL Schema
-- This stage generates PDF documents from Stage 8 embeddings data
-- PDFs are stored on NAS, this schema tracks generation metadata

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS earnings_transcripts;

-- Table to track PDF generation metadata
CREATE TABLE IF NOT EXISTS earnings_transcripts.stage_09_pdf_metadata (
    id SERIAL PRIMARY KEY,
    
    -- Transcript identifiers
    filename VARCHAR(255) NOT NULL,
    event_id VARCHAR(50),
    ticker VARCHAR(20) NOT NULL,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    
    -- PDF generation details
    pdf_filename VARCHAR(255) NOT NULL,
    pdf_path TEXT NOT NULL,
    pdf_size_bytes INTEGER,
    pages_count INTEGER,
    
    -- Generation metadata
    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generation_duration_ms INTEGER,
    
    -- Content statistics
    total_paragraphs INTEGER,
    md_speaker_blocks INTEGER,
    qa_blocks INTEGER,
    qa_speaker_blocks INTEGER,
    
    -- Status tracking
    generation_status VARCHAR(50) DEFAULT 'pending',  -- pending, completed, failed
    error_message TEXT,
    
    -- Indexing for lookups
    CONSTRAINT uk_filename_pdf UNIQUE(filename),
    INDEX idx_ticker (ticker),
    INDEX idx_fiscal_period (fiscal_year, fiscal_quarter),
    INDEX idx_generation_status (generation_status),
    INDEX idx_generation_timestamp (generation_timestamp)
);

-- Table to track PDF content structure
CREATE TABLE IF NOT EXISTS earnings_transcripts.stage_09_pdf_structure (
    id SERIAL PRIMARY KEY,
    pdf_metadata_id INTEGER REFERENCES earnings_transcripts.stage_09_pdf_metadata(id) ON DELETE CASCADE,
    
    -- Section information
    section_name VARCHAR(100) NOT NULL,  -- 'Management Discussion' or 'Q&A'
    section_page_start INTEGER NOT NULL,
    section_page_end INTEGER NOT NULL,
    
    -- Content organization
    speaker_count INTEGER,
    paragraph_count INTEGER,
    qa_block_count INTEGER,  -- NULL for MD section
    
    -- Layout details
    avg_paragraphs_per_page DECIMAL(10, 2),
    page_breaks_count INTEGER,
    
    INDEX idx_pdf_metadata (pdf_metadata_id),
    INDEX idx_section (section_name)
);

-- Table to track generation errors and retries
CREATE TABLE IF NOT EXISTS earnings_transcripts.stage_09_generation_errors (
    id SERIAL PRIMARY KEY,
    
    -- Transcript identifiers
    filename VARCHAR(255) NOT NULL,
    ticker VARCHAR(20),
    
    -- Error details
    error_type VARCHAR(50) NOT NULL,  -- pdf_generation, formatting, processing, validation
    error_message TEXT NOT NULL,
    error_details JSONB,
    
    -- Tracking
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    
    INDEX idx_filename_errors (filename),
    INDEX idx_error_type (error_type),
    INDEX idx_occurred_at (occurred_at),
    INDEX idx_resolved (resolved)
);

-- Table to track batch processing runs
CREATE TABLE IF NOT EXISTS earnings_transcripts.stage_09_batch_runs (
    id SERIAL PRIMARY KEY,
    
    -- Batch details
    batch_id UUID DEFAULT gen_random_uuid(),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    
    -- Processing statistics
    total_transcripts INTEGER,
    successful_pdfs INTEGER,
    failed_pdfs INTEGER,
    
    -- Performance metrics
    avg_generation_time_ms DECIMAL(10, 2),
    total_pages_generated INTEGER,
    total_size_bytes BIGINT,
    
    -- Configuration
    dev_mode BOOLEAN DEFAULT FALSE,
    dev_max_transcripts INTEGER,
    
    -- Status
    status VARCHAR(50) DEFAULT 'running',  -- running, completed, failed
    
    INDEX idx_batch_id (batch_id),
    INDEX idx_start_time (start_time),
    INDEX idx_status (status)
);

-- View to get latest PDF generation status for each transcript
CREATE OR REPLACE VIEW earnings_transcripts.v_stage_09_latest_pdfs AS
SELECT 
    pm.*,
    CASE 
        WHEN pm.generation_status = 'completed' THEN 'Available'
        WHEN pm.generation_status = 'failed' THEN 'Generation Failed'
        ELSE 'Pending'
    END as pdf_availability,
    br.batch_id as last_batch_id,
    br.start_time as last_batch_time
FROM earnings_transcripts.stage_09_pdf_metadata pm
LEFT JOIN earnings_transcripts.stage_09_batch_runs br 
    ON pm.generation_timestamp >= br.start_time 
    AND pm.generation_timestamp <= COALESCE(br.end_time, CURRENT_TIMESTAMP)
WHERE pm.id IN (
    SELECT MAX(id) 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    GROUP BY filename
);

-- View to get PDF generation statistics by fiscal period
CREATE OR REPLACE VIEW earnings_transcripts.v_stage_09_period_stats AS
SELECT 
    fiscal_year,
    fiscal_quarter,
    COUNT(DISTINCT filename) as total_transcripts,
    COUNT(DISTINCT CASE WHEN generation_status = 'completed' THEN filename END) as completed_pdfs,
    COUNT(DISTINCT CASE WHEN generation_status = 'failed' THEN filename END) as failed_pdfs,
    AVG(pages_count) as avg_pages_per_pdf,
    AVG(pdf_size_bytes) as avg_pdf_size_bytes,
    SUM(total_paragraphs) as total_paragraphs_processed,
    MAX(generation_timestamp) as last_generated
FROM earnings_transcripts.stage_09_pdf_metadata
GROUP BY fiscal_year, fiscal_quarter
ORDER BY fiscal_year DESC, fiscal_quarter DESC;

-- Function to get PDF generation summary
CREATE OR REPLACE FUNCTION earnings_transcripts.get_stage_09_summary()
RETURNS TABLE (
    metric_name VARCHAR(100),
    metric_value TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'Total PDFs Generated'::VARCHAR(100), 
           COUNT(*)::TEXT 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    WHERE generation_status = 'completed'
    
    UNION ALL
    
    SELECT 'Total Pages Created'::VARCHAR(100), 
           COALESCE(SUM(pages_count), 0)::TEXT 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    WHERE generation_status = 'completed'
    
    UNION ALL
    
    SELECT 'Average Pages per PDF'::VARCHAR(100), 
           ROUND(AVG(pages_count), 1)::TEXT 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    WHERE generation_status = 'completed'
    
    UNION ALL
    
    SELECT 'Total Storage Used (MB)'::VARCHAR(100), 
           ROUND(SUM(pdf_size_bytes) / 1024.0 / 1024.0, 2)::TEXT 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    WHERE generation_status = 'completed'
    
    UNION ALL
    
    SELECT 'Failed Generations'::VARCHAR(100), 
           COUNT(*)::TEXT 
    FROM earnings_transcripts.stage_09_pdf_metadata 
    WHERE generation_status = 'failed'
    
    UNION ALL
    
    SELECT 'Latest Generation'::VARCHAR(100), 
           COALESCE(MAX(generation_timestamp)::TEXT, 'Never') 
    FROM earnings_transcripts.stage_09_pdf_metadata;
END;
$$ LANGUAGE plpgsql;

-- Function to retry failed PDF generations
CREATE OR REPLACE FUNCTION earnings_transcripts.retry_failed_pdfs()
RETURNS TABLE (
    filename VARCHAR(255),
    ticker VARCHAR(20),
    retry_status TEXT
) AS $$
BEGIN
    -- Mark failed PDFs for retry
    UPDATE earnings_transcripts.stage_09_pdf_metadata
    SET generation_status = 'pending'
    WHERE generation_status = 'failed';
    
    -- Return list of PDFs marked for retry
    RETURN QUERY
    SELECT 
        pm.filename,
        pm.ticker,
        'Marked for retry'::TEXT as retry_status
    FROM earnings_transcripts.stage_09_pdf_metadata pm
    WHERE pm.generation_status = 'pending'
    ORDER BY pm.ticker;
END;
$$ LANGUAGE plpgsql;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_pdf_metadata_composite 
ON earnings_transcripts.stage_09_pdf_metadata(ticker, fiscal_year, fiscal_quarter, generation_status);

CREATE INDEX IF NOT EXISTS idx_pdf_structure_composite 
ON earnings_transcripts.stage_09_pdf_structure(pdf_metadata_id, section_name);

-- Comments
COMMENT ON TABLE earnings_transcripts.stage_09_pdf_metadata IS 
'Tracks PDF generation metadata for earnings transcripts from Stage 8 embeddings data';

COMMENT ON TABLE earnings_transcripts.stage_09_pdf_structure IS 
'Stores PDF content structure and layout information for generated documents';

COMMENT ON TABLE earnings_transcripts.stage_09_generation_errors IS 
'Logs errors encountered during PDF generation for debugging and retry purposes';

COMMENT ON TABLE earnings_transcripts.stage_09_batch_runs IS 
'Tracks batch processing runs for PDF generation with performance metrics';

COMMENT ON VIEW earnings_transcripts.v_stage_09_latest_pdfs IS 
'Shows the latest PDF generation status for each transcript';

COMMENT ON VIEW earnings_transcripts.v_stage_09_period_stats IS 
'Provides PDF generation statistics grouped by fiscal period';

-- Sample queries

-- Get all PDFs for a specific ticker
-- SELECT * FROM earnings_transcripts.v_stage_09_latest_pdfs WHERE ticker = 'AAPL' ORDER BY fiscal_year DESC, fiscal_quarter DESC;

-- Get generation statistics for current year
-- SELECT * FROM earnings_transcripts.v_stage_09_period_stats WHERE fiscal_year = EXTRACT(YEAR FROM CURRENT_DATE);

-- Check failed PDFs
-- SELECT filename, ticker, error_message FROM earnings_transcripts.stage_09_pdf_metadata WHERE generation_status = 'failed';

-- Get overall summary
-- SELECT * FROM earnings_transcripts.get_stage_09_summary();