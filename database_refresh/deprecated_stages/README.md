# Deprecated Stages

This folder contains archived legacy pipeline stages:
- `00_download_historical`
- `01_download_daily`

These stages were removed from the active database refresh workflow.
Active processing now starts at Stage 2 (`02_database_sync`) using pre-populated NAS XML input.
