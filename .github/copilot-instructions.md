# Copilot / AI Agent Instructions — Stock-monitoring-dashboard

Purpose: give an AI coding agent the minimal, concrete knowledge to be productive editing and extending this repo.

Quick start
- Install environment: `pip install -r requirements.txt` (repo root contains `requirements.txt`).
- Run the app: `streamlit run Home.py` (Home.py is the main entry; `pages/Stock_Detail.py` is the details page).

High-level architecture
- Single-process Streamlit dashboard. Entry: `Home.py` (main UI), additional tabbed pages under `pages/` (e.g., `pages/Stock_Detail.py`).
- Shared utility functions live in `utils.py` (data fetch, timezone normalization, and plotting helpers).
- Persistent data lives in `stock_data/` as CSV files created/updated by `utils.fetch_data` and (older) notebook code.

Data flow and persistence (canonical path)
- UI calls `utils.fetch_data(symbol)` which returns `(DataFrame, message)` and also writes a CSV.
- Canonical CSV naming used by `utils`: `stock_data/{symbol.replace('.', '_')}.csv`. Example: `S58.SI` -> `stock_data/S58_SI.csv`.
- `fetch_data` normalizes timestamps to UTC then converts to `Asia/Singapore` tz; plotting functions expect a tz-aware `Datetime` column.
- `fetch_data` handles yfinance MultiIndex columns and forces OHLCV to numeric. New rows are appended, and duplicates are removed using `Datetime` deduplication.

Important conventions & gotchas (do not assume defaults)
- Timezone: all components expect `Asia/Singapore` (SGT). If you change timezone handling, update `utils.fetch_data` and every plotting function that calls `.dt.tz_convert`.
- Datetime dtype: many functions require a tz-aware `Datetime` column. Use `pd.to_datetime(..., utc=True).dt.tz_convert('Asia/Singapore')` when ingesting.
- yfinance quirks: code handles `pd.MultiIndex` columns by collapsing to single-level names. Keep this behavior when adding alternate fetchers.
- Two naming patterns exist in the repo: the notebook `Financial Data Science Mastery.ipynb` uses per-stock `*_data_daily.csv` and `*_data_intraday.csv`, while `utils.fetch_data` uses a single `{symbol.replace('.', '_')}.csv`. Treat `utils.py` as canonical for the running app; the notebook is exploratory and contains a different approach.

Where to look for examples
- `utils.py` — canonical fetch, timezone normalization, plotting helpers (`plot_intraday_line`, `plot_candlestick_vwap`, `plot_detector`, `plot_3mths_candlestick`).
- `Home.py` — how the UI uses `fetch_data`, `plot_intraday_line`, and `st.session_state.stocks` to drive pages.
- `pages/Stock_Detail.py` — tabbed per-stock detailed views and how plotting functions are composed.
- `Financial Data Science Mastery.ipynb` — alternative data pipelines (daily vs intraday CSVs), useful reference but not the primary runtime flow.

Debugging tips / quick checks
- If visuals are blank: open `stock_data/` CSV for the symbol and check `Datetime` dtype and timezone. Use `pd.read_csv(path, parse_dates=['Datetime'])`.
- If no new data: `yfinance` limits intraday backfill (~7 days in some endpoints). The notebook explicitly limits intraday start to `end_date - timedelta(days=7)`; the app relies on `period`/`interval` args to `yf.download`.
- If plots error: ensure OHLCV columns are numeric; `utils._force_numeric` demonstrates the pattern to coerce columns.

Editing guidance for agents
- Preferred edit targets: `utils.py` for data/plotting changes, `Home.py` and `pages/Stock_Detail.py` for UI changes.
- When modifying CSV format, update every consumer (fetch, plotting, and notebook). Add unit-checks or assertions validating `Datetime` tz-awareness.
- Keep `DATA_DIR = os.path.join(os.path.dirname(__file__), 'stock_data')` intact to avoid path regressions.

Project-specific commands
- Install deps: `pip install -r requirements.txt`.
- Run app locally: `streamlit run Home.py`.

Tests & CI
- No test suite found. Keep changes minimal and run the app locally to validate visual/functional changes.

When in doubt, follow these heuristics
- Use `utils.fetch_data` as the source of truth for data layout.
- Prefer timezone-aware datetimes everywhere; convert at ingestion rather than at plot time (but plotting helpers already convert as a safety net).

If you need more context or want me to expand any section (examples, code snippets, or convert the notebook's flow into canonical functions), say which part to expand.
