# OF-FF Dashboard

Interactive Streamlit dashboard for TSN OF-FF defragmentation.

## Run

```bash
pip install streamlit pandas matplotlib
streamlit run dashboard.py
```

## What it shows
- JSON usecase selection and OF-FF controls (`p`, `n`, thresholds `c`)
- Summary cards (moved streams, violations, largest-hole delta)
- Full metric tables (largest hole, gap counts, threshold accommodation)
- Largest-hole-per-port before/after
- Per-stream moved start-time diffs (before vs after)
- Inferred stream ages from compact encoding
- Before/after schedule plots with optional stream/port filters
- Debug report per attempted stream (candidate checks and deltas)
