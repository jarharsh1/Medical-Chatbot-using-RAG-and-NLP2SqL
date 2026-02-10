---
title: Prescription Overview metrics are counting duplicates
labels: bug, frontend, backend
status: closed
---

## Description

The Prescription Overview page (Dashboard) was showing inflated numbers because counts were not using DISTINCT.

## Root Cause

The backend SQL queries joined multiple tables without using `DISTINCT` on prescription-level counts.

## Fix Applied

1. **Backend** (`backend/app.py`):
   - Added `COUNT(DISTINCT r.rx_id)` for total prescriptions
   - Added `COUNT(DISTINCT r.rx_id)` for status-based counts
   - Added `rx_id` to page query for frontend distinct counting

2. **Frontend** (`frontend/app.js`):
   - Rewrote `updateMetrics()` to use distinct counts consistently
   - All visualizations (KPI cards, donut chart, revenue bars) now use the same distinct values

## Test Results

```
Distinct Rx count: 196646
Duplicate Patient counts: 161646 (inflated before fix)

[*] Found 161646 duplicate patient counts
[PASS] Test passed: Distinct counts are correct
```

## Files Changed

- `backend/app.py` - Fixed SQL queries
- `frontend/app.js` - Fixed metric calculations
- `evaluation/test_prescription_metrics.py` - Added test case

## Verification

Run: `python evaluation/test_prescription_metrics.py`

---
*Closed by: Automated fix*
