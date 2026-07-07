# Dataset Report: CMU Keystroke Dynamics Benchmark

## Summary

| Metric | Value |
|---|---:|
| Users | 51 |
| Samples | 20400 |
| Features | 31 |
| Missing values | 0 |
| Min samples per user | 400 |
| Max samples per user | 400 |

## Feature Groups

| Group | Description | Count |
|---|---|---:|
| H.* | Hold / dwell time | 11 |
| DD.* | Press-press / down-down latency | 10 |
| UD.* | Release-press / up-down latency | 10 |

## Internal Format

The processed dataset uses the following metadata columns:

- user_id
- session_index
- rep
- sample_id

Feature columns are preserved from the original CMU dataset.

## First Feature Columns

- H.period
- DD.period.t
- UD.period.t
- H.t
- DD.t.i
- UD.t.i
- H.i
- DD.i.e
- UD.i.e
- H.e
- DD.e.five
- UD.e.five
- H.five
- DD.five.Shift.r
- UD.five.Shift.r
- H.Shift.r
- DD.Shift.r.o
- UD.Shift.r.o
- H.o
- DD.o.a
- ... 11 more
