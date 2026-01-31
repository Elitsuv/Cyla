# CYLA – Adaptive Self-Learning Search Prototype
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**v0.1.0-beta** – Early prototype of a self-adjusting linear search that learns better access order from frequency + recency signals.

> **Current status (transparent):**  
> Still slower than naive linear scan (≈ -67% efficiency on Zipf workloads). Work in progress.

## Overview

CYLA explores **online adaptive reordering** of a linear list:
- Small hot zone for fast first checks
- Momentum-based priority updates
- NumPy-accelerated
- Passing unit tests

## Performance (5k items, 20k Zipf α≈1.5 queries)

| Method            | Avg steps/query | Efficiency vs naive |
|-------------------|-----------------|---------------------|
| Naive linear scan | ~138–145        | baseline            |
| CYLA (current)    | ~220–250        | **-67%**            |

→ Learning is active, but convergence needs improvement.

## Requirements

```bash
Python ≥ 3.8
pip install numpy
```

```bash
git clone https://github.com/Elitsuv/cyla.git
cd cyla
pip install numpy
```
