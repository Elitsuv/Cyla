# Contributing to Cyla

Cyla is an open-source research prototype that evaluated online neural scoring on top of self-organizing lists. As documented in the project results, the neural scoring approach did not demonstrate gains over classic Move-to-Front.

We welcome analytical contributions, alternative algorithmic formulations, formal proofs, dataset evaluations, or bug fixes.

---

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Elitsuv/cyla.git
   cd cyla
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the test suite:
   ```bash
   python test.py
   # or
   python -m unittest discover tests
   ```

4. Run the benchmark suite:
   ```bash
   # Quick smoke run
   python benchmark.py --fast

   # Full 8.1M query evaluation
   python benchmark.py
   ```

---

## Guidelines

- **Simplicity**: Code should remain dependency-minimal (pure Python and NumPy for core logic).
- **Transparency**: Any modifications to scoring or list heuristics must be evaluated across the standard benchmark regimes (`stationary_zipf`, `drifting_workload`, `noise_sensitivity`) with full honesty regarding competitive ratios and CPU overhead.
- **Testing**: Add or update test coverage under `tests/` for any behavioral changes.
- **License**: All contributions are licensed under the MIT License.
