# Benchmark Results & Methodology

This directory archives the benchmark datasets, raw outputs, and evaluation metrics across 8.1 million simulated queries.

## Structure

- `stationary_zipf.csv`: Average search step comparisons across three Zipfian skew levels ($\alpha \in \{0.8, 1.2, 1.6\}$) with 95% confidence intervals and competitive ratio relative to static offline optimal (OPT).
- `drifting_workload.csv`: Adaptation across 3 discrete phases of 10,000 queries where the rank-popularity distribution is permuted abruptly at each phase boundary.
- `noise_sensitivity.csv`: Search step degradation under 0%, 10%, and 30% injected uniform singleton noise.
- `summary.json`: Machine-readable aggregated statistics and configuration parameters for all experiments.

## Methodology

- **Collection Size**: $N = 500$ elements.
- **Initialization**: All algorithms are initialized with identical randomly shuffled orderings per seed.
- **Seeds**: 10 random seeds for stationary and noise regimes; 5 random seeds for drifting regimes.
- **Evaluation Metric**: Search step count defined as 1-based index where target element was located ($pos + 1$), or $N$ on a miss.

## Algorithms Evaluated

1. **Naive (Static)**: Fixed array with linear scan, no reorganization ($O(1)$ updates).
2. **Move-to-Front (MTF)**: Sleator & Tarjan (1985); moves accessed item to position 0 ($O(1)$ operations on linked list, $O(k)$ on array).
3. **Transpose**: Swaps accessed item with its immediate predecessor ($O(1)$).
4. **Frequency Count (Freq-Count)**: Tracks access counts and periodically sorts by descending frequency.
5. **CYLA v1 (Unbounded Neural)**: Online SGD neural scoring applied across the entire prefix window.
6. **CYLA v2 (Cold-Gated Neural)**: Online SGD neural scoring restricted exclusively to cold items (`count == 0`), preserving MTF order for items with `count > 0`.

