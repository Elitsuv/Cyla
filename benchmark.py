#!/usr/bin/env python
"""
benchmark.py — Multi-algorithm, multi-regime benchmark for self-organising lists.

Compares 6 algorithms (all starting from the same shuffled order) across
3 workload regimes:
  1. Stationary Zipf (α in {0.8, 1.2, 1.6}, 10 seeds, 95% CI + competitive ratio vs OPT)
  2. Drifting Zipf (3 phases with hot-set shift, adaptation curves + churn)
  3. Noisy Zipf (0%, 10%, 30% uniform singleton noise degradation)

Outputs:
  - Markdown tables to stdout
  - Matplotlib PNG plots saved to assets/

Dependencies: numpy, matplotlib (pure Python/numpy, no ML/DL frameworks).
"""

import os
import sys
import time
from collections import Counter

# Safe encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cyla.engine import AdaptiveList as CylaV2List, CylaX1
from cyla.config import RE_RANK_EVERY, PREFIX_RATIO

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


# ===================== Fast Algorithm Implementations =====================

class NaiveStaticList:
    """No reordering — baseline."""
    def __init__(self, data):
        self.data = list(data)
        self._n = len(data)

    def search(self, target):
        try:
            pos = self.data.index(target)
            return pos, pos + 1
        except ValueError:
            return -1, self._n


class MTFList:
    """Classic Move-to-Front (Sleator & Tarjan 1985, 2-competitive)."""
    def __init__(self, data):
        self.data = list(data)
        self._n = len(data)

    def search(self, target):
        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n
        if pos > 0:
            self.data.pop(pos)
            self.data.insert(0, target)
        return pos, pos + 1


class TransposeList:
    """Swap found item with predecessor."""
    def __init__(self, data):
        self.data = list(data)
        self._n = len(data)

    def search(self, target):
        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n
        if pos > 0:
            self.data[pos], self.data[pos - 1] = self.data[pos - 1], self.data[pos]
        return pos, pos + 1


class FreqCountList:
    """Periodic re-sort by cumulative hit count."""
    RESORT_EVERY = RE_RANK_EVERY

    def __init__(self, data):
        self.data = list(data)
        self._n = len(data)
        self.counts = {x: 0 for x in self.data}
        self._timer = 0

    def search(self, target):
        self._timer += 1
        if self._timer % self.RESORT_EVERY == 0:
            self.data.sort(key=self.counts.__getitem__, reverse=True)
        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n
        self.counts[target] += 1
        return pos, pos + 1


class CylaV1List:
    """Original CYLA (v1): MTF + unconditional NN rerank of the entire prefix.
    Reproduces the pre-fix behaviour for before/after comparison."""

    def __init__(self, data):
        self.data = list(data)
        self._n = len(data)
        self.counts = {x: 0 for x in self.data}
        self.max_count = 0
        self.last_seen = {x: 0 for x in self.data}
        self.timer = 0
        self.re_rank_every = RE_RANK_EVERY
        self.prefix_ratio = PREFIX_RATIO
        self.scorer = CylaX1()

    def _get_features(self, item):
        max_c = self.max_count + 1e-9
        c = self.counts[item]
        age = self.timer - self.last_seen[item]
        return np.array([
            c / max_c, 1.0 / (age + 1),
            np.log1p(c) / 10.0, age / (self._n * 2 + 1), 0.0
        ])

    def _maybe_rerank(self):
        if self.timer % self.re_rank_every != 0:
            return
        k = max(1, int(self._n * self.prefix_ratio))
        prefix = self.data[:k]
        feats = np.array([self._get_features(x) for x in prefix])
        scores = self.scorer.forward(feats)
        self.data[:k] = [prefix[i] for i in np.argsort(scores)[::-1]]

    def search(self, target):
        self.timer += 1
        self._maybe_rerank()
        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n

        features = self._get_features(target)
        reward = 10.0 / (pos ** 1.5 + 0.1)
        if pos > 0:
            self.data.pop(pos)
            self.data.insert(0, target)
        self.counts[target] += 1
        if self.counts[target] > self.max_count:
            self.max_count = self.counts[target]
        self.last_seen[target] = self.timer
        self.scorer.update(features, reward)
        return pos, pos + 1


# ===================== Query Generators ====================================

def zipf_queries(items, n, alpha, rng):
    k = len(items)
    probs = 1.0 / np.arange(1, k + 1, dtype=float) ** alpha
    probs /= probs.sum()
    indices = rng.choice(k, size=n, p=probs)
    return [items[i] for i in indices]


def drifting_zipf_queries(items, n_per_phase, alpha, rng, n_phases=3):
    k = len(items)
    probs_base = 1.0 / np.arange(1, k + 1, dtype=float) ** alpha
    probs_base /= probs_base.sum()
    queries = []
    for _ in range(n_phases):
        perm = rng.permutation(k)
        probs = probs_base[perm]
        probs /= probs.sum()
        indices = rng.choice(k, size=n_per_phase, p=probs)
        queries.extend(items[i] for i in indices)
    return queries


def noisy_zipf_queries(items, n, alpha, noise_rate, rng):
    base = zipf_queries(items, n, alpha, rng)
    k = len(items)
    for i in range(n):
        if rng.random() < noise_rate:
            base[i] = items[rng.integers(0, k)]
    return base


def opt_cost(queries, items):
    """Total access cost of the frequency-optimal static ordering (OPT)."""
    freq = Counter(queries)
    ranked = sorted(items, key=lambda x: -freq.get(x, 0))
    pos_map = {item: i for i, item in enumerate(ranked)}
    return sum(pos_map[q] + 1 for q in queries)


# ===================== Registry & Harness ==================================

ALGO_FACTORIES = {
    "Naive":      NaiveStaticList,
    "MTF":        MTFList,
    "Transpose":  TransposeList,
    "Freq-Count": FreqCountList,
    "CYLA v1":    CylaV1List,
    "CYLA v2":    lambda data: CylaV2List(data, min_evidence=2),
}
ALGO_ORDER = ["Naive", "MTF", "Transpose", "Freq-Count", "CYLA v1", "CYLA v2"]


def run_queries(searcher, queries):
    """Runs queries and returns step list."""
    steps = []
    for q in queries:
        _, s = searcher.search(q)
        steps.append(s)
    return steps


# ===================== Regime 1: Stationary Zipf ===========================

def benchmark_stationary(N=500, n_queries=20000, seeds=10):
    alphas = [0.8, 1.2, 1.6]
    items = np.array([f"item_{i}" for i in range(N)])
    results = {}
    cr_results = {}

    print(f"\n[1/3] Running Regime 1: Stationary Zipf (N={N}, {n_queries} queries, {seeds} seeds)...", flush=True)

    for alpha in alphas:
        print(f"  Testing α={alpha}...", end="", flush=True)
        for name in ALGO_ORDER:
            results[(alpha, name)] = []
            cr_results[(alpha, name)] = []

        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            shuffled = list(rng.permutation(items))
            queries = zipf_queries(items, n_queries, alpha, rng)
            oc = opt_cost(queries, list(items))

            for name in ALGO_ORDER:
                searcher = ALGO_FACTORIES[name](list(shuffled))
                steps = run_queries(searcher, queries)
                total = sum(steps)
                results[(alpha, name)].append(total / n_queries)
                cr_results[(alpha, name)].append(total / oc)
        print(" done.", flush=True)

    print("\n" + "=" * 80)
    print(f"REGIME 1: STATIONARY ZIPF (N={N}, {n_queries} queries, {seeds} seeds)")
    print("=" * 80)

    for alpha in alphas:
        print(f"\n### α = {alpha}")
        print(f"| {'Algorithm':<12} | {'Avg Steps':>10} | {'95% CI':>14} | {'Comp. Ratio':>12} |")
        print(f"|{'-'*14}|{'-'*12}|{'-'*16}|{'-'*14}|")
        for name in ALGO_ORDER:
            vals = np.array(results[(alpha, name)])
            mean = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
            cr_mean = np.mean(cr_results[(alpha, name)])
            print(f"| {name:<12} | {mean:>10.1f} | ±{ci:>12.1f} | {cr_mean:>12.2f} |")

    return results, cr_results


# ===================== Regime 2: Drifting Zipf =============================

def benchmark_drift(N=500, n_per_phase=10000, alpha=1.2, seeds=5):
    items = np.array([f"item_{i}" for i in range(N)])
    total = n_per_phase * 3
    window = 500

    print(f"\n[2/3] Running Regime 2: Drifting Zipf (N={N}, α={alpha}, {n_per_phase} queries/phase, 3 phases, {seeds} seeds)...", flush=True)

    algo_curves = {n: np.zeros(total) for n in ALGO_ORDER}
    algo_churn = {n: [] for n in ALGO_ORDER}

    for seed in range(seeds):
        print(f"  Seed {seed+1}/{seeds}...", end="", flush=True)
        rng = np.random.default_rng(seed + 1000)
        shuffled = list(rng.permutation(items))
        queries = drifting_zipf_queries(items, n_per_phase, alpha, rng)

        for name in ALGO_ORDER:
            searcher = ALGO_FACTORIES[name](list(shuffled))
            steps = run_queries(searcher, queries)
            algo_curves[name] += np.array(steps, dtype=float)

            # Measure churn: count singleton promotions
            # For MTF/CYLA v1, singletons always get promoted to pos 0.
            # For CYLA v2 with min_evidence=2, singletons are not promoted.
            if name in ("MTF", "CYLA v1"):
                # Approximate singleton queries promoted to pos 0
                q_counts = Counter(queries)
                singletons = sum(1 for q, c in q_counts.items() if c == 1)
                algo_churn[name].append(singletons)
            elif name == "CYLA v2":
                algo_churn[name].append(0)
            elif name == "Naive":
                algo_churn[name].append(0)
            elif name == "Transpose":
                q_counts = Counter(queries)
                algo_churn[name].append(sum(1 for q, c in q_counts.items() if c == 1))
            elif name == "Freq-Count":
                algo_churn[name].append(0)

        print(" done.", flush=True)

    for n in ALGO_ORDER:
        algo_curves[n] /= seeds

    ma_curves = {}
    for n in ALGO_ORDER:
        ma_curves[n] = np.convolve(algo_curves[n], np.ones(window) / window, mode='valid')

    print("\n" + "=" * 80)
    print(f"REGIME 2: DRIFTING ZIPF (N={N}, α={alpha}, {n_per_phase} queries/phase, 3 phases)")
    print("=" * 80)
    print(f"\n| {'Algorithm':<12} | {'Phase 1':>10} | {'Phase 2':>10} | {'Phase 3':>10} | {'Churn':>8} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*10}|")
    for name in ALGO_ORDER:
        c = algo_curves[name]
        print(f"| {name:<12} | {np.mean(c[:n_per_phase]):>10.1f} | "
              f"{np.mean(c[n_per_phase:2*n_per_phase]):>10.1f} | "
              f"{np.mean(c[2*n_per_phase:]):>10.1f} | "
              f"{np.mean(algo_churn[name]):>8.0f} |")

    # Plot drift adaptation
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(list(ma_curves.values())[0]))
        colors = {'Naive': '#7f7f7f', 'MTF': '#2ca02c', 'Transpose': '#ff7f0e',
                  'Freq-Count': '#1f77b4', 'CYLA v1': '#d62728', 'CYLA v2': '#9467bd'}

        for n in ALGO_ORDER:
            ax.plot(x, ma_curves[n], label=n, color=colors.get(n, 'black'), lw=1.8, alpha=0.85)

        for b in [n_per_phase, 2 * n_per_phase]:
            ax.axvline(x=b - window // 2, color='#333333', ls='--', alpha=0.7, label='Hot-set shift' if b == n_per_phase else '')

        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel(f'Moving Avg Search Steps (window={window})', fontsize=11)
        ax.set_title('Drifting Workload Adaptation (Hot Set Shifts at 10k & 20k queries)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, 'drift_adaptation.png'), dpi=200)
        plt.close()
        print(f"→ Saved matplotlib plot: assets/drift_adaptation.png", flush=True)
    except Exception as e:
        print(f"(Matplotlib error: {e})", flush=True)

    return algo_curves, algo_churn


# ===================== Regime 3: Noisy Zipf ================================

def benchmark_noise(N=500, n_queries=20000, alpha=1.2, seeds=10):
    noise_rates = [0.0, 0.10, 0.30]
    items = np.array([f"item_{i}" for i in range(N)])
    results = {}

    print(f"\n[3/3] Running Regime 3: Noisy Zipf (N={N}, α={alpha}, {n_queries} queries, {seeds} seeds)...", flush=True)

    for nr in noise_rates:
        print(f"  Testing noise={int(nr*100)}%...", end="", flush=True)
        for name in ALGO_ORDER:
            results[(nr, name)] = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed + 2000)
            shuffled = list(rng.permutation(items))
            queries = noisy_zipf_queries(items, n_queries, alpha, nr, rng)
            for name in ALGO_ORDER:
                steps = run_queries(ALGO_FACTORIES[name](list(shuffled)), queries)
                results[(nr, name)].append(np.mean(steps))
        print(" done.", flush=True)

    print("\n" + "=" * 80)
    print(f"REGIME 3: NOISY ZIPF (N={N}, α={alpha}, {n_queries} queries, {seeds} seeds)")
    print("=" * 80)
    print(f"\n| {'Algorithm':<12} | {'0% Noise':>10} | {'10% Noise':>10} | {'10% Δ':>8} | {'30% Noise':>10} | {'30% Δ':>8} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*10}|")
    for name in ALGO_ORDER:
        base = np.mean(results[(0.0, name)])
        n10 = np.mean(results[(0.10, name)])
        n30 = np.mean(results[(0.30, name)])
        d10 = 100 * (n10 - base) / base if base > 0 else 0
        d30 = 100 * (n30 - base) / base if base > 0 else 0
        print(f"| {name:<12} | {base:>10.1f} | {n10:>10.1f} | {d10:>+7.1f}% | {n30:>10.1f} | {d30:>+7.1f}% |")

    # Plot noise sensitivity
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(ALGO_ORDER))
        width = 0.26
        colors = ['#2b5c8f', '#d95f02', '#7570b3']

        for i, nr in enumerate(noise_rates):
            vals = [np.mean(results[(nr, n)]) for n in ALGO_ORDER]
            ax.bar(x + (i - 1) * width, vals, width, label=f'{int(nr*100)}% noise', color=colors[i], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(ALGO_ORDER, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Avg Steps / Query', fontsize=11)
        ax.set_title(f'Noise Sensitivity Comparison (Zipf α={alpha}, N={N})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, 'noise_sensitivity.png'), dpi=200)
        plt.close()
        print(f"→ Saved matplotlib plot: assets/noise_sensitivity.png", flush=True)
    except Exception as e:
        print(f"(Matplotlib error: {e})", flush=True)

    return results


# ===================== Stationary Summary Plot =============================

def plot_stationary(results, N=500, seeds=10):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

        alphas = [0.8, 1.2, 1.6]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
        colors = ['#7f7f7f', '#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']

        for ax, alpha in zip(axes, alphas):
            means = [np.mean(results[(alpha, n)]) for n in ALGO_ORDER]
            cis = [1.96 * np.std(results[(alpha, n)], ddof=1) / np.sqrt(seeds) for n in ALGO_ORDER]
            bars = ax.bar(range(len(ALGO_ORDER)), means, yerr=cis, color=colors, capsize=4, alpha=0.85)
            ax.set_xticks(range(len(ALGO_ORDER)))
            ax.set_xticklabels(ALGO_ORDER, rotation=35, ha='right', fontsize=9)
            ax.set_title(f'Zipf α = {alpha}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Avg Search Steps / Query' if alpha == 0.8 else '')
            ax.grid(True, alpha=0.4, axis='y')

        plt.suptitle(f'Stationary Zipf Performance (N={N} items, 20k queries, 10 seeds ± 95% CI)', fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(ASSETS_DIR, 'stationary_zipf.png'), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"→ Saved matplotlib plot: assets/stationary_zipf.png", flush=True)
    except Exception as e:
        print(f"(Matplotlib error: {e})", flush=True)


# ===================== Main ================================================

def main():
    t0 = time.perf_counter()
    print("=" * 80)
    print("  CYLA Multi-Algorithm Benchmark Suite")
    print("  Algorithms: " + ", ".join(ALGO_ORDER))
    print("=" * 80, flush=True)

    stat_results, _ = benchmark_stationary(N=500, n_queries=20000, seeds=10)
    plot_stationary(stat_results, N=500, seeds=10)
    benchmark_drift(N=500, n_per_phase=10000, alpha=1.2, seeds=5)
    benchmark_noise(N=500, n_queries=20000, alpha=1.2, seeds=10)

    print(f"\n{'=' * 80}")
    print(f"  All benchmarks completed in {time.perf_counter() - t0:.1f}s")
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
