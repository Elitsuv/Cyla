import time
import random
import numpy as np


def generate_zipf_queries(items: list, n: int, alpha: float = 1.5) -> list:
    """
    Generate n queries following a Zipf distribution.
    A few items get searched a lot, most rarely — realistic workload.
    """
    k      = len(items)
    ranks  = np.arange(1, k + 1)
    probs  = 1.0 / ranks ** alpha
    probs /= probs.sum()
    return list(np.random.choice(items, size=n, p=probs))


def benchmark(searcher, queries: list) -> dict:
    """
    Run queries against a searcher and return performance stats.
    """
    steps_list = []
    start      = time.perf_counter()

    for q in queries:
        _, steps = searcher.search(q)
        steps_list.append(steps)

    elapsed = time.perf_counter() - start

    return {
        "total_queries" : len(queries),
        "avg_steps"     : round(sum(steps_list) / len(steps_list), 2),
        "min_steps"     : min(steps_list),
        "max_steps"     : max(steps_list),
        "elapsed_sec"   : round(elapsed, 4),
    }


def compare(searcher, baseline, queries: list) -> None:
    """
    Print a side by side comparison of CYLA vs a naive list.
    """
    cyla_stats   = benchmark(searcher, queries)
    naive_stats  = benchmark(baseline, queries)

    gain = 100 * (naive_stats["avg_steps"] - cyla_stats["avg_steps"]) / naive_stats["avg_steps"]

    print(f"{'Metric':<20} {'Naive':>10} {'CYLA':>10}")
    print("-" * 42)
    print(f"{'Avg steps':<20} {naive_stats['avg_steps']:>10} {cyla_stats['avg_steps']:>10}")
    print(f"{'Min steps':<20} {naive_stats['min_steps']:>10} {cyla_stats['min_steps']:>10}")
    print(f"{'Max steps':<20} {naive_stats['max_steps']:>10} {cyla_stats['max_steps']:>10}")
    print(f"{'Time (sec)':<20} {naive_stats['elapsed_sec']:>10} {cyla_stats['elapsed_sec']:>10}")
    print("-" * 42)
    print(f"Gain: {gain:+.1f}%")


def print_top(searcher, n: int = 5) -> None:
    """Print the top n items currently at the front of the list."""
    print(f"Top {n} items:", searcher.data[:n])


def shuffle_items(items: list) -> list:
    """Return a shuffled copy of items — useful for baseline testing."""
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled