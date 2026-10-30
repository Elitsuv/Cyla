"""
Utility functions for synthetic query generation, benchmarking, and result display.
"""

from typing import Any, Dict, List
import random
import time
import numpy as np


def generate_zipf_queries(items: List[Any], n: int, alpha: float = 1.5) -> List[Any]:
    """Generate n queries following a power-law (Zipfian) distribution.

    Parameters
    ----------
    items : List[Any]
        Universe of available items.
    n : int
        Number of query requests to generate.
    alpha : float, optional
        Zipfian skew parameter (alpha > 1 indicates heavy skew). Defaults to 1.5.

    Returns
    -------
    List[Any]
        Sequence of queried items.
    """
    k = len(items)
    ranks = np.arange(1, k + 1, dtype=float)
    probs = 1.0 / ranks ** alpha
    probs /= probs.sum()
    return list(np.random.choice(items, size=n, p=probs))


def benchmark(searcher: Any, queries: List[Any]) -> Dict[str, Any]:
    """Evaluate a searcher on a query sequence and return performance statistics.

    Parameters
    ----------
    searcher : Any
        Object implementing a search(target) method returning (pos, steps).
    queries : List[Any]
        Sequence of items to search.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing total queries, step metrics, and elapsed wall-clock time.
    """
    steps_list = []
    start = time.perf_counter()

    for q in queries:
        _, steps = searcher.search(q)
        steps_list.append(steps)

    elapsed = time.perf_counter() - start

    return {
        "total_queries": len(queries),
        "avg_steps": round(sum(steps_list) / len(steps_list), 2),
        "min_steps": min(steps_list) if steps_list else 0,
        "max_steps": max(steps_list) if steps_list else 0,
        "elapsed_sec": round(elapsed, 4),
    }


def compare(searcher: Any, baseline: Any, queries: List[Any]) -> None:
    """Print a side-by-side performance comparison of two searchers."""
    test_stats = benchmark(searcher, queries)
    base_stats = benchmark(baseline, queries)

    if base_stats["avg_steps"] > 0:
        gain = 100 * (base_stats["avg_steps"] - test_stats["avg_steps"]) / base_stats["avg_steps"]
    else:
        gain = 0.0

    print(f"{'Metric':<20} {'Baseline':>10} {'Searcher':>10}")
    print("-" * 44)
    print(f"{'Avg steps':<20} {base_stats['avg_steps']:>10} {test_stats['avg_steps']:>10}")
    print(f"{'Min steps':<20} {base_stats['min_steps']:>10} {test_stats['min_steps']:>10}")
    print(f"{'Max steps':<20} {base_stats['max_steps']:>10} {test_stats['max_steps']:>10}")
    print(f"{'Time (sec)':<20} {base_stats['elapsed_sec']:>10} {test_stats['elapsed_sec']:>10}")
    print("-" * 44)
    print(f"Gain in avg search steps: {gain:+.1f}%")


def print_top(searcher: Any, n: int = 5) -> None:
    """Print the top n items currently at the front of the list."""
    print(f"Top {n} items:", list(searcher)[:n])


def shuffle_items(items: List[Any]) -> List[Any]:
    """Return a randomly shuffled copy of items."""
    shuffled = list(items)
    random.shuffle(shuffled)
    return shuffled
