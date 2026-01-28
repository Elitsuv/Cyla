import time
import numpy as np
from cyla.engine import CYLA


def run_benchmark(n_items=1000, n_queries=2000, zipf_alpha=1.5, print_every=500):
    """
    Benchmark CYLA vs naive linear search on Zipf-distributed queries.
    """
    data = list(range(n_items))
    agent = CYLA(data)                      
    targets = np.random.zipf(zipf_alpha, n_queries)
    targets = np.mod(targets, n_items)      

    print(f"─{'─'*68}")
    print(f" CYLA Benchmark  |  N = {n_items:,} items  |  Queries = {n_queries:,}")
    print(f" Zipf α = {zipf_alpha:.2f}  |  Print every {print_every} queries")
    print(f"─{'─'*68}")

    total_steps_cyla = 0
    total_steps_naive = 0
    start = time.perf_counter()

    for i, target in enumerate(targets, 1):
        _, steps = agent.search(target)
        total_steps_cyla += steps
        total_steps_naive += target + 1       # 1-based position in naive scan

        if i % print_every == 0:
            print(f"  {i:4d} queries → CYLA avg steps: {total_steps_cyla / i:6.2f}")

    elapsed = time.perf_counter() - start

    avg_cyla  = total_steps_cyla / n_queries
    avg_naive = total_steps_naive / n_queries
    gain_pct  = (1 - avg_cyla / avg_naive) * 100

    print(f"\nFinal:")
    print(f"  Naive linear search : {avg_naive:6.2f} steps/query")
    print(f"  CYLA adaptive       : {avg_cyla:6.2f} steps/query")
    print(f"  Efficiency gain     : {gain_pct:5.1f}%")
    print(f"  Total time          : {elapsed:.3f} s")
    print(f"  Throughput          : {n_queries / elapsed:,.0f} q/s")


if __name__ == "__main__":
    run_benchmark(n_items=10_000, n_queries=50_000, zipf_alpha=1.4)