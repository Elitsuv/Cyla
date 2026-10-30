"""
Compares Cyla's AdaptiveList against a static (unreordered) list on a Zipfian workload.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyla.engine import AdaptiveList
from cyla.utils import generate_zipf_queries, shuffle_items, compare


class StaticList:
    """Fixed-order linear scan baseline."""
    def __init__(self, data):
        self.data = list(data)

    def search(self, target):
        try:
            pos = self.data.index(target)
            return pos, pos + 1
        except ValueError:
            return -1, len(self.data)


def main():
    # 100 elements, 2000 queries following Zipfian distribution (alpha=1.4)
    items = [f"item_{i}" for i in range(100)]
    shuffled_items = shuffle_items(items)
    queries = generate_zipf_queries(items, n=2000, alpha=1.4)

    static_searcher = StaticList(shuffled_items)
    cyla_searcher = AdaptiveList(shuffled_items)

    print("Running 2,000 queries (Zipf alpha=1.4) on 100 items starting from shuffled order...")
    compare(cyla_searcher, static_searcher, queries)


if __name__ == "__main__":
    main()
