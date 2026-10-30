"""
Basic usage example for Cyla's AdaptiveList.
Demonstrates linear search with online Move-to-Front adaptation.
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyla.engine import AdaptiveList

def main():
    items = [
        "iphone-15", "airpods-pro", "macbook-pro", "ipad-air",
        "apple-watch", "homepod-mini", "apple-tv", "magic-mouse",
        "usb-c-cable", "charger", "iphone-case", "wireless-charger"
    ]

    list_mgr = AdaptiveList(items)

    print("Initial list prefix (first 5 elements):")
    print(list_mgr[:5])
    print("-" * 60)

    # First search for 'charger' (near end of list, pos 9)
    pos, steps = list_mgr.search("charger")
    print(f"First search for 'charger': found at index {pos} ({steps} steps)")

    # Repeated queries for 'charger'
    for _ in range(3):
        pos, steps = list_mgr.search("charger")
    print(f"After repetition, 'charger' index: {pos} ({steps} step)")

    # Interleaved queries for 'airpods-pro' and 'iphone-15'
    workload = ["airpods-pro", "iphone-15", "airpods-pro", "iphone-15"]
    for item in workload:
        pos, steps = list_mgr.search(item)
        print(f"Search '{item}': found at index {pos} ({steps} steps)")

    print("-" * 60)
    print("Reorganized list prefix (first 5 elements):")
    print(list_mgr[:5])
    print(f"Access count for 'charger': {list_mgr.counts['charger']}")
    print(f"Access count for 'iphone-15': {list_mgr.counts['iphone-15']}")

if __name__ == "__main__":
    main()
