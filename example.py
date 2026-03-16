from cyla.engine import main
items = [
    "iphone-15", "airpods-pro", "macbook-pro", "ipad-air",
    "apple-watch", "homepod-mini", "apple-tv", "magic-mouse",
    "usb-c-cable", "charger", "iphone-case", "wireless-charger"
]

# Create the adaptive search list
searcher = main(items)

print("Initial order:")
print(searcher.data[:6], "...")  # show first few
print("-" * 60)

queries = [
    "iphone-15", "airpods-pro", "iphone-15", "macbook-pro",
    "airpods-pro", "iphone-15", "airpods-pro", "ipad-air",
    "iphone-15", "iphone-15", "airpods-pro", "iphone-15",
    "macbook-pro", "airpods-pro", "iphone-15", "iphone-15",
    "charger", "iphone-15", "airpods-pro", "iphone-15"
]

for q in queries:
    idx, steps = searcher.search(q)
    if idx != -1:
        print(f"Search '{q}': found at position {idx} (steps: {steps})")

print("-" * 60)
print("Final order (after learning):")
print(searcher.data[:6], "...")  # most accessed/recent items should be at front

query = "airpods-pro"
idx, steps = searcher.search(query)
print(f"\nNew search for '{query}': found at index {idx}, steps: {steps}")
print("Top 5 learned items:", searcher.data[:5])