<div align="center">
  <img src="assets/cyla.png" alt="CYLA Banner" width="800">
  <br><br>
  <h1>CYLA – Self-Organizing Adaptive List</h1>
</div>

# Contributing to CYLA

Thank you for your interest in CYLA — an open-source research prototype exploring lightweight online adaptive reordering in linear search structures.

We welcome contributions of all kinds: bug fixes, documentation, performance optimizations, new features, ablation studies, theoretical analysis, real-dataset experiments, or visualization tools.

### How to Contribute

1. **Found a bug or have a question?**  
   Open an issue with a clear title and description.

2. **Want to add a feature or improvement?**  
   - Open an issue first (label: feature)  
   - Describe the problem/opportunity, proposed solution, and expected impact  
   - Especially welcome: faster convergence, lower overhead, stability at scale.

3. **How to implement?**  
   - Fork the repo sitory 
   - Create a branch: `git checkout -b feature/your-idea`  
   - Make changes  
   - Add/update tests in `test.py`  
   - Run tests: `python test.py`  
   - Commit with clear message: `git commit -m "feat: add aggressive prefix sort"`  
   - Push and open a PR to `main`  
   - Include before/after improvements or plots if applicable in the desc of issue

### Guidelines

- Keep code simple, readable, and framework-free (only NumPy allowed)  
- All contributions are licensed under MIT

### High-Impact Areas

- Theoretical analysis (competitive ratio, amortized cost)  
- Comparisons (MTF, LRU, LFU, etc.)  
- Real datasets (words, queries, products, logs)  
- Visualizations (learning curves, top-k inspection)  
- Performance optimizations (reduce reorder cost)  
- Ports to C++/Rust for embedded/production

Every PR — big or small — helps make CYLA more valuable.
