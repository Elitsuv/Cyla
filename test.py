#!/usr/bin/env python
"""
test.py — Test runner discovering and executing all unit tests in tests/.
"""

import os
import sys
import unittest

if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
    suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
