import unittest
from cyla.engine import main

class TestCYLA(unittest.TestCase):
    def setUp(self):
        self.data = ["assam", "boron", "carbon", "date", "elderberry"]
        self.agent = main(self.data)

    def test_basic_search(self):
        idx, steps = self.agent.search("carbon")
        self.assertEqual(idx, 2)
        self.assertEqual(steps, 3)

    def test_learning_convergence(self):
        target = "elderberry"
        _, first_steps = self.agent.search(target)
        for _ in range(50):
            self.agent.search(target)
        _, final_steps = self.agent.search(target)
        self.assertLess(final_steps, first_steps,
                        f"Steps did not decrease: {first_steps} → {final_steps}")

    def test_count_update(self):
        """Search should increment count for accessed item"""
        before = self.agent.counts.copy()
        self.agent.search("assam")
        self.assertNotEqual(list(before), list(self.agent.counts))
        self.assertEqual(self.agent.counts[0], 1)

    def test_miss_gracefully(self):
        idx, steps = self.agent.search("unknown")
        self.assertEqual(idx, -1)
        self.assertEqual(steps, len(self.data))

if __name__ == "__main__":
    unittest.main(verbosity=2)