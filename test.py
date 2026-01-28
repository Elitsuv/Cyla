import unittest
import numpy as np
from cyla.engine import CYLA

class TestCYLA(unittest.TestCase):
    def setUp(self):
        self.data = ["assam", "boron", "carbon", "date", "elderberry"]
        self.agent = CYLA(self.data, lr=0.4, momentum=0.7)

    def test_basic_search(self):
        idx, _ = self.agent.search("carbon")
        self.assertEqual(idx, 2)

    def test_learning_convergence(self):
        target = "elderberry"
        _, first_steps = self.agent.search(target)
        for _ in range(25):
            self.agent.search(target)
        _, final_steps = self.agent.search(target, force_re_rank=True)
        self.assertLess(final_steps, first_steps)

    def test_weight_update(self):
        w_init = self.agent.weights.copy()
        self.agent.search("assam")
        self.assertFalse(np.allclose(w_init, self.agent.weights, atol=1e-7))

    def test_miss_gracefully(self):
        idx, steps = self.agent.search("unknown")
        self.assertEqual(idx, -1)
        self.assertEqual(steps, len(self.data))

if __name__ == "__main__":
    unittest.main(verbosity=2)