import unittest
import numpy as np
from cyla.engine import AdaptiveList, CylaX1
from cyla.config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE


class TestCylaX1(unittest.TestCase):
    """Unit tests for the CylaX1 pure-NumPy online neural scorer."""

    def setUp(self):
        self.model = CylaX1()

    def test_forward_shape(self):
        feats = np.random.randn(INPUT_SIZE)
        score = self.model.forward(feats)
        self.assertEqual(score.shape, (1,))

    def test_forward_batch_shape(self):
        batch = np.random.randn(4, INPUT_SIZE)
        scores = self.model.forward(batch)
        self.assertEqual(scores.shape, (4,))

    def test_update_modifies_weights(self):
        feats = np.ones(INPUT_SIZE)
        w1_before = self.model.W1.copy()
        w2_before = self.model.W2.copy()
        self.model.update(feats, reward=5.0)
        self.assertFalse(np.allclose(self.model.W1, w1_before))
        self.assertFalse(np.allclose(self.model.W2, w2_before))


class TestAdaptiveList(unittest.TestCase):
    """Unit tests for AdaptiveList behavior and invariants."""

    def setUp(self):
        self.data = ["assam", "boron", "carbon", "date", "elderberry"]
        self.agent = AdaptiveList(self.data)

    def test_found_correct_index(self):
        idx, steps = self.agent.search("carbon")
        self.assertEqual(idx, 2)
        self.assertEqual(steps, 3)

    def test_found_first_item(self):
        idx, steps = self.agent.search("assam")
        self.assertEqual(idx, 0)
        self.assertEqual(steps, 1)

    def test_found_last_item(self):
        idx, steps = self.agent.search("elderberry")
        self.assertEqual(idx, 4)
        self.assertEqual(steps, 5)

    def test_miss_returns_negative(self):
        idx, steps = self.agent.search("unknown")
        self.assertEqual(idx, -1)
        self.assertEqual(steps, len(self.data))

    def test_count_increments_on_hit(self):
        before = self.agent.counts["assam"]
        self.agent.search("assam")
        self.assertEqual(self.agent.counts["assam"], before + 1)

    def test_count_unchanged_on_miss(self):
        before = self.agent.counts.copy()
        self.agent.search("unknown")
        self.assertEqual(self.agent.counts, before)

    def test_last_seen_updates_on_hit(self):
        self.agent.search("boron")
        self.assertEqual(self.agent.last_seen["boron"], self.agent.timer)

    def test_last_seen_unchanged_on_miss(self):
        before = self.agent.last_seen.copy()
        self.agent.search("unknown")
        self.assertEqual(self.agent.last_seen, before)

    def test_timer_increments_every_search(self):
        for i in range(1, 6):
            self.agent.search("assam")
            self.assertEqual(self.agent.timer, i)

    def test_hot_item_moves_to_front(self):
        for _ in range(20):
            self.agent.search("elderberry")
        self.assertEqual(self.agent.data[0], "elderberry")

    def test_cold_item_stays_back(self):
        for _ in range(20):
            self.agent.search("assam")
        pos = self.agent.data.index("elderberry")
        self.assertGreater(pos, 0)

    def test_steps_decrease_after_repetition(self):
        target = "elderberry"
        _, first_steps = self.agent.search(target)
        for _ in range(50):
            self.agent.search(target)
        _, final_steps = self.agent.search(target)
        self.assertLess(final_steps, first_steps,
                        f"Steps did not decrease: {first_steps} -> {final_steps}")

    def test_all_items_findable(self):
        for item in self.data:
            agent = AdaptiveList(self.data)
            idx, _ = agent.search(item)
            self.assertGreaterEqual(idx, 0, f"'{item}' not found")

    def test_all_items_findable_after_heavy_use(self):
        for _ in range(100):
            self.agent.search("assam")
        for item in self.data:
            idx, _ = self.agent.search(item)
            self.assertGreaterEqual(idx, 0, f"'{item}' lost after heavy use")

    def test_append_new_item(self):
        self.agent.append("fig")
        self.assertIn("fig", self.agent)
        self.assertEqual(self.agent.counts["fig"], 0)
        idx, steps = self.agent.search("fig")
        self.assertEqual(idx, 5)

    def test_append_existing_item_is_noop(self):
        initial_len = len(self.agent)
        self.agent.append("assam")
        self.assertEqual(len(self.agent), initial_len)

    def test_container_dunder_methods(self):
        self.assertEqual(len(self.agent), 5)
        self.assertEqual(self.agent[0], "assam")
        self.assertEqual(list(iter(self.agent)), self.data)
        self.assertTrue("assam" in self.agent)
        self.assertFalse("nonexistent" in self.agent)

    # -- Noise guard & cold-gating --

    def test_min_evidence_gating(self):
        """min_evidence=3: cold-start promotes, sub-threshold does not."""
        agent = AdaptiveList(self.data, min_evidence=3)

        # First-ever hit (cold-start exception) -> promotes immediately
        agent.search("elderberry")
        self.assertEqual(agent.data[0], "elderberry",
                         "Cold-start (first-ever hit) should promote")

        # elderberry count=1; push it back
        agent2 = AdaptiveList(self.data, min_evidence=3)
        agent2.search("elderberry")     # count 0->1, cold-start promotes
        agent2.search("assam")          # pushes assam to front
        agent2.search("boron")          # pushes boron to front

        # count 1->2, below threshold -> should NOT move
        pos_before = agent2.data.index("elderberry")
        agent2.search("elderberry")
        pos_after = agent2.data.index("elderberry")
        self.assertEqual(pos_before, pos_after,
                         "Item below min_evidence should not be promoted")

        # count 2->3, meets threshold -> should promote
        agent2.search("elderberry")
        self.assertEqual(agent2.data[0], "elderberry",
                         "Item at min_evidence threshold should be promoted")

    def test_cold_only_reranking(self):
        """Rerank preserves hot items' MTF order; only cold items are NN-sorted."""
        agent = AdaptiveList(self.data)
        for _ in range(10):
            agent.search("elderberry")
        for _ in range(5):
            agent.search("carbon")

        k = max(1, int(agent._n * agent.prefix_ratio))

        # Align timer so next search triggers rerank
        agent.timer = agent.re_rank_every - 1
        prefix_before = agent.data[:k]
        hot_before = [x for x in prefix_before if agent.counts[x] > 0]

        agent.search("carbon")

        prefix_after = agent.data[:k]
        hot_after = [x for x in prefix_after if agent.counts[x] > 0]

        self.assertEqual(hot_before, hot_after,
                         "Hot items' relative order must be preserved by rerank")


if __name__ == "__main__":
    unittest.main(verbosity=2)

