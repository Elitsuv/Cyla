import unittest
from cyla.engine import AdaptiveList


class TestAdaptiveList(unittest.TestCase):

    def setUp(self):
        self.data  = ["assam", "boron", "carbon", "date", "elderberry"]
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
                        f"Steps did not decrease: {first_steps} → {final_steps}")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)