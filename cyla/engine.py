import numpy as np

class CYLA:
    def __init__(self, data, lr=0.4, momentum=0.7):
        self.data = np.array(data)
        self._n = len(data)
        self.lr = lr
        self.momentum = momentum
        self.counts = np.zeros(self._n, dtype=np.float32)
        self.last_seen = np.zeros(self._n, dtype=np.float32)
        self.weights = np.array([0.5, 0.5], dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.timer = 0
        self.re_rank_freq = 5
        self.threshold = 0.4
        self.hot_zone = np.arange(min(self._n, 5))

    def _refresh_priority(self):
        if self.timer == 0:
            return
        f_feat = self.counts / (np.max(self.counts) + 1e-9)
        r_feat = 1.0 / ((self.timer - self.last_seen) + 1.0)
        scores = np.stack([f_feat, r_feat], axis=1) @ self.weights
        k = max(1, int(self._n * self.threshold))
        self.hot_zone = np.argpartition(scores, -k)[-k:]
        self.hot_zone = self.hot_zone[np.argsort(scores[self.hot_zone])[::-1]]

    def search(self, target, force_re_rank=False):
        self.timer += 1
        if force_re_rank or (self.timer % self.re_rank_freq == 0):
            self._refresh_priority()
        hot_indices = self.hot_zone
        hit_mask = (self.data[hot_indices] == target)
        if np.any(hit_mask):
            hit_pos = np.where(hit_mask)[0][0]
            actual_idx = hot_indices[hit_pos]
            steps = hit_pos + 1
            self._update_model(actual_idx, steps)
            return actual_idx, steps
        for i, val in enumerate(self.data):
            if val == target:
                steps = i + len(hot_indices)
                self._update_model(i, steps)
                return i, steps
        return -1, self._n

    def _update_model(self, idx, steps):
        reward = 1.0 / (steps + 1e-5)
        f = self.counts[idx] / (np.max(self.counts) + 1e-9)
        r = 1.0 / ((self.timer - self.last_seen[idx]) + 1.0)
        grad = np.array([f, r])
        self.velocity = self.momentum * self.velocity + self.lr * reward * grad
        self.weights += self.velocity
        self.weights = np.clip(self.weights, -5.0, 5.0)
        self.counts[idx] += 1
        self.last_seen[idx] = self.timer