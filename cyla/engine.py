import numpy as np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
HIDDEN_SIZE   = 12
INPUT_SIZE    = 4
OUTPUT_SIZE   = 1
MOMENTUM      = 0.90
LR            = 0.09
WEIGHT_CLIP   = 7.0
REWARD_SCALE  = 1.0
RE_RANK_EVERY = 4
PREFIX_RATIO  = 0.22


# ---------------------------------------------------------------------------
# Neural scorer
# ---------------------------------------------------------------------------
class CylaX1:

    def __init__(self):
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        self._h = np.tanh(X @ self.W1 + self.b1)
        return (self._h @ self.W2 + self.b2).ravel()

    def update(self, features: np.ndarray, reward: float, lr: float = LR):
        features = np.asarray(features, dtype=float)

        h     = np.tanh(features @ self.W1 + self.b1)
        score = (h @ self.W2 + self.b2).ravel()
        error = REWARD_SCALE * reward - score

        dW2 = np.outer(h, error)
        db2 = error
        dh  = (error @ self.W2.T) * (1.0 - h ** 2)
        dW1 = np.outer(features, dh)
        db1 = dh

        self.vW1 = MOMENTUM * self.vW1 + (1 - MOMENTUM) * dW1
        self.vb1 = MOMENTUM * self.vb1 + (1 - MOMENTUM) * db1
        self.vW2 = MOMENTUM * self.vW2 + (1 - MOMENTUM) * dW2
        self.vb2 = MOMENTUM * self.vb2 + (1 - MOMENTUM) * db2

        self.W1 = np.clip(self.W1 + lr * self.vW1, -WEIGHT_CLIP, WEIGHT_CLIP)
        self.b1 = self.b1 + lr * self.vb1
        self.W2 = np.clip(self.W2 + lr * self.vW2, -WEIGHT_CLIP, WEIGHT_CLIP)
        self.b2 = self.b2 + lr * self.vb2


# ---------------------------------------------------------------------------
# Self-organising search list
# ---------------------------------------------------------------------------
class AdaptiveList:

    def __init__(self, data):
        self.data          = list(data)
        self._n            = len(data)
        self.counts        = {x: 0 for x in self.data}
        self.last_seen     = {x: 0 for x in self.data}
        self.timer         = 0
        self.re_rank_every = RE_RANK_EVERY
        self.prefix_ratio  = PREFIX_RATIO
        self.scorer        = CylaX1()

    def _get_features(self, item) -> np.ndarray:
        max_count = max(self.counts.values()) + 1e-9
        c         = self.counts[item]
        age       = self.timer - self.last_seen[item]
        return np.array([
            c / max_count,
            1.0 / (age + 1),
            np.log1p(c) / 10.0,
            age / (self._n * 2 + 1)
        ])

    def _maybe_rerank(self):
        if self.timer % self.re_rank_every != 0:
            return
        k             = max(1, int(self._n * self.prefix_ratio))
        prefix        = self.data[:k]
        feats         = np.array([self._get_features(x) for x in prefix])
        scores        = self.scorer.forward(feats)
        sorted_idx    = np.argsort(scores)[::-1]
        self.data[:k] = [prefix[i] for i in sorted_idx]

    def search(self, target) -> tuple[int, int]:
        self.timer += 1
        self._maybe_rerank()

        for pos, item in enumerate(self.data):
            if item != target:
                continue

            features = self._get_features(item)
            reward   = 10.0 / (pos ** 1.5 + 0.1)

            if pos >= max(1, int(self._n * self.prefix_ratio)):
                self.data.pop(pos)
                self.data.insert(0, item)

            self.counts[item]   += 1
            self.last_seen[item] = self.timer

            self.scorer.update(features, reward)

            return pos, pos + 1

        return -1, self._n