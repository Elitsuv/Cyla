import numpy as np
from .config import (
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    LR, MOMENTUM, WEIGHT_CLIP, REWARD_SCALE,
    RE_RANK_EVERY, PREFIX_RATIO,
    MIN_EVIDENCE, USE_CONTENT_PRIOR
)


class CylaX1:
    """Single-hidden-layer scorer: maps a 5-d feature vector to a scalar
    priority score used to rank cold items in the prefix window.

    Architecture: Linear(5→12) → tanh → Linear(12→1)
    Training: online SGD with momentum after every search() hit.
    """

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


class AdaptiveList:
    """Self-organising list: MTF + cold-gated NN reranking + noise guard.

    Key invariants
    --------------
    - Hot items (count > 0) are NEVER reordered by the NN; their positions
      are determined solely by MTF.
    - The NN only ranks cold items (count == 0) inside the prefix window,
      where MTF has no information yet.
    - The noise guard (MIN_EVIDENCE) prevents one-off queries from polluting
      the front of the list.  First-ever hits are always allowed (cold-start).

    Parameters
    ----------
    data : iterable
        Initial items.
    content_scores : dict | None
        Precomputed ``{item: float}`` from offline embeddings (e.g. MiniLM).
        Used as the 5th NN feature when USE_CONTENT_PRIOR is True.
    min_evidence : int | None
        Override for MIN_EVIDENCE from config.  None → use config default.
    """

    def __init__(self, data, *, content_scores=None, min_evidence=None):
        self.data          = list(data)
        self._n            = len(data)
        self.counts        = {x: 0 for x in self.data}
        self.max_count     = 0
        self.last_seen     = {x: 0 for x in self.data}
        self.timer         = 0
        self.re_rank_every = RE_RANK_EVERY
        self.prefix_ratio  = PREFIX_RATIO
        self.scorer        = CylaX1()
        self.content_scores = content_scores or {}
        self.min_evidence = min_evidence if min_evidence is not None else MIN_EVIDENCE

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> object:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item) -> bool:
        return item in self.counts

    def append(self, item):
        if item in self.counts:
            return
        self.data.append(item)
        self._n += 1
        self.counts[item] = 0
        self.last_seen[item] = self.timer

    def _get_features(self, item) -> np.ndarray:
        """5-d feature vector: [freq_ratio, recency, log_freq, age_ratio, content_sim]."""
        max_c   = self.max_count + 1e-9
        c       = self.counts[item]
        age     = self.timer - self.last_seen[item]
        content = (self.content_scores.get(item, 0.0)
                   if USE_CONTENT_PRIOR else 0.0)
        return np.array([
            c / max_c,
            1.0 / (age + 1),
            np.log1p(c) / 10.0,
            age / (self._n * 2 + 1),
            content
        ])

    def _maybe_rerank(self):
        """Cold-gated rerank: NN scores only count==0 items in the prefix.
        Hot items keep their MTF-determined order untouched."""
        if self.timer % self.re_rank_every != 0:
            return
        k      = max(1, int(self._n * self.prefix_ratio))
        prefix = self.data[:k]

        hot  = [x for x in prefix if self.counts[x] > 0]
        cold = [x for x in prefix if self.counts[x] == 0]

        if cold:
            feats       = np.array([self._get_features(x) for x in cold])
            scores      = self.scorer.forward(feats)
            cold_sorted = [cold[i] for i in np.argsort(scores)[::-1]]
        else:
            cold_sorted = cold

        self.data[:k] = hot + cold_sorted

    def search(self, target) -> tuple[int, int]:
        """Linear search with MTF promotion + noise guard.

        Returns (found_position, steps).  -1 if not found.
        """
        self.timer += 1
        self._maybe_rerank()

        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n

        features = self._get_features(target)
        reward   = 10.0 / (pos ** 1.5 + 0.1)

        # Promote via MTF only if: (a) first-ever hit, or
        # (b) item has accumulated >= min_evidence hits.
        current_count = self.counts[target]
        should_promote = (
            current_count == 0
            or current_count + 1 >= self.min_evidence
        )
        if pos > 0 and should_promote:
            self.data.pop(pos)
            self.data.insert(0, target)

        self.counts[target]   += 1
        if self.counts[target] > self.max_count:
            self.max_count = self.counts[target]
        self.last_seen[target] = self.timer
        self.scorer.update(features, reward)

        return pos, pos + 1