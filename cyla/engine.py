"""
Core engine implementing AdaptiveList and the online CylaX1 neural scorer.
"""

from typing import Any, Iterable, Optional, Tuple
import numpy as np

from .config import (
    INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    LR, MOMENTUM, WEIGHT_CLIP, REWARD_SCALE,
    RE_RANK_EVERY, PREFIX_RATIO,
    MIN_EVIDENCE, USE_CONTENT_PRIOR
)


class CylaX1:
    """Single-hidden-layer feedforward neural network scoring micro-engine.

    Maps a 5-dimensional access feature vector to a scalar priority score:
    Architecture: Linear(5 -> 12) -> tanh -> Linear(12 -> 1)
    Optimization: Online SGD with momentum, updated on each search hit.
    """

    def __init__(self):
        # Xavier / He style normal initialization
        self.W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
        self.b1 = np.zeros(HIDDEN_SIZE)
        self.W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
        self.b2 = np.zeros(OUTPUT_SIZE)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass predicting priority scores for one or more feature vectors."""
        X = np.atleast_2d(X)
        self._h = np.tanh(X @ self.W1 + self.b1)
        return (self._h @ self.W2 + self.b2).ravel()

    def update(self, features: np.ndarray, reward: float, lr: float = LR) -> None:
        """Online SGD parameter update based on observed search reward."""
        features = np.asarray(features, dtype=float)

        h = np.tanh(features @ self.W1 + self.b1)
        score = (h @ self.W2 + self.b2).ravel()
        error = REWARD_SCALE * reward - score

        # Backpropagation
        dW2 = np.outer(h, error)
        db2 = error
        dh = (error @ self.W2.T) * (1.0 - h ** 2)
        dW1 = np.outer(features, dh)
        db1 = dh

        # Momentum updates
        self.vW1 = MOMENTUM * self.vW1 + (1 - MOMENTUM) * dW1
        self.vb1 = MOMENTUM * self.vb1 + (1 - MOMENTUM) * db1
        self.vW2 = MOMENTUM * self.vW2 + (1 - MOMENTUM) * dW2
        self.vb2 = MOMENTUM * self.vb2 + (1 - MOMENTUM) * db2

        # Weight updates with clipping
        self.W1 = np.clip(self.W1 + lr * self.vW1, -WEIGHT_CLIP, WEIGHT_CLIP)
        self.b1 = self.b1 + lr * self.vb1
        self.W2 = np.clip(self.W2 + lr * self.vW2, -WEIGHT_CLIP, WEIGHT_CLIP)
        self.b2 = self.b2 + lr * self.vb2


class AdaptiveList:
    """Self-organizing list combining Move-to-Front with cold-gated neural scoring.

    Operational Design:
    - Hot items (count > 0): Strictly preserve their MTF-determined order.
    - Cold items (count == 0): Re-ranked within the prefix window using CylaX1.
    - Noise Guard: An optional threshold (min_evidence) requiring multiple hits
      before promoting items to position 0, while allowing cold-start promotions.

    Parameters
    ----------
    data : Iterable[Any]
        Initial collection of elements.
    content_scores : dict, optional
        Precomputed similarity priors {item: float} from content embeddings.
    min_evidence : int, optional
        Promotion threshold count. Defaults to config.MIN_EVIDENCE (1 = pure MTF).
    """

    def __init__(
        self,
        data: Iterable[Any],
        *,
        content_scores: Optional[dict] = None,
        min_evidence: Optional[int] = None
    ):
        self.data = list(data)
        self._n = len(self.data)
        self.counts = {x: 0 for x in self.data}
        self.max_count = 0
        self.last_seen = {x: 0 for x in self.data}
        self.timer = 0
        self.re_rank_every = RE_RANK_EVERY
        self.prefix_ratio = PREFIX_RATIO
        self.scorer = CylaX1()
        self.content_scores = content_scores or {}
        self.min_evidence = min_evidence if min_evidence is not None else MIN_EVIDENCE

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item: Any) -> bool:
        return item in self.counts

    def append(self, item: Any) -> None:
        """Append a new unique item to the tail of the list."""
        if item in self.counts:
            return
        self.data.append(item)
        self._n += 1
        self.counts[item] = 0
        self.last_seen[item] = self.timer

    def _get_features(self, item: Any) -> np.ndarray:
        """Construct 5-d feature vector: [freq_ratio, recency, log_freq, age_ratio, content_sim]."""
        max_c = self.max_count + 1e-9
        c = self.counts[item]
        age = self.timer - self.last_seen[item]
        content = (self.content_scores.get(item, 0.0) if USE_CONTENT_PRIOR else 0.0)
        return np.array([
            c / max_c,
            1.0 / (age + 1),
            np.log1p(c) / 10.0,
            age / (self._n * 2 + 1),
            content
        ])

    def _maybe_rerank(self) -> None:
        """Cold-gated reranking of prefix items.

        Preserves hot items' relative MTF ordering while sorting cold items
        according to CylaX1 scores.
        """
        if self.timer % self.re_rank_every != 0:
            return

        k = max(1, int(self._n * self.prefix_ratio))
        prefix = self.data[:k]

        hot = [x for x in prefix if self.counts[x] > 0]
        cold = [x for x in prefix if self.counts[x] == 0]

        if cold:
            feats = np.array([self._get_features(x) for x in cold])
            scores = self.scorer.forward(feats)
            cold_sorted = [cold[i] for i in np.argsort(scores)[::-1]]
        else:
            cold_sorted = cold

        self.data[:k] = hot + cold_sorted

    def search(self, target: Any) -> Tuple[int, int]:
        """Linear search for target element with adaptive reordering.

        Returns
        -------
        Tuple[int, int]
            (found_position, steps_taken). If target is missing, returns (-1, len).
        """
        self.timer += 1
        self._maybe_rerank()

        try:
            pos = self.data.index(target)
        except ValueError:
            return -1, self._n

        features = self._get_features(target)
        reward = 10.0 / (pos ** 1.5 + 0.1)

        # Move-to-Front promotion gating
        current_count = self.counts[target]
        should_promote = (
            current_count == 0
            or current_count + 1 >= self.min_evidence
        )
        if pos > 0 and should_promote:
            self.data.pop(pos)
            self.data.insert(0, target)

        self.counts[target] += 1
        if self.counts[target] > self.max_count:
            self.max_count = self.counts[target]
        self.last_seen[target] = self.timer
        self.scorer.update(features, reward)

        return pos, pos + 1
