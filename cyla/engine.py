import numpy as np

class CylaX1:
    def __init__(self):
        self.W1 = np.random.randn(4, 12) * 0.06
        self.b1 = np.zeros(12)

        self.W2 = np.random.randn(12, 1) * 0.06
        self.b2 = np.zeros(1)

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        def forward(self, X):
            h = np.tanh(X @ self.W1 + self.b1)
            scores = h @ self.W2 + self.b2
            return scores.ravel()
        
        def updater(self, features, reward, lr=0.09):
            h = np.tanh(features @ self.W1 + self.b1)
            score = h @ self.W2 + self.b2
            error = reward * 0.05 - score
            dW2 = h * error
            dh = error * self.W2.T * (1 - h ** 2)
            dW1 = np.outer(features, dh)
            self.vW1 = 0.65  * self.vW1 + 0.35 * dW1
            self.vb1 = 0.65  * self.vb1 + 0.35 * dW2

            self.W1 = np.clip(self.W1, -7, 7)
            self.W2 = np.clip(self.W2, -7, 7)

class main:
    def __init__(self, data):
        self.data = list(data)               
        self._n = len(data)
        self.counts = [0] * self._n          
        self.last_seen = [0] * self._n       
        self.timer = 0                       
        self.re_rank_every = 4               
        self.prefix_ratio = 0.22             
        self.scorer = CylaX1()         

    def _get_features(self, idx):
        maxc = max(self.counts) + 1e-9
        f = self.counts[idx] / maxc           
        r = 1.0 / (self.timer - self.last_seen[idx] + 1)  
        logf = np.log1p(self.counts[idx]) / 10.0   
        overdue = (self.timer - self.last_seen[idx]) / (self._n * 2 + 1) 
        return np.array([f, r, logf, overdue])

    def search(self, target):
        self.timer += 1

        if self.timer % self.re_rank_every == 0:
            k = max(1, int(self._n * self.prefix_ratio))
            prefix = self.data[:k]
            feats = np.array([self._get_features(self.data.index(x)) for x in prefix])
            scores = self.scorer.forward(feats)
            sorted_idx = np.argsort(scores)[::-1]  
            self.data[:k] = [prefix[i] for i in sorted_idx]

        for pos in range(self._n):
            if self.data[pos] == target:
                if pos < int(self._n * self.prefix_ratio):
                    item = self.data.pop(pos)
                    self.data.insert(0, item)

                self.counts[pos] += 1
                self.last_seen[pos] = self.timer
                features = self._get_features(0)
                reward = 10.0 / (pos ** 1.5 + 0.1)
                self.scorer.update(features, reward)

                return pos, pos + 1
        return -1, self._n