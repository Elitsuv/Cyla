"""
Configuration parameters for Cyla's online neural scorer and adaptive list engine.
"""

# -- CylaX1 neural-network architecture --
INPUT_SIZE = 5          # Features: [freq_ratio, recency, log_freq, age_ratio, content_sim]
HIDDEN_SIZE = 12        # Hidden units in single-layer MLP
OUTPUT_SIZE = 1         # Scalar priority score

# -- Training hyper-parameters (Online SGD) --
LR = 0.09               # Learning rate
MOMENTUM = 0.90         # Momentum factor
WEIGHT_CLIP = 7.0       # Gradient/weight divergence bounding
REWARD_SCALE = 1.0      # Scalar scaling for search position reward

# -- Periodic reranking schedule --
RE_RANK_EVERY = 4       # Number of search operations between prefix re-evaluations
PREFIX_RATIO = 0.22     # Fraction of total list capacity designated as prefix window

# -- Noise guard --
MIN_EVIDENCE = 1        # Hits required before promotion (1 = standard MTF behavior)

# -- Content-based prior --
USE_CONTENT_PRIOR = False  # Enable to use external embedding similarities as 5th feature
