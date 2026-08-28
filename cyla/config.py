# -- CylaX1 neural-network architecture --
INPUT_SIZE    = 5       # [freq_ratio, recency, log_freq, age_ratio, content_sim]
HIDDEN_SIZE   = 12
OUTPUT_SIZE   = 1

# -- Training hyper-parameters --
LR            = 0.09
MOMENTUM      = 0.90
WEIGHT_CLIP   = 7.0     # prevents weight divergence
REWARD_SCALE  = 1.0

# -- Periodic reranking schedule --
RE_RANK_EVERY = 4       # rerank prefix every N searches
PREFIX_RATIO  = 0.22    # fraction of the list treated as prefix

# -- Noise guard --
MIN_EVIDENCE  = 1       # hits required before MTF promotion (1 = pure MTF)

# -- Optional content-based prior --
USE_CONTENT_PRIOR = False   # set True to use precomputed embeddings as 5th feature