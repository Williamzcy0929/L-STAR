"""Default configuration and constants for L-STAR."""

# Default LLM model
DEFAULT_MODEL_NAME = "gpt-5-2025-08-07"

# Default hyperparameters for pairwise comparisons
DEFAULT_PAIRWISE_TEMPERATURE = 1.0
DEFAULT_PAIRWISE_REASONING_EFFORT = "medium"

# Default hyperparameters for second-round reasoning
DEFAULT_SECOND_ROUND_TEMPERATURE = 1.0
DEFAULT_SECOND_ROUND_REASONING_EFFORT = "high"

# Default output directory structure
DEFAULT_OUTPUT_DIR = "lstar_output"
PAIRWISE_SUBDIR = "pairwise"
RANKING_CSV_NAME = "ranking.csv"
CONSENSUS_CSV_NAME = "L_STAR_consensus.csv"

# Default H&E image basename
DEFAULT_HE_BASENAME = "he.png"

# Default k selection range
DEFAULT_K_RANGE = range(2, 16)

# Default consensus parameters
DEFAULT_REPS = 5
DEFAULT_TOP_K = 5

