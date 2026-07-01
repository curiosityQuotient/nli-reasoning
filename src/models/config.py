"""Model configuration constants."""

from pathlib import Path

# System prompt for reasoning tasks
SYSTEM_PROMPT = "You are a helpful AI assistant."

# Template for formatting prompts
TEMPLATE = "{system_prompt}\n\n{question}"

# Data configuration
TRAIN_DATA_DIR = Path("./data/train")
TEST_DATA_DIR = Path("./data/test")
TRAIN_FRACTION = 1.0

# LoRA configuration
RANK = 64
ALPHA = 64.0

# Sharding configuration
MESH = [(1, 4), ("fsdp", "tp")]

# GRPO configuration
MAX_PROMPT_LENGTH = 512
TOTAL_GENERATION_STEPS = 1024
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 4
NUM_ITERATIONS = 1
BETA = 0.08
EPSILON = 0.2

# Training configuration
TRAIN_MICRO_BATCH_SIZE = 2
NUM_BATCHES = 20
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.1
NUM_EPOCHS = 3
WARMUP_STEPS = 10
MAX_STEPS = 100
B1 = 0.9
B2 = 0.99

# Run configuration
RUN_CONFIG = {"reasoning": "uncert_aware", "dataset": "NLI"}
