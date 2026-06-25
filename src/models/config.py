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
