"""Model configuration constants."""

from pathlib import Path

# System prompt instructing the reasoning/certainty/answer format that the
# reward functions expect. Without these instructions the model will not
# emit the tags, so every reward collapses to its "malformed" branch.
SYSTEM_PROMPT = """A problem will follow. Think about the problem carefully and\
 provide your reasoning and its associated uncertainty. Place the reasoning and\
 uncertainty discussion between <reasoning> and </reasoning>. Then, provide an\
 uncertainty value for the answer placed between <certainty> and </certainty>\
 which should be between 1 and 10 where 10 is very certain. Finally, the answer\
 is placed between <answer> and </answer>.

The problem is: """

# Gemma chat template. The turn markers are required by the tokenizer so the
# model actually behaves like a chat model.
TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

# Data configuration
TRAIN_DATA_DIR = Path("./data/train")
TEST_DATA_DIR = Path("./data/test")
TRAIN_FRACTION = 1.0
NLI_TEST_SIZE = 200

# LoRA configuration
RANK = 64
ALPHA = 64.0

# Sharding configuration: (mesh_shape, axis_names) passed to jax.make_mesh.
# Uses the first prod(mesh_shape) devices when more are available.
MESH_SHAPE = (1, 4)
MESH_AXES = ("fsdp", "tp")

# GRPO rollout configuration
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
NUM_TEST_BATCHES = 100
EVAL_EVERY_N_STEPS = 10
LEARNING_RATE = 3e-6
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.1
NUM_EPOCHS = 3
WARMUP_STEPS = 10
MAX_STEPS = 100
B1 = 0.9
B2 = 0.99
CHECKPOINT_MAX_TO_KEEP = 2

# Run configuration
RUN_CONFIG = {"reasoning": "uncert_aware", "dataset": "NLI"}
