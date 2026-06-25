"""Main training module for NLI reasoning with uncertainty awareness."""

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import optax
import pandas as pd

# Add the src directory to the path to enable imports
sys.path.append("/kaggle/working/nli-reasoning/src")

from data.dataset import get_dataset, get_nli_dataset
from models.config import (
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
)
from utils.memory import show_hbm_usage

# Training configuration
TRAIN_MICRO_BATCH_SIZE = 4
NUM_BATCHES = 1000
NUM_TEST_BATCHES = 100
NUM_EPOCHS = 3


def load_nli_data(
    train_path: Path,
    train_fraction: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load NLI datasets for training and testing.

    Args:
        train_path: Path to training data JSONL file
        test_path: Path to test data JSONL file
        train_fraction: Fraction of data to use for training

    Returns:
        Tuple of (train_dataframe, test_dataframe)
    """
    # Validate and convert to Path object
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not train_path.is_file():
        raise ValueError(f"Training data path is not a file: {train_path}")
    else:
        print("Training data path exists as a file.")
    
    # Load training data
    df_nli = pd.read_json(train_path, lines=True)
    # Unpack premise and hypothesis
    df_nli["premise"] = [xx["premise"] for xx in df_nli.iloc[:, 6]]
    df_nli["hypothesis"] = [xx["hypothesis"] for xx in df_nli.iloc[:, 6]]
    # Convert Shannon Entropy to uncertainty
    def uncert_convert(x):
        return 1 + round(10*(1 - x/1.58496))
    df_nli["uncertainty"] = [uncert_convert(xx) for xx in df_nli.iloc[:, 5]]

    # Split into train/test
    N_all = df_nli.shape[0]
    N_train = N_all - 200
    req_cols = [2, 5, 9, 10, 11]

    df_test = df_nli.iloc[0:N_train, req_cols]
    df_train = df_nli.iloc[N_train:-1, req_cols]

    return df_train, df_test


def prepare_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    source: str = "kaggle"
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """Prepare all datasets for training.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        source: Data source ('tfds' or 'kaggle')

    Returns:
        Tuple of (train_nli, val_nli, test_nli, train_gsm8k, val_gsm8k, test_gsm8k)
    """
    # NLI dataset preparation
    nli_dataset = get_nli_dataset(df_train).batch(TRAIN_MICRO_BATCH_SIZE)[
        :2*NUM_BATCHES
    ]

    if TRAIN_FRACTION == 1.0:
        train_nli_dataset = nli_dataset.repeat(NUM_EPOCHS)
        val_nli_dataset = None
    else:
        train_nli_dataset = nli_dataset[:int(len(nli_dataset) * TRAIN_FRACTION)]
        train_nli_dataset = train_nli_dataset.repeat(NUM_EPOCHS)
        val_nli_dataset = nli_dataset[
            int(len(nli_dataset) * TRAIN_FRACTION):
        ].repeat(NUM_EPOCHS)

    test_nli_dataset = get_nli_dataset(df_test).batch(TRAIN_MICRO_BATCH_SIZE)[
        :NUM_TEST_BATCHES
    ]

    # GSM8K dataset preparation
    gsm8k_dataset = get_dataset(
        TRAIN_DATA_DIR, "train", source
    ).batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_BATCHES]

    if TRAIN_FRACTION == 1.0:
        train_gsm8k_dataset = gsm8k_dataset.repeat(NUM_EPOCHS)
        val_gsm8k_dataset = None
    else:
        train_gsm8k_dataset = gsm8k_dataset[:int(len(gsm8k_dataset) * TRAIN_FRACTION)]
        train_gsm8k_dataset = train_gsm8k_dataset.repeat(NUM_EPOCHS)
        val_gsm8k_dataset = gsm8k_dataset[
            int(len(gsm8k_dataset) * TRAIN_FRACTION):
        ].repeat(NUM_EPOCHS)

    test_gsm8k_dataset = get_dataset(
        TEST_DATA_DIR, "test", source
    ).batch(TRAIN_MICRO_BATCH_SIZE)[:NUM_TEST_BATCHES]

    return (
        train_nli_dataset,
        val_nli_dataset,
        test_nli_dataset,
        train_gsm8k_dataset,
        val_gsm8k_dataset,
        test_gsm8k_dataset
    )


def initialize_model_and_optimizer():
    """Initialize model and optimizer for training.

    Returns:
        Tuple of (model, optimizer, state)
    """
    # Model initialization would go here
    # This is a placeholder implementation
    model = None
    optimizer = optax.adam(learning_rate=1e-4)
    state = None

    return model, optimizer, state


def train_step(model, optimizer, state, batch):
    """Perform a single training step.

    Args:
        model: Model to train
        optimizer: Optimizer to use
        state: Training state
        batch: Batch of training data

    Returns:
        Updated model, optimizer, and state
    """
    # Training step implementation would go here
    # This is a placeholder implementation
    return model, optimizer, state


def main(training_config: Optional[Dict[str, Any]] = None) -> None:
    """Main training function.

    Args:
        training_config: Configuration dictionary for training
    """
    if training_config is None:
        training_config = {}

    # Set up WandB logging if configured
    if "wandb_api_key" in training_config:
        os.environ['WANDB_API_KEY'] = training_config["wandb_api_key"]

    # Display memory usage
    show_hbm_usage()

    # Load data
    print("Loading NLI data...")

    nli_data_path = Path("/kaggle/input/datasets/curiosityquotient/chaos-nli/chaosNLI_v1.0/chaosNLI_snli.jsonl")
    df_train, df_test = load_nli_data(nli_data_path)

    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")

    # Prepare datasets
    print("Preparing datasets...")
    datasets = prepare_datasets(df_train, df_test, source='kaggle')
    train_nli_dataset, val_nli_dataset, test_nli_dataset, \
        train_gsm8k_dataset, val_gsm8k_dataset, test_gsm8k_dataset = datasets

    dataset_lengths = (
        len(train_nli_dataset),
        len(val_nli_dataset) if val_nli_dataset is not None else 0,
        len(test_nli_dataset),
    )
    print(f"Dataset contains {dataset_lengths} batches")

    # Initialize model and optimizer
    print("Initializing model...")
    model, optimizer, state = initialize_model_and_optimizer()

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Iterate through batches
        for batch_idx, batch in enumerate(train_nli_dataset):
            model, optimizer, state = train_step(model, optimizer, state, batch)

            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}")
                show_hbm_usage()

            # Periodic garbage collection
            if batch_idx % 500 == 0:
                gc.collect()

    print("Training completed!")


if __name__ == "__main__":
    main()
