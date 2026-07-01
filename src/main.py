"""Main training module for NLI reasoning with uncertainty awareness."""

import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

sys.path.append("/kaggle/working/nli-reasoning")

from src.data.dataset import get_dataset, get_nli_dataset
from src.models.config import (
    NUM_BATCHES,
    NUM_EPOCHS,
    NUM_TEST_BATCHES,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
)
from src.models.loader import (
    download_gemma_model,
    get_gemma_ref_model,
    get_lora_model,
    get_tokenizer,
    resave_checkpoint,
)
from src.training.grpo_config import create_grpo_config_stage1, create_grpo_config_stage2
from src.training.trainer import train_two_stage
from src.utils.evaluations import evaluate, evaluate_nli
from src.utils.memory import show_hbm_usage


def load_nli_data(
    train_path: Path,
    train_fraction: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load NLI datasets for training and testing.

    Args:
        train_path: Path to training data JSONL file
        train_fraction: Fraction of data to use for training

    Returns:
        Tuple of (train_dataframe, test_dataframe)
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not train_path.is_file():
        raise ValueError(f"Training data path is not a file: {train_path}")
    
    print("Training data path exists as a file.")
    
    df_nli = pd.read_json(train_path, lines=True)
    
    df_nli["premise"] = [xx["premise"] for xx in df_nli.iloc[:, 6]]
    df_nli["hypothesis"] = [xx["hypothesis"] for xx in df_nli.iloc[:, 6]]
    
    def uncert_convert(x):
        return 1 + round(10 * (1 - x / 1.58496))
    
    df_nli["uncertainty"] = [uncert_convert(xx) for xx in df_nli.iloc[:, 5]]

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
    nli_dataset = get_nli_dataset(df_train).batch(TRAIN_MICRO_BATCH_SIZE)[
        :2 * NUM_BATCHES
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


def setup_model(
    checkpoint_dir: Path,
    model_family: str = "gemma2",
    model_version: str = "gemma2-2b-it"
) -> Tuple[Any, Any, Any]:
    """Download, setup, and load the model.
    
    Args:
        checkpoint_dir: Directory for model checkpoints
        model_family: Model family name
        model_version: Model version string
        
    Returns:
        Tuple of (reference_model, lora_model, tokenizer)
    """
    print("Downloading model from Kaggle...")
    kaggle_ckpt_path = download_gemma_model(model_family, model_version)
    
    resaved_ckpt_path = checkpoint_dir / "resaved_checkpoint"
    if not resaved_ckpt_path.exists():
        print("Resaving checkpoint in Flax NNX format...")
        resave_checkpoint(kaggle_ckpt_path, str(resaved_ckpt_path), model_family)
        gc.collect()
    
    print("Loading reference model...")
    ref_model = get_gemma_ref_model(str(resaved_ckpt_path / "model_params"))
    
    print("Applying LoRA to create policy model...")
    lora_model = get_lora_model(ref_model)
    
    print("Getting tokenizer...")
    tokenizer = get_tokenizer(model_version)
    
    return ref_model, lora_model, tokenizer


def run_pre_training_evaluation(
    test_nli_dataset: Any,
    test_gsm8k_dataset: Any,
    sampler: Any
) -> Tuple[Tuple, Tuple]:
    """Run evaluation before training to establish baseline.
    
    Args:
        test_nli_dataset: NLI test dataset
        test_gsm8k_dataset: GSM8K test dataset
        sampler: Text sampler
        
    Returns:
        Tuple of (nli_baseline_metrics, gsm8k_baseline_metrics)
    """
    print("=" * 60)
    print("Pre-training Evaluation (Baseline)")
    print("=" * 60)
    
    print("\nEvaluating on NLI test set...")
    nli_baseline = evaluate_nli(test_nli_dataset, sampler, make_lst=False)
    print(f"NLI Baseline: {nli_baseline}")
    
    print("\nEvaluating on GSM8K test set...")
    gsm8k_baseline = evaluate(test_gsm8k_dataset, sampler, make_lst=False)
    print(f"GSM8K Baseline: {gsm8k_baseline}")
    
    return nli_baseline, gsm8k_baseline


def run_post_training_evaluation(
    test_nli_dataset: Any,
    test_gsm8k_dataset: Any,
    sampler: Any
) -> Tuple[Tuple, Tuple]:
    """Run evaluation after training.
    
    Args:
        test_nli_dataset: NLI test dataset
        test_gsm8k_dataset: GSM8K test dataset
        sampler: Text sampler
        
    Returns:
        Tuple of (nli_metrics, gsm8k_metrics)
    """
    print("=" * 60)
    print("Post-training Evaluation")
    print("=" * 60)
    
    print("\nEvaluating on NLI test set...")
    nli_metrics = evaluate_nli(test_nli_dataset, sampler, make_lst=False)
    print(f"NLI Results: {nli_metrics}")
    
    print("\nEvaluating on GSM8K test set...")
    gsm8k_metrics = evaluate(test_gsm8k_dataset, sampler, make_lst=False)
    print(f"GSM8K Results: {gsm8k_metrics}")
    
    return nli_metrics, gsm8k_metrics


def main(training_config: Optional[Dict[str, Any]] = None) -> None:
    """Main training function.

    Args:
        training_config: Configuration dictionary for training
    """
    if training_config is None:
        training_config = {}

    if "wandb_api_key" in training_config:
        os.environ['WANDB_API_KEY'] = training_config["wandb_api_key"]
        use_wandb = True
    else:
        use_wandb = False

    show_hbm_usage()

    print("\n" + "=" * 60)
    print("Loading NLI data...")
    print("=" * 60)
    
    nli_data_path = Path(
        "/kaggle/input/datasets/curiosityquotient/chaos-nli/chaosNLI_v1.0/chaosNLI_snli.jsonl"
    )
    df_train, df_test = load_nli_data(nli_data_path)

    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")

    print("\n" + "=" * 60)
    print("Preparing datasets...")
    print("=" * 60)
    
    datasets = prepare_datasets(df_train, df_test, source='kaggle')
    train_nli_dataset, val_nli_dataset, test_nli_dataset, \
        train_gsm8k_dataset, val_gsm8k_dataset, test_gsm8k_dataset = datasets

    print(f"NLI dataset: {len(train_nli_dataset)} train batches")
    print(f"GSM8K dataset: {len(train_gsm8k_dataset)} train batches")

    print("\n" + "=" * 60)
    print("Setting up model...")
    print("=" * 60)
    
    checkpoint_dir = Path("/kaggle/working/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    ref_model, lora_model, tokenizer = setup_model(checkpoint_dir)

    print("\n" + "=" * 60)
    print("Running pre-training evaluation...")
    print("=" * 60)
    
    try:
        from tunix.generate import sampler as sampler_lib
        
        sampler = sampler_lib.Sampler(
            model=ref_model,
            tokenizer=tokenizer,
        )
        
        nli_baseline, gsm8k_baseline = run_pre_training_evaluation(
            test_nli_dataset, test_gsm8k_dataset, sampler
        )
    except Exception as e:
        print(f"Warning: Pre-training evaluation failed: {e}")
        nli_baseline = None
        gsm8k_baseline = None

    print("\n" + "=" * 60)
    print("Starting Two-Stage Training")
    print("=" * 60)
    
    grpo_config_stage1 = create_grpo_config_stage1()
    grpo_config_stage2 = create_grpo_config_stage2()
    
    trained_model, training_metrics = train_two_stage(
        model=lora_model,
        ref_model=ref_model,
        train_gsm8k_dataset=train_gsm8k_dataset,
        train_nli_dataset=train_nli_dataset,
        grpo_config_stage1=grpo_config_stage1,
        grpo_config_stage2=grpo_config_stage2,
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb,
    )
    
    print("\nTraining completed!")
    print(f"Training metrics: {training_metrics}")

    print("\n" + "=" * 60)
    print("Running post-training evaluation...")
    print("=" * 60)
    
    gc.collect()
    
    try:
        from tunix.generate import sampler as sampler_lib
        
        trained_sampler = sampler_lib.Sampler(
            model=trained_model,
            tokenizer=tokenizer,
        )
        
        nli_final, gsm8k_final = run_post_training_evaluation(
            test_nli_dataset, test_gsm8k_dataset, trained_sampler
        )
        
        print("\n" + "=" * 60)
        print("Final Results Comparison")
        print("=" * 60)
        
        if nli_baseline and gsm8k_baseline:
            print(f"\nNLI Accuracy: {nli_baseline[2]:.2f}% -> {nli_final[2]:.2f}%")
            print(f"GSM8K Accuracy: {gsm8k_baseline[2]:.2f}% -> {gsm8k_final[2]:.2f}%")
        
    except Exception as e:
        print(f"Warning: Post-training evaluation failed: {e}")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
