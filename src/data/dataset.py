"""Dataset loading and NLI data processing utilities."""

import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import grain
import kagglehub
import pandas as pd
import tensorflow_datasets as tfds

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""

# ChaosNLI stores full label names; the prompts (and therefore the rewards)
# use single-letter labels.
NLI_LABEL_MAP: Dict[str, str] = {
    "entailment": "e",
    "contradiction": "c",
    "neutral": "n",
}
VALID_NLI_LABELS = frozenset(NLI_LABEL_MAP.values())

# Maximum Shannon entropy for a 3-way classification (log2(3)).
NLI_MAX_ENTROPY: float = math.log2(3)


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract answer from text following the #### pattern.

    Args:
        text: Text containing the answer after #### marker

    Returns:
        Extracted answer or None if marker not found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def entropy_to_certainty(entropy: float) -> int:
    """Convert Shannon entropy over 3 labels to a 1-10 certainty score.

    Entropy 0 (full agreement) maps to 10; maximum entropy (1.585) maps
    to 1. The result is clamped to [1, 10] to protect against rounding
    overshoot so that actual certainty stays on the same 1-10 scale as
    the model's predicted certainty.

    Args:
        entropy: Shannon entropy in [0, log2(3)]

    Returns:
        Certainty score in [1, 10]
    """
    certainty = 1 + round(9 * (1 - entropy / NLI_MAX_ENTROPY))
    return int(min(10, max(1, certainty)))


def normalize_nli_label(label: str) -> str:
    """Normalize an NLI label to its single-letter form.

    Args:
        label: Label such as 'entailment', 'e', or 'Neutral'

    Returns:
        Single-letter label: 'e', 'c', or 'n'

    Raises:
        ValueError: If the label is not a recognized NLI label
    """
    lowered = str(label).strip().lower()
    normalized = NLI_LABEL_MAP.get(lowered, lowered)
    if normalized not in VALID_NLI_LABELS:
        raise ValueError(f"Unexpected NLI label: {label!r}")
    return normalized


def load_nli_data(
    train_path: Path,
    test_size: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load ChaosNLI data and split it into training and test frames.

    The last ``test_size`` rows form the test set; the remaining rows are
    used for training. This matches the notebook semantics (train on the
    bulk, evaluate on the tail), without dropping the final row.

    Args:
        train_path: Path to training data JSONL file
        test_size: Number of trailing rows to reserve for testing

    Returns:
        Tuple of (train_dataframe, test_dataframe)

    Raises:
        FileNotFoundError: If the data file does not exist
        ValueError: If the split sizes are infeasible
    """
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not train_path.is_file():
        raise ValueError(f"Training data path is not a file: {train_path}")

    df_nli = pd.read_json(train_path, lines=True)

    if "premise" not in df_nli.columns:
        df_nli["premise"] = [example["premise"] for example in df_nli["example"]]
        df_nli["hypothesis"] = [example["hypothesis"] for example in df_nli["example"]]

    entropy_col = "entropy" if "entropy" in df_nli.columns else df_nli.columns[5]
    label_col = (
        "majority_label" if "majority_label" in df_nli.columns else df_nli.columns[9]
    )

    df_nli["uncertainty"] = [entropy_to_certainty(v) for v in df_nli[entropy_col]]

    df_nli = df_nli[[label_col, "uncertainty", "premise", "hypothesis"]]

    if not 0 < test_size < len(df_nli):
        raise ValueError(
            f"test_size={test_size} leaves no training data for {len(df_nli)} rows"
        )

    df_train = df_nli.iloc[:-test_size]
    df_test = df_nli.iloc[-test_size:]
    return df_train, df_test


def _load_from_tfds(data_dir: str, split: str):
    """Load GSM8K dataset from TensorFlow Datasets.

    Args:
        data_dir: Directory to store/load data
        split: Dataset split to load

    Returns:
        TFDS data source
    """
    return tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )


def download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    """Download GSM8K dataset from Kaggle.

    Args:
        target_dir: Directory to download data to

    Returns:
        Path to downloaded dataset
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = Path(src)
    dst = target_path

    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return str(target_dir)


def get_dataset(
    data_dir: str, split: str = "train", source: str = "tfds"
) -> grain.MapDataset:
    """Get GSM8K dataset for training.

    Args:
        data_dir: Directory containing data
        split: Dataset split ('train' or 'test')
        source: Data source ('tfds' or 'kaggle')

    Returns:
        Processed dataset

    Raises:
        ValueError: If unknown source is provided
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        data_dir_path.mkdir(parents=True, exist_ok=True)
    elif not data_dir_path.is_dir():
        raise ValueError(f"data_dir is not a directory: {data_dir}")

    if source == "tfds":
        data = _load_from_tfds(str(data_dir_path), split)
    elif source == "kaggle":
        kaggle_dir = download_kaggle_dataset(str(data_dir_path))
        file_name = "main_" + split + ".csv"
        csv_path = Path(kaggle_dir) / file_name

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not csv_path.is_file():
            raise ValueError(f"CSV path is not a file: {csv_path}")

        rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                )
        data = rows
    else:
        raise ValueError(f"Unknown source: {source}")

    def _as_text(v: object) -> str:
        return v if isinstance(v, str) else bytes(v).decode("utf-8")

    from src.models.config import SYSTEM_PROMPT

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(
            lambda x: {
                # passed to model forward pass
                "prompts": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                # passed to reward functions
                "question": _as_text(x["question"]),
                # passed to reward functions
                "answer": extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )
    return dataset


def get_nli_dataset(data_df: pd.DataFrame) -> grain.MapDataset:
    """Process NLI dataset from DataFrame.

    Answers are normalized to the single-letter labels used in the
    prompts so that the reward functions can compare them directly.

    Args:
        data_df: DataFrame containing NLI data

    Returns:
        Processed dataset
    """

    def create_q(premise_text: str, hypothesis_text: str) -> str:
        question_prompt = f'''You are a textual entailment classifier.
        Input: A Premise and a Hypothesis.
        Output: A classification label.

        Allowed Labels: "e" for entailment, "c" for contradiction, "n" for neutral.

        Premise: "{premise_text}"
        Hypothesis: "{hypothesis_text}"
        '''
        return question_prompt

    def data_entry(row: pd.Series) -> Dict[str, object]:
        entry = {
            "question": create_q(str(row["premise"]), str(row["hypothesis"])),
            "uncertainty": int(row["uncertainty"]),
            "answer": normalize_nli_label(str(row["majority_label"])),
        }
        return entry

    data = list(data_df.apply(data_entry, axis=1))

    def _as_text(v: object) -> str:
        return v if isinstance(v, str) else bytes(v).decode("utf-8")

    from src.models.config import SYSTEM_PROMPT, TEMPLATE

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(
            lambda x: {
                # passed to model forward pass
                "prompts": TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    question=_as_text(x["question"]),
                ),
                # passed to reward functions
                "question": _as_text(x["question"]),
                # passed to reward functions
                "answer": x["answer"],
                "true_uncertainty": x["uncertainty"],
            }
        )
    )
    return dataset
