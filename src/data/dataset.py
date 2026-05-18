"""NLI dataset processing utilities."""

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import grain
import kagglehub
import pandas as pd
import tensorflow_datasets as tfds

TEMPLATE = """{system_prompt}\n\n{question}"""


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


def _load_from_tfds(data_dir: str, split: str) -> tfds.DatasetBuilder:
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
    os.makedirs(target_dir, exist_ok=True)
    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = Path(src)
    dst = Path(target_dir)

    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def get_dataset(
    data_dir: str,
    split: str = "train",
    source: str = "tfds"
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
    # Download data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if source == "tfds":
        data = _load_from_tfds(data_dir, split)
    elif source == "kaggle":
        kaggle_dir = download_kaggle_dataset(data_dir)
        file_name = "main_" + split + ".csv"
        csv_path = os.path.join(kaggle_dir, file_name)

        data = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append({
                    "question": row["question"],
                    "answer": row["answer"],
                })
    else:
        raise ValueError(f"Unknown source: {source}")

    def _as_text(v: Union[str, bytes]) -> str:
        return v if isinstance(v, str) else v.decode("utf-8")

    # Assuming SYSTEM_PROMPT is defined elsewhere
    from ..models.config import SYSTEM_PROMPT

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

    def data_entry(row: pd.Series) -> Dict:
        entry = {
            "question": create_q(row["premise"], row["hypothesis"]),
            "uncertainty": row["uncertainty"],
            "answer": row["majority_label"]
        }
        return entry

    data = list(data_df.apply(data_entry, axis=1))

    def _as_text(v: Union[str, bytes]) -> str:
        return v if isinstance(v, str) else v.decode("utf-8")

    # Assuming SYSTEM_PROMPT and TEMPLATE are defined elsewhere
    from ..models.config import SYSTEM_PROMPT, TEMPLATE

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
                "true_uncertainty": x["uncertainty"]
            }
        )
    )
    return dataset
