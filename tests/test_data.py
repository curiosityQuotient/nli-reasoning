"""Tests for dataset processing and NLI data loading."""

import math

import pandas as pd
import pytest

from src.data.dataset import (
    entropy_to_certainty,
    extract_hash_answer,
    get_nli_dataset,
    load_nli_data,
    normalize_nli_label,
)


def test_extract_hash_answer():
    """Test extracting answer from text with #### marker."""
    assert extract_hash_answer("Some reasoning here #### 42") == "42"
    assert extract_hash_answer("Some reasoning here without marker") is None
    # Multiple markers take the first
    assert extract_hash_answer("Reasoning #### 24 #### 42") == "24"


def test_entropy_to_certainty_bounds():
    """Entropy maps onto the 1-10 certainty scale and is clamped."""
    assert entropy_to_certainty(0.0) == 10
    assert entropy_to_certainty(math.log2(3)) == 1
    # Rounding overshoot (e.g. tiny negative numerical noise) clamps to 10
    assert entropy_to_certainty(-1e-9) == 10
    for entropy in [0.0, 0.2, 0.5, 0.8, 1.0, 1.3, 1.5849625002]:
        assert 1 <= entropy_to_certainty(entropy) <= 10


def test_normalize_nli_label():
    """Full label names map to single letters; bad labels raise."""
    assert normalize_nli_label("entailment") == "e"
    assert normalize_nli_label("contradiction") == "c"
    assert normalize_nli_label("neutral") == "n"
    assert normalize_nli_label("  NEUTRAL ") == "n"
    assert normalize_nli_label("e") == "e"
    with pytest.raises(ValueError):
        normalize_nli_label("yes")


@pytest.fixture
def nli_jsonl(tmp_path):
    """Create a small ChaosNLI-like JSONL file."""
    labels = ["entailment", "neutral", "contradiction"]
    rows = []
    for i in range(12):
        rows.append(
            {
                "uid": f"u{i}",
                "example": {"premise": f"premise {i}", "hypothesis": f"hypothesis {i}"},
                "entropy": 0.0 if i % 4 == 0 else 1.2,
                "majority_label": labels[i % 3],
                "filler_a": i,
                "filler_b": f"f{i}",
            }
        )
    path = tmp_path / "chaosNLI_snli.jsonl"
    pd.DataFrame(rows).to_json(path, orient="records", lines=True)
    return path


def test_load_nli_data_split(nli_jsonl):
    """Train on the bulk, test on the tail; no rows dropped."""
    df_train, df_test = load_nli_data(nli_jsonl, test_size=4)

    assert len(df_train) == 8
    assert len(df_test) == 4
    # The test set is the tail of the data
    assert list(df_test.index) == list(range(8, 12))
    # Only the required columns survive
    assert set(df_train.columns) == {
        "majority_label",
        "uncertainty",
        "premise",
        "hypothesis",
    }


def test_load_nli_data_uncertainty_in_range(nli_jsonl):
    """Certainty values derived from entropy stay on the 1-10 scale."""
    _, df_test = load_nli_data(nli_jsonl, test_size=4)
    assert df_test["uncertainty"].between(1, 10).all()


def test_load_nli_data_invalid_sizes(nli_jsonl):
    """A test split that swallows all data is rejected."""
    with pytest.raises(ValueError):
        load_nli_data(nli_jsonl, test_size=12)
    with pytest.raises(ValueError):
        load_nli_data(nli_jsonl, test_size=0)


def test_load_nli_data_missing_file(tmp_path):
    """A missing data file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_nli_data(tmp_path / "does_not_exist.jsonl")


def test_get_nli_dataset_normalizes_labels():
    """Dataset answers use single-letter labels and carry certainty."""
    df = pd.DataFrame(
        {
            "premise": ["Premise 1", "Premise 2"],
            "hypothesis": ["Hypothesis 1", "Hypothesis 2"],
            "uncertainty": [5, 8],
            "majority_label": ["entailment", "contradiction"],
        }
    )

    dataset = get_nli_dataset(df)
    entries = list(dataset)

    assert len(entries) == 2
    answers = {entry["answer"] for entry in entries}
    assert answers == {"e", "c"}
    for entry in entries:
        assert 1 <= entry["true_uncertainty"] <= 10
        assert "<reasoning>" in entry["prompts"]
        assert "Allowed Labels" in entry["question"]


def test_get_nli_dataset_rejects_unknown_label():
    """An unrecognized majority label fails fast at data-prep time."""
    df = pd.DataFrame(
        {
            "premise": ["p"],
            "hypothesis": ["h"],
            "uncertainty": [5],
            "majority_label": ["yes"],
        }
    )
    with pytest.raises(ValueError):
        list(get_nli_dataset(df))
