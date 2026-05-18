"""Tests for NLI reasoning modules."""

import pandas as pd
import pytest

from src.data.dataset import extract_hash_answer, get_nli_dataset


def test_extract_hash_answer():
    """Test extracting answer from text with #### marker."""
    # Test case with answer
    text_with_answer = "Some reasoning here #### 42"
    assert extract_hash_answer(text_with_answer) == "42"

    # Test case without answer
    text_without_answer = "Some reasoning here without marker"
    assert extract_hash_answer(text_without_answer) is None

    # Test case with multiple markers (should take first)
    text_multiple_markers = "Reasoning #### 24 #### 42"
    assert extract_hash_answer(text_multiple_markers) == "24"


def test_get_nli_dataset():
    """Test creating NLI dataset from DataFrame."""
    # Create sample DataFrame
    sample_data = {
        "premise": ["Premise 1", "Premise 2"],
        "hypothesis": ["Hypothesis 1", "Hypothesis 2"],
        "uncertainty": [5, 8],
        "majority_label": ["e", "c"]
    }
    df = pd.DataFrame(sample_data)

    # This would normally return a grain.MapDataset
    # We're just checking it doesn't crash with valid input
    try:
        dataset = get_nli_dataset(df)
        assert dataset is not None
    except ImportError:
        # Skip if grain is not properly installed
        pytest.skip("grain not available")
