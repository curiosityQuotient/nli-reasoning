"""Tests for reward functions (notebook-parity assertions)."""

import pytest

from src.rewards import (
    check_answer,
    check_answer_nli,
    check_numbers,
    match_format_approximately,
    match_format_exactly,
    uncert_aware_reward,
)

WELL_FORMED_GSM8K = "<reasoning> Two plus two is four. </reasoning>\n<answer>4</answer>"
WELL_FORMED_NLI = (
    "<reasoning> Indeed what a day! </reasoning>\n"
    "<certainty>10</certainty>\n"
    "<answer>n</answer>"
)


class TestMatchFormat:
    def test_exact_match(self):
        rewards = match_format_exactly(prompts=["p"], completions=[WELL_FORMED_GSM8K])
        assert rewards == [3.0]

    def test_no_match_scores_zero(self):
        rewards = match_format_exactly(prompts=["p"], completions=["The answer is 4"])
        assert rewards == [0.0]

    def test_approximately_all_single_tags(self):
        rewards = match_format_approximately(
            prompts=["p"], completions=[WELL_FORMED_NLI]
        )
        # Six structural tags present exactly once: 6 * 0.5
        assert rewards == [3.0]

    def test_approximately_penalizes_duplicates(self):
        duplicated = (
            "<reasoning>a<reasoning>b</reasoning>\n"
            "<certainty>5</certainty>\n<answer>n</answer>"
        )
        rewards = match_format_approximately(prompts=["p"], completions=[duplicated])
        # <reasoning> appears twice -> -0.5; the other five tags +0.5
        assert rewards == [2.0]

    def test_accepts_bytes(self):
        rewards = match_format_exactly(
            prompts=["p"], completions=[WELL_FORMED_GSM8K.encode()]
        )
        assert rewards == [3.0]


class TestUncertAwareReward:
    def test_notebook_case_high_certainty_correct(self):
        # Notebook assertion: correct answer, certainty 10 vs truth 9 -> 1.9
        assert uncert_aware_reward(1, "10", "9") == pytest.approx(1.9)

    def test_notebook_case_non_numeric_certainty(self):
        # Notebook assertion: unparseable certainty -> outcome * 0.5
        assert uncert_aware_reward(1, "High", "9") == 0.5
        assert uncert_aware_reward(0, "High", "9") == 0.0

    def test_wrong_low_confidence_relatively_good(self):
        low = uncert_aware_reward(0, 2, 3)
        high = uncert_aware_reward(0, 9, 3)
        assert low > high


class TestCheckAnswerNli:
    def test_notebook_expected_output(self):
        rewards = check_answer_nli(
            prompts=["What a day!"],
            completions=[WELL_FORMED_NLI],
            answer=["n"],
            true_uncertainty=["9"],
        )
        assert rewards == pytest.approx([1.9])

    def test_non_numeric_certainty_fallback(self):
        completion = (
            "<reasoning> Indeed what a day! </reasoning>\n"
            "<certainty>High</certainty>\n"
            "<answer>n</answer>"
        )
        rewards = check_answer_nli(
            prompts=["What a day!"],
            completions=[completion],
            answer=["n"],
            true_uncertainty=["9"],
        )
        assert rewards == [0.5]

    def test_malformed_response_scores_zero(self):
        rewards = check_answer_nli(
            prompts=["p"], completions=["no tags at all"], answer=["n"]
        )
        assert rewards == [0.0]

    def test_certainty_ten_not_truncated(self):
        # '10' must not be truncated to '1' by a single-character capture.
        high = check_answer_nli(
            prompts=["p"],
            completions=[WELL_FORMED_NLI],
            answer=["n"],
            true_uncertainty=[10],
        )
        low = check_answer_nli(
            prompts=["p"],
            completions=[WELL_FORMED_NLI.replace("<certainty>10", "<certainty>1")],
            answer=["n"],
            true_uncertainty=[10],
        )
        assert high[0] > low[0]

    def test_without_true_uncertainty(self):
        rewards = check_answer_nli(
            prompts=["p"], completions=[WELL_FORMED_NLI], answer=["n"]
        )
        assert rewards == [2.0]


class TestCheckAnswer:
    def test_exact_answer(self):
        rewards = check_answer(
            prompts=["p"], completions=[WELL_FORMED_GSM8K], answer=["4"]
        )
        assert rewards == [3.0]

    def test_number_in_reasoning_is_ignored(self):
        # Regression: the first number in the reasoning must not be
        # mistaken for the answer.
        completion = (
            "<reasoning> The total cost is 100 dollars. So the tax is 7 </reasoning>\n"
            "<answer>7</answer>"
        )
        rewards = check_answer(prompts=["p"], completions=[completion], answer=["7"])
        assert rewards == [3.0]

    def test_stripped_match(self):
        rewards = check_answer(
            prompts=["p"],
            completions=["<reasoning>x</reasoning>\n<answer> 4 </answer>"],
            answer=["4"],
        )
        assert rewards == [1.5]

    def test_close_ratio_gets_partial_credit(self):
        rewards = check_answer(
            prompts=["p"],
            completions=["<reasoning>x</reasoning>\n<answer>105</answer>"],
            answer=["100"],
        )
        assert rewards == [0.5]

    def test_wrong_answer_penalized(self):
        rewards = check_answer(
            prompts=["p"],
            completions=["<reasoning>x</reasoning>\n<answer>2</answer>"],
            answer=["100"],
        )
        assert rewards == [-1.0]

    def test_malformed_scores_zero(self):
        rewards = check_answer(prompts=["p"], completions=["garbage"], answer=["4"])
        assert rewards == [0.0]

    def test_thousands_separators(self):
        # Textually different ("1,000" vs "1000") but numerically exact:
        # the ratio band grants partial credit per notebook semantics.
        rewards = check_answer(
            prompts=["p"],
            completions=["<reasoning>x</reasoning>\n<answer>1,000</answer>"],
            answer=["1000"],
        )
        assert rewards == [0.5]


class TestCheckNumbers:
    def test_exact_number(self):
        rewards = check_numbers(
            prompts=["p"],
            completions=["<answer>0.34</answer>"],
            answer=["0.34"],
        )
        assert rewards == [2.0]

    def test_number_in_reasoning_is_ignored(self):
        rewards = check_numbers(
            prompts=["p"],
            completions=["<reasoning>1 apple</reasoning>\n<answer>2</answer>"],
            answer=["2"],
        )
        assert rewards == [2.0]

    def test_no_answer_tag_scores_zero(self):
        rewards = check_numbers(prompts=["p"], completions=["42"], answer=["42"])
        assert rewards == [0.0]
