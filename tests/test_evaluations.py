"""Tests for evaluation extraction helpers and metric loops."""

from types import SimpleNamespace

from src.utils.evaluations import (
    evaluate,
    evaluate_nli,
    extract_label_answer,
    extract_numeric_answer,
)

WELL_FORMED_GSM8K = "<reasoning> Two plus two is four. </reasoning>\n<answer>4</answer>"
WELL_FORMED_NLI = (
    "<reasoning> Indeed what a day! </reasoning>\n"
    "<certainty>10</certainty>\n"
    "<answer>n</answer>"
)


class FakeSampler:
    """Minimal sampler stand-in returning canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, input_strings, **kwargs):
        return SimpleNamespace(text=list(self.responses))


class TestExtraction:
    def test_numeric_prefers_answer_tag(self):
        # Regression: numbers quoted in the reasoning must be ignored.
        text = "<reasoning> The cost is 100 dollars. </reasoning>\n<answer>42</answer>"
        assert extract_numeric_answer(text) == 42.0

    def test_numeric_fallback_after_answer_tag(self):
        assert extract_numeric_answer("<answer> 3.5") == 3.5

    def test_numeric_none_without_answer(self):
        assert extract_numeric_answer("<reasoning>9</reasoning> plain text") is None

    def test_numeric_handles_thousands_separators(self):
        assert extract_numeric_answer("<answer>1,024</answer>") == 1024.0

    def test_numeric_accepts_bytes(self):
        assert extract_numeric_answer(b"<answer>7</answer>") == 7.0

    def test_label_prefers_answer_tag(self):
        assert extract_label_answer(WELL_FORMED_NLI) == "n"

    def test_label_fallback_after_answer_tag(self):
        assert extract_label_answer("<answer> e </answer>") == "e"

    def test_label_none_without_answer(self):
        assert extract_label_answer("<reasoning>t</reasoning> plain text") is None


class TestEvaluate:
    def test_correct_response(self):
        dataset = [{"question": ["q1"], "answer": ["4"]}]
        metrics = evaluate(dataset, FakeSampler([WELL_FORMED_GSM8K]))
        assert metrics == (1, 1, 100.0, 0.0, 100.0)

    def test_partial_accuracy_within_ratio(self):
        dataset = [{"question": ["q1"], "answer": ["100"]}]
        metrics = evaluate(
            dataset, FakeSampler(["<reasoning>x</reasoning>\n<answer>105</answer>"])
        )
        # Not exact, but within the 0.9-1.1 band: partial credit only
        assert metrics == (0, 1, 0.0, 100.0, 100.0)

    def test_number_in_reasoning_not_scored(self):
        # Regression: the first number in the reasoning (100) must not
        # be compared against the true answer (7).
        dataset = [{"question": ["q1"], "answer": ["7"]}]
        metrics = evaluate(
            dataset,
            FakeSampler(
                [
                    "<reasoning> The cost is 100 dollars. </reasoning>"
                    "\n<answer>7</answer>"
                ]
            ),
        )
        assert metrics == (1, 1, 100.0, 0.0, 100.0)

    def test_wrong_and_unformatted(self):
        dataset = [{"question": ["q1"], "answer": ["4"]}]
        metrics = evaluate(dataset, FakeSampler(["I think it is 5"]))
        assert metrics == (0, 1, 0.0, 0.0, 0.0)

    def test_multiple_questions(self):
        dataset = [{"question": ["q1", "q2"], "answer": ["4", "5"]}]
        metrics = evaluate(
            dataset,
            FakeSampler(
                [WELL_FORMED_GSM8K, "<reasoning>x</reasoning>\n<answer>9</answer>"]
            ),
        )
        assert metrics == (1, 2, 50.0, 0.0, 100.0)


class TestEvaluateNli:
    def test_correct_response(self):
        dataset = [{"question": ["q1"], "answer": ["n"]}]
        metrics = evaluate_nli(dataset, FakeSampler([WELL_FORMED_NLI]))
        assert metrics == (1, 1, 100.0, 100.0)

    def test_letter_in_reasoning_not_scored(self):
        # Regression: the first lowercase letter in the reasoning must
        # not be compared against the label.
        dataset = [{"question": ["q1"], "answer": ["n"]}]
        metrics = evaluate_nli(
            dataset,
            FakeSampler(["<reasoning> the answer is </reasoning>\n<answer>e</answer>"]),
        )
        assert metrics == (0, 1, 0.0, 100.0)

    def test_wrong_label(self):
        dataset = [{"question": ["q1"], "answer": ["e"]}]
        metrics = evaluate_nli(dataset, FakeSampler([WELL_FORMED_NLI]))
        assert metrics == (0, 1, 0.0, 100.0)

    def test_returns_four_metrics(self):
        dataset = [{"question": ["q1"], "answer": ["n"]}]
        metrics = evaluate_nli(dataset, FakeSampler([WELL_FORMED_NLI]))
        assert len(metrics) == 4
