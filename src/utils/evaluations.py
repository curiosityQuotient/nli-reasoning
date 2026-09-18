"""Evaluation utilities for GSM8K and NLI tasks."""

from typing import Any, Iterator, List, Optional, Tuple, Union

from tqdm import tqdm

from src.models.config import SYSTEM_PROMPT, TEMPLATE, TOTAL_GENERATION_STEPS
from src.rewards import (
    match_format,
    match_format_nli,
    match_letters,
    match_numbers,
    parse_number,
)


def generate(
    question: Union[str, List[str]],
    sampler: Any,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: Optional[int] = None,
) -> Union[str, List[str]]:
    """Given prompt, generates text.

    Args:
        question: Input question(s)
        sampler: Text sampler/generator
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        seed: Random seed

    Returns:
        Generated text(s)
    """
    questions = [question] if isinstance(question, str) else list(question)
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in questions
    ]

    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
    )

    output = out_data.text

    if isinstance(question, str):
        return output[0]
    return output


def _response_text(response: Any) -> str:
    """Normalize a model response to plain text."""
    if isinstance(response, bytes):
        return response.decode("utf-8", errors="replace")
    return str(response)


def extract_numeric_answer(response: Any) -> Optional[float]:
    """Extract the final numeric answer from a formatted response.

    Prefers the text between the <answer> tags; falls back to the first
    number found after the <answer> tag. Numbers quoted inside the
    reasoning are never extracted.

    Args:
        response: Model response (str or bytes)

    Returns:
        The extracted value, or None if no number can be parsed
    """
    text = _response_text(response)

    answer_match = match_format.search(text)
    if answer_match is not None:
        value = parse_number(answer_match.group(1))
        if value is not None:
            return value

    fallback = match_numbers.search(text)
    if fallback is not None:
        return parse_number(fallback.group(1))
    return None


def extract_label_answer(response: Any) -> Optional[str]:
    """Extract the final NLI label from a formatted response.

    Prefers the text between the <answer> tags; falls back to the first
    word-like token found after the <answer> tag.

    Args:
        response: Model response (str or bytes)

    Returns:
        The extracted label text, or None
    """
    text = _response_text(response)

    answer_match = match_format_nli.search(text)
    if answer_match is not None:
        return answer_match.group(2).strip()

    fallback = match_letters.search(text)
    if fallback is not None:
        return fallback.group(1).strip()
    return None


def _gsm8k_scores(
    text: str,
    answer_val: Optional[float],
) -> Tuple[bool, bool, bool]:
    """Score one GSM8K response.

    Args:
        text: Model response text
        answer_val: Parsed true answer, or None if unparseable

    Returns:
        Tuple of (correct, partial_correct, formatted)
    """
    extracted_val = extract_numeric_answer(text)

    correct = (
        extracted_val is not None
        and answer_val is not None
        and extracted_val == answer_val
    )
    partial = (
        extracted_val is not None
        and answer_val is not None
        and answer_val != 0
        and not correct
        and 0.9 <= extracted_val / answer_val <= 1.1
    )
    formatted = match_format.search(text) is not None
    return correct, partial, formatted


def _nli_scores(text: str, answer: str) -> Tuple[bool, bool]:
    """Score one NLI response.

    Args:
        text: Model response text
        answer: True single-letter label

    Returns:
        Tuple of (correct, formatted)
    """
    extracted = extract_label_answer(text)
    correct = extracted is not None and extracted == str(answer).strip()
    formatted = match_format.search(text) is not None
    return correct, formatted


class _Tally:
    """Accumulates per-question scoring across an evaluation run."""

    def __init__(self, corr_lst: bool, make_lst: bool, partial_enabled: bool):
        self.corr = 0
        self.partial = 0
        self.corr_format = 0
        self.total = 0
        self.response_lst: List[Any] = []
        self._corr_lst = corr_lst
        self._make_lst = make_lst
        self._partial_enabled = partial_enabled

    def add(
        self,
        question: str,
        answer: str,
        responses: List[Any],
        correct: bool,
        partial: bool,
        formatted: bool,
    ) -> None:
        """Record one evaluated question."""
        if correct:
            self.corr += 1
        if partial:
            self.partial += 1
        if formatted:
            self.corr_format += 1
        self.total += 1

        if self._make_lst and correct == self._corr_lst:
            self.response_lst.append((question, answer, responses))

        if self.total % 10 == 0:
            partial_pct = (
                self.partial / self.total * 100 if self._partial_enabled else 0.0
            )
            print(
                f"===> {self.corr=}, {self.total=}, "
                f"{self.corr / self.total * 100=}, {partial_pct}, "
                f"{self.corr_format / self.total * 100=}"
            )


def _iter_batches(
    dataset: Any,
    sampler: Any,
    temperature: float,
    top_k: int,
    top_p: float,
    num_passes: int,
) -> Iterator[Tuple[Any, List[str], List[List[str]]]]:
    """Yield (batch, questions, per-question responses) for each batch."""
    for batch in tqdm(dataset):
        questions = batch["question"]

        multiple_call_responses: List[List[Any]] = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(questions, sampler, temperature, top_k, top_p, seed=p)
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)

        yield batch, questions, multiple_call_responses


def evaluate(
    dataset: Any,
    sampler: Any,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    num_passes: int = 1,
    corr_lst: bool = False,
    make_lst: bool = False,
) -> Union[
    Tuple[int, int, float, float, float],
    Tuple[Tuple[int, int, float, float, float], List],
]:
    """Computes accuracy and percentage of outputs matching the format.

    Correct answers count towards accuracy; answers within a 0.9-1.1
    ratio of the truth (but not exact) count towards partial accuracy.

    Args:
        dataset: Evaluation dataset
        sampler: Text sampler
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_passes: Number of generation passes per question
        corr_lst: Whether to include correct outputs in response list
        make_lst: Whether to return response list

    Returns:
        Tuple of metrics or tuple of metrics and response list
    """
    tally = _Tally(corr_lst, make_lst, partial_enabled=True)

    for batch, questions, multiple_call_responses in _iter_batches(
        dataset, sampler, temperature, top_k, top_p, num_passes
    ):
        answers = batch["answer"]
        for question, responses, answer in zip(
            questions, multiple_call_responses, answers, strict=True
        ):
            answer_val = parse_number(answer)
            correct = False
            partial = False
            formatted = False

            for response in responses:
                is_correct, is_partial, is_formatted = _gsm8k_scores(
                    _response_text(response), answer_val
                )
                correct = correct or is_correct
                partial = partial or is_partial
                formatted = formatted or is_formatted
                if correct and partial and formatted:
                    break

            tally.add(question, answer, responses, correct, partial, formatted)

    to_return = (
        tally.corr,
        tally.total,
        tally.corr / tally.total * 100,
        tally.partial / tally.total * 100,
        tally.corr_format / tally.total * 100,
    )
    if make_lst:
        return to_return, tally.response_lst
    return to_return


def evaluate_nli(
    dataset: Any,
    sampler: Any,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    num_passes: int = 1,
    corr_lst: bool = False,
    make_lst: bool = False,
) -> Union[Tuple[int, int, float, float], Tuple[Tuple[int, int, float, float], List]]:
    """Computes NLI accuracy and percentage of outputs matching the format.

    Args:
        dataset: Evaluation dataset
        sampler: Text sampler
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        num_passes: Number of generation passes per question
        corr_lst: Whether to include correct outputs in response list
        make_lst: Whether to return response list

    Returns:
        Tuple of (correct, total, accuracy %, format %) metrics, or a
        tuple of metrics and response list
    """
    tally = _Tally(corr_lst, make_lst, partial_enabled=False)

    for batch, questions, multiple_call_responses in _iter_batches(
        dataset, sampler, temperature, top_k, top_p, num_passes
    ):
        answers = batch["answer"]
        for question, responses, answer in zip(
            questions, multiple_call_responses, answers, strict=True
        ):
            correct = False
            formatted = False

            for response in responses:
                is_correct, is_formatted = _nli_scores(
                    _response_text(response), str(answer)
                )
                correct = correct or is_correct
                formatted = formatted or is_formatted
                if correct and formatted:
                    break

            tally.add(question, answer, responses, correct, False, formatted)

    to_return = (
        tally.corr,
        tally.total,
        tally.corr / tally.total * 100,
        tally.corr_format / tally.total * 100,
    )
    if make_lst:
        return to_return, tally.response_lst
    return to_return
