"""Reward functions for GRPO training.

These mirror the notebook's reward semantics:

- ``match_format_exactly``: 3.0 if the reasoning/answer tags are present,
  else 0.
- ``match_format_approximately``: +/-0.5 per structural tag, penalising
  duplicated tags.
- ``check_answer``: GSM8K answer checking with partial credit for answers
  within a ratio band of the truth.
- ``check_answer_nli``: uncertainty-aware NLI scoring.

All patterns are plain string regexes. Tunix passes completions as
strings; the helpers tolerate bytes defensively.
"""

import re
from typing import List, Optional, Union

Text = Union[str, bytes]

REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
CERTAINTY_START = "<certainty>"
CERTAINTY_END = "</certainty>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

# Format matches require reasoning followed by the answer, optionally a
# certainty tag. Capture group 1 holds the answer text; for the NLI
# variant group 1 is the certainty and group 2 the answer.
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{ANSWER_START}(.+?){ANSWER_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format_nli = re.compile(
    rf"^[\s]{{0,}}"
    rf"{REASONING_START}.+?{REASONING_END}.*?"
    rf"{CERTAINTY_START}(.+?){CERTAINTY_END}.*?"
    rf"{ANSWER_START}(.+?){ANSWER_END}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Extractors anchored at the answer tag so that numbers or labels quoted
# inside the reasoning are never mistaken for the final answer.
match_numbers = re.compile(
    rf"{ANSWER_START}.*?([\d\.,]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

match_letters = re.compile(
    rf"{ANSWER_START}.*?([\w\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)

STRUCTURAL_TAGS = (
    REASONING_START,
    REASONING_END,
    CERTAINTY_START,
    CERTAINTY_END,
    ANSWER_START,
    ANSWER_END,
)


def _as_str(text: Text) -> str:
    """Decode completions that arrive as bytes."""
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text


def parse_number(text: Text) -> Optional[float]:
    """Parse a float, tolerating thousands separators and stray trailing
    punctuation (e.g. ``"1,024."``)."""
    try:
        cleaned = str(text).strip().rstrip(".,").replace(",", "")
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def match_format_exactly(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Reward +3.0 if output format matches exactly, else 0.

    Args:
        prompts: List of input prompts
        completions: List of model completions

    Returns:
        List of reward values
    """
    return [
        3.0 if match_format.search(_as_str(completion)) is not None else 0.0
        for completion in completions
    ]


def match_format_approximately(
    prompts: List[str], completions: List[str], **kwargs
) -> List[float]:
    """Reward +0.5 per structural tag present exactly once, else -0.5.

    Tags: reasoning open/close, certainty open/close, answer open/close.
    Duplicated tags are penalised to discourage repetition.

    Args:
        prompts: List of input prompts
        completions: List of model completions

    Returns:
        List of reward values
    """
    rewards = []
    for completion in completions:
        text = _as_str(completion)
        reward = 0.0
        for tag in STRUCTURAL_TAGS:
            reward += 0.5 if text.count(tag) == 1 else -0.5
        rewards.append(reward)
    return rewards


def uncert_aware_reward(outcome: int, certainty: Text, actual_certainty: Text) -> float:
    """Core uncertainty-aware reward calculation.

    Rewards correct predictions made with high certainty and incorrect
    predictions made with low certainty, plus alignment between the
    predicted and actual certainty. Both certainties are on a 1-10 scale
    and normalised to [0, 1] internally.

    Args:
        outcome: 1 if correct, 0 if incorrect
        certainty: Model's predicted certainty (1-10 scale)
        actual_certainty: Ground truth certainty (1-10 scale)

    Returns:
        Reward value
    """
    if not isinstance(certainty, float):
        try:
            certainty = float(certainty)  # type: ignore[assignment]
        except (ValueError, TypeError):
            # Unparseable certainty yields a small outcome-only reward,
            # matching the notebook's fallback.
            return outcome * 0.5
    if not isinstance(actual_certainty, float):
        actual_certainty = float(actual_certainty)  # type: ignore[assignment]

    certainty = certainty / 10.0  # type: ignore[operator]
    actual_certainty = actual_certainty / 10.0  # type: ignore[operator]

    reward = 0.5 * (outcome * (1 + certainty) + (1 - outcome) * (1 - certainty)) + (
        1 - abs(certainty - actual_certainty)
    )
    return reward


def check_answer_nli(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    true_uncertainty: Optional[List[int]] = None,
    **kwargs,
) -> List[float]:
    """NLI-specific answer checking with uncertainty awareness.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers ('e', 'c', or 'n')
        true_uncertainty: Ground truth certainty values (1-10)

    Returns:
        List of reward values
    """
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        text = _as_str(completion)
        match = match_format_nli.search(text)
        if match is None:
            # Malformed response: neither answer nor certainty extracted.
            rewards.append(0.0)
            continue

        predicted_certainty = match.group(1)
        predicted_answer = match.group(2)

        if true_uncertainty is not None and idx < len(true_uncertainty):
            rewards.append(
                uncert_aware_reward(
                    1 if predicted_answer.strip() == str(answer[idx]).strip() else 0,
                    predicted_certainty,
                    true_uncertainty[idx],
                )
            )
        else:
            is_correct = predicted_answer.strip() == str(answer[idx]).strip()
            rewards.append(2.0 if is_correct else -1.0)

    return rewards


def check_answer(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[float]:
    """GSM8K answer checking.

    Extracts the text between the <answer> tags and scores it: exact
    string match earns 3.0, stripped match 1.5, and answers within a
    ratio band of the truth earn partial credit. Numbers appearing in
    the reasoning are never considered.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers

    Returns:
        List of reward values
    """
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        text = _as_str(completion)
        match = match_format.search(text)
        if match is None:
            rewards.append(0.0)
            continue

        guess = match.group(1)
        true_answer = str(answer[idx])

        if guess == true_answer:
            rewards.append(3.0)
            continue
        if guess.strip() == true_answer.strip():
            rewards.append(1.5)
            continue

        predicted = parse_number(guess)
        actual = parse_number(true_answer)
        if predicted is None or actual is None or actual == 0:
            rewards.append(-0.5)
            continue

        ratio = predicted / actual
        if 0.9 <= ratio <= 1.1:
            rewards.append(0.5)
        elif 0.8 <= ratio <= 1.2:
            rewards.append(0.25)
        else:
            rewards.append(-1.0)

    return rewards


def check_numbers(
    prompts: List[str], completions: List[str], answer: List[str], **kwargs
) -> List[float]:
    """Extract and compare numeric answers found after the answer tag.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers

    Returns:
        List of reward values
    """
    rewards: List[float] = []

    for idx, completion in enumerate(completions):
        text = _as_str(completion)
        predicted_match = match_numbers.search(text)

        if predicted_match is None:
            rewards.append(0.0)
            continue

        predicted = parse_number(predicted_match.group(1))
        actual = parse_number(answer[idx])
        if predicted is None or actual is None:
            rewards.append(0.0)
            continue

        if predicted == actual:
            rewards.append(2.0)
        elif actual != 0 and 0.9 <= predicted / actual <= 1.1:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)

    return rewards
