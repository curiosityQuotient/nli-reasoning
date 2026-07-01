"""Reward functions for GRPO training."""

import re
from typing import Dict, List, Optional

match_format = re.compile(
    rb"<reasoning>[\s\S]*?<\/reasoning>\n<answer>[\s\S]*?$"
)

match_format_nli = re.compile(
    rb"<reasoning>[\s\S]*?<\/reasoning>\n<certainty>(.?)<\/certainty>\n<answer>(.?)$",
    re.DOTALL
)

match_numbers = re.compile(
    rb"[\s]{0,5}([-]?[\d\.,]{1,20})"
)

match_letters = re.compile(
    rb"[\s]{0,5}([a-z])"
)


def match_format_exactly(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """Reward +3.0 if output format matches exactly.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        
    Returns:
        List of reward values
    """
    rewards = []
    for completion in completions:
        if match_format.search(completion) is not None:
            rewards.append(3.0)
        else:
            rewards.append(-1.0)
    return rewards


def match_format_approximately(
    prompts: List[str],
    completions: List[str],
    **kwargs
) -> List[float]:
    """Reward based on presence of correct tags.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        
    Returns:
        List of reward values
    """
    rewards = []
    for completion in completions:
        reward = 0.0
        if b"<reasoning>" in completion:
            reward += 0.5
        if b"</reasoning>" in completion:
            reward += 0.5
        if b"<answer>" in completion:
            reward += 0.5
        if b"<certainty>" in completion:
            reward += 0.5
        rewards.append(reward)
    return rewards


def uncert_aware_reward(
    outcome: int,
    certainty: float,
    actual_certainty: float
) -> float:
    """Core uncertainty-aware reward calculation.
    
    Rewards:
    - Correct predictions with high confidence
    - Wrong predictions with low confidence
    - Alignment between predicted and actual uncertainty
    
    Args:
        outcome: 1 if correct, 0 if incorrect
        certainty: Model's predicted certainty (1-10 scale)
        actual_certainty: Ground truth certainty (1-10 scale)
        
    Returns:
        Reward value
    """
    certainty = certainty / 10.0
    actual_certainty = actual_certainty / 10.0
    
    reward = (
        0.5 * (outcome * (1 + certainty) + (1 - outcome) * (1 - certainty))
        + (1 - abs(certainty - actual_certainty))
    )
    return reward


def check_answer_nli(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    true_uncertainty: Optional[List[float]] = None,
    **kwargs
) -> List[float]:
    """NLI-specific answer checking with uncertainty awareness.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers
        true_uncertainty: Optional list of ground truth uncertainty values
        
    Returns:
        List of reward values
    """
    rewards = []
    
    for idx, completion in enumerate(completions):
        match = match_format_nli.search(completion)
        if match is None:
            rewards.append(-2.0)
            continue
            
        predicted_certainty = match.group(1)
        predicted_answer = match.group(2)
        
        try:
            predicted_certainty = float(predicted_certainty.decode())
        except (ValueError, AttributeError):
            predicted_certainty = 5.0
            
        outcome = 1.0 if predicted_answer.decode().strip() == answer[idx].strip() else 0.0
        
        if true_uncertainty is not None and idx < len(true_uncertainty):
            reward = uncert_aware_reward(
                outcome,
                predicted_certainty,
                true_uncertainty[idx]
            )
        else:
            reward = 2.0 if outcome == 1.0 else -1.0
            
        rewards.append(reward)
        
    return rewards


def check_answer(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs
) -> List[float]:
    """GSM8K answer checking.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers
        
    Returns:
        List of reward values
    """
    rewards = []
    
    for idx, completion in enumerate(completions):
        match = match_format.search(completion)
        if match is None:
            rewards.append(-2.0)
            continue
            
        predicted_match = match_numbers.search(match.group(0))
        if predicted_match is None:
            rewards.append(-1.0)
            continue
            
        try:
            predicted = float(predicted_match.group(1).decode().replace(",", ""))
            actual = float(answer[idx].replace(",", ""))
            
            ratio = predicted / actual if actual != 0 else 0
            if ratio >= 0.9 and ratio <= 1.1:
                rewards.append(2.0)
            else:
                rewards.append(-0.5)
        except (ValueError, AttributeError):
            rewards.append(-1.0)
            
    return rewards


def check_numbers(
    prompts: List[str],
    completions: List[str],
    answer: List[str],
    **kwargs
) -> List[float]:
    """Extract and compare numeric answers.
    
    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of correct answers
        
    Returns:
        List of reward values
    """
    rewards = []
    
    for idx, completion in enumerate(completions):
        predicted_match = match_numbers.search(completion)
        
        if predicted_match is None:
            rewards.append(-1.0)
            continue
            
        try:
            predicted = float(predicted_match.group(1).decode().replace(",", ""))
            actual = float(str(answer[idx]).replace(",", ""))
            
            if predicted == actual:
                rewards.append(2.0)
            else:
                ratio = predicted / actual if actual != 0 else 0
                if ratio >= 0.9 and ratio <= 1.1:
                    rewards.append(1.0)
                else:
                    rewards.append(-0.5)
        except (ValueError, AttributeError):
            rewards.append(-1.0)
            
    return rewards
