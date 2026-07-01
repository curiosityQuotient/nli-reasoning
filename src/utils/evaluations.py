"""
evaluations.py 

Evaluations that are of use in this script
"""

import re
from typing import Any, List, Optional, Tuple, Union

from tqdm import tqdm

from src.models.config import SYSTEM_PROMPT, TEMPLATE


match_format = re.compile(
    rb"<reasoning>[\s\S]*?<\/reasoning>\n<answer>[\s\S]*?$"
)

match_numbers = re.compile(
    rb"[\s]{0,5}([-]?[\d\.,]{1,20})"
)

match_letters = re.compile(
    rb"[\s]{0,5}([a-z])"
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
    if isinstance(question, str):
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=question,
            ),
        ]
    else:
        input_batch = [
            TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=q,
            )
            for q in question
        ]

    out_data = sampler(
        input_strings=input_batch,
        max_generation_steps=1024,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed if seed is not None else None,
    )

    output = out_data.text

    if isinstance(question, str):
        return output[0]
    return output


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
    Tuple[Tuple[int, int, float, float, float], List]
]:
    """Computes accuracy and percentage of outputs matching the format.
    
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
    response_lst = []
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]

        multiple_call_responses = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(
                questions, sampler, temperature, top_k, top_p, seed=p
            )
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)

        for question, multiple_call_response, answer in zip(
            questions, multiple_call_responses, answers
        ):
            corr_ctr_per_question = 0
            partially_corr_per_question = 0
            corr_format_per_question = 0
            
            for response in multiple_call_response:
                if isinstance(response, str):
                    response = response.encode()
                    
                extracted_response = (
                    guess.group(1)
                    if (guess := match_numbers.search(response)) is not None
                    else b"-1000000"
                )

                try:
                    extracted_val = float(extracted_response.decode().strip())
                    answer_val = float(str(answer).strip())
                    
                    if extracted_val == answer_val:
                        corr_ctr_per_question += 1

                        ratio = extracted_val / answer_val if answer_val != 0 else 0
                        if ratio >= 0.9 and ratio <= 1.1:
                            partially_corr_per_question += 1
                except (ValueError, AttributeError):
                    print("SKIPPED")

                if match_format.search(response) is not None:
                    corr_format_per_question += 1

                if (
                    corr_ctr_per_question > 0
                    and partially_corr_per_question > 0
                    and corr_format_per_question > 0
                ):
                    break

            if corr_ctr_per_question > 0:
                corr += 1
                if corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            else:
                if not corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            if partially_corr_per_question > 0:
                partially_corr += 1
            if corr_format_per_question > 0:
                corr_format += 1

            total += 1
            if total % 10 == 0:
                print(
                    f"===> {corr=}, {total=}, {corr / total * 100=}, "
                    f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
                )

    to_return = (
        corr,
        total,
        corr / total * 100,
        partially_corr / total * 100,
        corr_format / total * 100,
    )
    if make_lst:
        return to_return, response_lst
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
) -> Union[
    Tuple[int, int, float, float, float],
    Tuple[Tuple[int, int, float, float, float], List]
]:
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
        Tuple of metrics or tuple of metrics and response list
    """
    response_lst = []
    corr = 0
    partially_corr = 0
    corr_format = 0
    total = 0

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]

        multiple_call_responses = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(
                questions, sampler, temperature, top_k, top_p, seed=p
            )
            for idx, response in enumerate(responses):
                multiple_call_responses[idx].append(response)

        for question, multiple_call_response, answer in zip(
            questions, multiple_call_responses, answers
        ):
            corr_ctr_per_question = 0
            partially_corr_per_question = 0
            corr_format_per_question = 0
            
            for response in multiple_call_response:
                if isinstance(response, str):
                    response = response.encode()
                    
                extracted_response = (
                    guess.group(1)
                    if (guess := match_letters.search(response)) is not None
                    else b"-1000000"
                )

                try:
                    if extracted_response.decode().strip() == str(answer).strip():
                        corr_ctr_per_question += 1
                except (ValueError, AttributeError):
                    print("SKIPPED")

                if match_format.search(response) is not None:
                    corr_format_per_question += 1

                if (
                    corr_ctr_per_question > 0
                    and partially_corr_per_question > 0
                    and corr_format_per_question > 0
                ):
                    break

            if corr_ctr_per_question > 0:
                corr += 1
                if corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            else:
                if not corr_lst and make_lst:
                    response_lst.append((question, answer, multiple_call_response))
            if partially_corr_per_question > 0:
                partially_corr += 1
            if corr_format_per_question > 0:
                corr_format += 1

            total += 1
            if total % 10 == 0:
                print(
                    f"===> {corr=}, {total=}, {corr / total * 100=}, "
                    f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
                )

    to_return = (
        corr,
        total,
        corr / total * 100,
        partially_corr / total * 100,
        corr_format / total * 100,
    )
    if make_lst:
        return to_return, response_lst
    return to_return
