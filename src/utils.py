from typing import Tuple, List, Union, Dict
Prompt = Union[Dict[str, str], List[Dict[str, str]]]

import os
from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(organization=openai_org, 
                project=openai_project, 
                api_key=openai_key)


def assemble_context(mu_h: Tuple, mu_m: Tuple, C: List) -> List:
    """
    Assemble context for the machine.

    Args:
        mu_h (Tuple): human's response
        mu_m (Tuple): machine's response
        C (List): context uptil now

    Returns:
        Tuple: context
    """
    _, y_h, e_h = mu_h
    _, y_m, e_m = mu_m
    matchOK = match(y_h, y_m)
    agreeOK = agree(e_h, e_m)
    C.append((y_h, y_m, e_h, e_m, matchOK, agreeOK))
    return C


def assemble_prompt(x, c_j) -> Prompt:
    """
    Assemble prompt for the LLM.

    Args:
        x: example
        c_j: context

    Returns:
        Prompt: prompt
    """
    if c_j is None:
        message = ""


def match(y, pred) -> bool:
    """
    Match the prediction with the example.

    Args:
        y: prediction
        pred: prediction

    Returns:
        bool: True if the prediction matches the example, False otherwise
    """
    ys = str(y).lower().split(",")
    preds = str(pred).lower().split(",")
    return any(y in preds for y in ys)


def agree(e, e_pred) -> bool:
    """
    Check if the explanation agrees with the prediction.

    Args:
        e: explanation
        e_pred: explanation

    Returns:
        bool: True if the explanation agrees with the prediction, False otherwise
    """
    pass
