from typing import Tuple, List

def assemble_context(mu_h: Tuple, mu_m: Tuple) -> List:
    """
    Assemble context for the machine.

    Args:
        mu_h (Tuple): human's response
        mu_m (Tuple): machine's response

    Returns:
        Tuple: context
    """
    pass


def assemble_prompt(x, c_j) -> str:
    """
    Assemble prompt for the LLM.

    Args:
        x: example
        c_j: context

    Returns:
        str: prompt
    """
    pass


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
