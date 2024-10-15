
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