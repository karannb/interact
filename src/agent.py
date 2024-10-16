from typing import Tuple
from functools import partial
from src.utils import assemble_context, assemble_prompt

import os
from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(organization=openai_org, 
                project=openai_project, 
                api_key=openai_key)


class Agent:
    """
    Generic class for agents,
    which can be either a human or a machine.
    """
    def __init__(self, type: str, id: int):
        self.type = type
        self.id = id

    def ask(self, j: int, k: int, delta: Tuple) -> Tuple:
        """
        Ask the agent for a response.

        Args:
            j (int): interaction identifier
            k (int): the minimum number of interactions after which the "Reject" tag can be sent
            delta (Tuple): context of the conversation

        Returns:
            Tuple: response (tag, pred, expl)
        """
        raise NotImplementedError, "Subclass must implement abstract method."


class Machine(Agent):
    """
    Machine agent class.
    """
    def __init__(self, id: int):
        super().__init__("Machine", id)
        self.llm = partial(client.chat.completions.create, 
                           model="gpt-4o",
                           max_tokens=300)

    def ask(self, j: int, k: int, delta: Tuple) -> Tuple:
        """
        Ask the machine for a response.

        Args:
            j (int): interaction identifier
            k (int): the minimum number of interactions after which the "Reject" tag can be sent
            delta (Tuple): context of the conversation

        Returns:
            Tuple: response (tag, pred, expl)
        """
        D, M, C = delta

        # current session & example
        sess, x = D[-1]

        # assemble context for the machine if j > 2
        if j > 2:
            mu_h = M[-1][3]
            mu_m = M[-2][3]
            c_j = assemble_context(mu_h, mu_m, C)
        else:
            c_j = None

        # assemble prompt and ask LLM
        P_j = assemble_prompt(x, c_j)
        y, e = self.llm(messages=P_j)

        # get label
        l_m = None
        # if j <= 2, we don't have human's response yet
        if (j > 2):
            _, y_h, e_h = mu_h
            if y != y_h and e != e_h:
                if j > k:
                    l_m = "reject"
                else:
                    l_m = "refute"
            else:
                l_m = "revise"
        else:
            l_m = "revise"

        return l_m, y, e


class Human(Agent):
    """
    Human agent class.
    """
    def __init__(self, id: int):
        super().__init__("Human", id)

    def ask(self, j: int, k: int, delta: Tuple) -> Tuple:
        """
        Ask the human for a response.

        Args:
            j (int): interaction identifier
            k (int): the minimum number of interactions after which the "Reject" tag can be sent
            delta (Tuple): context of the conversation

        Returns:
            Tuple: response (tag, pred, expl)
        """
        D, M, _ = delta

        # current session & example
        sess, x = D[-1]
        e, y = x

        # get prev machine response
        mu_m = M[-1][3]
        _, y_m, e_m = mu_m

        # get label
        l_h = None
        if y_m == y and e_m == e:
            l_h = "ratify"
        elif y_m != y and e_m != e:
            if j > k:
                l_h = "reject"
            else:
                l_h = "refute"

        return l_h, y, e
