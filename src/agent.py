from typing import Tuple
from functools import partial
from src.utils import (
    agree, 
    match, 
    encode_image, 
    parse_response, 
    assemble_prompt)

import os
from openai import OpenAI
# openai_org = os.getenv("OPENAI_ORG")
# openai_project = os.getenv("OPENAI_PROJECT")
# openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization="org-FS3BNL7yaD4kX7b68zAMckVr",
    project="proj_eHzIByecPCksPXepXpcAB9cT",
    api_key="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"
)


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
        raise NotImplementedError("Subclass must implement abstract method.")


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
        x, sess = D[-1]

        # assemble prompt and ask LLM
        P_j = assemble_prompt(x, C)
        response = self.llm(messages=P_j)
        try:
            y, e, C = parse_response(response, C)
        except AssertionError as e:
            return ("problem", 0, 0), C

        # get label for the machine
        l_m = None
        # if j <= 2, we don't have human's response yet
        if (j > 2):
            _, y_h, e_h = M[-1][3]
            matchOK = match(y, y_h)
            agreeOK = agree(e, e_h)
            if not matchOK and not agreeOK:
                if j > k:
                    l_m = "reject"
                else:
                    l_m = "refute"
            else:
                l_m = "revise"
        else:
            l_m = "revise"

        return (l_m, y, e), C


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
        D, M, C = delta

        # current session & example
        x, sess = D[-1]
        y, img, e = x
        encoded_img = encode_image(img)

        # get prev machine response
        mu_m = M[-1][3]
        _, y_m, e_m = mu_m

        # get label
        l_h, human_response = None, None
        matchOK = match(y, y_m)
        agreeOK = agree(e, e_m)
        if matchOK and agreeOK:
            l_h = "ratify"
            human_response = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Both the diagnosis and the explanation are correct. Great job!"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_img}"
                        }
                    }
                ]
            }
        elif not matchOK and agreeOK:
            l_h = "refute"
            human_response = {
                "role": "user",
                "content": f"The diagnosis about {y} is incorrect, but the explanation is correct. Please correct the diagnosis."
            }
        elif matchOK and not agreeOK:
            l_h = "refute"
            human_response = {
                "role": "user",
                "content": f"The diagnosis about {y} is correct, but the explanation is incorrect. Please correct the explanation. The correct explanation is {e}."
            }
        else:
            if j > k:
                l_h = "reject"
            else:
                l_h = "refute"
                human_response = {
                    "role": "user",
                    "content": f"Both the diagnosis and the explanation are incorrect. Please correct both. The correct explanation is {e}."
                }

        # append the human response to the conversation
        if human_response is not None:
            C.append(human_response)

        return (l_h, y, e), C
