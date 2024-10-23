from copy import deepcopy
from functools import partial
from typing import Tuple, List, Dict
from src.utils import (
    agree, 
    match, 
    encode_image, 
    parse_response, 
    assemble_prompt)

import os
from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_project = os.getenv("OPENAI_PROJECT")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    project=openai_project,
    api_key=openai_key
)


class Agent:
    """
    Generic class for agents,
    which can be either a human or a machine.
    """
    def __init__(self, type: str, id: int):
        self.type = type
        self.id = id

    def call(self, j: int, k: int, delta: Tuple) -> Tuple:
        """
        Call the agent for a response.

        Args:
            j (int): interaction identifier
            k (int): the minimum number of interactions after which the "Reject" tag can be sent
            delta (Tuple): context of the conversation

        Returns:
            Tuple: response (tag, pred, expl)
        """
        D, M, C = delta # we don't need to set C_0 = None, because that's already done.

        # current session & example
        x, sess = D[-1]

        # ask the agent, the "problem" part is needed to 
        # handle the case when the response is invalid
        # for the machine agent
        y_hat = "problem"
        while y_hat == "problem":
            y_hat, e_hat, C = self.ask(x, C)

        # check if we have crossed 2 interactions
        if j >= 2:
            _, yp, ep = M[-1][3] # message from prev agent
            if j == 2:
                ypp, epp = y_hat, e_hat
            else:
                _, ypp, epp = M[-2][3] # message from prev prev agent
            # now check for categories
            if j % 2 == 0: # human
                # check `match` for why this is done
                matchOK = match(ypp, yp)
            else: # machine
                matchOK = match(yp, ypp)
            agreeOK = agree(ep, epp)
            catA = matchOK and agreeOK
            catB = matchOK and not agreeOK
            catC = not matchOK and agreeOK
            catD = not matchOK and not agreeOK
            change = (not match(ypp, y_hat)) or (not agree(epp, e_hat))

            # assign label
            if catA:
                l_hat = "ratify"
            elif catB or catC:
                if change:
                    l_hat = "revise"
                else:
                    l_hat = "refute"
            elif catD:
                if j > k:
                    l_hat = "reject"
                elif change:
                    l_hat = "revise"
                else:
                    l_hat = "refute"
        else:
            l_hat = "init"

        # update context
        C = self.update_context(C, x, l_hat, j)

        return (l_hat, y_hat, e_hat), C
    
    def update_context(self, C: List[Dict], x: Tuple, l_hat, j: int) -> List[Dict]:
        """
        Updates context, i.e. the conversation to make it more conversation-like.

        Args:
            C (List[Dict]): Current context
            x (Tuple): example of (y, img, e)
            mu (Tuple): tuple of (l, y, e)
            j (int): interaction identifier

        Returns:
            List[Dict]: new context
        """
        _, img, _ = x
        if j == 1 or j == 2:
            # first messages are just information about the models
            C[-1]["content"] = "This is my initial opinion on the X-Ray: " + C[-1]["content"]
        else:
            if l_hat == "ratify":
                C[-1]["content"] = "Our opinions match. I agree with you. This conversation is ratified. "
                # now for ratify, we need to add the image (if it has not been added before)
                if not isinstance(C[-2]["content"], list):
                    encoded_img = encode_image(img)
                    new_content = {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": C[-1]["content"]
                                },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_img}"
                                }
                            }
                        ]
                    }
                    C.append(new_content)
            elif l_hat == "reject":
                C[-1]["content"] = "I disagree with you and reject your opinion. " + C[-1]["content"]
            elif l_hat == "revise":
                C[-1]["content"] = "I think I made a mistake. I will revise my opinion. " + C[-1]["content"]
            elif l_hat == "refute":
                C[-1]["content"] = "I think you made a mistake. I refute your opinion. " + C[-1]["content"]

        return C

    def ask(self, x: Tuple, C: List[Dict]) -> Tuple:
        """
        Ask the agent for a response.

        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context

        Returns:
            Tuple: prediction and explanation
        """
        raise NotImplementedError("Subclass must implement abstract method.")


class Machine(Agent):
    """
    Machine agent class.
    """
    def __init__(self, id: int):
        super().__init__("Machine", id)
        self.llm = partial(client.chat.completions.create, 
                           model="gpt-4o-mini",
                           max_tokens=300)

    def ask(self, x: Tuple, C: List[Dict]) -> Tuple:
        """
        Ask the machine for a response.
        
        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context

        Returns:
            Tuple: prediction, explanation and updated context
        """

        # assemble prompt and ask LLM
        P_j = assemble_prompt(x, C)
        response = self.llm(messages=P_j)
        try:
            copied_C = deepcopy(C)
            y_m, e_m, new_C = parse_response(response, copied_C)
        except AssertionError as e:
            print(f"Problem {e} in response {response}, redoing...")
            return "problem", -1, C

        # update context if the response is valid
        C = new_C

        return y_m, e_m, C


class Human(Agent):
    """
    Human agent class.
    """
    def __init__(self, id: int):
        super().__init__("Human", id)

    def ask(self, x: Tuple, C: List[Dict]) -> Tuple:
        """
        Ask the human for a response.

        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context

        Returns:
            Tuple: prediction, explanation and updated context
        """
        # current session & example
        y, _, e = x

        # add the human's response to the context
        human_response = {
            "role": "user",
            "content": f"""*Prediction: Yes*
            *Explanation: {e}*"""
        }
        C.append(human_response)

        return y, e, C
