from copy import deepcopy
from functools import partial
from abc import abstractmethod

from tasks import RAD, DRUG
from utils import bcolors, draw_smiles

import os
from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)

from typing import Tuple, List, Dict, Callable


class Agent:
    """
    Generic class for agents,
    which can be either a human or a machine.
    """
    def __init__(self, type: str, id: int):
        self.type = type
        self.id = id
        self.performance = -1.0
        self.preds = -1.0
        self.expls = -1.0

        # these are set in the subclasses
        self.match: Callable = None
        self.agree: Callable = None

    def __call__(self, j: int, k: int, delta: Tuple) -> Tuple:
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
                # we decide to put the ground truth first
                # and then the prediction / generation
                matchOK = self.match(ypp, yp)
                agreeOK = self.agree(epp, ep)
            else: # machine
                matchOK = self.match(yp, ypp)
                agreeOK = self.agree(ep, epp)
            catA = matchOK and agreeOK
            catB = matchOK and not agreeOK
            catC = not matchOK and agreeOK
            catD = not matchOK and not agreeOK
            change = (not self.match(ypp, y_hat)) or (not self.agree(epp, e_hat))

            # because change is an approximate test, we make sure that the there are no 
            # ambiguities by manually resetting y_hat and e_hat to 2 turns ago
            if not change:
                y_hat, e_hat = ypp, epp

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
            matchOK, agreeOK, change = None, None, None

        # update context
        C = self.update_context(C, x, l_hat, j)

        return (l_hat, y_hat, e_hat), C

    @abstractmethod
    def update_context(self, C: List[Dict], x: Tuple, l_hat, j: int) -> List[Dict]:
        """
        Updates context, i.e. the conversation to make it more conversation-like.

        Args:
            C (List[Dict]): Current context
            x (Tuple): example of (y, img, e)
            l_hat (str): predicted label
            j (int): interaction identifier

        Returns:
            List[Dict]: new context
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the agent for a response.

        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction and explanation
        """
        raise NotImplementedError("Subclass must implement abstract method.")


"""
RAD Task agents.
"""


class RADAgent(Agent):
    """
    Agent class for the RAD task.
    """
    def __init__(self, type: str, id: int):
        super().__init__(type, id)

        # Task-wise parse_response and assemble_prompt are defined in utils.py
        self.parse_response = RAD.parse_response
        self.assemble_prompt = RAD.assemble_prompt
        self.match = RAD.match
        self.agree = RAD.agree

    def update_context(self, C: List[Dict], x: Tuple, l_hat: str, j: int) -> List[Dict]:
        """
        Updates context, i.e. the conversation to make it more conversation-like.

        Args:
            C (List[Dict]): Current context
            x (Tuple): example of (y, img, e)
            l_hat (str): predicted label
            j (int): interaction identifier

        Returns:
            List[Dict]: new context
        """
        if j == 1:
            # first messages are just information about the models
            C[-1]["content"] = "This is my initial diagnosis of the X-Ray: " + C[-1]["content"]
        elif j == 2:
            # need to handle human's first response specially.
            if l_hat == "ratify":
                C[-1]["content"] = "Our diagnoses match. I agree with you. This conversation is ratified. "
            elif l_hat == "reject":
                C[-1]["content"] = "I disagree with you and reject your diagnosis. My diagnosis of this XRay is: " + C[-1]["content"]
            elif l_hat == "revise":
                C[-1]["content"] = "I think I made a mistake. I revise my diagnosis to: " + C[-1]["content"]
            elif l_hat == "refute":
                C[-1]["content"] = "I think you made a mistake. I refute your diagnosis. I think: " + C[-1]["content"]
        else:
            if l_hat == "ratify":
                C[-1]["content"] = "Our diagnoses match. I agree with you. This conversation is ratified. "
            elif l_hat == "reject":
                C[-1]["content"] = "I disagree with you and reject your diagnosis. My diagnosis of this XRay is: " + C[-1]["content"]
            elif l_hat == "revise":
                C[-1]["content"] = "I think I made a mistake. I revise my diagnosis to: " + C[-1]["content"]
            elif l_hat == "refute":
                # because nothing changes, we send the same diagnosis again as 2 steps ago
                if "I think you made a mistake" in C[-3]["content"]:
                    # if we had refuted two steps ago, we need to send the same diagnosis again
                    # without the refutation added again
                    C[-1]["content"] = C[-3]["content"]
                elif isinstance(C[-3]["content"], list):
                    C[-1]["content"] = "I think you made a mistake. I refute your diagnosis. I think: " + C[-1]["content"]
                else:
                    C[-1]["content"] = "I think you made a mistake. I refute your diagnosis. I think: " + C[-3]["content"]

        return C

    @abstractmethod
    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the agent for a response.

        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction and explanation
        """
        raise NotImplementedError("Subclass must implement abstract method.")


class RADMachine(RADAgent):
    """
    Machine agent class.
    """
    def __init__(self, id: int):
        super().__init__("Machine", id)
        self.llm = partial(client.chat.completions.create, 
                           model="gpt-4o-mini",
                           max_tokens=300,
                           seed=42)

    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the machine for a response.
        
        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction, explanation and updated context
        """
        # copy the context to avoid modifying the original (reduntant but safer)
        copied_C = deepcopy(C)

        # assemble prompt and ask LLM
        if is_prompt:
            P_j = C
        else:
            P_j = self.assemble_prompt(x, copied_C)
        response = self.llm(messages=P_j)

        try:
            copied_C = deepcopy(C)
            y_m, e_m, new_C = self.parse_response(response, copied_C)
        except AssertionError as e:
            print()
            print(f"Problem {e} in response {response}, redoing...")
            print()
            return "problem", -1, C

        # update context if the response is valid
        if not is_prompt:
            C = new_C

        return y_m, e_m, C


class RADHuman(RADAgent):
    """
    Human agent class.
    """
    def __init__(self, id: int):
        super().__init__("Human", id)

    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the human for a response.

        Args:
            x (Tuple): example of (y, img, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction, explanation and updated context
        """
        # current session & example
        y, _, e = x

        # add the human's response to the context
        human_response = {
            "role": "user",
            "content": f"""
            *Prediction: {str(y).lower()}*
            *Explanation: {e}*
            """
        }
        C.append(human_response)

        return y, e, C


"""
DRUG Task agents.
"""


class DRUGAgent(Agent):
    """
    Agent class for the DRUG task.
    """
    def __init__(self, type: str, id: int):
        super().__init__(type, id)

        # Task-wise parse_response and assemble_prompt are defined in utils.py
        self.parse_response = DRUG.parse_response
        self.assemble_prompt = DRUG.assemble_prompt
        self.match = DRUG.match
        self.agree = DRUG.agree

    def update_context(self, C: List[Dict], x: Tuple, l_hat, j: int) -> List[Dict]:
        """
        Updates context, i.e. the conversation to make it more conversation-like.

        Args:
            C (List[Dict]): Current context
            x (Tuple): example of (y, mol, e)
            mu (Tuple): tuple of (l, y, e)
            j (int): interaction identifier

        Returns:
            List[Dict]: new context
        """

        if j == 1:
            # first messages are just information about the models
            C[-1]["content"] = "This is the preliminary retrosynthesis pathway I propose: " + C[-1]["content"]
        elif j == 2:
            # need to handle human's first response specially.
            if l_hat == "ratify":
                C[-1]["content"] = "Our opinions match. I agree with you. This conversation is ratified. "
            elif l_hat == "reject":
                C[-1]["content"] = "I disagree with you and reject your opinion. I think the retrosynthesis pathway is: " + C[-1]["content"]
            elif l_hat == "revise":
                C[-1]["content"] = "I think I made a mistake. I revise my opinion to: " + C[-1]["content"]
            elif l_hat == "refute":
                C[-1]["content"] = "I think you made a mistake. I refute your opinion. I think: " + C[-1]["content"]
        else:
            if l_hat == "ratify":
                C[-1]["content"] = "Our opinions match. I agree with you. This conversation is ratified. "
            elif l_hat == "reject":
                C[-1]["content"] = "I disagree with you and reject your opinion. I think the retrosynthesis pathway is: " + C[-1]["content"]
            elif l_hat == "revise":
                C[-1]["content"] = "I think I made a mistake. I revise my opinion to: " + C[-1]["content"]
            elif l_hat == "refute":
                # because nothing changes, we send the same opinion again as 2 steps ago
                if "I think you made a mistake" in C[-3]["content"]:
                    # if we had refuted two steps ago, we need to send the same opinion again
                    # without the refutation added again
                    C[-1]["content"] = C[-3]["content"]
                else:
                    C[-1]["content"] = "I think you made a mistake. I refute your opinion. I think: " + C[-1]["content"]
                

        return C

    @abstractmethod
    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the agent for a response.

        Args:
            x (Tuple): example of (y, mol, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction and explanation
        """
        raise NotImplementedError("Subclass must implement abstract method.")


class DRUGMachine(DRUGAgent):
    """
    Machine agent class.
    """
    def __init__(self, id: int):
        super().__init__("Machine", id)
        self.llm = partial(client.chat.completions.create, 
                           model="gpt-4o",
                           max_tokens=1024,
                           seed=42)

    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the machine for a response.

        Args:
            x (Tuple): example of (y, mol, e)
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction, explanation and updated context
        """

        # copy the context to avoid modifying the original (reduntant but safer)
        copied_C = deepcopy(C)

        # assemble prompt and ask LLM
        if is_prompt:
            P_j = copied_C
        else:
            P_j = self.assemble_prompt(x, copied_C)
        response = self.llm(messages=P_j)
        try:
            # recopy context to remove the query from the context
            copied_C = deepcopy(C)
            y_m, e_m, new_C = self.parse_response(response, copied_C)
        except AssertionError as e:
            print()
            print(f"Problem {e} in response {response}, redoing...")
            print()
            return "problem", -1, C

        # update context if the response is valid
        if not is_prompt:
            C = new_C

        return y_m, e_m, C
    

class DRUGHuman(DRUGAgent):
    """
    Human agent that interacts using a CLI (Command Line Interface).
    """
    def __init__(self, id: int):
        super().__init__("Human", id)

    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask a Human for a response on the Command Line.

        This method prompts a human user to provide a response on the command line.

        Args:
            x (Tuple): (y, mol) tuple.
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or

        Returns:
            Tuple: prediction, explanation and updated context
        """
        # current session & example
        _, mol, _ = x
        print("*"*10)
        print(f"The molecule: {mol}")
        print("*"*10)

        # show the current conversation (pretty print)
        for c in C:
            print("*"*20)
            if c["role"] == "assistant":
                print(bcolors.OKGREEN + c["role"].capitalize() + bcolors.ENDC)
            else:
                print(bcolors.OKBLUE + c["role"].capitalize() + bcolors.ENDC)
            print("*"*20)
            content = c["content"].replace("*", "")
            try:
                pre_prediction = content.split("Prediction:")[0].strip()
                print(pre_prediction)
                prediction = content.split("Prediction:")[1].split("Pathway:")[0].strip()
                print("\n" + bcolors.WARNING + "Prediction: " + bcolors.ENDC + prediction.strip("*"))
                pathway = content.split("Pathway:")[1].strip()
                print(bcolors.WARNING + "Pathway: " + bcolors.ENDC + pathway)
            except:
                print(content)

        print("*"*20)
        print(bcolors.OKBLUE + "human".capitalize() + bcolors.ENDC)
        print("*"*20)
        try:
            # try to print the reaction
            mol_to_print =  prediction.strip("*") + ">>" + mol 
            draw_smiles(mol_to_print)
        except:
            # if the prediction is not available, print the molecules
            draw_smiles(mol, reaction=False)

        # take prediction input from the user
        done_with_prediction = False
        while not done_with_prediction:
            y_h = input(bcolors.WARNING + "Prediction: " + bcolors.ENDC)
            safety = input("Are you sure about this prediction? ([y]/n): ")
            if safety == "y" or safety == "" or safety.lower() == "y":
                done_with_prediction = True

        # ask for explanation
        done_with_explanation = False
        while not done_with_explanation:
            e_h = input(bcolors.WARNING + "Explanation: " + bcolors.ENDC)
            safety = input("Are you sure about this explanation? ([y]/n): ")
            if safety == "y" or safety == "" or safety.lower() == "y":
                done_with_explanation = True

        # add the human's response to the context
        human_response = {
            "role": "user",
            "content": f"""
            Prediction: {y_h}
            Pathway: {e_h}
            """
        }
        C.append(human_response)

        return y_h, e_h, C


class DRUGHumanStatic(DRUGAgent):
    """
    Human agent class.
    """
    def __init__(self, id: int):
        super().__init__("Human", id)

    def ask(self, x: Tuple, C: List[Dict], is_prompt: bool = False) -> Tuple:
        """
        Ask the human for a response.

        Args:
            x (Tuple): (y, mol, e) tuple.
            C (List[Dict]): context
            is_prompt (bool): whether the C is a prompt or not

        Returns:
            Tuple: prediction, explanation and updated context
        """
        # current session & example
        y_h, _, e_h = x

        # add the human's response to the context
        human_response = {
            "role": "user",
            "content": f"""
            Prediction: {y_h}
            Pathway: {e_h}
            """
        }
        C.append(human_response)

        return y_h, e_h, C


"""
Factory method to create agents.
"""
def create_agent(task: str, type: str, human_type: str, id: int):
    """
    Factory method to create agents.

    Args:
        task (str): task type
        type (str): agent type
        id (int): agent id

    Returns:
        Agent: agent object
    """
    if task == "RAD":
        if type == "Machine":
            return RADMachine(id)
        elif type == "Human":
            return RADHuman(id)
        else:
            raise ValueError(f"Invalid agent type: {type}")
    elif task == "DRUG":
        if type == "Machine":
            return DRUGMachine(id)
        elif type == "Human":
            if human_type == "real-time":
                return DRUGHuman(id)
            elif human_type == "static":
                return DRUGHumanStatic(id)
        else:
            raise ValueError(f"Invalid agent type: {type}")
    else:
        raise ValueError(f"Invalid task: {task}")
