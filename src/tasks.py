import os
import re
import pandas as pd
from copy import deepcopy
from abc import abstractmethod
from utils import encode_image
from typing import Tuple, List, Union, Dict, Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from anthropic import Anthropic

from variables import RAD_ASSEMBLE_SYS_PROMPT, RAD_ASSEMBLE_USER_PROMPT, RAD_AGREE_SYS_PROMPT
from variables import DRUG_ASSEMBLE_SYS_PROMPT, DRUG_ASSEMBLE_USER_PROMPT, DRUG_AGREE_SYS_PROMPT

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=anthropic_key, )

Prompt = Union[Dict[str, str], List[Dict[str, str]]]


class Task:
    """
    Base class for a task (e.g. RAD, DRUG);
    the task decides 4 functions, namely, assemble_prompt, match, agree, and parse_response.
    """

    def __init__(self):
        pass

    @abstractmethod
    def assemble_prompt(x, c_j) -> Prompt:
        """
        Assemble prompt for the LLM.

        Args:
            x: example
            c_j: context

        Returns:
            Prompt: prompt
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def learn(C, val_data: pd.DataFrame, machine):
        """
        Learn the task.
        A task is learnt when the performance on the validation set after processing a given instance of data increases
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def match(y, pred) -> bool:
        """
        Match the prediction with the example.

        Args:
            y: ground truth
            pred: prediction

        Returns:
            bool: True if the prediction matches the example, False otherwise
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def agree(e, e_pred) -> bool:
        """
        Check if the explanation agrees with the prediction.

        Args:
            e: explanation
            e_pred: explanation

        Returns:
            bool: True if the explanation agrees with the prediction, False otherwise
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def parse_response(response, C: Optional[List]) -> Tuple:
        """
        Parse the response from the LLM.

        Args:
            response (str): response from the LLM
            C (List): context

        Returns:
            Tuple: prediction and explanation
        """
        raise NotImplementedError("Subclass must implement abstract method.")

    @abstractmethod
    def evaluate(C: Optional[List], test_df: pd.DataFrame, machine,
                 **kwargs) -> Tuple:
        """
        Evaluates the model on the given test data.

        Args:
            context (Prompt): Current context for the model
            test_df (pd.DataFrame): Test data
            machine (Agent): The machine agent
            kwargs: additional arguments (e.g. ailment for RAD, use-case specific)

        Returns:
            float: The overall accuracy of the model
        """
        raise NotImplementedError("Subclass must implement abstract method.")


class RAD(Task):
    """
    Task for Radiology Diagnosis.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def assemble_prompt(x, c_j) -> Prompt:
        """
        Assemble prompt for the LLM.

        Args:
            x: example
            c_j: context

        Returns:
            Prompt: prompt
        """
        _, img, _ = x
        encoded_img = encode_image(img)

        # extension is needed for claude API
        extension = img.split('.')[-1]
        if extension == "jpg":
            extension = "jpeg"

        # messages to be sent to the LLM
        messages = [{
            "role":
            "user",
            "content": RAD_ASSEMBLE_USER_PROMPT
            
        }, {
            "role":
            "user",
            "content": [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{extension}",
                    "data": encoded_img
                }
            }]
        }]

        # pass the system message separately to claude
        system = RAD_ASSEMBLE_SYS_PROMPT

        if c_j is None:
            c_j = []

        c_j += messages
        return c_j, system

    @staticmethod
    def learn(C: List, val_data: pd.DataFrame, machine, **kwargs) -> bool:
        """
        Learn the task.

        Args:
            C (List): context
            machine (Agent): machine

        Returns:
            bool: True if the performance is improved, False otherwise
        """
        new_performance = RAD.evaluate(C, val_data, machine, set="VAL")
        # return true if the performance is improved
        if machine.performance <= new_performance:
            machine.performance = new_performance
            return True
        else:
            return False

    @staticmethod
    def match(y, pred) -> bool:
        """
        Match the prediction with the example.

        Args:
            y: ground truth
            pred: prediction

        Returns:
            bool: True if the prediction matches the example, False otherwise
        """
        pred = str(pred).lower()
        y = str(y).lower()
        if y == pred or y in pred or pred in y:
            return True
        else:
            return False

    @staticmethod
    def agree(e, e_pred) -> bool:
        """
        Check if the explanation agrees with the prediction.

        Args:
            e: explanation
            e_pred: explanation

        Returns:
            bool: True if the explanation agrees with the prediction, False otherwise
        """
        # NOTE: for this task, we need to use GPT-4, 3.5 is not enough
        completion = client.messages.create(
            model="claude-3-opus-latest",
            temperature=0.0,
            max_tokens=10,
            system=RAD_AGREE_SYS_PROMPT,
            messages=[
                {
                    "role":
                    "user",
                    "content":
                    f"Given A: {e} is a explanation of an XRay, and B: {e_pred} is another explanation of an XRay, are these two consistent?"
                },
            ])

        # parse the response
        out = completion.content[0].text.lower()
        if "yes" in out:
            return True
        else:
            return False

    @staticmethod
    def parse_response(response, C: Optional[List]) -> Tuple:
        """
        Parse the response from the LLM.

        Args:
            response (str): response from the LLM
            C (List): context

        Returns:
            Tuple: prediction and explanation
        """
        response = response.content[0].text
        pred_and_expl = response.split("\n")
        # extract prediction and explanation
        prediction, explanation = "", ""
        for text in pred_and_expl:
            if "Prediction" in text:
                prediction = text
            if "Explanation" in text:
                explanation = text
        assert prediction != "" or explanation != "", "Prediction or Explanation not found in the response."
        if prediction != "":
            assert "Prediction" in prediction, f"Prediction not found in the response, expected 'Prediction: Yes/No', got {prediction}."
            pred = prediction.split(
                ":")[1].strip()[:-1]  # remove the '*' at the end
        else:
            pred = None
        if explanation != "":
            assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation
            expl = explanation.split(
                ":")[1].strip()[:-1]  # remove the '*' at the end
        else:
            expl = None

        # add to the context
        response_conv = {"role": "assistant", "content": response}
        if C != None:
            C.append(response_conv)
        else:
            C = [response_conv]

        return pred, expl, C

    @staticmethod
    def evaluate(context: Prompt, test_df: pd.DataFrame, machine,
                 **kwargs) -> float:
        """
        Evaluates the model on the given test data.

        Args:
            context (Prompt): Current context for the model
            test_df (pd.DataFrame): Test data
            machine (Agent): The machine agent
            kwargs:
                ailment (str): The ailment to evaluate the model for

        Returns:
            float: The overall accuracy of the model
        """
        set_name = kwargs.pop("set", None)
        print(f"Evaluating on {set_name} set...")

        # initialize the counters
        total = 0
        correct_preds = 0
        correct_expls = 0
        correct = 0

        # iterate over the test data
        for _, row in test_df.iterrows():
            total += 1
            y = row["label"]
            img = row["img_path"]
            e = row["report"]

            x = (y, img, e)

            # get the prompt
            prompt = deepcopy(context)
            prompt, system = RAD.assemble_prompt(x, prompt)

            # get output from the model
            y_pred = "problem"
            while y_pred == "problem":
                y_pred, e_pred, _ = machine.ask(x, [prompt, system],
                                                is_prompt=True)
                if y_pred == "problem":
                    print("Problem with prediction, retrying...")

            # check if the prediction is correct
            matchOK = RAD.match(y, y_pred)
            agreeOK = RAD.agree(e, e_pred)
            correct_preds += 1 if matchOK else 0
            correct_expls += 1 if agreeOK else 0
            correct += 1 if matchOK and agreeOK else 0

        # print the results
        print(f"Total: {total}")
        print(f"Correct Predictions: {correct_preds}")
        print(f"Correct Explanations: {correct_expls}")
        print(f"Correct Overall: {correct}")
        print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
        print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
        print(f"Overall Accuracy: {100*correct / total:.2f}")

        # log the results
        log_str = f"""
        ***********************************************************
        Accuracy on {set_name}
        Total: {total}
        Correct Predictions: {correct_preds}
        Correct Explanations: {correct_expls}
        Correct Overall: {correct}
        Prediction Accuracy: {100*correct_preds / total:.2f}
        Explanation Accuracy: {100*correct_expls / total:.2f}
        Overall Accuracy: {100*correct / total:.2f}
        ***********************************************************
        """
        f = open("results/accuracy_log.txt", "a+")
        f.write(log_str)
        f.close()

        return correct / total


class DRUG(Task):
    """
    Task for Retrosynthesis.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_few_shot() -> str:
        """
        This method loads a few examples of single-step retrosynthetic pathways.
        """
        # load the few-shot examples
        examples = pd.read_csv("data/few_shot_DRUG.csv")
        few_shot = []
        for _, row in examples.iterrows():
            y = row["output"]
            mol = row["input"]
            e = row["explanation"]
            prompt = f"""Generate a single-step retrosynthetic pathway for {mol}.

                    *Prediction: {y}*
                    *Pathway: {e}*
                    """
            few_shot.append(prompt)

        few_shot = "\n\n".join(few_shot)
        return few_shot

    @staticmethod
    def assemble_prompt(x, c_j) -> Tuple[Prompt, str]:
        """
        Assemble prompt for the LLM.

        Args:
            x: example
            c_j: context

        Returns:
            Prompt: prompt
        """
        _, mol, _ = x

        # load few-shot examples
        few_shot = DRUG.get_few_shot()

        messages = [{
            "role":
            "user",
            "content": DRUG_ASSEMBLE_USER_PROMPT.format(mol=mol)
        }]

        # pass the system message separately to claude
        system = DRUG_ASSEMBLE_SYS_PROMPT.format(few_shot=few_shot)

        if c_j is None:
            c_j = []

        c_j += messages
        return c_j, system

    @staticmethod
    def learn(C: List, val_data: pd.DataFrame, machine, **kwargs) -> bool:
        """
        Check if the model has learned a better understanding of the task.
        
        Args:
            C (List): context
            x (Tuple): example of (y, mol, e)
            machine (Agent): machine
            agree_fn (function): agree_fn

        Returns:
            bool: True if the performance is improved, False otherwise
        """
        new_performance = DRUG.evaluate(C, val_data, machine, set="VAL")
        # return true if the performance is improved
        if machine.performance <= new_performance:
            machine.performance = new_performance
            return True
        else:
            return False

    @staticmethod
    def match(y, pred) -> bool:
        """
        Match the prediction with the example.

        Args:
            y: ground truth
            pred: prediction

        Returns:
            bool: True if the prediction matches the example, False otherwise
        """
        try:
            # y can have multiple molecules, separated by a "."
            y = y.split(".")
            y = [Chem.MolFromSmiles(mol) for mol in y]
        except Exception as e:
            print(f"Exception: {e} while processing {y}: first SMILES string.")
            y = None

        try:
            pred = pred.split(".")
            pred = [Chem.MolFromSmiles(mol) for mol in pred]
        except Exception as e:
            print(
                f"Exception: {e} while processing {pred}: second SMILES string."
            )
            pred = None

        if pred is None or y is None:
            raise ValueError("Invalid SMILES string provided.")

        # Get canonical SMILES for both molecules
        canonical_smiles_pred = set(
            [Chem.MolToSmiles(mol, canonical=True) for mol in pred])
        canonical_smiles_y = set(
            [Chem.MolToSmiles(mol, canonical=True) for mol in y])

        # Alternatively, compare molecular fingerprints
        fingerprints_pred = set([
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                           radius=2,
                                                           nBits=1024)
            for mol in pred
        ])
        fingerprint_y = set([
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
                                                           radius=2,
                                                           nBits=1024)
            for mol in y
        ])

        # Check if canonical SMILES or fingerprints match
        if len(canonical_smiles_pred.intersection(canonical_smiles_y)) > 0:
            return True
        elif len(fingerprints_pred.intersection(fingerprint_y)) > 0:
            return True
        else:
            return False

    @staticmethod
    def agree(e, e_pred) -> bool:
        """
        Check if the explanation agrees with the prediction.

        Args:
            e: explanation
            e_pred: explanation

        Returns:
            bool: True if the explanation agrees with the prediction, False otherwise
        """
        # NOTE: for this task, we need to use GPT-4, 3.5 is not enough
        completion = client.messages.create(
            model="claude-3-opus-latest",
            temperature=0.0,
            max_tokens=10,
            system=DRUG_AGREE_SYS_PROMPT,
            messages=[
                {
                    "role":
                    "user",
                    "content":
                    f"Given A: {e} is a retrosynthesis pathway, and B: {e_pred} is another retrosynthesis pathway, do these describe similar reactions?"
                },
            ])

        # parse the response
        out = completion.content[0].text.lower()
        pattern = re.compile(r"judgement:\s*yes")
        out = pattern.findall(out)
        out = out[0] if len(out) > 0 else "no"
        if "yes" in out:
            return True
        else:
            return False

    @staticmethod
    def parse_response(response, C: Optional[List]) -> Tuple:
        """
        Parse the response from the LLM.

        Args:
            response (str): response from the LLM
            C (List): context

        Returns:
            Tuple: prediction and explanation
        """
        # parse LLM response
        response = response.content[0].text
        pred_and_expl = response.split("\n")
        prediction, explanation = "", ""
        for text in pred_and_expl:
            if "Prediction" in text:
                prediction = text
            if "Pathway" in text:
                explanation = text
        assert prediction != "" or explanation != "", "Prediction or Pathway not found in the response."

        # extract prediction
        if prediction != "":
            assert "Prediction" in prediction, f"Prediction not found in the response, expected 'Prediction: <smiles of retrosynthesis input>', got {prediction}."
            pred = prediction.split(":", 1)[1].split("*")[0].strip()
            # check if the SMILES string is valid
            mols = pred.split(".")
            smiles = []
            # to store only valid SMILES strings
            new_pred = []
            # iterate to find valid SMILES strings
            for mol in mols:
                mol_obj = Chem.MolFromSmiles(mol)
                smiles.append(mol_obj)
                if mol_obj is not None:
                    new_pred.append(mol)
            # if all are None, raise an error
            if all([x == None for x in smiles]):
                raise AssertionError(
                    f"Invalid SMILES string provided: {pred}.")
            # if some are None, print a warning
            elif any([x == None for x in smiles]):
                print(
                    f"A few invalid SMILES strings were provided: {pred}, but the rest are valid."
                )
                pred = ".".join(new_pred)
                print(f"New prediction (with valid SMILES) : {pred}")
        else:
            pred = None

        # extract explanation
        if explanation != "":
            assert "Pathway" in explanation, f"Pathway not found in the response, expected 'Pathway: <Your explanation here>', got {explanation}."
            expl = explanation.split(
                ":", 1)[1].strip()[:-1]  # remove the ending '*'
        else:
            expl = None

        # add to the context
        response_conv = {"role": "assistant", "content": response}
        if C != None:
            C.append(response_conv)
        else:
            C = [response_conv]

        return pred, expl, C

    @staticmethod
    def evaluate(context: Prompt, test_df: pd.DataFrame, machine, **kwargs):
        """
        Evaluates the model on the given test data.

        Args:
            context (Prompt): Current context for the model
            test_df (pd.DataFrame): Test data
            machine (Agent): The machine agent
            set(str): which set the model is being evaluated on

        Returns:
            float: The overall accuracy of the model
        """
        set_name = kwargs.pop("set", None)
        print(f"Evaluating on {set_name} set...")

        # initialize the counters
        total = 0
        correct_preds = 0
        correct_expls = 0
        correct = 0

        # iterate over the test data
        for _, row in test_df.iterrows():
            total += 1
            y = row["output"]
            mol = row["input"]
            e = row["explanation"]

            x = (y, mol, e)

            # pred+expl prompt
            prompt = deepcopy(context)
            prompt, system = DRUG.assemble_prompt(x, prompt)

            # get prediction
            y_pred = "problem"
            while y_pred == "problem":
                y_pred, e_pred, _ = machine.ask(x, [prompt, system],
                                                is_prompt=True)
                if y_pred == "problem":
                    print("Problem with prediction, retrying...")

            # check if the prediction is correct
            matchOK = DRUG.match(y, y_pred)
            agreeOK = DRUG.agree(e, e_pred)
            correct_preds += 1 if matchOK else 0  # this is kept as a check so that other things don't get matched
            correct_expls += 1 if agreeOK else 0
            correct += 1 if matchOK and agreeOK else 0

        # print the results
        print(f"Accuracy on {set_name}")
        print(f"Total: {total}")
        print(f"Correct Predictions: {correct_preds}")
        print(f"Correct Explanations: {correct_expls}")
        print(f"Correct Overall: {correct}")
        print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
        print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
        print(f"Overall Accuracy: {100*correct / total:.2f}")

        # log the results
        log_str = f"""
        ***********************************************************
        Accuracy on {set_name}
        Total: {total}
        Correct Predictions: {correct_preds}
        Correct Explanations: {correct_expls}
        Correct Overall: {correct}
        Prediction Accuracy: {100*correct_preds / total:.2f}
        Explanation Accuracy: {100*correct_expls / total:.2f}
        Overall Accuracy: {100*correct / total:.2f}
        ***********************************************************
        """
        f = open("results/accuracy_log.txt", "a+")
        f.write(log_str)
        f.close()

        return correct / total
