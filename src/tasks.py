import os
from openai import OpenAI
from abc import abstractmethod
from utils import encode_image
from typing import Tuple, List, Union, Dict, Optional

Prompt = Union[Dict[str, str], List[Dict[str, str]]]

openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)

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
        y, img, _ = x
        encoded_img = encode_image(img)
        messages = [
            {
                "role": "system",
                "content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
                It is known that the user's predictions are always correct, i.e., ground truth.
                You have to strictly response in, no extra text:
                *Prediction: Yes/No*
                *Explanation: <Your explanation here>*
                """
            },
            {
                "role": "user",
                "content": f"""
                Given the following chest XRay, you have to predict the presence of {y}.
                """
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_img}"
                        }
                    }
                ]
            }
        ]
        if c_j is None:
            return messages
        else:
            c_j += messages
            return c_j

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
        if "yes" in pred or y == pred:
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
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            seed=42,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
                    Your task is to check consistency between two given diagnoses/explanations of an XRay.
                    1. Ignore any personal patient information mentioned in either diagnosis/explanation, e.g. age, name, etc.
                    2. Consider consistency in terms of the symptoms only and not the causes, e.g. if a report mentions xyz can be
                    diagnosed from follow-up and another report just mentions xyz, then this is no problem, it's not necessary to mention follow-up.
                    3. VERY IMPORTANT, your answer should be the same if the two reports are swapped, i.e., independent of the order of the two reports.
                    4. Respond only in Yes/No."""
                },
                {
                    "role": "user",
                    "content": f"Given A: {e} is a diagnosis/explanation of an XRay, and B: {e_pred} is another diagnosis/explanation of an XRay, are these two consistent?"
                },
            ]
        )

        # parse the response
        out = completion.choices[0].message.content.lower()
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
        response = response.choices[0].message.content
        pred_and_expl = response.split("\n")
        prediction, explanation = "", ""
        for text in pred_and_expl:
            if "Prediction" in text:
                prediction = text
            if "Explanation" in text:
                explanation = text
        assert prediction != "", "Prediction not found in the response"
        assert explanation != "", "Explanation not found in the response" 
        assert "Prediction" in prediction, "Prediction not found in the response, expected 'Prediction: Yes/No', got " + prediction
        assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation

        # add to the context
        response_conv = {
            "role": "assistant",
            "content": response
        }
        if C != None:
            C.append(response_conv)
        else:
            C = [response_conv]

        return prediction.split(":")[1].strip(), explanation.split(":")[1].strip(), C


class DRUG(Task):
    """
    Task for Retrosynthesis.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def assemble_prompt(x, c_j) -> Prompt:
        pass

    @staticmethod
    def match(y, pred) -> bool:
        pass

    @staticmethod
    def agree(e, e_pred) -> bool:
        pass

    @staticmethod
    def parse_response(response, C: Optional[List]) -> Tuple:
        pass
