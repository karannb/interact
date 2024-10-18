from typing import Tuple, List, Union, Dict
Prompt = Union[Dict[str, str], List[Dict[str, str]]]

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

import re
import base64


def encode_image(image_path):
    """
    Default encoding for images is base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def assemble_prompt(x, c_j) -> Prompt:
    """
    Assemble prompt for the LLM.

    Args:
        x: example
        c_j: context

    Returns:
        Prompt: prompt
    """
    img, y, _ = x
    encoded_img = encode_image(img)
    messages = [
        {
            "role": "system",
            "content": "You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax."
        },
        {
            "role": "user",
            "content": f"Given the following chest XRay, you have to predict the existence of {y}."
        },
        {
            "role": "user",
            "content": """Adhere to the following output format strictly, no extra text:
                        Prediction: Yes/No
                        Explanation: <Your explanation here>
            """},
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
    y = str(y).lower()
    if y == "yes":
        return True
    else:
        return False


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
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax."
            },
            {
                "role": "user",
                "content": f"Given {e} is a correct diagnosis/explanation of an XRay, do you agree that {e_pred} is also a correct diagnosis/explanation? Respond only in Yes/No."
            },
        ]
    )

    # parse the response
    out = completion.choices[0].message.content.lower()
    if "yes" in out:
        return True
    else:
        return False


def parse_response(response, C: List) -> Tuple:
    """
    Parse the response from the LLM.

    Args:
        response (str): response from the LLM
        C (List): context

    Returns:
        Tuple: prediction and explanation
    """
    response = response.choices[0].message.content
    prediction, explanation = response.split("\n")
    assert "Prediction" in prediction, "Prediction not found in the response, expected 'Prediction: Yes/No', got " + prediction
    assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation

    # add to the context
    response_conv = {
        "role": "system",
        "content": response
    }
    C.append(response_conv)

    return prediction.split(":")[1].strip(), explanation.split(":")[1].strip(), C

def summarize(report: str, ailment: str) -> str:
    """
    Generates a summary of the report, in context of the ailment.

    Args:
        report (str): A radiology report
        ailment (str): The ailment to summarize the report for

    Returns:
        str: The summary of the report
    """
    # NOTE: for this task, 3.5 is just as good as any advanced model
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system", 
                "content": "You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax."
                },
            {
                "role": "user",
                "content": "Summarize the given radiology report in context of " 
                + ailment + 
                ". Also, you must (requirement) omit information about the patient age, name, as well as any links. You can also skip the 'report by' information, basically anything not related to the ailment."
                + " Only include information that explicitly mentions the ailment or is close to such a mention."
                + " Strictly do not write 'summary' anywhere, i.e., summarize the report as if you are generating it."
                + " The report is as follows: "
                },
            {
                "role": "user",
                "content": report
                }
            ]
        )

    # parse the response
    summary = completion.choices[0].message.content

    return summary
