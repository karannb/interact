import os
import base64
import pandas as pd
from copy import deepcopy
from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)

from typing import List, Union, Dict, Callable
Prompt = Union[Dict[str, str], List[Dict[str, str]]]


def encode_image(image_path):
    """
    Default encoding for images is base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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
        seed=42,
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


def evaluate(context: Prompt, test_df: pd.DataFrame, ailment: str, machine, agree_fn: Callable) -> float:
    """
    Evaluates the model on the given test data.

    Args:
        context (Prompt): Current context for the model
        test_df (pd.DataFrame): Test data
        ailment (str): The ailment to evaluate the model for
        machine (Agent): The machine agent
        agree_fn (Callable): Function to check if the explanation agrees with the prediction

    Returns:
        float: The overall accuracy of the model
    """
    print(f"Evaluating for {ailment}...")

    # initialize the counters
    total = 0
    correct_preds = 0
    correct_expls = 0
    correct = 0

    # prediction prompt
    pred_prompt = deepcopy(context)
    pred_prompt.extend([
        {
            "role": "system",
            "content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
            You have to strictly response in, no extra text:
            *Prediction: Yes/No*
            """
        },
        {
            "role": "user",
            "content": f"Given the following chest XRay, you have to predict the presence of {ailment}."
        }
    ]
    )

    expl_prompt = deepcopy(context)
    expl_prompt.extend([
        {
            # TODO: check prompt
            "role": "system",
            "content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
            You have to strictly response in, no extra text:
            *Prediction: Yes*
            *Explanation: <Your explanation here>*
            """
        },
        {
            "role": "user",
            "content": f"Given the following chest XRay, you have to explain the presence of {ailment}."
        },
        {
            "role": "assistant",
            "content": "*Prediction: Yes*"
        }
    ]
    )

    # iterate over the test data
    for _, row in test_df.iterrows():
        total += 1
        report = row["report"]
        label = row["label"]
        img = encode_image(row["img_path"])

        # get prediction
        img_msg = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                }
            ]
        }
        y = "problem"
        while y == "problem":
            y, _, _ = machine.ask(None, pred_prompt + [img_msg], is_prompt=True)
            if y == "problem":
                print("Problem with prediction, retrying...")

        prediction = y

        # if label != ailment, we don't care about explanation, only prediction
        if label != ailment:
            matchOK = "no" in prediction.lower()
            correct_preds += 1 if matchOK else 0
            correct_expls += 1 if matchOK else 0
            correct += 1 if matchOK else 0
        else:
            if "no" in prediction.lower():
                # the ailment IS present, but the model predicted it as absent
                pass
            else:
                # get explanation
                y = "problem"
                while y == "problem":
                    y, explanation, _ = machine.ask(None, expl_prompt + [img_msg], is_prompt=True)
                    if y == "problem":
                        print("Problem with explanation, retrying...")

                # check if the prediction is correct
                matchOK = "yes" in prediction.lower()
                agreeOK = agree_fn(explanation, report)
                correct_preds += 1 if matchOK else 0 # this is kept as a check so that other things don't get matched
                correct_expls += 1 if agreeOK else 0
                correct += 1 if matchOK and agreeOK else 0

    # print the results
    print(f"Ailment: {ailment}")
    print(f"Total: {total}")
    print(f"Correct Predictions: {correct_preds}")
    print(f"Correct Explanations: {correct_expls}")
    print(f"Correct Overall: {correct}")
    print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
    print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
    print(f"Overall Accuracy: {100*correct / total:.2f}")

    return correct / total


def evaluate_many(context: Prompt, test_data, machine, agree_fn: Callable) -> None:
    """
    Evaluates the model on multiple test data.

    Args:
        context (Prompt): Current context for the model
        test_data: Overall test data
        machine (Agent): The machine agent
        agree_fn (Callable): Function to check if the explanation agrees with the prediction
    """

    ailments = ["Atelectasis", "Pneumonia", "Pleural Effusion", "Cardiomegaly", "Pneumothorax"]

    for ailment in ailments:
        evaluate(context, test_data, ailment, machine, agree_fn)

    return


def are_molecules_same(smiles1: str, smiles2: str) -> bool:
    """Function to check if two molecules are the same.

    Parameters
    ----------
    smiles1 : str
        SMILES string for molecule 1
    smiles2 : str
        SMILES string for molecule 2

    Returns
    -------
    bool
        True if the molecules are the same, False otherwise

    Raises
    ------
    ValueError
        If invalid SMILES strings are provided
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
    except Exception as e:
        mol1 = None
    
    try:
        mol2 = Chem.MolFromSmiles(smiles2)
    except Exception as e:
        mol2 = None

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES string provided.")

    # Get canonical SMILES for both molecules
    canonical_smiles1 = Chem.MolToSmiles(mol1, canonical=True)
    canonical_smiles2 = Chem.MolToSmiles(mol2, canonical=True)

    # Alternatively, compare molecular fingerprints
    fingerprint1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1,
                                                                  radius=2,
                                                                  nBits=1024)
    fingerprint2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2,
                                                                  radius=2,
                                                                  nBits=1024)

    # Check if canonical SMILES or fingerprints match
    if canonical_smiles1 == canonical_smiles2:
        return True
    elif fingerprint1 == fingerprint2:
        return True
    else:
        return False
