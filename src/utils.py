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
