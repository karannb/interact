import os
import base64
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as Reactions


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
	organization=openai_org,
	api_key=openai_key,
)

from typing import List, Union, Dict
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


def draw_smiles(smiles: str, reaction: bool = True):
    """
    Simple utility to draw a molecule (or reaction) from a SMILES string.

    Args:
        smiles (str): The SMILES string to draw
        reaction (bool): Whether the SMILES string represents a reaction or not

    Returns:
        None: Plots the molecule (or reaction) using matplotlib
    """
    # Convert SMILES to a molecule object
    try:
        mol = Reactions.ReactionFromSmarts(smiles, useSmiles=True)
    except:
        mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        print("Invalid SMILES string")
        return
    
    # Draw the molecule and display it
    if reaction:
        img = Draw.ReactionToImage(mol)
    else:
        img = Draw.MolToImage(mol)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
