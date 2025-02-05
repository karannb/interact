import os
import base64
from typing import List, Union, Dict
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdChemReactions as Reactions
import matplotlib.pyplot as plt
from litellm import completion
from dotenv import load_dotenv
from variables import RAD_SUMMARIZE_SYS_PROMPT, RAD_SUMMARIZE_USER_PROMPT
load_dotenv()


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
    respoonse = completion(
        model="gpt-3.5-turbo-0125",
        seed=42,
        messages=[{
            "role":
            "system",
            "content": RAD_SUMMARIZE_SYS_PROMPT
            
        }, {
            "role":
            "user",
            "content": RAD_SUMMARIZE_USER_PROMPT.format(ailment=ailment)
            
        }, {
            "role": "user",
            "content": report
        }])

    # parse the response
    summary = respoonse.choices[0].message.content

    return summary


def draw_smiles(smiles, reaction=True):
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
