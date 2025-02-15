import base64
import litellm
import argparse
from dotenv import load_dotenv
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


# set API keys
load_dotenv()
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
    completion = litellm.completion(
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


def parse_args():
    """
    Parse command line arguments for the interaction system.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Configuration for interactive learning system")

    # Core parameters
    parser.add_argument(
        "--n", "--num_iter",
        type=int,
        default=3,
        help="Number of iterations for the interaction loop"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="RAD",
        choices=["RAD", "DRUG"],
        help="Task type to perform: RAD (Radiology) or DRUG (Retrosynthesis)."
    )
    parser.add_argument(
        "--machine",
        type=str,
        required=True,
        default="gpt-4o",
        choices=[
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229"
        ],
        help="Language model to use for the interaction"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        choices=[
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20240620",
            "claude-3-sonnet-20240229"
        ],
        help="Specify the evaluator to use for assessing performance, will use the --machine model if not provided."
    )
    parser.add_argument(
        "--human_type",
        type=str,
        default="real-time",
        choices=["real-time", "static"],
        help="Type of human interaction: real-time for live interaction, static for pre-defined responses"
    )

    # Flags for controlling system behavior
    parser.add_argument(
        "--eval_at_start",
        default=False,
        action="store_true",
        help="Evaluate the agent at the start of the interaction to get base performance"
    )
    parser.add_argument(
        "--no_learn",
        default=False,
        action="store_true",
        help="Flag to decide to use a validation set which decides whether to learn or not"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Debug mode for the interaction system, clips the train, val and test sets to 5 examples, 2 examples and 2 examples respectively."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume the interaction from a saved state"
    )

    # Resume-related parameters
    resume_group = parser.add_argument_group('Resume Parameters', 'Parameters used when resuming from a saved state')
    resume_group.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index for the interaction when resuming"
    )
    resume_group.add_argument(
        "--D",
        type=str,
        default=None,
        help="Path to the relational database file"
    )
    resume_group.add_argument(
        "--M",
        type=str,
        default=None,
        help="Path to the messages file"
    )
    resume_group.add_argument(
        "--C",
        type=str,
        default=None,
        help="Path to the context file"
    )
    
    args = parser.parse_args()

    # assign evaluator to machine if not provided
    if args.evaluator is None:
        args.evaluator = args.machine
    
    # Validate resume-related arguments
    if args.resume:
        if not all([args.D, args.M, args.C]):
            parser.error("When --resume is set, --D, --M, and --C must all be provided")
    
    return args
