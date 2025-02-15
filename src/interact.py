"""
This is the core module and runs the whole process of interaction between the human and the machine.
Described in Procedure 1 of the paper.
"""

import sys
sys.path.append("./")

import os
import uuid
import pickle
import pandas as pd
from copy import deepcopy
from dotenv import load_dotenv

from tasks import RAD, DRUG
from utils import parse_args
from agent import create_agent

from typing import Optional, Dict, List, Union

Prompt = Union[Dict[str, str], List[Dict[str, str]]]
# set API keys
load_dotenv()


def Interact(train_data: pd.DataFrame,
             val_data: Optional[pd.DataFrame],
             test_data: Optional[pd.DataFrame],
             machine_llm: str,
             evaluator: str,
             resume: bool,
             D: Optional[List],
             M: Optional[List],
             C: Optional[Prompt],
             human_type: str,
             eval_at_start: bool,
             no_learn: bool,
             task: str,
             h: int,
             m: int,
             n: int,
             k: int = 3) -> List:
    """
	Core function that simulates an interaction between the human and the machine.

	Args:
		train_data (pd.DataFrame): Training data for the agent
		val_data (pd.DataFrame): Validation data for the agent and task, used in learn
		test_data (Optional, pd.DataFrame): Test data for the agent and task, used in evaluate
		machine (str): The LLM to use as the machine agent
		evaluator (str): The evaluator to use for assessing performance, will use the --machine model if not provided
		resume (bool): Resume the interaction from a saved state
		D (Optional, List): List of relational databases
		M (Optional, List): List of messages
		C (Optional, Prompt): Context for the interaction
		human_type (str): Type of human agent, either "real-time" or "static", Only used in DRUG task
		eval_at_start (bool): Evaluate the agent at the start of the interaction to get base performance
		no_learn (bool): If True, the agent will always add a session to the context
		task (str): Task to perform, either "RAD" or "DRUG"
		h (int): human-identifier
		m (int): machine-identifier
		n (int): upper bound on interactions
		k (int): minimum number of interactions after which the "Reject" tag can be sent

	Returns:
		List: List of relational databases, Input and Messages.
	"""

    # Initialize the relational databases
    if not resume:
        D, M, C = [], [], None
    n *= 2  # number of interactions need to be doubled
    n += 1  # and 1 is added because the first interaction is "initialization"

    # Initialize the agents
    assert task in ["RAD", "DRUG"
                    ], f"Invalid task, expected 'RAD' or 'DRUG', got {task}."
    human = create_agent(task, "Human", human_type, evaluator, h)
    machine = create_agent(task, "Machine", None, m, evaluator, machine_llm)

    # Select the task-specific functions
    learn_fn = RAD.learn if task == "RAD" else DRUG.learn
    evaluate_fn = RAD.evaluate if task == "RAD" else DRUG.evaluate

    # initial performance on the test data
    if eval_at_start:
        if test_data is None:
            print(
                "eval_at_start is set to True, but test_data is None. Continuing without evaluation..."
            )
        elif task == "RAD":
            print(
                "We only support performance evaluation for DRUG, but got task RAD. Continuing without evaluation..."
            )
        elif task == "DRUG":
            evaluate_fn([],
                        test_data,
                        machine,
                        set="TEST_START",
                        evaluator=evaluator)
            print("Evaluation at start complete.")

    # metrics
    total_sessions = 0
    one_way_human, one_way_machine = 0, 0
    two_way = 0
    l_m_revision = False

    # Iterate over all input data
    for _, x in train_data.iterrows():

        # Generate a random session identifier and store the input data
        sess = uuid.uuid4().hex[:4]
        total_sessions += 1
        label, _, _ = x
        D.append((x, sess))

        # loop variables
        j = 1
        tags = []
        done = False
        human_ratified, machine_ratified = False, False

        C_ = deepcopy(C)  # copy the context so we don't modify the original

        while not done:
            # ask the machine
            mu_m, C_ = machine(j, k,
                               (D, M, C_))  # (tag, pred, expl) and context
            M += [(sess, j, m, mu_m, h)]
            j += 1
            if mu_m[0] == "revise":
                l_m_revision = True
            machine_ratified = (mu_m[0] == "ratify") or machine_ratified
            # stopping condition
            done = (j > n) or (human_ratified and machine_ratified)
            tags.extend([f"Machine: {mu_m[0]}"])

            if not done:
                # ask the human
                mu_h, C_ = human(j, k,
                                 (D, M, C_))  # (tag, pred, expl) and context
                M += [(sess, j, h, mu_h, m)]
                j += 1
                human_ratified = (mu_h[0] == "ratify") or human_ratified
                # stopping condition
                done = (j > n) or (human_ratified
                                   and machine_ratified) or (mu_h[0]
                                                             == "reject")
                tags.extend([f"Human: {mu_h[0]}"])

        # store the tags
        with open("tags.txt", "a") as f:
            if task == "RAD":
                f.write(f"sessionID-{sess}, ailment-{label} ::: tags-{tags}\n")
            elif task == "DRUG":
                f.write(f"sessionID-{sess}, mol-{label} ::: tags-{tags}\n")

        # decide if the context is helpful
        if no_learn:
            print("no_learn is set to True, always adding the session to the context.")
            learnOK = True
        else:
            if val_data is None:
                print(
                    "val_data is None, but no_learn was not set to True, continuing without learning..."
                )
                learnOK = False
            else:
                learnOK = learn_fn(C_, val_data, machine, evaluator)

        # update the context
        if learnOK:
            C = C_
        else:
            print("Session not helpful, not adding to the context.")

        if (test_data is not None) and (l_m_revision) and learnOK:
            if task == "DRUG":
                evaluate_fn(C, test_data, machine, set="TEST")
            else:
                print(
                    f"We only support performance evaluation for DRUG, but got task {task}. Continuing without evaluation..."
                )

        # only check for ratify because, in this special case,
        # human agent can never revise.
        l_h, l_m = mu_h[0], mu_m[0]
        if l_h == "ratify":
            one_way_human += 1
        if l_m == "ratify" or l_m_revision:
            one_way_machine += 1
        if (l_h == "ratify") and (l_m == "ratify" or l_m_revision):
            two_way += 1

        # save the state of the interaction
        with open("results/data.pkl", "wb") as f:
            pickle.dump(D, f)
        with open("results/messages.pkl", "wb") as f:
            pickle.dump(M, f)
        with open("results/context.pkl", "wb") as f:
            pickle.dump(C, f)

    print(f"Total Sessions: {total_sessions}")
    print(f"One-way Human: {one_way_human}")
    print(f"One-way Machine: {one_way_machine}")
    print(f"Two-way: {two_way}")

    # final performance on the test data
    if test_data is not None:
        if task == "DRUG":
            evaluate_fn(C, test_data, machine, set="TEST_END")
        else:
            print(
                f"We only support performance evaluation for DRUG, but got task {task}. Continuing without evaluation..."
            )

    log_str = f"""
	###################################
	SESSION INTELLIGIBILITY DATA:
	Total Sessions: {total_sessions}
	One-way Human: {one_way_human}
	One-way Machine: {one_way_machine}
	Two-way: {two_way}
	"""

    with open("results/accuracy_log.txt", "a+") as f:
        f.write(log_str)

    return D, M, C


def main():

    args = parse_args()

    if args.task == "RAD":
        train_data = pd.read_csv("data/train_xray_data.csv", index_col=None)
        val_data = pd.read_csv("data/val_xray_data.csv", index_col=None)
        train_data = train_data.drop(columns=["case", "label_short", "link"],
                                     inplace=False)
        val_data = val_data.drop(columns=["case", "label_short", "link"],
                                 inplace=False)
        test_data = None
        print("*" * 50)
        print(
            "We have not curated test data for the RAD task, and as such only report the intelligibility metrics."
        )
        print("*" * 50)
    elif args.task == "DRUG":
        data = pd.read_csv(
            "data/retro_match_sorted.csv",
            index_col=None)  # by default in y, x, e format, i.e. y ->^e x
        data.drop(columns=["matchOK"], inplace=True)
        # shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        data = data[["output", "input", "explanation"]]
        # split the data into train, val and test
        total = len(data)
        train_data = data[:int(total * 0.1)]
        val_data = data[int(total * 0.6):int(total * 0.8)]
        test_data = data[int(total * 0.8):]
    else:
        raise ValueError("Invalid task, expected 'RAD' or 'DRUG', got " +
                         args.task)

    # create a new accuracy log file for each run
    with open("results/accuracy_log.txt", "w") as f:
        f.write("")

    # resume from a saved state
    if args.resume and args.start_idx > 0:
        train_data = train_data.iloc[args.start_idx:]
        with open(args.D, "rb") as f:
            D = pickle.load(f)
        with open(args.M, "rb") as f:
            M = pickle.load(f)
        with open(args.C, "rb") as f:
            C = pickle.load(f)
    else:
        D, M, C = [], [], None

    # Interact with the agents
    D, M, C = Interact(train_data=train_data,
                       val_data=val_data,
                       test_data=test_data,
                       machine_llm=args.machine,
                       evaluator=args.evaluator,
                       D=D,
                       M=M,
                       C=C,
                       resume=args.resume,
                       human_type=args.human_type,
                       eval_at_start=args.eval_at_start,
                       task=args.task,
                       h=1,
                       m=2,
                       n=args.n,
                       no_learn=args.no_learn)

    # save the relational databases
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/data_final.pkl", "wb") as f:
        pickle.dump(D, f)
    with open("results/messages_final.pkl", "wb") as f:
        pickle.dump(M, f)
    with open("results/context_final.pkl", "wb") as f:
        pickle.dump(C, f)


if __name__ == "__main__":
    main()
