import sys
sys.path.append("./")

import uuid
import pandas as pd
from typing import List
from argparse import ArgumentParser
from src.agent import Machine, Human


def Interact(data, h: int, m: int, n: int, k: int = 3) -> List:
    """
    Core interact function between the human and the machine.

    Args:
        data : Input Data instances, a list of data.
        h (int): human-identifier
        m (int): machine-identifier
        n (int): upper bound on interactions
        k (int): minimum number of interactions after which the "Reject" tag can be sent

    Returns:
        List: List of relational databases, Input and Messages.
    """

    # Initialize the relational databases
    D, M, C = [], [], None
    n *= 2 # number of interactions need to be doubled

    # Initialize the agents
    human = Human(h)
    machine = Machine(m)

    # metrics
    total_sessions = 0
    one_way_human, one_way_machine = 0, 0
    two_way_human, two_way_machine = 0, 0
    l_m_revision = False

    # Iterate over all input data
    for idx, x in data:
        # Generate a random session identifier and store the input data
        sess = uuid.uuid4().hex[:4]
        total_sessions += 1
        D.append((x, sess))

        j = 0
        tags = []
        done = False
        while not done:
            # ask the machine
            mu_m, C = machine.ask(j, k, (D, M, C)) # (tag, pred, expl) and context
            M += [(sess, j, m, mu_m, h)]
            j += 1
            if mu_m[0] == "problem":
                j -= 1
                continue
            if mu_m[0] == "revise":
                l_m_revision = True

            # ask the human
            mu_h, C = human.ask(j, k, (D, M, C))
            M += [(sess, j, h, mu_h, m)]
            j += 1

            # stopping condition
            done = (j > n) or (mu_h[0] == "ratify") or (mu_h[0] == "reject")
            tags.extend([f"Machine: {mu_m[0]}", f"Human: {mu_h[0]}"])

        # only check for ratify because, in this special case,
        # human agent can never revise.
        l_h, l_m = mu_h[0], mu_m[0]
        if l_h == "ratify":
            one_way_human += 1
        if l_m == "ratify" or l_m_revision:
            one_way_machine += 1
        if (l_h == "ratify") and (l_m == "ratify" or l_m_revision):
            two_way_human += 1
            two_way_machine += 1

        # store the tags
        with open("tags.txt", "a") as f:
            f.write(f"sessionID-{sess} ::: tags-{tags}\n")

    print(f"Total Sessions: {total_sessions}")
    print(f"One-way Human: {one_way_human}")
    print(f"One-way Machine: {one_way_machine}")
    print(f"Two-way Human: {two_way_human}")
    print(f"Two-way Machine: {two_way_machine}")
    return D, M, C


if __name__ == "__main__":
    data = pd.read_csv("data/xray_data_filtered.csv", index_col=None)
    data = data.drop(columns=["case", "label_short"], inplace=False)
    iterdata = data.iterrows()

    parser = ArgumentParser()
    parser.add_argument("--n", "-num_iter", type=int, default=3)
    args = parser.parse_args()

    Interact(iterdata, h=1, m=2, n=args.n)