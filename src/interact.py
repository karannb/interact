import sys
sys.path.append("./")

import os
import uuid
import pickle
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
    two_way = 0
    l_m_revision, l_h_revision = False, False

    # Iterate over all input data
    for idx, x in data:
        # Generate a random session identifier and store the input data
        sess = uuid.uuid4().hex[:4]
        total_sessions += 1
        label, _, _ = x
        D.append((x, sess))

        j = 1
        tags = []
        done = False
        while not done:
            # ask the machine
            mu_m, C = machine.call(j, k, (D, M, C)) # (tag, pred, expl) and context
            M += [(sess, j, m, mu_m, h)]
            j += 1
            if mu_m[0] == "revise":
                l_m_revision = True

            # stopping condition
            done = (j > n) or (mu_m[0] == "ratify") or (mu_m[0] == "reject")

            if not done:
                # ask the human
                mu_h, C = human.call(j, k, (D, M, C)) # (tag, pred, expl, human_response) and context
                M += [(sess, j, h, mu_h, m)]
                j += 1
                if mu_h[0] == "revise":
                    l_h_revision = True

                # stopping condition
                done = (j > n) or (mu_h[0] == "ratify") or (mu_h[0] == "reject")

            if (mu_m[0] == "ratify") or (mu_m[0] == "reject"):
                tags.extend([f"Machine: {mu_m[0]}"])
            else:
                tags.extend([f"Machine: {mu_m[0]}", f"Human: {mu_h[0]}"])

        # only check for ratify because, in this special case,
        # human agent can never revise.
        l_h, l_m = mu_h[0], mu_m[0]
        if l_h == "ratify" or l_h_revision:
            one_way_human += 1
        if l_m == "ratify" or l_m_revision:
            one_way_machine += 1
        if (l_h == "ratify") and (l_m == "ratify" or l_m_revision):
            two_way += 1

        # store the tags
        with open("tags.txt", "a") as f:
            f.write(f"sessionID-{sess}, ailment-{label} ::: tags-{tags}\n")

    print(f"Total Sessions: {total_sessions}")
    print(f"One-way Human: {one_way_human}")
    print(f"One-way Machine: {one_way_machine}")
    print(f"Two-way: {two_way}")
    return D, M, C


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n", "--num_iter", type=int, default=3)
    parser.add_argument("--num_ailments", type=int, default=5, choices=[3, 5, 7])
    args = parser.parse_args()

    data = pd.read_csv(f"data/xray_data_{args.num_ailments}.csv", index_col=None)
    data = data.drop(columns=["case", "label_short"], inplace=False)
    iterdata = data.iterrows()
    D, M, C = Interact(iterdata, h=1, m=2, n=args.n)
    # save the relational databases
    if not os.path.exists("results"):
        os.makedirs("results")
    with open("results/data.pkl", "wb") as f:
        pickle.dump(D, f)
    with open("results/messages.pkl", "wb") as f:
        pickle.dump(M, f)
    with open("results/context.pkl", "wb") as f:
        pickle.dump(C, f)
