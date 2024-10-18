import uuid
from typing import List
from src.agent import Machine, Human


def Interact(data: List, h: int, m: int, n: int, k: int = 3) -> List:
    """
    Core interact function between the human and the machine.

    Args:
        data (List): Input Data instances, a list of data.
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

    # Iterate over all input data
    for x in data:

        # Generate a random session identifier and store the input data
        sess = uuid.uuid4().hex[:4]
        D.append((x, sess))

        j = 0
        done = False
        while not done:
            # ask the machine
            mu_m, C = machine.ask(j, k, (D, M, C)) # (tag, pred, expl) and context
            M += [(sess, j, m, mu_m, h)]
            j += 1

            # ask the human
            mu_h, C = human.ask(j, k, (D, M, C))
            M += [(sess, j, h, mu_h, m)]
            j += 1

            # stopping condition
            done = (j > n) or (mu_h[0] == "ratify") or (mu_h[0] == "reject")

        # only check for ratify because, in this special case,
        # human agent can never revise.
        if mu_h[0] == "ratify":
            print("Strongly intelligible for the Human.")
        elif mu_h[0] == "reject" or mu_h[0] == "refute":
            print("Not Strongly intelligible for the Human.")

    return D, M, C


if __name__ == "__main__":
    data = [1, 2, 3]
    Interact(data, 1, 2, 3)