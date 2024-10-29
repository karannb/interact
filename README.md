# interact
Official Implementation for the intelligibility protocol, PXP.

[![arXiv](https://img.shields.io/badge/arXiv-2410.20600-b31b1b.svg)](https://arxiv.org/abs/2410.20600)

### Installation
We have very minimal dependencies, and you can install them using the following command:
```bash
pip install -r requirements.txt
```
You might want to create a virtual environment (or use conda) to avoid conflicts with your system packages.
We use **python3.9.18** for all experiments.

### Data
For the RAD task, please write to Prof. Sidong Liu [[email](mailto:sidong.liu@mq.edu.au)].
For the DRUG task, please write to Shreyas V [[email](mailto:shreyas.college@gmail.com)].
Please mention "[INTERACT]" in the subject.
You can then use `src/preprocess.py` to generate the data in the correct format, for the experiments.
This will also summarize the data, using the `summarize` function from `src/utils.py`.

### Reproducing our results
To reproduce our RAD results, you can run the following command:
```bash
python src/interact.py --num_iter=5
```
This will output the counts of one-way and two-way intelligible sessions, create a `tags.txt` file of the actual tags exchanged between the two agents and also save the D (`data.pkl`), M (`messages.pkl`) and C (`context.pkl`) (from Procedure 1 in the paper) to the `results/` folder.
To reproduce the trend in Figure 3 from the paper, we ran the above command 5 times and manually extracted how many one-way intelligible sessions (upto an interaction limit) were generated per agent.
Reproducing the DRUG results requires an expert and so the outcome may be stochastic.

### Static / Real-time feedback
In general the code allows for interaction between both static and real-time human feedback and an LLM (interfaced by the `XMachine`).
To use the approach with custom data, 
- you can use some form of static human feedback (like RAD), stored in data as a CSV,
- as with the DRUG task, one can create an analogous real-time feedback system, using the command line and a real expert human for feedback.

### How to use the code for a different task
Here, we precisely describe how to use the code for a different task, say MATS (i.e. Materials Science).
- Decide the type of feedback you have access to, static (CSV with some predictions and explanations) or real-time (human expert)
- If it is static then you would need to add the data to the `data/` folder.
- Now, depeding on the type of feedback, you should implement a `MATSAgent` class in `src/agents.py` which should inherit from `Agent`, and borrow code from `RADAgent` (if static) and `DRUGAgent` (if real-time).
- Following this, implement `MATSMachine` and `MATSHuman` classes in the same file.
- With this, you need to change the `create_agent` in `src/agent.py` to also be compatible with the new task.
- Finally, you have to implement the `MATS` class in `src/utils.py` which should inherit from `Task` and borrow code from `RAD` and `DRUG` appropriately.
- Now, you can run the code using the following command: (add this task to the choices for the `--task` argument)
```bash
python src/interact.py --num_iter=5 --task=MATS
```

### Example
This is an example interaction (from the RAD task) generated by using the PXP protocol and our implementation
(As explained in the paper, this is a special case of the protocol such that the human-agent can never revise it's internal model).
<p align="center">
  <img src="assets/conv.png" width="350" alt="example of PXP">
</p>

### Citation
```
@misc{srinivasan2024implementationapplicationintelligibilityprotocol,
      title={Implementation and Application of an Intelligibility Protocol for Interaction with an LLM}, 
      author={Ashwin Srinivasan and Karan Bania and Shreyas V and Harshvardhan Mestha and Sidong Liu},
      year={2024},
      eprint={2410.20600},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.20600}, 
}
```
