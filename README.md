# Adversarial agents for IDS
## Proposal
The current repository hosts the source code for an adversarial architecture designed to test the predictive capabilities of agentic AI agents in network connection intrusion detection.

The architecture is comprised of two agents: an attacker agent, which reads data from a given dataset and may modify specific values under some limitations in an attempt to hide attacks, and a defender agent, which receives the data sent by the attacker agent and attempts to determine whether it is a benign connection, an attack, or which attack, based on execution mode.

Execution modes are binary, in which the defender agent only attempts to discern between benign and malicious traffic, and multiclass, in which the defender agent attempts to predict the specific family of attack.

Moreover, the architecture includes a RandomForest-based arbiter that allows to determine whether the defender's predictions are correct or not.

Both agents are trained using reinforcement learnign (LoRA), so that the attacker agent achieves a higher reward if it manages to fool both the RandomForest arbiter and the defender agent, and the defender agent achieves a higher reward if it properly predicts the class of a given attack.

All data is pre-processed to fix missing and mistaken values and encode every categorical attribute. The model runs for 1000 epochs before stopping, and is compatible with several HuggingFace LLMs as a baseline for the agents.

In an attempt to analyze the capabilities of the model to predict zero-day attacks, the training follows the zero-shot paradigm.
