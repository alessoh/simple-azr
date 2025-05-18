# Simple AZR - Absolute Zero Reasoner
A simplified implementation of the Absolute Zero Reasoner (AZR) concept from the paper "Absolute Zero: Reinforced Self-play Reasoning with Zero Data". This project demonstrates how a language model can learn to reason through self-play without any external data.
# Overview
The Absolute Zero paradigm enables models to learn reasoning capabilities by:

# Proposing their own challenging tasks
Solving these self-proposed tasks
Learning from the results through reinforcement learning

Unlike traditional approaches that rely on human-curated datasets, AZR allows a model to continuously improve by generating its own learning curriculum.
# Project Structure
simple-azr/
├── requirements.txt
├── README.md
├── main.py
├── configs/
│   └── default.yaml
├── azr/
│   ├── __init__.py
│   ├── model.py
│   ├── data.py
│   ├── environment.py
│   ├── utils.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── abduction.py
│   │   ├── deduction.py
│   │   └── induction.py
│   └── training/
│       ├── __init__.py
│       ├── policy.py
│       └── rewards.py
└── outputs/  # Created during runtime
#File Descriptions

main.py: Entry point that orchestrates the training process
configs/default.yaml: Configuration settings for the training
azr/model.py: Wrapper for the language model (acting as both proposer and solver)
azr/data.py: Buffer management for storing task examples
azr/environment.py: Safe Python code execution environment
azr/utils.py: Helper functions for code parsing and manipulation
azr/tasks/*.py: Implementation of the three reasoning task types:

abduction.py: Given a function and output, predict the input
deduction.py: Given a function and input, predict the output
induction.py: Given input-output pairs, synthesize the function


azr/training/policy.py: Policy update mechanism for reinforcement learning
azr/training/rewards.py: Reward calculation functions

# Installation
bash# Clone the repository
git clone https://github.com/yourusername/simple-azr.git
cd simple-azr

# Create and activate conda environment
conda create -n simple-azr python=3.10
conda activate simple-azr

# Install dependencies
pip install -r requirements.txt
Usage
Run
python main.py --model_name gpt2-medium --num_steps 5
Run the training process:
bashpython main.py --model_name distilgpt2 --num_steps 500
Arguments:

--config: Path to configuration file (default: configs/default.yaml)
--model_name: HuggingFace model to use (default: distilgpt2)
--output_dir: Directory to save outputs (default: outputs)
--num_steps: Number of training steps (default: 500)

# How It Works

Initialization: The model starts with a minimal set of seed tasks
Propose Phase: The model generates new challenging tasks of three types
Solve Phase: The model attempts to solve its proposed tasks
Update Phase: The model is updated based on rewards from solving performance
Repeat: The loop continues, with tasks growing in complexity over time

# Extending the Project
This simplified implementation can be extended in various ways:

Use more advanced language models
Add more complex task types
Implement more sophisticated reward functions
Scale up the training process

# Reference
This project is based on concepts from the paper:
@article{zhao2025absolute,
  title={Absolute Zero: Reinforced Self-play Reasoning with Zero Data},
  author={Zhao, Andrew and Wu, Yiran and et al.},
  journal={arXiv preprint},
  year={2025}
}