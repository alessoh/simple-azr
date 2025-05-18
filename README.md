# Simple AZR - Absolute Zero Reasoner

A simplified implementation of the Absolute Zero Reasoner (AZR) concept from the paper "Absolute Zero: Reinforced Self-play Reasoning with Zero Data". This project demonstrates how a language model can learn to reason through self-play without any external data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Extending the Project](#extending-the-project)
- [Reference](#reference)
- [License](#license)

## Overview

The Absolute Zero paradigm enables models to learn reasoning capabilities by:

- Proposing their own challenging tasks
- Solving these self-proposed tasks
- Learning from the results through reinforcement learning

Unlike traditional approaches that rely on human-curated datasets, AZR allows a model to continuously improve by generating its own learning curriculum.

## Project Structure

```
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
```

## File Descriptions

- **main.py**: Entry point that orchestrates the training process
- **configs/default.yaml**: Configuration settings for the training
- **azr/model.py**: Wrapper for the language model (acting as both proposer and solver)
- **azr/data.py**: Buffer management for storing task examples
- **azr/environment.py**: Safe Python code execution environment
- **azr/utils.py**: Helper functions for code parsing and manipulation
- **azr/tasks/*.py**: Implementation of the three reasoning task types:
  - **abduction.py**: Given a function and output, predict the input
  - **deduction.py**: Given a function and input, predict the output
  - **induction.py**: Given input-output pairs, synthesize the function
- **azr/training/policy.py**: Policy update mechanism for reinforcement learning
- **azr/training/rewards.py**: Reward calculation functions

## Installation

```bash
# Clone the repository
git clone https://github.com/alessoh/simple-azr.git
cd simple-azr

# Create and activate conda environment
conda create -n simple-azr python=3.10
conda activate simple-azr

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the training process:

```bash
python main.py --model_name distilgpt2 --num_steps 500
```

Arguments:

- `--config`: Path to configuration file (default: configs/default.yaml)
- `--model_name`: HuggingFace model to use (default: distilgpt2)
- `--output_dir`: Directory to save outputs (default: outputs)
- `--num_steps`: Number of training steps (default: 500)

## How It Works

1. **Initialization**: The model starts with a minimal set of seed tasks
2. **Propose Phase**: The model generates new challenging tasks of three types
3. **Solve Phase**: The model attempts to solve its proposed tasks
4. **Update Phase**: The model is updated based on rewards from solving performance
5. **Repeat**: The loop continues, with tasks growing in complexity over time

## Extending the Project

This simplified implementation can be extended in various ways:

- Use more advanced language models
- Add more complex task types
- Implement more sophisticated reward functions
- Scale up the training process

## Reference

This project is based on concepts from the paper:

Paper URL: [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)

## License

MIT License

Copyright (c) 2025 H. Peter Alesso, AI HIVE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.