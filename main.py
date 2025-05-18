#!/usr/bin/env python
# Main entry point for Simple AZR training

import argparse
import yaml
import os
from tqdm import tqdm

from azr.model import SimpleAZRModel
from azr.data import TaskBuffer
from azr.environment import PythonExecutor
from azr.training.policy import update_policy
from azr.tasks.abduction import AbductionTask
from azr.tasks.deduction import DeductionTask  
from azr.tasks.induction import InductionTask

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Simple AZR Model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                        help='Path to config file')
    parser.add_argument('--model_name', type=str, 
                        default='distilgpt2',
                        help='Hugging Face model name')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--num_steps', type=int, default=500,
                        help='Number of training steps')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    executor = PythonExecutor()
    model = SimpleAZRModel(args.model_name)
    
    # Initialize task buffers
    abduction_buffer = TaskBuffer(max_size=config['buffer_size'])
    deduction_buffer = TaskBuffer(max_size=config['buffer_size'])
    induction_buffer = TaskBuffer(max_size=config['buffer_size'])
    
    # Initialize task handlers
    abduction_task = AbductionTask(model, executor, abduction_buffer)
    deduction_task = DeductionTask(model, executor, deduction_buffer)
    induction_task = InductionTask(model, executor, induction_buffer)
    
    # Seed the buffers
    print("Seeding task buffers...")
    abduction_task.seed_buffer(config['seed_examples'])
    deduction_task.seed_buffer(config['seed_examples'])
    induction_task.seed_buffer(config['seed_examples'])
    
    # Main training loop
    for step in tqdm(range(args.num_steps)):
        # PROPOSE phase
        print(f"\nStep {step+1}: Proposing tasks")
        
        # Propose abduction tasks
        abduction_examples = abduction_task.propose_tasks(config['batch_size'])
        
        # Propose deduction tasks
        deduction_examples = deduction_task.propose_tasks(config['batch_size'])
        
        # Propose induction tasks
        induction_examples = induction_task.propose_tasks(config['batch_size'])
        
        # SOLVE phase
        print(f"Step {step+1}: Solving tasks")
        
        # Solve abduction tasks
        abduction_rewards = abduction_task.solve_tasks(abduction_examples)
        
        # Solve deduction tasks
        deduction_rewards = deduction_task.solve_tasks(deduction_examples)
        
        # Solve induction tasks
        induction_rewards = induction_task.solve_tasks(induction_examples)
        
        # Update policy - Fix: Ensure learning_rate is a float
        learning_rate = float(config['learning_rate'])
        update_policy(
            model,
            abduction_examples + deduction_examples + induction_examples,
            abduction_rewards + deduction_rewards + induction_rewards,
            learning_rate=learning_rate
        )
        
        # Save checkpoint
        if (step + 1) % config['save_interval'] == 0:
            model.save(os.path.join(args.output_dir, f"model_step_{step+1}"))
            
        # Log metrics
        if (step + 1) % config['log_interval'] == 0:
            avg_abduction_reward = sum(abduction_rewards) / len(abduction_rewards) if abduction_rewards else 0
            avg_deduction_reward = sum(deduction_rewards) / len(deduction_rewards) if deduction_rewards else 0
            avg_induction_reward = sum(induction_rewards) / len(induction_rewards) if induction_rewards else 0
            
            print(f"\nStep {step+1} Metrics:")
            print(f"Abduction Reward: {avg_abduction_reward:.4f}")
            print(f"Deduction Reward: {avg_deduction_reward:.4f}")
            print(f"Induction Reward: {avg_induction_reward:.4f}")
    
    # Save final model
    model.save(os.path.join(args.output_dir, "model_final"))
    print("Training complete!")

if __name__ == "__main__":
    main()