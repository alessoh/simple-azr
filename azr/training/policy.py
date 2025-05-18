def update_policy(model, examples, rewards, learning_rate=1e-5):
    """
    Update the model policy based on the rewards.
    
    Args:
        model: The language model
        examples: List of examples with prompts and responses
        rewards: List of reward values for the examples
        learning_rate: Learning rate for the update
    """
    # Ensure learning_rate is a float
    learning_rate = float(learning_rate)
    
    total_loss = 0.0
    
    # Process all examples
    for example, reward in zip(examples, rewards):
        # Get the prompt and response
        prompt = example.get('prompt', '')
        response = example.get('response', '')
        
        # For solving tasks, use the solving prompt and proposed solution
        if 'solving_prompt' in example:
            prompt = example['solving_prompt']
            
            # Get the appropriate response based on task type
            if 'proposed_solution' in example:
                response = example['proposed_solution']
            elif 'proposed_function' in example:
                response = f"```python\n{example['proposed_function']}\n```"
        
        # Skip if prompt or response is missing
        if not prompt or not response:
            continue
        
        # Update model
        loss = model.update(prompt, response, reward, learning_rate)
        total_loss += loss
    
    return total_loss