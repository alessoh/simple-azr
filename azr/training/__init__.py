from azr.data import format_code_examples

class InductionTask:
    """
    Task for function synthesis (induction): given input-output pairs, predict the function.
    """
    
    def __init__(self, model, executor, buffer):
        """
        Initialize the induction task.
        
        Args:
            model: Language model wrapper
            executor: Python executor
            buffer: Task buffer
        """
        self.model = model
        self.executor = executor
        self.buffer = buffer
    
    def seed_buffer(self, num_examples=5):
        """
        Seed the buffer with initial examples.
        
        Args:
            num_examples: Number of examples to generate
        """
        # Start with one very simple example
        seed_pairs = [
            ('1', '2'),
            ('2', '4'),
            ('3', '6'),
            ('4', '8'),
            ('5', '10')
        ]
        
        seed_example = {
            'function': 'def f(x):\n    return x * 2',
            'pairs': seed_pairs
        }
        self.buffer.add(seed_example)
        
        # Generate additional examples if needed
        if num_examples > 1:
            # Simple functions
            functions = [
                {
                    'function': 'def f(x):\n    return x + 5',
                    'pairs': [('1', '6'), ('2', '7'), ('3', '8'), ('10', '15'), ('0', '5')]
                },
                {
                    'function': 'def f(s):\n    return s[::-1]',
                    'pairs': [('"hello"', '"olleh"'), ('"python"', '"nohtyp"'), ('"12345"', '"54321"')]
                },
                {
                    'function': 'def f(numbers):\n    return sum(numbers)',
                    'pairs': [('[1, 2]', '3'), ('[1, 2, 3]', '6'), ('[5, 10, 15]', '30')]
                },
                {
                    'function': 'def f(text):\n    return len(text)',
                    'pairs': [('"hello"', '5'), ('"a"', '1'), ('"python"', '6')]
                }
            ]
            
            import random
            for _ in range(num_examples - 1):
                idx = random.randint(0, len(functions) - 1)
                self.buffer.add(functions[idx])
    
    def propose_tasks(self, num_tasks=1):
        """
        Propose new induction tasks.
        
        Args:
            num_tasks: Number of tasks to propose
            
        Returns:
            List of proposed task examples
        """
        proposed_examples = []
        
        # Sample reference examples from buffer
        reference_examples = self.buffer.sample(2)
        references_formatted = format_code_examples(reference_examples, "induction")
        
        # Build the prompt for task proposal
        proposal_prompt = f"""
# Propose Input-Output Pairs for a Python Function

Your task is to create a hidden Python function 'f' and generate 5 input-output pairs that demonstrate its behavior.
The function should be challenging but solvable by analyzing the patterns in the input-output pairs.

Here are some example functions with input-output pairs:
{references_formatted}

Create a new function and generate 5 input-output pairs for it.
The function should be named 'f' and should be deterministic (same input always gives same output).
Avoid using modules like os, sys, random, datetime, which can make functions non-deterministic.

Your response should be in this format:

```python
def f(parameter):
    # Your function implementation
    return result
```

Input-Output Pairs:
1. Input: value1, Output: result1
2. Input: value2, Output: result2
3. Input: value3, Output: result3
4. Input: value4, Output: result4
5. Input: value5, Output: result5
"""
        
        for _ in range(num_tasks):
            # Generate function and input-output pairs
            proposal = self.model.generate(proposal_prompt, max_length=500)
            
            # Parse the proposal
            try:
                # Extract function code
                function_code = proposal.split('```python')[1].split('```')[0].strip()
                
                # Extract input-output pairs
                pairs_section = proposal.split('Input-Output Pairs:')[1].strip()
                lines = pairs_section.split('\n')
                pairs = []
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Extract input and output
                    try:
                        input_part = line.split('Input:')[1].split('Output:')[0].strip().rstrip(',')
                        output_part = line.split('Output:')[1].strip()
                        
                        # Validate the pair
                        success, actual_output = self.executor.validate_task(function_code, input_part)
                        
                        if success and actual_output == output_part:
                            pairs.append((input_part, output_part))
                    except:
                        continue
                
                # Need at least 4 valid pairs
                if len(pairs) >= 4:
                    # Add the example to the proposed list
                    example = {
                        'function': function_code,
                        'pairs': pairs,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': function_code,
                        'pairs': pairs
                    })
                    
            except Exception as e:
                print(f"Error parsing proposal: {str(e)}")
                continue
        
        return proposed_examples
    
    def solve_tasks(self, examples):
        """
        Solve the proposed induction tasks.
        
        Args:
            examples: List of task examples to solve
            
        Returns:
            List of rewards
        """
        rewards = []
        
        for example in examples:
            # Split pairs for training and testing
            all_pairs = example['pairs']
            
            # Use first half for training, second half for testing
            mid = len(all_pairs) // 2
            train_pairs = all_pairs[:mid]
            test_pairs = all_pairs[mid:]
            
            # Format training pairs
            train_pairs_formatted = "\n".join([f"Input: {pair[0]}, Output: {pair[1]}" for pair in train_pairs])
            
            # Build the prompt for solving
            solving_prompt = f"""
# Synthesize a Python Function from Input-Output Pairs

Given several input-output pairs, write a Python function 'f' that produces the given outputs for the inputs.
Then, the function will be tested on new inputs to check if it correctly generalizes the pattern.

Input-Output Pairs:
{train_pairs_formatted}

Write a Python function 'f' that implements the pattern shown in these examples.
Your answer should be a complete Python function definition.

```python
def f(parameter):
    # Your implementation here
    return result
```
"""
            
            # Generate solution
            solution = self.model.generate(solving_prompt, max_length=500)
            
            # Parse and validate the solution
            try:
                # Extract the function code
                proposed_function = solution.split('```python')[1].split('```')[0].strip()
                
                # Test on held-out test pairs
                correct_count = 0
                
                for input_val, expected_output in test_pairs:
                    success, actual_output = self.executor.validate_task(proposed_function, input_val)
                    
                    if success and actual_output == expected_output:
                        correct_count += 1
                
                # Calculate reward based on accuracy
                if test_pairs:
                    accuracy = correct_count / len(test_pairs)
                    reward = accuracy
                else:
                    reward = 0.0
                
                # Save the results
                example['proposed_function'] = proposed_function
                example['solving_prompt'] = solving_prompt
                example['accuracy'] = accuracy if test_pairs else 0.0
                example['reward'] = reward
                
                rewards.append(reward)
                
            except Exception as e:
                print(f"Error solving task: {str(e)}")
                rewards.append(0.0)
        
        return rewards