import ast
import traceback
import random
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
            
            for _ in range(num_examples - 1):
                idx = random.randint(0, len(functions) - 1)
                self.buffer.add(functions[idx])
    
    def extract_function_code(self, text):
        """
        Extract function code from the generated text.
        
        Args:
            text: Generated text containing a function
            
        Returns:
            Extracted function code or None if extraction fails
        """
        try:
            # Try to extract code from Markdown code blocks
            if "```python" in text and "```" in text.split("```python", 1)[1]:
                return text.split("```python", 1)[1].split("```", 1)[0].strip()
            
            # Try to extract code without Markdown
            if "def f(" in text:
                lines = text.split("\n")
                func_start_idx = -1
                
                # Find where the function starts
                for i, line in enumerate(lines):
                    if "def f(" in line:
                        func_start_idx = i
                        break
                
                if func_start_idx == -1:
                    return None
                
                # Extract the function lines
                function_lines = [lines[func_start_idx]]
                indent_level = None
                
                for i in range(func_start_idx + 1, len(lines)):
                    line = lines[i]
                    
                    # Skip empty lines
                    if not line.strip():
                        function_lines.append(line)
                        continue
                    
                    # Detect indentation from first indented line after def
                    if indent_level is None and line.strip() and (line.startswith(' ') or line.startswith('\t')):
                        indent_level = len(line) - len(line.lstrip())
                    
                    # Include lines with proper indentation
                    if indent_level is not None:
                        if line.strip() and len(line) - len(line.lstrip()) >= indent_level:
                            function_lines.append(line)
                        elif not line.startswith(' ') and not line.startswith('\t'):
                            # If we hit a non-indented line, we're done
                            break
                
                return "\n".join(function_lines)
            
            return None
        
        except Exception as e:
            print(f"Error extracting function code: {str(e)}")
            return None
    
    def extract_input_output_pairs(self, text):
        """
        Extract input-output pairs from the generated text.
        
        Args:
            text: Generated text containing input-output pairs
            
        Returns:
            List of (input, output) pairs or None if extraction fails
        """
        try:
            pairs = []
            lines = text.split('\n')
            
            # Look for a section that contains input-output pairs
            in_pairs_section = False
            for line in lines:
                line = line.strip()
                
                # Check if we're entering a section with pairs
                if not in_pairs_section and ('Input-Output Pairs:' in line or 'Examples:' in line):
                    in_pairs_section = True
                    continue
                
                # Skip empty lines
                if not line:
                    continue
                
                # Look for input-output pairs
                if in_pairs_section:
                    # Check for lines with both input and output
                    if 'Input:' in line and 'Output:' in line:
                        try:
                            input_part = line.split('Input:')[1].split('Output:')[0].strip().rstrip(',')
                            output_part = line.split('Output:')[1].strip()
                            pairs.append((input_part, output_part))
                        except Exception:
                            pass
                    
                    # Check for numbered examples
                    elif '.' in line and ('Input:' in line or 'input:' in line):
                        try:
                            # Remove the number prefix
                            parts = line.split('.', 1)[1].strip()
                            if 'Input:' in parts and 'Output:' in parts:
                                input_part = parts.split('Input:')[1].split('Output:')[0].strip().rstrip(',')
                                output_part = parts.split('Output:')[1].strip()
                                pairs.append((input_part, output_part))
                        except Exception:
                            pass
            
            # If we found no pairs using the above methods, try a more generic approach
            if not pairs:
                for line in lines:
                    if 'Input' in line and 'Output' in line:
                        try:
                            # Try to extract values directly
                            input_output = line.lower().replace('input', '').replace('output', '')
                            parts = input_output.split(',')
                            if len(parts) == 2:
                                input_part = parts[0].strip()
                                output_part = parts[1].strip()
                                pairs.append((input_part, output_part))
                        except Exception:
                            pass
            
            return pairs if pairs else None
        
        except Exception as e:
            print(f"Error extracting input-output pairs: {str(e)}")
            return None
    
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
        
        # Build the prompt for task proposal - SIMPLIFIED for easier parsing
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
            
            # Debug print the raw proposal
            print(f"Raw proposal (first 150 chars): {proposal[:150]}...")
            
            # Parse the proposal with robust error handling
            try:
                # Extract function code using the helper
                function_code = self.extract_function_code(proposal)
                if not function_code:
                    # If we can't find a function, use a default function
                    function_code = """def f(x):
    return x * 2"""
                    print("Using default function since none was generated")
                
                # Validate function syntax
                try:
                    ast.parse(function_code)
                except SyntaxError as e:
                    # If syntax is invalid, use a default function
                    function_code = """def f(x):
    return x * 2"""
                    print(f"Using default function due to syntax error: {str(e)}")
                
                # Extract input-output pairs using the helper
                pairs = self.extract_input_output_pairs(proposal)
                
                # If no valid pairs found, create default pairs
                if not pairs or len(pairs) < 4:
                    print(f"Could not extract enough input-output pairs, using defaults")
                    pairs = [
                        ('1', '2'),
                        ('2', '4'),
                        ('3', '6'),
                        ('4', '8'),
                        ('5', '10')
                    ]
                
                print(f"Using function: {function_code[:50]}...")
                print(f"Using {len(pairs)} input-output pairs")
                
                # Validate the pairs
                valid_pairs = []
                for input_val, expected_output in pairs:
                    # Execute the function with the input
                    success, actual_output = self.executor.validate_task(function_code, input_val)
                    
                    if success:
                        # Clean up for comparison
                        expected_clean = str(expected_output).strip('"\'`)( \t\n')
                        actual_clean = str(actual_output).strip('"\'`)( \t\n')
                        
                        # Check if the output matches the expected output
                        if expected_clean == actual_clean:
                            valid_pairs.append((input_val, expected_output))
                            print(f"Valid pair: Input={input_val}, Output={expected_output}")
                        else:
                            print(f"Invalid pair: Input={input_val}, Expected={expected_output}, Got={actual_output}")
                
                # Need at least 4 valid pairs
                if len(valid_pairs) >= 4:
                    # Add the example to the proposed list
                    example = {
                        'function': function_code,
                        'pairs': valid_pairs,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': function_code,
                        'pairs': valid_pairs
                    })
                    
                    print(f"Successfully added induction task with {len(valid_pairs)} valid pairs")
                else:
                    # Not enough valid pairs, use default function and pairs
                    default_function = """def f(x):
    return x * 2"""
                    default_pairs = [
                        ('1', '2'),
                        ('2', '4'),
                        ('3', '6'),
                        ('4', '8'),
                        ('5', '10')
                    ]
                    
                    example = {
                        'function': default_function,
                        'pairs': default_pairs,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': default_function,
                        'pairs': default_pairs
                    })
                    
                    print(f"Added fallback induction task to buffer")
            except Exception as e:
                # More detailed error logging
                print(f"Error parsing proposal: {str(e)}")
                print(traceback.format_exc())
                
                # Add a default example as fallback
                default_function = """def f(x):
    return x * 2"""
                default_pairs = [
                    ('1', '2'),
                    ('2', '4'),
                    ('3', '6'),
                    ('4', '8'),
                    ('5', '10')
                ]
                
                example = {
                    'function': default_function,
                    'pairs': default_pairs,
                    'prompt': proposal_prompt,
                    'response': proposal
                }
                
                proposed_examples.append(example)
                
                # Add to buffer
                self.buffer.add({
                    'function': default_function,
                    'pairs': default_pairs
                })
                
                print(f"Added fallback induction task to buffer")
                
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
            
            # Build the prompt for solving - SIMPLIFIED
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
                # Extract the function code using the helper
                proposed_function = self.extract_function_code(solution)
                if not proposed_function:
                    # If extraction fails, use a simple function based on the training pairs
                    if train_pairs and all(isinstance(eval(pair[0]), (int, float)) for pair in train_pairs):
                        # For numeric inputs, try a simple multiplication rule
                        input1, output1 = train_pairs[0]
                        factor = float(output1) / float(input1)
                        proposed_function = f"def f(x):\n    return x * {factor}"
                    else:
                        # Default fallback
                        proposed_function = "def f(x):\n    return x"
                    
                    print(f"Could not extract valid function code, using fallback: {proposed_function}")
                
                # Validate function syntax
                try:
                    ast.parse(proposed_function)
                except SyntaxError as e:
                    # If syntax is invalid, use a simple default function
                    proposed_function = "def f(x):\n    return x"
                    print(f"Invalid function syntax, using fallback: {proposed_function}")
                
                print(f"Proposed function: {proposed_function[:50]}...")
                
                # Test on held-out test pairs
                correct_count = 0
                test_results = []
                
                for input_val, expected_output in test_pairs:
                    # Execute the function with the input
                    success, actual_output = self.executor.validate_task(proposed_function, input_val)
                    
                    if success:
                        # Clean up for comparison
                        expected_clean = str(expected_output).strip('"\'`)( \t\n')
                        actual_clean = str(actual_output).strip('"\'`)( \t\n')
                        
                        # Check if the output matches the expected output
                        if expected_clean == actual_clean:
                            correct_count += 1
                            test_results.append(True)
                        else:
                            test_results.append(False)
                        
                        print(f"Test pair: Input={input_val}, Expected={expected_output}, Got={actual_output}, Match={test_results[-1]}")
                    else:
                        test_results.append(False)
                        print(f"Execution error for input: {input_val}")
                
                # Calculate reward based on accuracy
                if test_pairs:
                    accuracy = correct_count / len(test_pairs)
                    reward = accuracy
                else:
                    reward = 0.0
                
                # Save the results
                example['proposed_function'] = proposed_function
                example['solving_prompt'] = solving_prompt
                example['test_results'] = test_results
                example['accuracy'] = accuracy if test_pairs else 0.0
                example['reward'] = reward
                
                rewards.append(reward)
                
                print(f"Induction task solved with accuracy: {accuracy:.2f}, reward: {reward}")
                
            except Exception as e:
                print(f"Error solving task: {str(e)}")
                print(traceback.format_exc())
                rewards.append(0.0)
        
        return rewards
