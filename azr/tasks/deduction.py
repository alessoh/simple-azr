import ast
import traceback
from azr.data import format_code_examples

class DeductionTask:
    """
    Task for output prediction (deduction): given a function and input, predict the output.
    """
    
    def __init__(self, model, executor, buffer):
        """
        Initialize the deduction task.
        
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
        seed_example = {
            'function': 'def f(x):\n    return x * 2',
            'input': '5',
            'output': '10'
        }
        self.buffer.add(seed_example)
        
        # Generate additional examples if needed
        if num_examples > 1:
            for _ in range(num_examples - 1):
                # Generate a simple function
                functions = [
                    'def f(x):\n    return x + 10',
                    'def f(s):\n    return "".join(sorted(s))',
                    'def f(numbers):\n    return [n * 2 for n in numbers]',
                    'def f(x, y):\n    return x ** y',
                    'def f(text):\n    return text.upper()'
                ]
                
                inputs = [
                    '7',
                    '"hello"',
                    '[1, 2, 3]',
                    '2, 3',
                    '"python"'
                ]
                
                import random
                idx = random.randint(0, len(functions) - 1)
                
                function = functions[idx]
                input_str = inputs[idx]
                
                # Execute the function to get the output
                success, output = self.executor.validate_task(function, input_str)
                
                if success:
                    self.buffer.add({
                        'function': function,
                        'input': input_str,
                        'output': output
                    })
    
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
    
    def extract_input(self, text):
        """
        Extract the input value from the generated text.
        
        Args:
            text: Generated text containing an input specification
            
        Returns:
            Extracted input or None if extraction fails
        """
        try:
            # Try to extract input after "Input:" label
            if "Input:" in text:
                input_part = text.split("Input:", 1)[1].strip()
                return input_part.split("\n", 1)[0].strip()
            
            # Try to find a line containing "input" and a colon
            lines = text.split("\n")
            for line in lines:
                if ":" in line and "input" in line.lower():
                    return line.split(":", 1)[1].strip()
            
            # Look for "input =" pattern
            for line in lines:
                if "input =" in line.lower():
                    return line.split("input =", 1)[1].strip()
            
            return None
        
        except Exception as e:
            print(f"Error extracting input: {str(e)}")
            return None
    
    def propose_tasks(self, num_tasks=1):
        """
        Propose new deduction tasks.
        
        Args:
            num_tasks: Number of tasks to propose
            
        Returns:
            List of proposed task examples
        """
        proposed_examples = []
        
        # Sample reference examples from buffer
        reference_examples = self.buffer.sample(3)
        references_formatted = format_code_examples(reference_examples, "deduction")
        
        # Build the prompt for task proposal - SIMPLIFIED for easier parsing
        proposal_prompt = f"""
<|user|>
Write a simple, complete Python function named 'f' followed by an input value:
1. Function should take exactly one parameter
2. Function should return a value
3. Must be short and valid Python
4. No indentation errors
5. No comments or explanations

Here are some examples:
{references_formatted}

Only respond with:
1. The complete function
2. A single input value on a new line

<|assistant|>
def f(
"""
        
        for _ in range(num_tasks):
            # Generate function and input
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
                
                # Extract input using the helper
                input_str = self.extract_input(proposal)
                if not input_str:
                    # If no input, use a default
                    input_str = "5"
                    print("Using default input since none was generated")
                
                print(f"Using function: {function_code}")
                print(f"Using input: {input_str}")
                
                # Validate the task
                success, output = self.executor.validate_task(function_code, input_str)
                
                if success:
                    # Add the example to the proposed list
                    example = {
                        'function': function_code,
                        'input': input_str,
                        'output': output,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': function_code,
                        'input': input_str,
                        'output': output
                    })
                    
                    print(f"Successfully added deduction task to buffer: input={input_str}, output={output}")
                else:
                    # If validation fails, add a default example
                    default_function = """def f(x):
    return x * 2"""
                    default_input = "5"
                    success, output = self.executor.validate_task(default_function, default_input)
                    
                    example = {
                        'function': default_function,
                        'input': default_input,
                        'output': output,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': default_function,
                        'input': default_input,
                        'output': output
                    })
                    
                    print(f"Added fallback deduction task to buffer")
            except Exception as e:
                # More detailed error logging
                print(f"Error parsing proposal: {str(e)}")
                print(traceback.format_exc())
                
                # Add a default example as fallback
                default_function = """def f(x):
    return x * 2"""
                default_input = "5"
                success, output = self.executor.validate_task(default_function, default_input)
                
                if success:
                    example = {
                        'function': default_function,
                        'input': default_input,
                        'output': output,
                        'prompt': proposal_prompt,
                        'response': proposal
                    }
                    
                    proposed_examples.append(example)
                    
                    # Add to buffer
                    self.buffer.add({
                        'function': default_function,
                        'input': default_input,
                        'output': output
                    })
                    
                    print(f"Added fallback deduction task to buffer")
                
                continue
        
        return proposed_examples
    
    def solve_tasks(self, examples):
        """
        Solve the proposed deduction tasks.
        
        Args:
            examples: List of task examples to solve
            
        Returns:
            List of rewards
        """
        rewards = []
        
        for example in examples:
            # Build the prompt for solving
            solving_prompt = f"""
# Predict Output from Function and Input

Given a Python function and its input, predict what output the function will produce.

Function:
```python
{example['function']}
```

Input: {example['input']}

Think step by step to determine what output would be produced when this input is passed to the function.
Your answer should be just the output value in Python syntax.
"""
            
            # Generate solution
            solution = self.model.generate(solving_prompt, max_length=300)
            
            # Parse and validate the solution
            try:
                # Clean up the solution to get just the output
                cleaned_solution = solution.strip()
                
                # Try to remove any explanation text and get only the Python value
                if '\n' in cleaned_solution:
                    # Take the last non-empty line as the answer
                    lines = [line for line in cleaned_solution.split('\n') if line.strip()]
                    if lines:
                        cleaned_solution = lines[-1].strip()
                
                # Remove common prefixes/suffixes that might be present
                for prefix in ["Output:", "Answer:", "The output is:"]:
                    if cleaned_solution.startswith(prefix):
                        cleaned_solution = cleaned_solution[len(prefix):].strip()
                
                # Compare with the actual output
                actual_output = example['output']
                
                # Clean up for comparison
                cleaned_solution = cleaned_solution.strip('"\'`)( \t\n')
                actual_output = str(actual_output).strip('"\'`)( \t\n')
                
                # Check if the output matches
                if cleaned_solution == actual_output:
                    reward = 1.0
                else:
                    reward = 0.0
                
                # Save the results
                example['proposed_solution'] = cleaned_solution
                example['solving_prompt'] = solving_prompt
                example['reward'] = reward
                
                rewards.append(reward)
                
                print(f"Solving task - Expected: {actual_output}, Got: {cleaned_solution}, Reward: {reward}")
                
            except Exception as e:
                print(f"Error solving task: {str(e)}")
                print(traceback.format_exc())
                rewards.append(0.0)
        
        return rewards