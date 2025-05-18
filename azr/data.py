class TaskBuffer:
    """
    Buffer to store task examples.
    """
    
    def __init__(self, max_size=1000):
        """
        Initialize the task buffer.
        
        Args:
            max_size: Maximum number of examples to store
        """
        self.buffer = []
        self.max_size = max_size
    
    def add(self, example):
        """
        Add an example to the buffer.
        
        Args:
            example: Task example to add
        """
        self.buffer.append(example)
        
        # If buffer exceeds max size, remove oldest examples
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]
    
    def sample(self, n=1):
        """
        Sample n examples from the buffer.
        
        Args:
            n: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        import random
        
        if not self.buffer:
            return []
        
        # Sample min(n, buffer_size) examples
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)
    
    def size(self):
        """
        Get the number of examples in the buffer.
        
        Returns:
            Buffer size
        """
        return len(self.buffer)

def format_code_examples(examples, task_type):
    """
    Format a list of task examples for prompting.
    
    Args:
        examples: List of task examples
        task_type: Type of task ('abduction', 'deduction', or 'induction')
        
    Returns:
        Formatted examples as string
    """
    result = ""
    
    for i, example in enumerate(examples):
        result += f"Example {i+1}:\n"
        
        if task_type == "abduction":
            # Input prediction: given code and output, predict input
            result += f"Function:\n{example['function']}\n"
            result += f"Output: {example['output']}\n"
            result += f"Input: {example['input']}\n\n"
            
        elif task_type == "deduction":
            # Output prediction: given code and input, predict output
            result += f"Function:\n{example['function']}\n"
            result += f"Input: {example['input']}\n"
            result += f"Output: {example['output']}\n\n"
            
        elif task_type == "induction":
            # Function synthesis: given input-output pairs, predict function
            result += f"Input-Output Pairs:\n"
            for pair in example['pairs']:
                result += f"Input: {pair[0]}, Output: {pair[1]}\n"
            result += f"Function:\n{example['function']}\n\n"
    
    return result