import subprocess
import ast
import tempfile
import traceback
import os

class PythonExecutor:
    """
    Environment for executing Python code safely and verifying task validity.
    """
    
    def __init__(self, timeout=5, max_output_size=1000):
        """
        Initialize the Python executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_size: Maximum allowed output size
        """
        self.timeout = timeout
        self.max_output_size = max_output_size
        
        # Forbidden modules for security
        self.forbidden_modules = [
            'os', 'sys', 'subprocess', 'shutil', 'importlib',
            'multiprocessing', 'threading', 'socket'
        ]
    
    def check_security(self, code):
        """
        Check if the code contains forbidden modules.
        
        Args:
            code: Python code string
            
        Returns:
            True if code is safe, False otherwise
        """
        try:
            # Parse code to AST
            tree = ast.parse(code)
            
            # Check for imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in self.forbidden_modules:
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.forbidden_modules:
                        return False
            
            return True
        
        except SyntaxError:
            return False
    
    def check_determinism(self, function_code, input_str, num_runs=2):
        """
        Check if the function is deterministic by running it multiple times.
        
        Args:
            function_code: Python function code
            input_str: Input string
            num_runs: Number of runs to check
            
        Returns:
            True if deterministic, False otherwise
        """
        outputs = []
        
        # Run function multiple times
        for _ in range(num_runs):
            output = self.execute_code(
                f"{function_code}\n"
                f"result = f({input_str})\n"
                f"print(repr(result))"
            )
            outputs.append(output)
        
        # Check if all outputs are the same
        return all(output == outputs[0] for output in outputs)
    
    def execute_code(self, code, input_data=None):
        """
        Execute Python code safely in a subprocess.
        
        Args:
            code: Python code to execute
            input_data: Optional input data for stdin
            
        Returns:
            Output of the code execution or error message
        """
        # Check for security violations
        if not self.check_security(code):
            return "Error: Security violation detected."
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp_filename = temp.name
            temp.write(code.encode('utf-8'))
        
        try:
            # Execute the code
            process = subprocess.Popen(
                ['python', temp_filename],
                stdin=subprocess.PIPE if input_data else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Get output with timeout
            stdout, stderr = process.communicate(
                input=input_data, 
                timeout=self.timeout
            )
            
            # Check output size
            if len(stdout) > self.max_output_size or len(stderr) > self.max_output_size:
                return "Error: Output size exceeded."
            
            # Return output or error
            if process.returncode != 0:
                return f"Error: {stderr.strip()}"
            
            return stdout.strip()
        
        except subprocess.TimeoutExpired:
            process.kill()
            return "Error: Execution timed out."
        
        except Exception as e:
            return f"Error: {str(e)}"
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    def validate_task(self, function_code, input_str):
        """
        Validate a task by executing the function with the given input.
        
        Args:
            function_code: Python function code
            input_str: Input string
            
        Returns:
            (success, output) tuple
        """
        try:
            # Check if code is secure
            if not self.check_security(function_code):
                return False, "Security violation detected."
            
            # Validate function syntax
            try:
                ast.parse(function_code)
            except SyntaxError:
                return False, "Invalid function syntax."
            
            # Execute the function
            result = self.execute_code(
                f"{function_code}\n"
                f"result = f({input_str})\n"
                f"print(repr(result))"
            )
            
            # Check for errors
            if result.startswith("Error:"):
                return False, result
            
            # Check determinism
            if not self.check_determinism(function_code, input_str):
                return False, "Function is not deterministic."
            
            return True, result
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"