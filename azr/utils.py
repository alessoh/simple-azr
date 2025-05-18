def extract_function_code(text):
    """
    Extract function code from text.
    
    Args:
        text: Text containing Python code
        
    Returns:
        Extracted function code or None
    """
    try:
        # Try to extract code from Markdown code blocks
        if "```python" in text and "```" in text.split("```python", 1)[1]:
            return text.split("```python", 1)[1].split("```", 1)[0].strip()
        
        # Try to extract code without explicit Markdown
        if "def f(" in text:
            lines = text.split("def f(", 1)[1].split("\n")
            function_lines = ["def f(" + lines[0]]
            
            for line in lines[1:]:
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    break
                function_lines.append(line)
            
            return "\n".join(function_lines)
        
        return None
    
    except Exception:
        return None

def clean_output(output):
    """
    Clean up generated output text.
    
    Args:
        output: Raw output text
        
    Returns:
        Cleaned output
    """
    # Remove leading/trailing whitespace
    output = output.strip()
    
    # Try to extract just the answer part if there's explanation text
    if "answer is" in output.lower():
        output = output.split("answer is", 1)[1].strip()
    
    # Remove quotes around outputs if they're added by the model
    if output.startswith('"') and output.endswith('"'):
        output = output[1:-1]
    elif output.startswith("'") and output.endswith("'"):
        output = output[1:-1]
    
    return output

def parse_code(code):
    """
    Parse code to extract its features.
    
    Args:
        code: Python code string
        
    Returns:
        Dict of code features
    """
    import ast
    import re
    
    features = {
        'lines': len(code.split('\n')),
        'chars': len(code),
        'functions': 0,
        'loops': 0,
        'conditionals': 0,
        'variables': set()
    }
    
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['functions'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                features['loops'] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                features['conditionals'] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                features['variables'].add(node.id)
    
    except Exception:
        pass
    
    features['variables'] = len(features['variables'])
    
    return features

def is_valid_python_code(code):
    """
    Check if a string is valid Python code.
    
    Args:
        code: String to check
        
    Returns:
        True if valid Python code, False otherwise
    """
    try:
        import ast
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception:
        return False