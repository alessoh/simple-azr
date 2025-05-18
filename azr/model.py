import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleAZRModel:
    """
    A simple wrapper around a language model for the AZR system.
    This represents both the proposer and solver roles using the same model.
    """
    
    def __init__(self, model_name):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the HuggingFace model to use
        """
        try:
            print(f"Loading model: {model_name}")
            # Load model with efficient settings for CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Save the model name for reference in generate method
            self.model_name = model_name
            
            print(f"Successfully loaded {model_name}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to gpt2-medium...")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
            self.model_name = "gpt2-medium"
        
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to training mode
        self.model.train()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
    
    def generate(self, prompt, max_length=300, temperature=0.7, top_p=0.9):
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of the generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text string
        """
        # Use lower temperature for code generation tasks
        if "def f" in prompt or "function" in prompt.lower():
            temperature = 0.3  # Lower temperature for more focused code generation
        
        # Special handling for phi models
        if "phi" in self.model_name.lower():
            # Format prompt for phi model if it's not already formatted
            if not prompt.strip().startswith("<|user|>"):
                # Check if this is likely a code generation task
                if "def f" in prompt or "function" in prompt.lower():
                    prompt = f"<|user|>\nWrite a simple Python function:\n{prompt}\n<|assistant|>\n"
                else:
                    prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create proper attention mask if it doesn't exist
        if 'attention_mask' not in inputs:
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        
        # Calculate total max length (prompt + new tokens)
        total_max_length = len(inputs['input_ids'][0]) + max_length
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=total_max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the entire generated sequence
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the newly generated part (everything after the prompt)
            prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt_text):
                result = generated_text[len(prompt_text):]
            else:
                # Fallback in case exact matching fails
                result = generated_text
            
            # Clean up the result
            result = self._clean_generated_text(result)
            
            return result.strip()
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return "Error generating response."
    
    def _clean_generated_text(self, text):
        """
        Clean up generated text, especially for code.
        
        Args:
            text: The generated text to clean
        
        Returns:
            Cleaned text
        """
        # Remove common stop patterns for code
        stop_patterns = [
            "```\n",
            "<|endoftext|>",
            "# Test the function",
            "# Example usage",
            "# Let's test",
            "# Output:",
            "# Result:",
        ]
        
        result = text
        for pattern in stop_patterns:
            if pattern in result:
                result = result.split(pattern)[0]
        
        # For phi models, clean up any special tokens
        if hasattr(self, 'model_name') and "phi" in self.model_name.lower():
            if "<|assistant|>" in result:
                result = result.split("<|assistant|>")[1]
            if "<|user|>" in result:
                result = result.split("<|user|>")[0]
        
        return result
    
    def compute_loss(self, prompt, target):
        """
        Compute the loss for a prompt-target pair.
        
        Args:
            prompt: The input prompt
            target: The target completion
            
        Returns:
            Loss value
        """
        try:
            # Tokenize inputs
            input_text = prompt + target
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            # Move inputs to the same device as the model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate labels (shift inputs to right)
            labels = inputs['input_ids'].clone()
            
            # Mask prompt tokens in labels (-100 is the token ignored in loss computation)
            prompt_length = len(self.tokenizer(prompt, return_tensors="pt")['input_ids'][0])
            labels[0, :prompt_length] = -100
            
            # Forward pass
            outputs = self.model(inputs['input_ids'], labels=labels)
            
            return outputs.loss
        
        except Exception as e:
            print(f"Error computing loss: {str(e)}")
            # Return a dummy loss if computation fails
            return torch.tensor(0.0, requires_grad=True)
    
    def update(self, prompt, target, reward, learning_rate=1e-5):
        """
        Update model parameters using reinforcement learning.
        This is a simplified version that uses a weighted loss.
        
        Args:
            prompt: Input prompt
            target: Generated target
            reward: Reward value
            learning_rate: Learning rate
        """
        # Ensure learning_rate is a float
        learning_rate = float(learning_rate)
        
        try:
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
                
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute loss
            loss = self.compute_loss(prompt, target)
            
            # Scale loss by reward
            # In RL, we want to maximize reward, so we minimize -reward*loss
            scaled_loss = -reward * loss
            
            # Backward pass
            scaled_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            
            return loss.item()
        
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return 0.0
    
    def save(self, path):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Model successfully saved to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            print("Continuing without saving...")