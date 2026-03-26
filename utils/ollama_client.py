import requests
import time
import numpy as np
from typing import List, Dict

class OllamaClient:
    """Client for Ollama API to get logprobs for multiple-choice questions."""
    
    def __init__(self, model_name: str = "qwen3.5:0.8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def get_choice_probs(self, prompt: str, num_samples: int = 50) -> List[float]:
        """
        Get probability distribution over choices using sampling method.
        This approximates softmax probabilities (as in paper Section 6.7).
        
        Returns:
            List of 6 probabilities for options A, B, C, D, E, F
        """
        counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
        
        for _ in range(num_samples):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt + "\n\nAnswer with only the letter of the correct choice:",
                        "stream": False,
                        "temperature": 1.0,
                        "max_tokens": 5
                    },
                    timeout=30
                )
                result = response.json()
                answer = result.get("response", "").strip().upper()
                
                for choice in ["A", "B", "C", "D", "E", "F"]:
                    if choice in answer:
                        counts[choice] += 1
                        break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
                continue
        
        total = sum(counts.values())
        if total == 0:
            return [1.0/6] * 6
        
        return [counts[choice] / total for choice in ["A", "B", "C", "D", "E", "F"]]
    
    def get_choice_logits(self, prompt: str, num_samples: int = 50) -> np.ndarray:
        """
        Get logits (log of probabilities) for each choice.
        """
        probs = self.get_choice_probs(prompt, num_samples)
        logits = np.log(np.array(probs) + 1e-10)
        return logits