import requests
import numpy as np

CHOICES = ["A", "B", "C", "D", "E", "F"]


class OllamaClient:
    """Ollama API client — scores each choice via continuation logprob."""

    def __init__(self, model_name: str = "qwen3.5:4b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url   = base_url
        self.api_url    = f"{base_url}/api/generate"

    def _get_logprob(self, prompt: str) -> float:
        """Return the logprob of the last token in the prompt."""
        response = requests.post(
            self.api_url,
            json={
                "model"     : self.model_name,
                "prompt"    : prompt,
                "stream"    : False,
                "logprobs"  : True,
                "think"     : False,
                "options"   : {"temperature": 0, "num_predict": 1},
            },
            timeout=60,
        )
        logprobs_list = response.json().get("logprobs", [])
        if logprobs_list:
            return logprobs_list[-1].get("logprob", -1e9)
        return -1e9

    def get_choice_logits(self, prompt: str) -> np.ndarray:
        """
        Score each choice A-F by appending it to the prompt and reading
        the logprob of that continuation token.

        Returns:
            np.ndarray of shape (6,) -- logprobs for A, B, C, D, E, F.
        """
        logits = np.full(len(CHOICES), -1e9, dtype=np.float32)
        for i, choice in enumerate(CHOICES):
            logits[i] = self._get_logprob(prompt + " " + choice)
        return logits
