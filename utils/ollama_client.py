import requests
import numpy as np

CHOICES = ["A", "B", "C", "D", "E", "F"]


class OllamaClient:
    """Ollama API client — scores each choice via continuation logprob."""

    def __init__(self, model_name: str = "qwen3.5:4b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url   = base_url
        self.api_url    = f"{base_url}/api/generate"

    def _get_logprob(self, prompt: str, keep_alive: str = "5m") -> float:
        """Return the logprob of the last token in the prompt."""
        response = requests.post(
            self.api_url,
            json={
                "model"      : self.model_name,
                "prompt"     : prompt,
                "stream"     : False,
                "logprobs"   : True,
                "think"      : False,
                "keep_alive" : keep_alive,
                "options"    : {"temperature": 0, "num_predict": 1},
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

        keep_alive="5m" for all calls except the last, which uses "0"
        to release the model from RAM immediately after the question is done.

        Returns:
            np.ndarray of shape (6,) -- logprobs for A, B, C, D, E, F.
        """
        logits = np.full(len(CHOICES), -1e9, dtype=np.float32)
        last   = len(CHOICES) - 1
        for i, choice in enumerate(CHOICES):
            keep_alive  = "0" if i == last else "5m"
            logits[i]   = self._get_logprob(prompt + " " + choice, keep_alive)
        return logits

    def unload(self) -> None:
        """Explicitly unload the model from RAM (keep_alive=0 with empty prompt)."""
        requests.post(
            self.api_url,
            json={"model": self.model_name, "prompt": "", "keep_alive": "0"},
            timeout=30,
        )
