import numpy as np
import requests

CHOICES = ["A", "B", "C", "D", "E", "F"]

# NVIDIA NIM models available via build.nvidia.com
NIM_MODELS = {
    "llama3.1-8b":   "meta/llama-3.1-8b-instruct",
    "llama3.1-70b":  "meta/llama-3.1-70b-instruct",
    "llama3.3-70b":  "meta/llama-3.3-70b-instruct",
    "mistral-7b":    "mistralai/mistral-7b-instruct-v0.3",
    "qwen2.5-7b":    "qwen/qwen2.5-7b-instruct",
    "qwen2.5-72b":   "qwen/qwen2.5-72b-instruct",
    "gemma2-9b":     "google/gemma-2-9b-it",
    "gemma2-27b":    "google/gemma-2-27b-it",
}


class NIMClient:
    """
    NVIDIA NIM client — scores each choice A-F via real token logprobs.

    Unlike OllamaClient (continuation scoring), NIM returns logprobs for the
    first generated token, which is the model's direct distribution over the
    vocabulary.  We read logprob("A"), logprob("B"), ... from a single API call,
    which is both faster and more accurate than 6 separate continuation calls.

    Args:
        model_name: NIM model ID (e.g. "meta/llama-3.1-8b-instruct") or
                    a shorthand key from NIM_MODELS dict above.
        api_key:    NVIDIA API key (nvapi-...).
        base_url:   NIM endpoint, default is NVIDIA's hosted API.
    """

    def __init__(
        self,
        model_name: str = "meta/llama-3.1-8b-instruct",
        api_key: str = "",
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ):
        self.model_name = NIM_MODELS.get(model_name, model_name)
        self.base_url   = base_url.rstrip("/")
        self.headers    = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        }

    def get_choice_logits(self, prompt: str) -> np.ndarray:
        """
        Get logprobs for choices A-F in a single API call.

        The NIM chat completions API supports `logprobs=True` + `top_logprobs=20`,
        returning the top-20 token logprobs for the first generated token.
        We extract logprob for each of A, B, C, D, E, F from that list.

        Returns:
            np.ndarray of shape (6,) — logprobs for A, B, C, D, E, F.
            Missing tokens get -1e9.
        """
        payload = {
            "model":       self.model_name,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  1,
            "temperature": 0,
            "logprobs":    True,
            "top_logprobs": 20,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        token_logprobs = (
            data["choices"][0]
            .get("logprobs", {})
            .get("content", [{}])[0]
            .get("top_logprobs", [])
        )

        logprob_map = {entry["token"].strip(): entry["logprob"] for entry in token_logprobs}

        logits = np.full(len(CHOICES), -1e9, dtype=np.float32)
        for i, choice in enumerate(CHOICES):
            if choice in logprob_map:
                logits[i] = logprob_map[choice]

        return logits
