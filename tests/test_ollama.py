import requests
import json

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model":    "qwen3.5:4b",
        "prompt":   "Question: What is 2+2?\nChoices:\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
        "stream":   False,
        "logprobs": True,
        "think":    False,
        "options":  {"temperature": 0, "num_predict": 1},
    },
)
data = response.json()
print("response:", data.get("response"))
print("logprobs:", json.dumps(data.get("logprobs"), indent=2))
