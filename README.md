# SecureAI

SecureAI gives your DGX/Open WebUI/vLLM setup a small API-key gateway, a Python SDK, and a built-in examples page.

## Install

Local development:

```bash
python3 -m pip install -e ".[server]"
```

GitHub-style install after pushing this repo:

```bash
python3 -m pip install git+https://github.com/360Secure/SecureAI.git
```

## API Keys

Create API keys from Open WebUI:

```text
http://spark.tail4ba90a.ts.net/secureai/api-keys
```

The SDK auto-connects to:

```text
http://spark.tail4ba90a.ts.net/secureai/v1
```

## Python SDK

```python
from SecureAI import SecureAI

ai = SecureAI(
    api_key="sk-secureai-your-key",
)

print(ai.ask("What is DGX Spark?"))
```

Streaming:

```python
from SecureAI import SecureAI

ai = SecureAI(api_key="sk-secureai-your-key")

for token in ai.stream("Tell me about DGX Spark"):
    print(token, end="", flush=True)
```

Custom SecureAI syntax:

```python
import SecureAI as AI

API = "sk-secureai-your-key"

print(AI.AskAI(API)["What is DGX Spark?"])

for token in AI.StreamAI(API)["Tell me what DGX Spark is"]:
    print(token, end="", flush=True)

for letter in AI.StreamLettersAI(API)["Say hello"]:
    print(letter, end="", flush=True)
```

## Notes

- Open WebUI on the DGX stores SecureAI API keys in SQLite.
- API keys are hashed at rest.
- The `/secureai/v1/chat/completions` and `/secureai/v1/models` endpoints are OpenAI-compatible.
- The SDK can also talk to any OpenAI-compatible server by passing `base_url=...`.
