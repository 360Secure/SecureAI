from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx


API = os.getenv("SECUREAI_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("SECUREAI_BASE_URL", "http://spark.tail4ba90a.ts.net/secureai/v1")


class SecureAIError(RuntimeError):
    """Raised when the SecureAI API returns an error."""


@dataclass
class ChatMessage:
    role: str
    content: Any

    def as_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


class SecureAI:
    """Small Python SDK for SecureAI/OpenAI-compatible local model APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "qwen72b-vl",
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SECUREAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout

    def ask(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> str:
        response = self.chat(
            prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        return response["choices"][0]["message"]["content"]

    def chat(
        self,
        prompt: str | List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        messages = self._messages(prompt, system=system)
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            **extra,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        with httpx.Client(timeout=self.timeout) as client:
            res = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
        return self._json_or_error(res)

    def stream(
        self,
        prompt: str | List[Dict[str, Any]],
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **extra: Any,
    ) -> Iterable[str]:
        messages = self._messages(prompt, system=system)
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            **extra,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        ) as res:
            if res.status_code >= 400:
                raise SecureAIError(res.text)

            for line in res.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = event.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token

    def stream_letters(self, *args: Any, **kwargs: Any) -> Iterable[str]:
        """Yield streaming output character-by-character."""
        for token in self.stream(*args, **kwargs):
            for char in token:
                yield char

    def vision(
        self,
        prompt: str,
        image_url: str,
        *,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
        return self.ask(messages, model=model, temperature=temperature)

    def models(self) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            res = client.get(f"{self.base_url}/models", headers=self._headers())
        return self._json_or_error(res)

    def _messages(self, prompt: str | List[Dict[str, Any]], *, system: Optional[str]) -> List[Dict[str, Any]]:
        if isinstance(prompt, list):
            return prompt
        messages = []
        if system:
            messages.append(ChatMessage("system", system).as_dict())
        messages.append(ChatMessage("user", prompt).as_dict())
        return messages

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _json_or_error(res: httpx.Response) -> Dict[str, Any]:
        if res.status_code >= 400:
            raise SecureAIError(res.text)
        try:
            return res.json()
        except json.JSONDecodeError as exc:
            raise SecureAIError(res.text) from exc


class _Command:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: str = "qwen72b-vl",
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.client = SecureAI(api_key=api_key or API, base_url=base_url, model=model)
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _prompt(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return "\n".join(str(part) for part in value)
        return value


class AskAI(_Command):
    """Square-bracket syntax for one-shot answers: AskAI(API)["question"]."""

    def __getitem__(self, prompt: Any) -> str:
        return self.client.ask(
            self._prompt(prompt),
            system=self.system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class ChatAI(_Command):
    """Square-bracket syntax returning the raw chat completion JSON."""

    def __getitem__(self, prompt: Any) -> Dict[str, Any]:
        return self.client.chat(
            self._prompt(prompt),
            system=self.system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class StreamAI(_Command):
    """Square-bracket syntax for token streaming: StreamAI(API)["question"]."""

    def __getitem__(self, prompt: Any) -> Iterable[str]:
        return self.client.stream(
            self._prompt(prompt),
            system=self.system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class StreamLettersAI(_Command):
    """Square-bracket syntax for character-by-character streaming."""

    def __getitem__(self, prompt: Any) -> Iterable[str]:
        return self.client.stream_letters(
            self._prompt(prompt),
            system=self.system,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


class VisionAI(_Command):
    """Vision helper: VisionAI(API, image_url='...')['describe this'].""" 

    def __init__(self, api_key: Optional[str] = None, *, image_url: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)
        self.image_url = image_url

    def __getitem__(self, prompt: Any) -> str:
        return self.client.vision(
            str(self._prompt(prompt)),
            self.image_url,
            system=self.system,
            temperature=self.temperature,
        )


class ModelAI(_Command):
    """List available models: ModelAI(API)[] is invalid Python, use ModelAI(API).list()."""

    def list(self) -> Dict[str, Any]:
        return self.client.models()
