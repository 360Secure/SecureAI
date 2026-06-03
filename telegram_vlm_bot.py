from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TEXT_API_BASE = os.getenv("TEXT_API_BASE", "http://127.0.0.1:8001/v1").rstrip("/")
VLM_API_BASE = os.getenv("VLM_API_BASE", "http://127.0.0.1:8002/v1").rstrip("/")
SEARCH_BASE = os.getenv("SEARCH_BASE", "http://127.0.0.1:8081").rstrip("/")
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen32b")
VLM_MODEL = os.getenv("VLM_MODEL", "fast-vlm")
TIMEZONE = os.getenv("BOT_TIMEZONE", "Asia/Kolkata")
MAX_TELEGRAM_LEN = 3900
OFFSET_FILE = Path(os.getenv("BOT_OFFSET_FILE", "/home/secure360/.vlm-smart-bot-offset"))

API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"


def log(message: str) -> None:
    print("%s %s" % (now_text(), message), flush=True)


def now_text() -> str:
    now = datetime.now(ZoneInfo(TIMEZONE))
    return now.strftime("%A, %B %d, %Y at %H:%M:%S %Z")


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 120) -> Dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def telegram(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return http_json(f"{API_URL}/{method}", payload, timeout=60)


def send_message(chat_id: int, text: str) -> None:
    if not text.strip():
        text = "(empty response)"
    for start in range(0, len(text), MAX_TELEGRAM_LEN):
        telegram("sendMessage", {"chat_id": chat_id, "text": text[start:start + MAX_TELEGRAM_LEN]})


def send_typing(chat_id: int) -> None:
    try:
        telegram("sendChatAction", {"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


def load_offset() -> int:
    try:
        return int(OFFSET_FILE.read_text().strip())
    except Exception:
        return 0


def save_offset(offset: int) -> None:
    try:
        OFFSET_FILE.write_text(str(offset))
    except Exception as exc:
        log("offset save error: %s" % exc)


def latest_update_offset() -> int:
    try:
        body = telegram("getUpdates", {"timeout": 1, "limit": 100, "allowed_updates": ["message"]})
    except Exception as exc:
        log("latest offset error: %s" % exc)
        return 0
    updates = body.get("result", [])
    if not updates:
        return 0
    return max(int(update.get("update_id", 0)) for update in updates) + 1


def message_summary(message: Dict[str, Any]) -> str:
    if message.get("text"):
        return str(message["text"]).strip()
    if message.get("caption"):
        return str(message["caption"]).strip()
    if "photo" in message:
        return "[photo]"
    if "document" in message:
        return "[document] " + str(message["document"].get("file_name") or "")
    if "voice" in message:
        return "[voice]"
    return "[message]"


def reply_context(message: Dict[str, Any]) -> str:
    replied = message.get("reply_to_message")
    if not isinstance(replied, dict):
        return ""
    who = "assistant" if replied.get("from", {}).get("is_bot") else "user"
    summary = message_summary(replied)
    if not summary:
        return ""
    return "The user is replying to this %s message: %s\n\n" % (who, summary[:1200])


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 800,
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = http_json(f"{base_url}/chat/completions", payload, timeout=timeout)
    return body["choices"][0]["message"]["content"]


def text_completion(messages: List[Dict[str, Any]], *, max_tokens: int = 800, temperature: float = 0.2) -> str:
    try:
        return chat_completion(TEXT_API_BASE, TEXT_MODEL, messages, max_tokens=max_tokens, temperature=temperature, timeout=12)
    except Exception as exc:
        log("text model fallback to %s: %s" % (VLM_MODEL, exc))
        return chat_completion(VLM_API_BASE, VLM_MODEL, messages, max_tokens=max_tokens, temperature=temperature, timeout=180)


def strict_system_prompt() -> str:
    return (
        "You are VLM Smart Bot running on the user's DGX Spark.\n"
        f"The current real date and time is {now_text()}.\n"
        "Never claim the year is 2023 unless the user is asking about that historical year.\n"
        "Be direct, useful, and accurate. If web search results are provided, use them and cite URLs.\n"
        "If the user asks about an image, inspect the image carefully and answer what is visible.\n"
    )


def needs_web_search(question: str, memory: List[Dict[str, str]]) -> bool:
    recent_context = "\n".join("%s: %s" % (m["role"], m["content"]) for m in memory[-6:])
    prompt = (
        "You are a strict web-search router. Decide if the user's next question needs web search.\n"
        "Answer exactly one lowercase word: yes or no.\n\n"
        "Return yes when the user asks you to search, look up, find, verify, cite sources, use websites, "
        "or asks about current/latest/recent/today/live information.\n"
        "Return no only when the answer can be handled from normal reasoning or prior conversation.\n\n"
        "Examples:\n"
        "Question: search it up and find it\nAnswer: yes\n"
        "Question: look this up online\nAnswer: yes\n"
        "Question: what is the latest price of bitcoin\nAnswer: yes\n"
        "Question: find sources for BrainGym360 activities\nAnswer: yes\n"
        "Question: explain what a GPU is\nAnswer: no\n"
        "Question: write a Python loop\nAnswer: no\n\n"
        "Recent conversation:\n%s\n\nQuestion: %s\nAnswer:" % (recent_context, question)
    )
    answer = text_completion(
        [{"role": "system", "content": strict_system_prompt()}, {"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0,
    )
    return answer.strip().lower().startswith("y")


def search_web(query: str, count: int = 5) -> List[Dict[str, str]]:
    params = urllib.parse.urlencode({"q": query, "format": "json", "language": "en", "safesearch": "1"})
    try:
        body = http_json(f"{SEARCH_BASE}/search?{params}", timeout=30)
    except Exception:
        return []
    results = []
    for item in body.get("results", [])[:count]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet[:700]})
    return results


def sources_text(sources: List[Dict[str, str]]) -> str:
    parts = []
    for index, source in enumerate(sources, 1):
        parts.append(
            "[%d] %s\nURL: %s\nSnippet: %s"
            % (index, source["title"], source["url"], source.get("snippet") or "No snippet available.")
        )
    return "\n\n".join(parts)


def telegram_file_data_url(file_id: str) -> str:
    file_info = telegram("getFile", {"file_id": file_id})
    file_path = file_info["result"]["file_path"]
    with urllib.request.urlopen(f"{FILE_URL}/{file_path}", timeout=90) as response:
        raw = response.read()
    mime = "image/jpeg"
    if file_path.lower().endswith(".png"):
        mime = "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def photo_data_url_from_message(message: Dict[str, Any]) -> Optional[str]:
    photos = message.get("photo")
    if not photos:
        return None
    return telegram_file_data_url(photos[-1]["file_id"])


def answer_text(question: str, memory: List[Dict[str, str]]) -> str:
    use_search = needs_web_search(question, memory)
    messages: List[Dict[str, Any]] = [{"role": "system", "content": strict_system_prompt()}]
    if use_search:
        sources = search_web(question)
        if sources:
            messages.append({
                "role": "system",
                "content": "Web search is enabled. Use these sources and cite URLs:\n\n" + sources_text(sources),
            })
    messages.extend(memory[-10:])
    messages.append({"role": "user", "content": question})
    prefix = "web search = %s\n\n" % str(use_search).lower()
    return prefix + text_completion(messages, max_tokens=1100)


def answer_image(prompt: str, image_data_url: str) -> str:
    content = [
        {"type": "text", "text": prompt or "Describe this image carefully."},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
    messages = [{"role": "system", "content": strict_system_prompt()}, {"role": "user", "content": content}]
    return chat_completion(VLM_API_BASE, VLM_MODEL, messages, max_tokens=900, timeout=180)


def handle_message(message: Dict[str, Any], memory_by_chat: Dict[int, List[Dict[str, str]]]) -> None:
    chat_id = int(message["chat"]["id"])
    memory = memory_by_chat.setdefault(chat_id, [])
    text = (message.get("text") or "").strip()
    if text in {"/start", "/help"}:
        log("ignored command chat=%s text=%r" % (chat_id, text))
        return

    send_typing(chat_id)
    context = reply_context(message)
    if "photo" in message:
        log("photo message chat=%s reply=%s" % (chat_id, bool(context)))
        prompt = context + (message.get("caption") or "Describe this image and answer any question in the caption.")
        image_data_url = photo_data_url_from_message(message)
        if not image_data_url:
            answer = "I could not load that image."
            send_message(chat_id, answer)
            log("sent answer chat=%s chars=%s" % (chat_id, len(answer)))
            return
        answer = answer_image(prompt, image_data_url)
    else:
        log("text message chat=%s reply=%s text=%r" % (chat_id, bool(context), text[:80]))
        if text == "/clear":
            memory.clear()
            answer = "Memory cleared."
        elif text:
            replied = message.get("reply_to_message") if isinstance(message.get("reply_to_message"), dict) else {}
            replied_image = photo_data_url_from_message(replied)
            question = context + text
            if replied_image:
                log("text reply to photo chat=%s" % chat_id)
                answer = answer_image(question or "Answer about this image.", replied_image)
            else:
                answer = answer_text(question, memory)
                memory.append({"role": "user", "content": question})
                memory.append({"role": "assistant", "content": answer})
                if len(memory) > 12:
                    memory[:] = memory[-12:]
        else:
            answer = "Send text or a photo."
    send_message(chat_id, answer)
    log("sent answer chat=%s chars=%s" % (chat_id, len(answer)))


def main() -> None:
    offset = load_offset()
    if offset <= 0:
        offset = latest_update_offset()
        save_offset(offset)
        log("initialized offset=%s" % offset)
    memory_by_chat: Dict[int, List[Dict[str, str]]] = {}
    log("bot started offset=%s" % offset)
    while True:
        try:
            updates = telegram("getUpdates", {"offset": offset, "timeout": 30, "allowed_updates": ["message"]})
            for update in updates.get("result", []):
                offset = max(offset, int(update["update_id"]) + 1)
                save_offset(offset)
                message = update.get("message")
                if message:
                    try:
                        handle_message(message, memory_by_chat)
                    except Exception as exc:
                        log("message error: %s" % exc)
                        send_message(int(message["chat"]["id"]), "Error: %s" % exc)
        except urllib.error.URLError as exc:
            log("telegram network error: %s" % exc)
            time.sleep(5)
        except Exception as exc:
            log("bot loop error: %s" % exc)
            time.sleep(3)


if __name__ == "__main__":
    main()
