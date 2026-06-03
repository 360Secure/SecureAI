from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import time
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from open_webui.utils.auth import get_verified_user

router = APIRouter()

KEYS_DB = os.getenv('SECUREAI_KEYS_DB', '/app/backend/data/secureai_api_keys.db')
DEFAULT_BASE_URL = os.getenv('SECUREAI_DEFAULT_BASE_URL', 'http://spark.tail4ba90a.ts.net/secureai/v1')
GITHUB_INSTALL = os.getenv('SECUREAI_GITHUB_INSTALL', 'python3 -m pip install git+https://github.com/360Secure/SecureAI.git')
UPSTREAM_BASE_URL = os.getenv('SECUREAI_UPSTREAM_BASE_URL', 'http://172.17.0.1:8001/v1').rstrip('/')
VLM_UPSTREAM_BASE_URL = os.getenv('SECUREAI_VLM_UPSTREAM_BASE_URL', 'http://172.17.0.1:8002/v1').rstrip('/')
UPSTREAM_API_KEY = os.getenv('SECUREAI_UPSTREAM_API_KEY', 'none')
VLM_MODEL = os.getenv('SECUREAI_VLM_MODEL', 'fast-vlm')
SEARCH_BASE_URL = os.getenv('SECUREAI_SEARCH_BASE_URL', 'http://172.17.0.1:8081').rstrip('/')
MAX_INPUT_TOKENS = int(os.getenv('SECUREAI_MAX_INPUT_TOKENS', '8192'))
SAFE_INPUT_TOKENS = int(os.getenv('SECUREAI_SAFE_INPUT_TOKENS', '7400'))
CHARS_PER_TOKEN = 4


class KeyForm(BaseModel):
    name: str = 'python-sdk'


class RevokeForm(BaseModel):
    key_id: str


class AnnotateForm(BaseModel):
    image_url: str
    instruction: str = 'circle all humans'
    color: str = 'red'


def db():
    conn = sqlite3.connect(KEYS_DB)
    conn.row_factory = sqlite3.Row
    conn.execute(
        'create table if not exists api_keys ('
        'id text primary key, user_id text not null, name text not null, '
        'key_hash text not null unique, prefix text not null, '
        'created_at integer not null, revoked_at integer)'
    )
    conn.commit()
    return conn


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def upstream_headers():
    headers = {'Content-Type': 'application/json'}
    if UPSTREAM_API_KEY:
        headers['Authorization'] = f'Bearer {UPSTREAM_API_KEY}'
    return headers


def payload_needs_vlm(payload: Dict[str, Any]) -> bool:
    model = str(payload.get('model') or '').lower()
    if model in {'fast-vlm', 'qwen3vl-fast', 'qwen3-vl-8b', 'qwen3-vl-32b', 'qwen3vl32', 'qwen/qwen3-vl-32b-instruct'}:
        return True
    for message in payload.get('messages', []) or []:
        content = message.get('content') if isinstance(message, dict) else None
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and (item.get('type') == 'image_url' or 'image_url' in item):
                    return True
    return False


def upstream_for_payload(payload: Dict[str, Any]) -> str:
    return VLM_UPSTREAM_BASE_URL if payload_needs_vlm(payload) else UPSTREAM_BASE_URL


def vlm_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    fallback = dict(payload)
    fallback['model'] = VLM_MODEL
    return fallback


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text') or item.get('content')
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return '\n'.join(parts)
    return str(content or '')


def host_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ''
    return host.removeprefix('www.')


def query_domains(query: str) -> List[str]:
    domains = []
    for token in query.replace('https://', ' ').replace('http://', ' ').split():
        token = token.strip('.,;:!?()[]{}"\'').lower().removeprefix('www.')
        if '.' in token and ' ' not in token:
            domains.append(token.split('/')[0])
    return domains


def source_reliability(source: Dict[str, str], query: str) -> tuple[int, str]:
    host = host_from_url(source.get('url', ''))
    q_domains = query_domains(query)
    if any(host == domain or host.endswith('.' + domain) for domain in q_domains):
        return 0, 'primary/official site named in the question'
    if host and not any(part in host for part in ('wikipedia.org', 'reddit.com', 'facebook.com', 'instagram.com', 'x.com', 'twitter.com', 'linkedin.com', 'medium.com', 'quora.com')):
        return 1, 'direct or topic-specific source'
    if 'wikipedia.org' in host:
        return 2, 'general reference'
    return 3, 'third-party/community source'


def latest_user_text(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get('role') == 'user':
            return message_text(message.get('content'))
    return ''


async def search_web(query: str, count: int = 6) -> List[Dict[str, str]]:
    count = max(1, min(int(count or 6), 10))
    params = {'q': query, 'format': 'json', 'language': 'en', 'safesearch': '1'}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        async with session.get(f'{SEARCH_BASE_URL}/search', params=params) as response:
            if response.status >= 400:
                return []
            data = await response.json(content_type=None)
    results = []
    for domain in query_domains(query):
        url = 'https://' + domain + '/'
        title = domain
        snippet = 'Official site named directly in the user question.'
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, allow_redirects=True) as response:
                    text = await response.text(errors='ignore')
                    title_match = re.search(r'<title[^>]*>(.*?)</title>', text, flags=re.I | re.S)
                    desc_match = re.search(r'<meta[^>]+name=["\\\']description["\\\'][^>]+content=["\\\'](.*?)["\\\']', text, flags=re.I | re.S)
                    if title_match:
                        title = re.sub(r'\s+', ' ', title_match.group(1)).strip()[:140] or title
                    if desc_match:
                        snippet = re.sub(r'\s+', ' ', desc_match.group(1)).strip()[:700] or snippet
                    url = str(response.url)
        except Exception:
            pass
        source = {'title': title, 'url': url, 'snippet': snippet}
        _, label = source_reliability(source, query)
        source['reliability'] = label
        source['_rank'] = '0'
        results.append(source)
    for item in data.get('results', [])[: max(count * 2, count)]:
        title = str(item.get('title') or '').strip()
        url = str(item.get('url') or '').strip()
        snippet = str(item.get('content') or item.get('snippet') or '').strip()
        if title and url:
            source = {'title': title, 'url': url, 'snippet': snippet[:700]}
            rank, label = source_reliability(source, query)
            source['reliability'] = label
            source['_rank'] = str(rank)
            results.append(source)
    seen_hosts = set()
    unique_results = []
    for source in results:
        host = host_from_url(source.get('url', ''))
        key = host or source.get('url', '')
        if key in seen_hosts:
            continue
        seen_hosts.add(key)
        unique_results.append(source)
    results = unique_results
    results.sort(key=lambda source: int(source.get('_rank', '9')))
    for source in results:
        source.pop('_rank', None)
    return results[:count]


def sources_block(sources: List[Dict[str, str]]) -> str:
    lines = ['Web search results:']
    for index, source in enumerate(sources, 1):
        snippet = source.get('snippet') or 'No snippet available.'
        reliability = source.get('reliability') or 'unranked source'
        lines.append(f'[{index}] {source["title"]}\nURL: {source["url"]}\nReliability: {reliability}\nSnippet: {snippet}')
    return '\n\n'.join(lines)


def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        total_chars += len(str(message.get('role', ''))) + len(message_text(message.get('content')))
    return max(1, total_chars // CHARS_PER_TOKEN)


def trimmed_message(message: Dict[str, Any], max_chars: int) -> Dict[str, Any]:
    message = dict(message)
    content = message_text(message.get('content'))
    if len(content) > max_chars:
        content = '[older text compressed]\n' + content[-max_chars:]
    message['content'] = content
    return message


def compact_payload_context(payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    messages = payload.get('messages')
    if not isinstance(messages, list) or estimate_message_tokens(messages) <= SAFE_INPUT_TOKENS:
        return payload, False

    max_chars = SAFE_INPUT_TOKENS * CHARS_PER_TOKEN
    system_messages = [dict(message) for message in messages if message.get('role') == 'system']
    chat_messages = [dict(message) for message in messages if message.get('role') != 'system']
    compacted: List[Dict[str, Any]] = []
    used_chars = 0

    for message in system_messages[-3:]:
        compacted_message = trimmed_message(message, 1800)
        compacted.append(compacted_message)
        used_chars += len(message_text(compacted_message.get('content'))) + 32

    compression_note = {
        'role': 'system',
        'content': (
            'Some older conversation was compressed automatically to stay within the 8192 token context limit. '
            'Use the newest messages as the strongest source of truth.'
        ),
    }
    compacted.append(compression_note)
    used_chars += len(compression_note['content']) + 32

    kept_newest: List[Dict[str, Any]] = []
    for message in reversed(chat_messages):
        per_message_limit = 2800 if message.get('role') == 'assistant' else 4200
        compacted_message = trimmed_message(message, per_message_limit)
        message_chars = len(message_text(compacted_message.get('content'))) + 32
        if kept_newest and used_chars + message_chars > max_chars:
            break
        if used_chars + message_chars > max_chars:
            compacted_message = trimmed_message(message, max(1000, max_chars - used_chars - 64))
            message_chars = len(message_text(compacted_message.get('content'))) + 32
        kept_newest.append(compacted_message)
        used_chars += message_chars

    payload = dict(payload)
    payload['messages'] = compacted + list(reversed(kept_newest))
    return payload, True


def context_status_event(model: str) -> bytes:
    event = {
        'id': 'secureai-context-compressed',
        'object': 'chat.completion.chunk',
        'model': model,
        'choices': [{'index': 0, 'delta': {'content': 'compressing context...\n'}, 'finish_reason': None}],
    }
    return f'data: {json.dumps(event)}\n\n'.encode('utf-8')


async def image_bytes_from_url(image_url: str) -> bytes:
    if image_url.startswith('data:image/'):
        _, encoded = image_url.split(',', 1)
        return base64.b64decode(encoded)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.get(image_url) as response:
            if response.status >= 400:
                raise HTTPException(status_code=400, detail='Could not download image')
            return await response.read()


def parse_boxes(text: str) -> List[Dict[str, Any]]:
    match = re.search(r'```(?:json)?\s*(.*?)```', text, flags=re.S)
    if match:
        text = match.group(1)
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return []
    boxes = []
    if isinstance(data, dict):
        data = data.get('boxes') or data.get('objects') or []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        box = item.get('box') or item.get('bbox') or item.get('coordinates')
        if isinstance(box, dict):
            x1 = box.get('x1') or box.get('left') or box.get('xmin')
            y1 = box.get('y1') or box.get('top') or box.get('ymin')
            x2 = box.get('x2') or box.get('right') or box.get('xmax')
            y2 = box.get('y2') or box.get('bottom') or box.get('ymax')
            box = [x1, y1, x2, y2]
        if isinstance(box, list) and len(box) >= 4:
            try:
                boxes.append({'label': str(item.get('label') or item.get('name') or 'object'), 'box': [float(v) for v in box[:4]]})
            except Exception:
                pass
    return boxes


def draw_box_circles(image_bytes: bytes, boxes: List[Dict[str, Any]], color: str) -> bytes:
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    line_width = max(3, min(width, height) // 180)
    for item in boxes:
        x1, y1, x2, y2 = item['box']
        if max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        if max(x1, y1, x2, y2) <= 1000:
            x1, x2 = x1 / 1000 * width, x2 / 1000 * width
            y1, y2 = y1 / 1000 * height, y2 / 1000 * height
        pad_x = max(8, (x2 - x1) * 0.12)
        pad_y = max(8, (y2 - y1) * 0.12)
        draw.ellipse((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), outline=color, width=line_width)
    output = io.BytesIO()
    image.save(output, format='PNG')
    return output.getvalue()


async def prepare_payload(payload: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, str]], bool]:
    payload = dict(payload)
    web_search = bool(payload.pop('web_search', False) or payload.pop('search', False))
    search_query = payload.pop('search_query', None)
    search_count = payload.pop('search_count', 6)
    if not web_search:
        compacted_payload, compressed = compact_payload_context(payload)
        return compacted_payload, [], compressed

    messages = [dict(message) for message in payload.get('messages', [])]
    query = str(search_query or latest_user_text(messages)).strip()
    if not query:
        compacted_payload, compressed = compact_payload_context(payload)
        return compacted_payload, [], compressed

    sources = await search_web(query, search_count)
    if not sources:
        compacted_payload, compressed = compact_payload_context(payload)
        return compacted_payload, [], compressed

    instruction = (
        'Use the web search results below when they are relevant. Sources include reliability labels. '
        'Treat a primary/official site named in the question as the strongest source of truth for what that site says or offers. '
        'Use third-party/community/general-reference sources only as secondary context, criticism, or corroboration. '
        'If third-party sources conflict with the official site, clearly separate official claims from outside commentary. '
        'Answer the user clearly, cite source URLs, and say when the results do not contain enough evidence.'
    )
    messages.insert(0, {'role': 'system', 'content': instruction})
    messages.append({'role': 'user', 'content': f'{sources_block(sources)}\n\nUser question: {query}'})
    payload['messages'] = messages
    compacted_payload, compressed = compact_payload_context(payload)
    return compacted_payload, sources, compressed


def verify_bearer(authorization: Optional[str]):
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing SecureAI API key')
    key_hash = hash_key(authorization.removeprefix('Bearer ').strip())
    conn = db()
    row = conn.execute('select * from api_keys where key_hash = ? and revoked_at is null', (key_hash,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail='Invalid SecureAI API key')
    return row


def page_html() -> str:
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>API Keys | Open WebUI</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0f0f0f; --panel:#171717; --panel2:#202020; --text:#f4f4f5; --muted:#a1a1aa; --line:#2f2f2f; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ border-bottom:1px solid var(--line); background:#111; padding:18px 22px; position:sticky; top:0; z-index:2; }}
    nav {{ display:flex; align-items:center; gap:10px; margin-bottom:16px; }} a {{ color:inherit; text-decoration:none; }} .back {{ border:1px solid var(--line); background:var(--panel2); padding:8px 12px; border-radius:8px; font-weight:700; }}
    main {{ max-width:1120px; margin:0 auto; padding:22px; }} h1 {{ margin:0; font-size:32px; letter-spacing:0; }} h2 {{ font-size:20px; margin:0 0 12px; }} p {{ color:var(--muted); line-height:1.55; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }} .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; margin-bottom:16px; }}
    label {{ display:block; font-weight:700; margin:12px 0 8px; }} input {{ width:100%; min-height:42px; border-radius:8px; border:1px solid var(--line); background:#0b0b0b; color:var(--text); padding:10px 12px; font-size:15px; }}
    button {{ min-height:40px; border:1px solid var(--line); border-radius:8px; background:#f4f4f5; color:#111; font-weight:800; padding:0 13px; cursor:pointer; }} button.secondary {{ background:var(--panel2); color:var(--text); }}
    .actions {{ display:flex; gap:10px; flex-wrap:wrap; margin:12px 0; }} pre {{ margin:0; white-space:pre-wrap; overflow:auto; background:#0b0b0b; border:1px solid var(--line); border-radius:8px; padding:13px; color:#e5e5e5; line-height:1.45; }}
    code {{ font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; }} .small {{ font-size:13px; color:var(--muted); }} .warn {{ color:#fbbf24; }}
    @media (max-width: 860px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header><nav><a class="back" href="/">Open WebUI</a><a class="back" href="/settings/account">Account</a></nav><h1>API Keys</h1><p>Create SecureAI keys for Python apps. Keys authorize calls to this DGX through Open WebUI.</p></header>
  <main>
    <section class="card"><h2>1. Install SecureAI</h2><p>One command. Paste this into any computer on your tailnet.</p><div class="actions"><button onclick="copyText('install')">Copy Install Command</button></div><pre><code id="install">{GITHUB_INSTALL}</code></pre></section>
    <section class="grid"><div class="card"><h2>2. Create New API Key</h2><p>Secret keys show once. They are stored hashed on the DGX.</p><label>Key name</label><input id="keyName" value="python-sdk" /><div class="actions"><button onclick="createKey()">Create New API Key</button><button class="secondary" onclick="listKeys()">View API Keys</button></div><pre id="createdKey">No key created yet.</pre></div><div class="card"><h2>3. View API Keys</h2><p>Existing key prefixes only. The full secret is hidden.</p><pre id="keyList">Click View API Keys.</pre></div></section>
    <section class="card"><h2>4. Make app.py</h2><p>Paste this code into <code>app.py</code>. After creating a key, the placeholder updates automatically.</p><div class="actions"><button onclick="copyText('askExample')">Copy Python Example</button><button class="secondary" onclick="copyText('streamExample')">Copy Streaming Search Example</button><button class="secondary" onclick="copyText('vlmExample')">Copy VLM Example</button><button class="secondary" onclick="copyText('annotateExample')">Copy Circle Image Example</button><button class="secondary" onclick="copyText('autoSearchExample')">Copy Memory Chat</button><button class="secondary" onclick="copyText('persistentMemoryExample')">Copy Saved Memory Chat</button><button class="secondary" onclick="copyText('run')">Copy Run Command</button></div><pre><code id="askExample">from SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nai = SecureAI(api_key=API)\n\nprint(ai.ask("What is DGX Spark? Answer with sources.", web_search=True))</code></pre><p class="small">Run it:</p><pre><code id="run">python3 app.py</code></pre></section>
    <section class="card"><h2>Streaming Web Search</h2><pre><code id="streamExample">from SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nai = SecureAI(api_key=API)\n\nfor token in ai.stream(\n    "Search the web and tell me the latest about DGX Spark with sources.",\n    web_search=True,\n    search_count=5,\n):\n    print(token, end="", flush=True)</code></pre></section>
    <section class="card"><h2>Qwen 32B VLM Image</h2><pre><code id="vlmExample">from SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nai = SecureAI(api_key=API)\n\nprint(ai.vision(\n    "Describe this image carefully.",\n    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg",\n    model="qwen32b",\n))</code></pre></section>
    <section class="card"><h2>Circle Objects In Image</h2><pre><code id="annotateExample">from SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nai = SecureAI(api_key=API)\n\npng = ai.annotate_image(\n    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/People_at_a_crosswalk.jpg/640px-People_at_a_crosswalk.jpg",\n    instruction="circle all humans",\n    color="red",\n)\n\nwith open("circled.png", "wb") as file:\n    file.write(png)\n\nprint("saved circled.png")</code></pre></section>
    <section class="card"><h2>Auto Search Memory Chat</h2><pre><code id="autoSearchExample">from SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nai = SecureAI(api_key=API)\nmemory = []\nMAX_MEMORY_MESSAGES = 12\n\n\ndef needs_web_search(question):\n    recent_context = "\\n".join("%s: %s" % (m['role'], m['content']) for m in memory[-6:])\n    decision = ai.ask(\n        "You are a strict web-search router. Decide if the user's next question needs web search.\\n"\n        "Answer exactly one lowercase word: yes or no.\\n\\n"\n        "Return yes when the user asks you to search, look up, find, verify, cite sources, use websites, "\n        "or asks about current/latest/recent/today/live information.\\n"\n        "Return no only when the answer can be handled from normal reasoning or prior conversation.\\n\\n"\n        "Examples:\\n"\n        "Question: search it up and find it\\nAnswer: yes\\n"\n        "Question: look this up online\\nAnswer: yes\\n"\n        "Question: what is the latest price of bitcoin\\nAnswer: yes\\n"\n        "Question: find sources for BrainGym360 activities\\nAnswer: yes\\n"\n        "Question: explain what a GPU is\\nAnswer: no\\n"\n        "Question: write a Python loop\\nAnswer: no\\n\\n"\n        "Recent conversation:\\n%s\\n\\nQuestion: %s\\nAnswer:" % (recent_context, question),\n        temperature=0,\n        max_tokens=3,\n    )\n    return decision.strip().lower().startswith("y")\n\n\nprint("SecureAI terminal chat. Type bye to stop.")\n\nwhile True:\n    question = input("\\nYou: ").strip()\n    if question.lower() in {"bye", "exit", "quit"}:\n        print("SecureAI: bye")\n        break\n    if not question:\n        continue\n\n    use_search = needs_web_search(question)\n    print("web search = " + str(use_search).lower())\n    print("SecureAI: ", end="", flush=True)\n\n    messages = memory + [dict(role="user", content=question)]\n    answer_parts = []\n    for token in ai.stream(messages, web_search=use_search, search_count=5, temperature=0.2):\n        answer_parts.append(token)\n        print(token, end="", flush=True)\n\n    memory.append(dict(role="user", content=question))\n    memory.append(dict(role="assistant", content="".join(answer_parts)))\n    if len(memory) > MAX_MEMORY_MESSAGES:\n        print("compressing context...")\n        memory = memory[-MAX_MEMORY_MESSAGES:]\n    print()</code></pre></section>
    <section class="card"><h2>Saved Memory Chat</h2><pre><code id="persistentMemoryExample">import json\nfrom pathlib import Path\nfrom SecureAI import SecureAI\n\nAPI = "sk-secureai-your-key"\nMEMORY_FILE = Path("secureai_memory.json")\nMAX_MEMORY_MESSAGES = 20\nai = SecureAI(api_key=API)\n\n\ndef load_memory():\n    if not MEMORY_FILE.exists():\n        return []\n    try:\n        data = json.loads(MEMORY_FILE.read_text())\n    except json.JSONDecodeError:\n        return []\n    return data[-MAX_MEMORY_MESSAGES:] if isinstance(data, list) else []\n\n\ndef save_memory(memory):\n    MEMORY_FILE.write_text(json.dumps(memory[-MAX_MEMORY_MESSAGES:], indent=2))\n\n\ndef needs_web_search(question, memory):\n    recent_context = "\\n".join("%s: %s" % (m['role'], m['content']) for m in memory[-8:])\n    decision = ai.ask(\n        "You are a strict web-search router. Decide if the user's next question needs web search.\\n"\n        "Answer exactly one lowercase word: yes or no.\\n\\n"\n        "Return yes when the user asks you to search, look up, find, verify, cite sources, use websites, "\n        "or asks about current/latest/recent/today/live information.\\n"\n        "Return no only when the answer can be handled from normal reasoning or prior conversation.\\n\\n"\n        "Examples:\\n"\n        "Question: search it up and find it\\nAnswer: yes\\n"\n        "Question: look this up online\\nAnswer: yes\\n"\n        "Question: what is the latest price of bitcoin\\nAnswer: yes\\n"\n        "Question: find sources for BrainGym360 activities\\nAnswer: yes\\n"\n        "Question: explain what a GPU is\\nAnswer: no\\n"\n        "Question: write a Python loop\\nAnswer: no\\n\\n"\n        "Recent conversation:\\n%s\\n\\nQuestion: %s\\nAnswer:" % (recent_context, question),\n        temperature=0,\n        max_tokens=3,\n    )\n    return decision.strip().lower().startswith("y")\n\n\nmemory = load_memory()\nprint("SecureAI saved memory chat. Type bye to stop. Type /clear to erase memory.")\n\nwhile True:\n    question = input("\\nYou: ").strip()\n    if question.lower() in {"bye", "exit", "quit"}:\n        print("SecureAI: bye")\n        break\n    if question == "/clear":\n        memory = []\n        save_memory(memory)\n        print("memory cleared")\n        continue\n    if not question:\n        continue\n\n    use_search = needs_web_search(question, memory)\n    print("web search = " + str(use_search).lower())\n    print("SecureAI: ", end="", flush=True)\n\n    messages = memory + [dict(role="user", content=question)]\n    answer_parts = []\n    for token in ai.stream(messages, web_search=use_search, search_count=5, temperature=0.2):\n        answer_parts.append(token)\n        print(token, end="", flush=True)\n\n    memory.append(dict(role="user", content=question))\n    memory.append(dict(role="assistant", content="".join(answer_parts)))\n    if len(memory) > MAX_MEMORY_MESSAGES:\n        print("compressing context...")\n        memory = memory[-MAX_MEMORY_MESSAGES:]\n    save_memory(memory)\n    print()</code></pre></section>
    <section class="card"><h2>Letter Streaming</h2><div class="actions"><button onclick="copyText('letterExample')">Copy Letter Example</button></div><pre><code id="letterExample">import SecureAI as AI\n\nAPI = "sk-secureai-your-key"\n\nfor letter in AI.StreamLettersAI(API)["Say SecureAI"]:\n    print(letter, end="", flush=True)</code></pre></section>
    <p class="warn">Default SDK endpoint: {DEFAULT_BASE_URL}</p>
  </main>
<script>
function copyText(id) {{ navigator.clipboard.writeText(document.getElementById(id).innerText || document.getElementById(id).textContent); }}
function applyKey(key) {{ for (const id of ['askExample','streamExample','vlmExample','annotateExample','autoSearchExample','persistentMemoryExample','letterExample']) {{ const el = document.getElementById(id); el.textContent = el.textContent.replaceAll('sk-secureai-your-key', key); }} }}
async function createKey() {{ const res = await fetch('/secureai/api-keys', {{ method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{name:document.getElementById('keyName').value || 'python-sdk'}}) }}); const data = await res.json(); document.getElementById('createdKey').textContent = JSON.stringify(data, null, 2); if (data.api_key) applyKey(data.api_key); await listKeys(); }}
async function listKeys() {{ const res = await fetch('/secureai/api-keys/list'); const data = await res.json(); document.getElementById('keyList').textContent = JSON.stringify(data, null, 2); }}
listKeys().catch(() => {{}});
</script>
</body>
</html>'''


@router.get('/api-keys', response_class=HTMLResponse)
async def api_key_page(user=Depends(get_verified_user)):
    return page_html()


@router.get('/api-keys/list')
async def list_keys(user=Depends(get_verified_user)):
    conn = db()
    rows = conn.execute('select id, name, prefix, created_at, revoked_at from api_keys where user_id = ? order by created_at desc', (user.id,)).fetchall()
    conn.close()
    return {'data': [dict(row) for row in rows]}


@router.post('/api-keys')
async def create_key(form_data: KeyForm, user=Depends(get_verified_user)):
    key = 'sk-secureai-' + secrets.token_urlsafe(32)
    key_id = 'key_' + secrets.token_hex(8)
    now = int(time.time())
    conn = db()
    conn.execute('insert into api_keys (id, user_id, name, key_hash, prefix, created_at, revoked_at) values (?, ?, ?, ?, ?, ?, null)', (key_id, user.id, form_data.name, hash_key(key), key[:22], now))
    conn.commit()
    conn.close()
    return {'id': key_id, 'name': form_data.name, 'api_key': key, 'created_at': now}


@router.post('/api-keys/revoke')
async def revoke_key(form_data: RevokeForm, user=Depends(get_verified_user)):
    conn = db()
    conn.execute('update api_keys set revoked_at = ? where id = ? and user_id = ?', (int(time.time()), form_data.key_id, user.id))
    conn.commit()
    conn.close()
    return {'status': 'revoked', 'id': form_data.key_id}


@router.get('/v1/models')
async def models(authorization: Optional[str] = Header(default=None)):
    verify_bearer(authorization)
    data = []
    async with aiohttp.ClientSession() as session:
        for base_url in (UPSTREAM_BASE_URL, VLM_UPSTREAM_BASE_URL):
            try:
                async with session.get(f'{base_url}/models', headers=upstream_headers()) as response:
                    if response.status < 400:
                        body = await response.json()
                        data.extend(body.get('data', []))
            except Exception:
                pass
    return JSONResponse(content={'object': 'list', 'data': data})


@router.post('/v1/annotate-image')
async def annotate_image(form_data: AnnotateForm, authorization: Optional[str] = Header(default=None)):
    verify_bearer(authorization)
    image_bytes = await image_bytes_from_url(form_data.image_url)
    prompt = (
        'Find every object requested by this instruction: "%s". '
        'Return ONLY valid JSON as a list of objects. '
        'Each object must be {"label": "...", "box": [x1, y1, x2, y2]}. '
        'Use absolute pixel coordinates if possible; otherwise use normalized 0-1000 coordinates. '
        'Do not include explanations.'
    ) % form_data.instruction
    payload = {
        'model': VLM_MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': form_data.image_url}},
            ],
        }],
        'temperature': 0,
        'max_tokens': 700,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async with session.post(f'{VLM_UPSTREAM_BASE_URL}/chat/completions', headers=upstream_headers(), json=payload) as response:
            body = await response.json(content_type=None)
    text = body.get('choices', [{}])[0].get('message', {}).get('content', '')
    boxes = parse_boxes(text)
    if not boxes:
        raise HTTPException(status_code=422, detail={'message': 'No boxes returned by VLM', 'model_output': text[:1200]})
    annotated = draw_box_circles(image_bytes, boxes, form_data.color)
    return Response(content=annotated, media_type='image/png', headers={'Content-Disposition': 'attachment; filename="secureai-annotated.png"'})


@router.post('/v1/chat/completions')
async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
    verify_bearer(authorization)
    payload, sources, compressed = await prepare_payload(await request.json())
    upstream_base_url = upstream_for_payload(payload)
    if upstream_base_url == VLM_UPSTREAM_BASE_URL:
        payload = vlm_payload(payload)
    if payload.get('stream'):
        async def stream_proxy():
            if compressed:
                yield context_status_event(str(payload.get('model') or 'qwen32b'))
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f'{upstream_base_url}/chat/completions', headers=upstream_headers(), json=payload) as response:
                        if response.status >= 500 and upstream_base_url != VLM_UPSTREAM_BASE_URL:
                            raise aiohttp.ClientError(f'upstream status {response.status}')
                        async for chunk in response.content.iter_any():
                            yield chunk
            except aiohttp.ClientError:
                if upstream_base_url == VLM_UPSTREAM_BASE_URL:
                    raise
                fallback = vlm_payload(payload)
                async with aiohttp.ClientSession() as session:
                    async with session.post(f'{VLM_UPSTREAM_BASE_URL}/chat/completions', headers=upstream_headers(), json=fallback) as response:
                        async for chunk in response.content.iter_any():
                            yield chunk
        return StreamingResponse(stream_proxy(), media_type='text/event-stream')
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f'{upstream_base_url}/chat/completions', headers=upstream_headers(), json=payload) as response:
                if response.status >= 500 and upstream_base_url != VLM_UPSTREAM_BASE_URL:
                    raise aiohttp.ClientError(f'upstream status {response.status}')
                try:
                    body = await response.json()
                    if sources and isinstance(body, dict):
                        body['secureai_search_sources'] = sources
                    if compressed and isinstance(body, dict):
                        body['secureai_context_compressed'] = True
                except Exception:
                    body = {'error': await response.text()}
                return JSONResponse(status_code=response.status, content=body)
        except aiohttp.ClientError:
            if upstream_base_url == VLM_UPSTREAM_BASE_URL:
                raise
            fallback = vlm_payload(payload)
            async with session.post(f'{VLM_UPSTREAM_BASE_URL}/chat/completions', headers=upstream_headers(), json=fallback) as response:
                try:
                    body = await response.json()
                    if sources and isinstance(body, dict):
                        body['secureai_search_sources'] = sources
                    if compressed and isinstance(body, dict):
                        body['secureai_context_compressed'] = True
                    if isinstance(body, dict):
                        body['secureai_model_fallback'] = VLM_MODEL
                except Exception:
                    body = {'error': await response.text()}
                return JSONResponse(status_code=response.status, content=body)
