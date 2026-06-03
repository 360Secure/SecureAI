from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel


APP_NAME = "SecureAI"
DB_PATH = Path(os.getenv("SECUREAI_DB", "secureai.db"))
ADMIN_TOKEN = os.getenv("SECUREAI_ADMIN_TOKEN", "change-this-admin-token")
UPSTREAM_BASE_URL = os.getenv("SECUREAI_UPSTREAM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
UPSTREAM_API_KEY = os.getenv("SECUREAI_UPSTREAM_API_KEY", "none")

app = FastAPI(title=APP_NAME, version="0.1.0")


class CreateKeyRequest(BaseModel):
    name: str


class RevokeKeyRequest(BaseModel):
    key_id: str


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        create table if not exists api_keys (
            id text primary key,
            name text not null,
            key_hash text not null unique,
            prefix text not null,
            created_at integer not null,
            revoked_at integer
        )
        """
    )
    conn.commit()
    return conn


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def require_admin(authorization: Optional[str] = Header(default=None)) -> None:
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid admin token")


def require_api_key(authorization: Optional[str] = Header(default=None)) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    key = authorization.removeprefix("Bearer ").strip()
    key_hash = hash_key(key)
    conn = db()
    row = conn.execute(
        "select * from api_keys where key_hash = ? and revoked_at is null",
        (key_hash,),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return row


def upstream_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HOME_PAGE


@app.get("/settings", response_class=HTMLResponse)
def settings() -> str:
    return SETTINGS_PAGE


@app.get("/settings/account", response_class=HTMLResponse)
def account_settings() -> str:
    return ACCOUNT_PAGE


@app.get("/settings/account/api-keys", response_class=HTMLResponse)
def api_key_page() -> str:
    return API_KEYS_PAGE


@app.get("/api-keys", response_class=HTMLResponse)
def api_keys_shortcut() -> str:
    return API_KEYS_PAGE


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "upstream": UPSTREAM_BASE_URL}


@app.get("/admin/api-keys")
def list_api_keys(_: None = Depends(require_admin)) -> Dict[str, Any]:
    conn = db()
    rows = conn.execute(
        "select id, name, prefix, created_at, revoked_at from api_keys order by created_at desc"
    ).fetchall()
    conn.close()
    return {"data": [dict(row) for row in rows]}


@app.post("/admin/api-keys")
def create_api_key(payload: CreateKeyRequest, _: None = Depends(require_admin)) -> Dict[str, Any]:
    key = "sk-secureai-" + secrets.token_urlsafe(32)
    key_id = "key_" + secrets.token_hex(8)
    now = int(time.time())
    conn = db()
    conn.execute(
        "insert into api_keys (id, name, key_hash, prefix, created_at, revoked_at) values (?, ?, ?, ?, ?, null)",
        (key_id, payload.name, hash_key(key), key[:18], now),
    )
    conn.commit()
    conn.close()
    return {"id": key_id, "name": payload.name, "api_key": key, "created_at": now}


@app.post("/admin/api-keys/revoke")
def revoke_api_key(payload: RevokeKeyRequest, _: None = Depends(require_admin)) -> Dict[str, Any]:
    conn = db()
    conn.execute("update api_keys set revoked_at = ? where id = ?", (int(time.time()), payload.key_id))
    conn.commit()
    conn.close()
    return {"status": "revoked", "id": payload.key_id}


@app.get("/v1/models")
async def models(_: sqlite3.Row = Depends(require_api_key)) -> JSONResponse:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(f"{UPSTREAM_BASE_URL}/models", headers=upstream_headers())
    return JSONResponse(status_code=res.status_code, content=res.json())


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, _: sqlite3.Row = Depends(require_api_key)) -> Any:
    body = await request.json()
    stream = bool(body.get("stream"))
    if stream:
        return StreamingResponse(proxy_stream(body), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=180.0) as client:
        res = await client.post(
            f"{UPSTREAM_BASE_URL}/chat/completions",
            headers=upstream_headers(),
            json=body,
        )
    try:
        return JSONResponse(status_code=res.status_code, content=res.json())
    except Exception:
        return JSONResponse(status_code=res.status_code, content={"error": res.text})


async def proxy_stream(body: Dict[str, Any]):
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream(
            "POST",
            f"{UPSTREAM_BASE_URL}/chat/completions",
            headers=upstream_headers(),
            json=body,
        ) as res:
            async for chunk in res.aiter_bytes():
                yield chunk


EXAMPLES = [
    ("Basic ask", "print(ai.ask('What is DGX Spark?'))"),
    ("Your custom import", "import SecureAI as AI\nAPI = 'sk-secureai-...'\nprint(AI.AskAI(API)['What is DGX Spark?'])"),
    ("Your custom streaming", "import SecureAI as AI\nAPI = 'sk-secureai-...'\nfor t in AI.StreamAI(API)['Tell me about DGX Spark']:\n    print(t, end='', flush=True)"),
    ("Every letter streaming", "import SecureAI as AI\nAPI = 'sk-secureai-...'\nfor ch in AI.StreamLettersAI(API)['Say SecureAI']:\n    print(ch, end='', flush=True)"),
    ("System prompt", "print(ai.ask('Explain briefly', system='You are concise.'))"),
    ("Streaming tokens", "for token in ai.stream('Tell me a story'): print(token, end='', flush=True)"),
    ("Streaming letters", "for ch in ai.stream_letters('Say hello slowly'): print(ch, end='', flush=True)"),
    ("List models", "print(ai.models())"),
    ("Low temperature", "print(ai.ask('Write a factual summary', temperature=0.0))"),
    ("Creative mode", "print(ai.ask('Name an AI product', temperature=0.9))"),
    ("Limit output", "print(ai.ask('Give 3 bullets', max_tokens=120))"),
    ("Custom model", "print(ai.ask('Hello', model='qwen72b-vl'))"),
    ("Multi-message chat", "print(ai.chat([{'role':'user','content':'Hi'}]))"),
    ("Developer style system", "print(ai.ask('Plan my setup', system='Act like a senior engineer.'))"),
    ("JSON request", "print(ai.ask('Return JSON with name and summary'))"),
    ("Classification", "print(ai.ask('Classify: urgent or normal? Server down.'))"),
    ("Extraction", "print(ai.ask('Extract names from: Rajesh emailed SecureAI.'))"),
    ("Summarization", "print(ai.ask('Summarize this paragraph: ...'))"),
    ("Rewrite", "print(ai.ask('Rewrite professionally: hey fix this'))"),
    ("Translation", "print(ai.ask('Translate to Hindi: The server is ready.'))"),
    ("Coding help", "print(ai.ask('Write a Python retry function.'))"),
    ("Debug logs", "print(ai.ask('Explain this log: Connection refused on 8001'))"),
    ("Shell command help", "print(ai.ask('Give a safe command to check Docker containers.'))"),
    ("RAG-style context", "print(ai.ask('Use this context: ... Question: ...'))"),
    ("Web-search prompt", "print(ai.ask('Search context says: ... Answer with sources.'))"),
    ("SQL helper", "print(ai.ask('Write SQL to list active API keys.'))"),
    ("Regex helper", "print(ai.ask('Regex for sk-secureai keys.'))"),
    ("Security review", "print(ai.ask('Review this API design for security risks.'))"),
    ("Prompt template", "print(ai.ask('Create a prompt template for support tickets.'))"),
    ("Agent instruction", "print(ai.ask('Make a step-by-step plan to deploy.'))"),
    ("Short answer", "print(ai.ask('What is vLLM? Answer in one sentence.'))"),
    ("Long answer", "print(ai.ask('Explain local AI gateways in detail.'))"),
    ("Bullet points", "print(ai.ask('Give 5 bullet points about API keys.'))"),
    ("Table output", "print(ai.ask('Make a markdown table comparing HTTP and HTTPS.'))"),
    ("Error handling", "try: print(ai.ask('Hi'))\nexcept Exception as e: print(e)"),
    ("Environment config", "ai = SecureAI()  # uses SECUREAI_API_KEY and SECUREAI_BASE_URL"),
    ("Direct base URL", "ai = SecureAI(api_key='sk...', base_url='http://spark:8787/v1')"),
    ("OpenAI-compatible curl", "curl http://spark:8787/v1/chat/completions -H 'Authorization: Bearer sk...'"),
    ("Create key with curl", "curl -X POST /admin/api-keys -H 'Authorization: Bearer ADMIN' -d '{\"name\":\"test\"}'"),
    ("Revoke key", "curl -X POST /admin/api-keys/revoke -H 'Authorization: Bearer ADMIN' -d '{\"key_id\":\"key_...\"}'"),
    ("Vision URL", "print(ai.vision('Describe image', 'https://example.com/image.png'))"),
    ("Local image data URL", "print(ai.vision('Read text', 'data:image/png;base64,...'))"),
    ("OCR-style prompt", "print(ai.vision('Extract all visible text', image_url))"),
    ("API docs generation", "print(ai.ask('Write docs for POST /v1/chat/completions.'))"),
    ("Unit test generation", "print(ai.ask('Write pytest tests for this function: ...'))"),
    ("Config explanation", "print(ai.ask('Explain max_model_len=8192.'))"),
    ("Token-safe prompt", "print(ai.ask('Summarize in under 500 tokens: ...'))"),
    ("Streaming UI", "for token in ai.stream(prompt): websocket.send_text(token)"),
    ("Batch prompts", "for p in prompts: print(ai.ask(p))"),
    ("CLI assistant", "while True: print(ai.ask(input('> ')))"),
]


def examples_html() -> str:
    rows = []
    for i, (title, code) in enumerate(EXAMPLES, 1):
        rows.append(
            f"""
            <article class="example">
              <div class="num">{i:02d}</div>
              <div>
                <h3>{title}</h3>
                <pre><code>{code}</code></pre>
              </div>
            </article>
            """
        )
    return "\n".join(rows)


BASE_CSS = """
    :root {
      color-scheme: dark;
      --bg: #101312;
      --panel: #191d1b;
      --text: #eef3ef;
      --muted: #9aa59e;
      --line: #2d3430;
      --accent: #6ee7b7;
      --warn: #fbbf24;
      --danger: #fb7185;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 28px 24px 18px;
      border-bottom: 1px solid var(--line);
      background: #141816;
    }
    nav {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    nav a, .link-button {
      color: var(--text);
      text-decoration: none;
      border: 1px solid var(--line);
      background: #101312;
      border-radius: 6px;
      padding: 9px 12px;
      font-weight: 700;
    }
    nav a:hover, .link-button:hover { border-color: var(--accent); }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: clamp(28px, 4vw, 48px); letter-spacing: 0; }
    h2 { margin: 28px 0 14px; font-size: 24px; }
    h3 { margin: 0 0 8px; font-size: 16px; }
    p { color: var(--muted); line-height: 1.55; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .panel, .example, .setting-row {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .setting-row {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 12px;
    }
    label { display: block; font-weight: 700; margin-bottom: 8px; }
    input {
      width: 100%;
      min-height: 42px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #0c0f0e;
      color: var(--text);
      padding: 10px 12px;
      font-size: 15px;
    }
    button {
      min-height: 42px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #062116;
      font-weight: 800;
      padding: 0 14px;
      cursor: pointer;
    }
    button.secondary {
      background: #26302b;
      color: var(--text);
      border: 1px solid var(--line);
    }
    pre {
      margin: 0;
      overflow: auto;
      background: #0b0e0d;
      border: 1px solid #26302b;
      border-radius: 6px;
      padding: 12px;
      color: #d8f8e7;
      line-height: 1.45;
    }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 13px; }
    .example { display: grid; grid-template-columns: 44px 1fr; gap: 12px; margin-bottom: 12px; }
    .num { color: var(--accent); font-weight: 900; }
    .warning { color: var(--warn); }
    .danger { color: var(--danger); }
    .stack { display: grid; gap: 12px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    #createdKey, #keyList { word-break: break-all; }
    @media (max-width: 800px) {
      .grid { grid-template-columns: 1fr; }
      .setting-row { align-items: flex-start; flex-direction: column; }
    }
"""


NAV = """
    <nav>
      <a href="/">Home</a>
      <a href="/settings">Settings</a>
      <a href="/settings/account">Account</a>
      <a href="/settings/account/api-keys">API Keys</a>
    </nav>
"""


HOME_PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SecureAI</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header>
    {NAV}
    <h1>SecureAI</h1>
    <p>Your local DGX API gateway and Python SDK.</p>
  </header>
  <main>
    <section class="grid">
      <div class="panel">
        <h2>Settings</h2>
        <p>Go to settings, open account, then use the API Keys link to generate keys and copy examples.</p>
        <p><a class="link-button" href="/settings/account">Open Account Settings</a></p>
      </div>
      <div class="panel">
        <h2>API Keys</h2>
        <p>Create keys, view existing keys, copy the download command, and paste a ready Python example.</p>
        <p><a class="link-button" href="/settings/account/api-keys">Open API Keys</a></p>
      </div>
    </section>
  </main>
</body>
</html>
"""


SETTINGS_PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Settings - SecureAI</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header>
    {NAV}
    <h1>Settings</h1>
    <p>Manage SecureAI preferences and account tools.</p>
  </header>
  <main>
    <section class="stack">
      <article class="setting-row">
        <div>
          <h2>Account</h2>
          <p>Profile, API keys, and developer access.</p>
        </div>
        <a class="link-button" href="/settings/account">Open Account</a>
      </article>
    </section>
  </main>
</body>
</html>
"""


ACCOUNT_PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Account - SecureAI</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header>
    {NAV}
    <h1>Account</h1>
    <p>Account settings for SecureAI.</p>
  </header>
  <main>
    <section class="stack">
      <article class="setting-row">
        <div>
          <h2>API Keys</h2>
          <p>Create API keys, view key prefixes, install SecureAI, and copy runnable examples.</p>
        </div>
        <a class="link-button" href="/settings/account/api-keys">Open API Keys</a>
      </article>
    </section>
  </main>
</body>
</html>
"""


API_KEYS_PAGE = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>API Keys - SecureAI</title>
  <style>{BASE_CSS}</style>
</head>
<body>
  <header>
    {NAV}
    <h1>API Keys</h1>
    <p>Create a key, copy the install command, paste an example into a new file, and run it.</p>
  </header>
  <main>
    <section class="panel">
      <h2>1. Download SecureAI</h2>
      <p>Copy this command into Terminal.</p>
      <div class="actions">
        <button onclick="copyText('installCommand')">Copy Download Command</button>
      </div>
      <pre><code id="installCommand">python3 -m pip install "secureai-dgx[server] @ git+https://github.com/YOUR_USERNAME/SecureAI.git"</code></pre>
      <p>Local install while this repo is on your Mac:</p>
      <pre><code>cd /Users/360secure/Documents/Doorlock
python3 -m pip install -e ".[server]"</code></pre>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>2. Run The API Server</h2>
        <p>Start SecureAI and connect it to your DGX model endpoint.</p>
        <div class="actions">
          <button onclick="copyText('serverCommand')">Copy Server Command</button>
        </div>
        <pre><code id="serverCommand">export SECUREAI_ADMIN_TOKEN="change-this-admin-password"
export SECUREAI_UPSTREAM_BASE_URL="http://127.0.0.1:8001/v1"
python3 -m secureai_server.cli serve --host 0.0.0.0 --port 8787</code></pre>
      </div>
      <div class="panel">
        <h2>3. Make A New Python File</h2>
        <p>Create <code>app.py</code>, paste an example, replace the placeholder key, then run it.</p>
        <div class="actions">
          <button onclick="copyText('runCommand')">Copy Run Command</button>
        </div>
        <pre><code id="runCommand">python3 app.py</code></pre>
      </div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>4. Create New API Key</h2>
        <p>Use your admin token to generate a key. The full key is shown once.</p>
        <label for="adminToken">Admin Token</label>
        <input id="adminToken" type="password" placeholder="SECUREAI_ADMIN_TOKEN">
        <label for="keyName">Key Name</label>
        <input id="keyName" value="python-sdk">
        <p class="actions">
          <button onclick="createKey()">Create New API Key</button>
          <button class="secondary" onclick="listKeys()">View API Keys</button>
        </p>
        <pre id="createdKey">No key created yet.</pre>
      </div>
      <div class="panel">
        <h2>5. View API Keys</h2>
        <p>Only prefixes are shown. Secret API keys are never shown again after creation.</p>
        <pre id="keyList">Click View API Keys.</pre>
      </div>
    </section>

    <section>
      <h2>6. Copy A Runnable Example</h2>
      <p>After generating a key, this example updates automatically with your key.</p>
      <div class="actions">
        <button onclick="copyText('mainExample')">Copy Python Example</button>
        <button onclick="copyText('streamExample')">Copy Streaming Example</button>
      </div>
      <div class="grid">
        <div class="panel">
          <h3>Normal Answer</h3>
          <pre><code id="mainExample">import SecureAI as AI

API = "sk-secureai-your-key"
BASE = "http://spark:8787/v1"

print(AI.AskAI(API, base_url=BASE)["What is DGX Spark?"])
</code></pre>
        </div>
        <div class="panel">
          <h3>Streaming Answer</h3>
          <pre><code id="streamExample">import SecureAI as AI

API = "sk-secureai-your-key"
BASE = "http://spark:8787/v1"

for token in AI.StreamAI(API, base_url=BASE)["Tell me about DGX Spark"]:
    print(token, end="", flush=True)</code></pre>
        </div>
      </div>
    </section>

    <section>
      <h2>Letter-By-Letter Streaming</h2>
      <div class="actions">
        <button onclick="copyText('lettersExample')">Copy Letter Streaming Example</button>
      </div>
      <pre><code id="lettersExample">import SecureAI as AI

API = "sk-secureai-your-key"
BASE = "http://spark:8787/v1"

for letter in AI.StreamLettersAI(API, base_url=BASE)["Say SecureAI"]:
    print(letter, end="", flush=True)</code></pre>
    </section>

    <section>
      <h2>OpenAI-Compatible API</h2>
      <pre><code>from SecureAI import SecureAI

ai = SecureAI(api_key="sk-secureai-...", base_url="http://spark:8787/v1")
print(ai.ask("What is DGX Spark?"))</code></pre>
      <p class="warning">Keep keys private. Use HTTPS or Tailscale for remote access.</p>
    </section>

    <section>
      <h2>50 Examples</h2>
      {examples_html()}
    </section>
  </main>
  <script>
    let latestKey = "";

    function withLatestKey(template) {{
      if (!latestKey) {{
        return template;
      }}
      return template.replaceAll("sk-secureai-your-key", latestKey);
    }}

    async function copyText(id) {{
      const element = document.getElementById(id);
      const text = element.innerText || element.textContent;
      await navigator.clipboard.writeText(text);
    }}

    function updateExamples(apiKey) {{
      latestKey = apiKey;
      for (const id of ["mainExample", "streamExample", "lettersExample"]) {{
        const element = document.getElementById(id);
        element.textContent = withLatestKey(element.textContent);
      }}
    }}

    async function createKey() {{
      const adminToken = document.getElementById("adminToken").value;
      const name = document.getElementById("keyName").value || "api-key";
      const res = await fetch("/admin/api-keys", {{
        method: "POST",
        headers: {{
          "Content-Type": "application/json",
          "Authorization": "Bearer " + adminToken
        }},
        body: JSON.stringify({{ name }})
      }});
      const data = await res.json();
      document.getElementById("createdKey").textContent = JSON.stringify(data, null, 2);
      if (data.api_key) {{
        updateExamples(data.api_key);
      }}
      await listKeys();
    }}

    async function listKeys() {{
      const adminToken = document.getElementById("adminToken").value;
      const res = await fetch("/admin/api-keys", {{
        method: "GET",
        headers: {{
          "Authorization": "Bearer " + adminToken
        }}
      }});
      const data = await res.json();
      document.getElementById("keyList").textContent = JSON.stringify(data, null, 2);
    }}
  </script>
</body>
</html>
"""
