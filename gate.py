"""llm-gate: a fair queue in front of llama-server that tells waiting clients where they stand.

Generation requests run MAX_ACTIVE at a time, in arrival order, except that PRIORITY clients go
first and, after each of their requests, everyone else is held back for HOLD seconds (so their model
stays loaded between back-to-back requests). While a streaming chat request waits, its position is
streamed as reasoning_content (Open WebUI and Cline show it in the thinking section, so it never
becomes part of the answer). Model switches, image routing and the long-chat archive (archive.py)
report there too.

Clients authenticate with their own keys (KEYS_FILE); upstream gets UPSTREAM_KEY_FILE's key, which
llama-server is the only other holder of, so client keys don't work on llama-server directly.
See README.md for the settings.
"""
import asyncio
import json
import logging
import os
import re
import time

from aiohttp import ClientSession, ClientTimeout, web

from archive import Archive, normalize

UPSTREAM = os.environ.get("UPSTREAM", "http://host.docker.internal:8081")
KEYS_FILE = os.environ.get("KEYS_FILE", "/keys/server-keys.txt")
MAX_ACTIVE = int(os.environ.get("MAX_ACTIVE", "1"))
MAX_WAITING = int(os.environ.get("MAX_WAITING", "8"))
UPSTREAM_KEY_FILE = os.environ.get("UPSTREAM_KEY_FILE", "/keys/upstream-key.txt")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")  # for requests without a model; empty = pass them on as they are
PRIORITY = set(filter(None, os.environ.get("PRIORITY", "").split(",")))  # key names from KEYS_FILE
HOLD = float(os.environ.get("HOLD", "30"))  # seconds others wait after a priority request ends
AUTO_MODEL = os.environ.get("AUTO_MODEL", "")  # extra model name meaning DEFAULT_MODEL, listed first; empty = off
VISION_MODELS = set(filter(None, os.environ.get("VISION_MODELS", "").split(",")))  # models that can read images
VISION_FALLBACK = os.environ.get("VISION_FALLBACK", "")  # where image requests for other models go; empty = off
IMAGE_NOTE = ("[The user attached an image here. You cannot see it, but the image edit tool receives it automatically: "
              "call that tool if they want the image changed. If they ask what is in it, say you can't see images "
              "and suggest picking a model that can.]")
# models only some keys may use, e.g. "big-model:alice|bob;other-model:carol"
RESTRICTED = {m: set(names.split("|")) for m, _, names in
              (x.partition(":") for x in os.environ.get("RESTRICTED", "").split(";") if x.strip())}
# owner's rules for agents on client machines, so nothing is installed there: "key name:file" pairs, files in
# CLIENT_RULES_DIR, added to Cline's system prompt for that key (e.g. laptop:cline-windows.md)
CLIENT_RULES = {k.strip(): f.strip() for k, _, f in (x.partition(":") for x in os.environ.get("CLIENT_RULES", "").split(",")) if f.strip()}
CLIENT_RULES_DIR = os.environ.get("CLIENT_RULES_DIR", "/rules")
# long chats: archive old messages instead of overflowing (see archive.py). Limits are prompt tokens per model.
ARCHIVE_DB = os.environ.get("ARCHIVE_DB", "")  # empty = off
ARCHIVE_KEYS = set(filter(None, os.environ.get("ARCHIVE_KEYS", "").split(",")))
ARCHIVE_LIMITS = {m: int(n) for m, _, n in (x.partition(":") for x in os.environ.get("ARCHIVE_LIMITS", "").split(",") if x.strip())}
ARCHIVE_DEFAULT_LIMIT = int(os.environ.get("ARCHIVE_DEFAULT_LIMIT", "20000"))
EMBED_URL = os.environ.get("EMBED_URL", "")  # CPU embedding server for queries (~0.2 s); empty = keywords only
EMBED_QUERY = "Instruct: Given a question, retrieve passages from earlier in the same conversation that help answer it\nQuery: "
# models that get archive_read/archive_search in archived chats (the gate runs those calls itself; clients never see them)
ARCHIVE_TOOL_MODELS = set(filter(None, os.environ.get("ARCHIVE_TOOL_MODELS", "").split(",")))
MAX_LOOKUPS = 4  # archive tool rounds per request; later calls are told to answer now (the tools stay in the request:
# removing them changes the top of the prompt for models that list tools there, so the whole chat would be read again)
LOOKUP_LIMIT_NOTE = "No more archive lookups for this answer: answer now from what you already have."
GAVE_UP = "I couldn't finish looking this up in the archive. Please ask again, more specifically."
INDEX_BATCH, INDEX_TIMEOUT = 30, 90  # sections per summary request; seconds before the index goes without summaries
ARCHIVE_TOOLS = [
    {"type": "function", "function": {
        "name": "archive_read",
        "description": "Read archived messages of this conversation word for word, by message number as shown in the archive "
                       "index in the system message. Up to about 20 messages per call.",
        "parameters": {"type": "object", "properties": {"start": {"type": "integer", "description": "first message number"},
                                                        "end": {"type": "integer", "description": "last message number"}},
                       "required": ["start", "end"]}}},
    {"type": "function", "function": {
        "name": "archive_search",
        "description": "Search the archived part of this conversation by keywords and meaning. Returns the best-matching "
                       "passages with their message numbers.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
ARCHIVE_TOOL_NAMES = {t["function"]["name"] for t in ARCHIVE_TOOLS}
INDEX_PROMPT = ("Below are sections from the earlier part of a chat between a user and an assistant (each message is cut "
                "to its start). Write an index line for each section, at most 25 words, naming every distinct topic, "
                "decision, fact, name and number in it, including personal details, plans and purchases, so a reader can "
                "tell which section holds a given detail. Describe each section on its own, never by reference to "
                "another. Reply with exactly one line per section in the form 'S<number>: <index line>' and nothing "
                "else.\n\n")
RETURN_AFTER = float(os.environ.get("RETURN_AFTER", "0"))  # idle seconds before reloading DEFAULT_MODEL (0 = never)
# addresses (prefixes) of reverse proxies whose CF-Connecting-IP / X-Real-IP header names the real client, e.g. "172."
TRUSTED_PROXIES = tuple(filter(None, os.environ.get("TRUSTED_PROXIES", "").split(",")))
HEARTBEAT = 10  # seconds; keeps Cloudflare (100 s idle limit) and nginx from dropping waiting clients
QUEUED_PATHS = {"/v1/chat/completions", "/chat/completions", "/v1/completions", "/completions", "/completion",
                "/v1/messages", "/v1/responses", "/infill"}
# everything else clients may reach; the rest of llama-server (loading/unloading models, slots, props changes,
# metrics) needs the upstream key, which only the gate holds, so clients must not get it there via the gate
OPEN_GETS = {"/v1/models", "/models", "/health", "/v1/health", "/props"}
OPEN_POSTS = {"/tokenize", "/detokenize", "/apply-template"}
BAD_KEY_LIMIT, BAD_KEY_WINDOW = 10, 600  # wrong keys per client address before it is locked out for the window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("llm-gate")


def has_images(payload):
    return any(isinstance(m.get("content"), list) and any(p.get("type") in ("image_url", "input_image") for p in m["content"])
               for m in payload.get("messages") or [] if isinstance(m, dict))


def has_edit_tool(payload):
    names = [(t.get("function") or {}).get("name", "") for t in payload.get("tools") or [] if isinstance(t, dict)]
    return any(re.search(r"edit.*image|image.*edit", n, re.I) for n in names)


def strip_images(payload):
    for m in payload.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            parts = [{"type": "text", "text": IMAGE_NOTE} if p.get("type") in ("image_url", "input_image") else p for p in m["content"]]
            m["content"] = parts


def route(payload, who):
    """Pick the real model. AUTO_MODEL becomes DEFAULT_MODEL. Image input for a text-only model: keep the model and
    hide the image if it can call an image edit tool (the tool gets the picture from the chat app, not from the
    model); otherwise send it to a model that can see. Returns a note for the user, or None."""
    if AUTO_MODEL and payload.get("model") == AUTO_MODEL:
        payload["model"] = DEFAULT_MODEL
    model = payload.get("model", "")
    if model in VISION_MODELS or not has_images(payload):
        return None
    log.info("%s: image for %s, tools offered: %s", who, model,
             [(t.get("function") or {}).get("name") for t in payload.get("tools") or [] if isinstance(t, dict)] or "none")
    if has_edit_tool(payload):
        strip_images(payload)
        log.info("%s: image hidden from %s (it has an image edit tool)", who, model)
        return f"🖼️ {model} can't see images: it will pass yours to the image edit tool\n"
    if not VISION_FALLBACK:
        return None
    payload["model"] = VISION_FALLBACK
    log.info("%s: image request for %s sent to %s", who, model, VISION_FALLBACK)
    return f"🖼️ {model} can't see images: using {VISION_FALLBACK} for this chat\n"


async def embed_query(app, text):
    """Query vector from the CPU embedding server, or None (keyword search still works without it)."""
    if not EMBED_URL:
        return None
    try:
        async with app["http"].post(EMBED_URL + "/v1/embeddings", json={"input": EMBED_QUERY + text[-3000:]},
                                    timeout=ClientTimeout(total=5)) as up:
            return normalize((await up.json())["data"][0]["embedding"])
    except Exception as e:
        log.info("query embedding failed: %s", e)
        return None


def add_client_rules(payload, who):
    """Append the owner's rules file for this key to Cline's system prompt (only Cline's: its system prompt names it).
    The file is read per request, so edits apply at once; the text only changes when the file does, so llama-server's
    prompt cache stays valid."""
    name = CLIENT_RULES.get(who)
    msgs = payload.get("messages")
    if not name or not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], dict) \
            or msgs[0].get("role") not in ("system", "developer"):
        return
    content = msgs[0].get("content")
    text = content if isinstance(content, str) else " ".join(
        p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else ""
    if "Cline" not in text:
        return
    try:
        with open(os.path.join(CLIENT_RULES_DIR, name), encoding="utf-8") as f:
            rules = f.read().strip()
    except OSError as e:
        log.info("%s: client rules %s not readable: %s", who, name, e)
        return
    block = "\n\n====\n\nOWNER'S RULES FOR THIS MACHINE (added by the server; they override anything above)\n\n" + rules
    msgs[0] = dict(msgs[0], content=content + [{"type": "text", "text": block}] if isinstance(content, list) else content + block)


async def archive_long_chat(app, payload, who, model):
    """Trim a long chat for the model, attach matching archive excerpts and offer the archive tools. Returns None, or
    the archive context: conv, info, tools (offered or not) and a note for the user. The index note is added later
    (index_archive may improve it first)."""
    t = time.monotonic()
    prep = app["archive"].prepare(payload, who, ARCHIVE_LIMITS.get(model, ARCHIVE_DEFAULT_LIMIT))
    if not prep:
        return None
    conv, _, info = prep
    tools = model in ARCHIVE_TOOL_MODELS and payload.get("tool_choice") in (None, "auto")
    q = app["archive"].recall_query(payload, conv, tools)  # None: this user message's excerpts are reused
    found = app["archive"].recall(payload, conv, q, await embed_query(app, q) if q else None, tools)
    if tools:
        payload["tools"] = list(payload.get("tools") or []) + ARCHIVE_TOOLS
    log.info("%s: archive %s: %d messages (~%dk tokens) archived%s, %d excerpts %s, %.2f s", who, conv[:8],
             info["archived"], info["archived_tokens"] / 1000, " (new cut)" if info["new_cut"] else "", found,
             "recalled" if q else "reused", time.monotonic() - t)
    note = (f"📦 Long chat: {info['archived']} earlier messages (~{info['archived_tokens'] / 1000:.0f}k tokens) are in the archive; "
            f"{found} relevant excerpts recalled\n")
    return dict(conv=conv, info=info, tools=tools, note=note, who=who)


async def index_archive(app, ctx, model, resp):
    """Add one-line summaries to the index, using the model about to answer. Done in the turn that made
    the cut: that turn re-reads the whole prompt anyway, and a separate job would push the chat out of llama-server's
    single cache slot. On failure or timeout the index keeps just the quoted openings."""
    pending = app["archive"].unsummarized(ctx["conv"])
    if not pending:
        return
    if resp:
        await resp.write(chunk(model, f"📇 Indexing the archived part ({len(pending)} sections)…\n").encode())
    t = time.monotonic()
    headers = {"Authorization": "Bearer " + upstream_key()}
    try:
        for i in range(0, len(pending), INDEX_BATCH):
            part = pending[i:i + INDEX_BATCH]
            prompt = INDEX_PROMPT + "\n\n".join(f"=== S{k + 1} ===\n{text}" for k, (_, _, text) in enumerate(part))
            body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 3000,
                    "chat_template_kwargs": {"reasoning_effort": "low"}}
            async with app["http"].post(UPSTREAM + "/v1/chat/completions", json=body, headers=headers,
                                        timeout=ClientTimeout(total=max(5, INDEX_TIMEOUT - (time.monotonic() - t)))) as up:
                text = (await up.json())["choices"][0]["message"].get("content") or ""
            summaries = {}
            for k, line in re.findall(r"(?m)^\W*S(\d+)\W*[:.)-]\s*(.+)$", text):
                if 0 < int(k) <= len(part):
                    summaries[part[int(k) - 1][0]] = line.strip()
            app["archive"].set_summaries(ctx["conv"], summaries)
        log.info("%s: archive %s indexed: %d sections in %.0f s", ctx["who"], ctx["conv"][:8], len(pending), time.monotonic() - t)
    except Exception as e:
        log.info("%s: archive %s indexing failed after %.0f s (%r); index without summaries", ctx["who"], ctx["conv"][:8],
                 time.monotonic() - t, e)


async def archive_tool(app, ctx, name, arguments):
    """Run one archive tool call. Returns (result for the model, line for the user)."""
    try:
        args = json.loads(arguments or "{}")
        if name == "archive_read":
            start, end = int(args["start"]), int(args.get("end", args["start"]))
            return app["archive"].read(ctx["conv"], start, end), f"🔎 Reading archived messages {start}–{end}\n"
        query = str(args["query"])
        return (app["archive"].search_text(ctx["conv"], query, await embed_query(app, query)),
                f"🔎 Searching the archive for “{query[:60]}”\n")
    except (ValueError, KeyError, TypeError) as e:
        return f"Invalid arguments for {name}: {e}", f"🔎 {name}: invalid arguments\n"


async def run_with_archive(app, url, headers, payload, ctx, resp, model):
    """Send the request; while the model only calls archive tools, run them here and send it again with the results.
    Streaming: text and reasoning reach the client as they come; archive tool calls (and the end of those rounds)
    don't. Tool calls meant for the client end the request as usual. Returns (response, upstream status)."""
    for rnd in range(MAX_LOOKUPS + 2):
        async with app["http"].post(url, data=json.dumps(payload).encode(), headers=headers) as up:
            if resp is None:  # not streaming
                data = await up.read()
                if up.status != 200:
                    return web.Response(body=data, status=up.status, content_type=up.content_type), up.status
                reply = json.loads(data)
                msg = reply["choices"][0]["message"]
                calls = msg.get("tool_calls") or []
                ours = [c for c in calls if c["function"]["name"] in ARCHIVE_TOOL_NAMES]
                if not ours or len(ours) < len(calls) or rnd > MAX_LOOKUPS:
                    if ours:
                        msg["tool_calls"] = [c for c in calls if c not in ours] or None
                        if not msg["tool_calls"]:
                            msg["content"], reply["choices"][0]["finish_reason"] = GAVE_UP, "stop"
                    return web.json_response(reply), up.status
                content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
            else:
                if up.status != 200:
                    await resp.write(f"data: {await up.text()}\n\ndata: [DONE]\n\n".encode())
                    return resp, up.status
                calls, content, reasoning, held = {}, "", "", []
                async for line in up.content:
                    if not line.startswith(b"data: "):
                        if line.strip():
                            await resp.write(line + b"\n")
                        continue
                    if line.strip() == b"data: [DONE]" or held:
                        held.append(line.strip() + b"\n\n")
                        continue
                    event = json.loads(line[6:])
                    choice = (event.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or ""
                    if delta.get("tool_calls"):
                        theirs = []
                        for tc in delta["tool_calls"]:
                            c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                            f = tc.get("function") or {}
                            c["id"] = tc.get("id") or c["id"]
                            c["name"] += f.get("name") or ""
                            c["arguments"] += f.get("arguments") or ""
                            if c["name"] not in ARCHIVE_TOOL_NAMES:
                                theirs.append(tc)
                        if len(theirs) < len(delta["tool_calls"]):
                            delta["tool_calls"] = theirs
                            if not theirs:
                                delta.pop("tool_calls")
                            if not delta and not choice.get("finish_reason"):
                                continue
                            line = b"data: " + json.dumps(event).encode()
                    if choice.get("finish_reason") == "tool_calls" and any(c["name"] in ARCHIVE_TOOL_NAMES for c in calls.values()):
                        held.append(line.strip() + b"\n\n")  # this round may be ours: decide at the end
                        continue
                    await resp.write(line.strip() + b"\n\n")
                ours = [c for c in calls.values() if c["name"] in ARCHIVE_TOOL_NAMES]
                if not ours or len(ours) < len(calls) or not held:
                    for line in held:
                        await resp.write(line)
                    return resp, up.status
                if rnd > MAX_LOOKUPS:
                    await resp.write(("data: " + json.dumps({
                        "id": "llm-gate", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                        "choices": [{"index": 0, "delta": {"content": GAVE_UP}, "finish_reason": "stop"}]}) + "\n\ndata: [DONE]\n\n").encode())
                    log.info("%s: archive %s: gave up after %d lookup rounds", ctx["who"], ctx["conv"][:8], rnd)
                    return resp, up.status
                calls = [{"id": c["id"] or f"call_{rnd}_{n}", "type": "function",
                          "function": {"name": c["name"], "arguments": c["arguments"]}} for n, c in enumerate(ours)]
        payload["messages"].append({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls})
        for c in calls:
            if rnd < MAX_LOOKUPS:
                result, line = await archive_tool(app, ctx, c["function"]["name"], c["function"]["arguments"])
            else:
                result, line = LOOKUP_LIMIT_NOTE, "🔎 Lookup limit reached\n"
            if resp is not None:
                await resp.write(chunk(model, "\n" + line).encode())  # own line: reasoning before it has no newline
            payload["messages"].append({"role": "tool", "tool_call_id": c["id"], "content": result})
            log.info("%s: archive %s: %s", ctx["who"], ctx["conv"][:8], line.strip())
    raise RuntimeError("unreachable")


async def model_status(app, name):
    """'loaded', 'loading', 'unloaded'... from llama-server's router, or None if it can't tell."""
    try:
        async with app["http"].get(UPSTREAM + "/models", headers={"Authorization": "Bearer " + upstream_key()}) as up:
            data = await up.json()
        return next((m["status"]["value"] for m in data.get("data", []) if m.get("id") == name), None)
    except Exception:
        return None


async def switch_model(app, resp, model):
    """Load the model before the request, telling the user (a load sends nothing for 30-60 s otherwise)."""
    state = await model_status(app, model)
    if state in ("loaded", None):
        return
    await resp.write(chunk(model, f"🔄 Switching model to {model} (usually 30–60 s)…\n").encode())
    t = time.monotonic()
    try:
        async with app["http"].post(UPSTREAM + "/models/load", json={"model": model},
                                    headers={"Authorization": "Bearer " + upstream_key()}) as up:
            await up.read()
    except Exception as e:
        log.info("loading %s failed: %s", model, e)
    while time.monotonic() - t < 240 and await model_status(app, model) != "loaded":
        await resp.write(b": loading\n\n")  # keeps Cloudflare/nginx from dropping the stream
        await asyncio.sleep(3)
    await resp.write(chunk(model, f"✅ {model} ready after {time.monotonic() - t:.0f} s\n").encode())
    log.info("switched to %s in %.0f s", model, time.monotonic() - t)


class Keys:
    """API key -> name, from llama-server's key file ('# name' line above each key). Reloads on change."""

    def __init__(self, path):
        self.path, self.mtime, self.names = path, None, {}

    def name(self, key):
        mtime = os.stat(self.path).st_mtime
        if mtime != self.mtime:
            names, label = {}, "unnamed"
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#"):
                        label = line[1:].strip()
                    elif line:
                        names[line] = label
            self.names, self.mtime = names, mtime
        return self.names.get(key)


def upstream_key():
    with open(UPSTREAM_KEY_FILE) as f:
        return next(line.strip() for line in f if line.strip() and not line.startswith("#"))


class Queue:
    def __init__(self, max_active, hold=0.0):
        self.max_active, self.active, self.waiting = max_active, 0, []
        self.hold, self.hold_until, self.timer = hold, 0.0, None
        self.durations = []  # recent request run times, for the wait estimate

    def held(self, ticket=None):
        """True while non-priority requests are kept back after a priority request."""
        return (ticket is None or not ticket.priority) and time.monotonic() < self.hold_until

    def _changed(self):
        for t in self.waiting:
            t.changed.set()

    def position(self, ticket):
        return self.waiting.index(ticket) + 1

    def eta(self, position):
        if not self.durations:
            return None
        avg = sum(self.durations) / len(self.durations)
        return round(avg * position / self.max_active)

    async def acquire(self, ticket):
        if self.active < self.max_active and not self.waiting and not self.held(ticket):
            self.active += 1
            return
        if ticket.priority:  # ahead of everyone except earlier priority requests
            i = next((n for n, t in enumerate(self.waiting) if not t.priority), len(self.waiting))
            self.waiting.insert(i, ticket)
        else:
            self.waiting.append(ticket)
        self._dispatch()  # a priority request can start at once during a hold
        try:
            await ticket.turn.wait()
        except BaseException:
            if ticket in self.waiting:
                self.waiting.remove(ticket)
                self._changed()
            elif ticket.turn.is_set():  # given a slot as the client left: hand it on
                self.release(None)
            raise

    def release(self, duration, priority=False):
        if duration is not None:
            self.durations = (self.durations + [duration])[-20:]
        if priority and self.hold:
            self.hold_until = time.monotonic() + self.hold
        self.active -= 1
        self._dispatch()

    def _dispatch(self):
        """Start waiting requests while there are free slots; re-run when a hold ends."""
        while self.waiting and self.active < self.max_active:
            if self.held(self.waiting[0]):  # priority requests are always first, so none are waiting
                loop = asyncio.get_running_loop()
                if self.timer:
                    self.timer.cancel()
                self.timer = loop.call_later(self.hold_until - time.monotonic() + 0.05, self._dispatch)
                break
            self.active += 1
            self.waiting.pop(0).turn.set()
        self._changed()


class Ticket:
    def __init__(self, who, priority=False):
        self.who, self.priority = who, priority
        self.turn, self.changed = asyncio.Event(), asyncio.Event()


def chunk(model, text):
    return "data: " + json.dumps({
        "id": "llm-gate", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": text}, "finish_reason": None}],
    }) + "\n\n"


def status_text(queue, ticket):
    pos = queue.position(ticket)
    eta = queue.eta(pos)
    line = f"⏳ Waiting in queue: position {pos} of {len(queue.waiting)}"
    if queue.held(ticket):
        line += ", priority job running"
    return line + (f", about {eta} s\n" if eta is not None and not queue.held(ticket) else "\n")


def client_addr(request):
    """Real client address. Only TRUSTED_PROXIES may name it in a header: anyone else could otherwise lock out
    someone else by sending their address."""
    peer = request.remote or "?"
    if TRUSTED_PROXIES and peer.startswith(TRUSTED_PROXIES):
        return (request.headers.get("CF-Connecting-IP") or request.headers.get("X-Real-IP") or peer).strip()
    return peer


async def handle(request):
    app = request.app
    addr = client_addr(request)
    bad = app["bad_keys"]
    now = time.monotonic()
    recent = [t for t in bad.get(addr, []) if now - t < BAD_KEY_WINDOW]
    if len(recent) >= BAD_KEY_LIMIT:
        return web.json_response({"error": {"message": "Too many invalid API keys, try again later", "type": "rate_limited", "code": 429}}, status=429)
    auth = request.headers.get("Authorization", "")
    who = app["keys"].name(auth.removeprefix("Bearer ").strip())
    if who is None:
        if len(bad) > 10000:  # many addresses: forget the old ones rather than grow without limit
            bad.clear()
        bad[addr] = recent + [now]
        log.info("invalid API key from %s for %s %s (%d in %d s)", addr, request.method, request.path, len(bad[addr]), BAD_KEY_WINDOW)
        return web.json_response({"error": {"message": "Invalid API Key", "type": "authentication_error", "code": 401}}, status=401)
    bad.pop(addr, None)
    allowed = (request.path in QUEUED_PATHS and request.method == "POST"
               or request.path in OPEN_GETS and request.method == "GET"
               or request.path in OPEN_POSTS and request.method == "POST")
    if not allowed:
        log.info("%s blocked: %s %s", who, request.method, request.path)
        return web.json_response({"error": {"message": "Not available through the gate", "type": "forbidden", "code": 403}}, status=403)

    body = await request.read()
    headers = {"Authorization": "Bearer " + upstream_key(), "Content-Type": request.headers.get("Content-Type", "application/json")}
    url = UPSTREAM + (request.path if request.method == "GET" else request.path_qs)  # no ?reload=1 (unloads every model)
    if request.method != "POST" or request.path not in QUEUED_PATHS:
        async with app["http"].request(request.method, url, data=body or None, headers=headers) as up:
            data = await up.read()
            if request.method == "GET" and request.path in ("/v1/models", "/models") and up.status == 200 and (AUTO_MODEL or RESTRICTED):
                try:  # list the auto model alongside the real ones; hide models this key may not use
                    listing = json.loads(data)
                    listing["data"] = [m for m in listing["data"] if who in RESTRICTED.get(m.get("id"), {who})]
                    if AUTO_MODEL:
                        listing["data"].insert(0, {"id": AUTO_MODEL, "object": "model", "created": 0, "owned_by": "llm-gate"})
                    data = json.dumps(listing).encode()
                except (ValueError, KeyError, AttributeError):
                    pass
            return web.Response(body=data, status=up.status, content_type=up.content_type)

    try:
        payload = json.loads(body)
    except ValueError:
        payload = {}
    streaming = bool(payload.get("stream"))
    chat = request.path.endswith("chat/completions")
    # llama-server in router mode needs a model name; some older clients send none
    if DEFAULT_MODEL and isinstance(payload, dict) and payload and not payload.get("model"):
        payload["model"] = DEFAULT_MODEL
        body = json.dumps(payload).encode()
    note = None
    if isinstance(payload, dict) and payload:
        before = json.dumps(payload)
        note = route(payload, who) if chat else None
        if chat:
            add_client_rules(payload, who)
        if AUTO_MODEL and payload.get("model") == AUTO_MODEL:
            payload["model"] = DEFAULT_MODEL
        if json.dumps(payload) != before:
            body = json.dumps(payload).encode()
    model = payload.get("model", "")
    if who not in RESTRICTED.get(model, {who}):
        log.info("%s blocked: model %s", who, model)
        return web.json_response({"error": {"message": f"Model {model} is not available for this key", "type": "forbidden", "code": 403}}, status=403)
    archived = None
    if chat and "archive" in app and who in ARCHIVE_KEYS and isinstance(payload, dict):
        original = json.loads(body)
        try:
            archived = await archive_long_chat(app, payload, who, model)
        except Exception:
            log.exception("archive failed; sending the chat unchanged")
            payload = original

    queue = app["queue"]
    priority = who in PRIORITY
    if len(queue.waiting) >= MAX_WAITING and not priority:
        log.info("%s rejected: queue full (%d waiting)", who, len(queue.waiting))
        return web.json_response({"error": {"message": f"Server busy: {len(queue.waiting)} requests waiting. Try again shortly.",
                                            "type": "server_busy", "code": 503}}, status=503)

    ticket = Ticket(who, priority)
    arrived = time.monotonic()
    resp = None
    if streaming:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        await resp.prepare(request)

    wait = asyncio.ensure_future(queue.acquire(ticket))
    try:
        last = None
        while not wait.done():
            if request.transport is None or request.transport.is_closing():
                raise ConnectionResetError
            if streaming and ticket in queue.waiting:
                text = status_text(queue, ticket)
                if text != last:
                    await resp.write((chunk(model, text) if chat else ": " + text + "\n").encode())
                    last = text
                else:
                    await resp.write(b": waiting\n\n")
            ticket.changed.clear()
            changed = asyncio.ensure_future(ticket.changed.wait())
            await asyncio.wait([wait, changed], timeout=HEARTBEAT, return_when=asyncio.FIRST_COMPLETED)
            changed.cancel()
        await wait
    except ConnectionResetError:
        wait.cancel()
        log.info("%s left the queue after %.0f s", who, time.monotonic() - arrived)
        return resp or web.Response(status=499)
    except asyncio.CancelledError:
        wait.cancel()
        raise

    waited = time.monotonic() - arrived
    started = time.monotonic()
    status = 0
    try:
        if streaming and chat and last is not None:
            await resp.write(chunk(model, "✅ Your turn, generating…\n").encode())
        if streaming and chat:
            if note:
                await resp.write(chunk(model, note).encode())
            if archived:
                await resp.write(chunk(model, archived["note"]).encode())
            await switch_model(app, resp, model)
        if archived:
            if archived["info"]["new_cut"]:
                await index_archive(app, archived, model, resp if streaming and chat else None)
            app["archive"].add_note(payload, archived["conv"], archived["info"], archived["tools"])
            if archived["tools"]:
                result, status = await run_with_archive(app, url, headers, payload, archived, resp if streaming else None, model)
                if streaming:
                    await resp.write_eof()
                return result
            body = json.dumps(payload).encode()
        async with app["http"].post(url, data=body, headers=headers) as up:
            status = up.status
            if not streaming:
                return web.Response(body=await up.read(), status=up.status, content_type=up.content_type)
            if up.status != 200:
                err = await up.text()
                await resp.write(f"data: {err}\n\ndata: [DONE]\n\n".encode())
            else:
                async for data in up.content.iter_any():
                    await resp.write(data)
            await resp.write_eof()
            return resp
    except ConnectionResetError:  # client went away; closing upstream makes llama-server stop generating
        status = "client disconnected"
        return resp
    finally:
        took = time.monotonic() - started
        app["idle"].update(model=model, since=time.monotonic())
        queue.release(took, priority)
        log.info("%s %s waited %.0f s, ran %.0f s, status %s", who, request.path, waited, took, status)


async def status(request):
    q = request.app["queue"]
    held = max(0, round(q.hold_until - time.monotonic()))
    return web.json_response({"active": q.active, "waiting": len(q.waiting), "max_active": q.max_active, "held_s": held})


async def return_to_default(app):
    """Once nothing has run for RETURN_AFTER seconds, load DEFAULT_MODEL again so the next chat doesn't wait for it."""
    queue = app["queue"]
    while True:
        await asyncio.sleep(15)
        last = app["idle"].get("model")
        if not last or last == DEFAULT_MODEL or queue.active or queue.waiting or queue.held():
            continue
        if time.monotonic() - app["idle"]["since"] < RETURN_AFTER:
            continue
        app["idle"]["model"] = DEFAULT_MODEL
        try:
            async with app["http"].post(UPSTREAM + "/models/load", json={"model": DEFAULT_MODEL},
                                        headers={"Authorization": "Bearer " + upstream_key()}) as up:
                log.info("idle after %s: loading %s (status %s)", last, DEFAULT_MODEL, up.status)
        except Exception as e:
            log.info("idle after %s: loading %s failed: %s", last, DEFAULT_MODEL, e)


async def startup(app):
    app["http"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
    if RETURN_AFTER > 0 and DEFAULT_MODEL:
        app["returner"] = asyncio.ensure_future(return_to_default(app))


async def cleanup(app):
    if "returner" in app:
        app["returner"].cancel()
    await app["http"].close()


def main():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["keys"] = Keys(KEYS_FILE)
    app["queue"] = Queue(MAX_ACTIVE, HOLD)
    app["idle"] = {}  # last model used and when its request ended
    app["bad_keys"] = {}  # client address -> times of recent invalid keys
    if ARCHIVE_DB:
        app["archive"] = Archive(ARCHIVE_DB)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_get("/gate/status", status)
    app.router.add_route("*", "/{tail:.*}", handle)
    web.run_app(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8090")), access_log=None)


if __name__ == "__main__":
    main()
