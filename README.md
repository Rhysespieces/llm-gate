# llm-gate

A small proxy in front of [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` for one GPU shared by
several people and apps. It queues requests fairly and shows waiting users where they stand. Long chats keep
working past the model's context: old messages go to a searchable archive instead of being cut off or compressed.

It speaks the OpenAI API, so Open WebUI, Cline and scripts work with no changes. What the gate does shows up in
the thinking/reasoning section of the reply, never in the answer:

```
⏳ Waiting in queue: position 2 of 3, about 40 s
🔄 Switching model to gpt-oss-20b (usually 30–60 s)…
📦 Long chat: 97 earlier messages (~68k tokens) are in the archive; 12 relevant excerpts recalled
🔎 Reading archived messages 91–94
```

One Python file for the gate, one for the archive; aiohttp is the only dependency.

## Features

**Queue**
- Runs `MAX_ACTIVE` generation requests at a time (match llama-server's `-np`); the rest wait in arrival order.
- Streaming chat requests see their position and an estimated wait. Heartbeats every 10 s keep Cloudflare and
  nginx from dropping waiting clients.
- `PRIORITY` keys go first. After each of their requests, others are held back for `HOLD` seconds, so a batch job's
  model stays loaded between its back-to-back requests.
- `MAX_WAITING`: past this the gate answers 503 instead of queueing forever.

**Models** (llama-server router mode, optional)
- Announces model switches while they load, instead of sending nothing for a minute.
- `RETURN_AFTER`: reloads `DEFAULT_MODEL` after the GPU has been idle, so the next chat doesn't wait.
- `AUTO_MODEL`: an extra model name meaning "the default", listed first.
- Images sent to a text-only model go to `VISION_FALLBACK`. If the request has an image-edit tool, the image is
  hidden from the model instead (the tool gets it from the client).
- `RESTRICTED`: models only some keys may use; they're hidden from everyone else's model list.

**Security**
- Each client has its own key (`KEYS_FILE`). llama-server only accepts the gate's upstream key, so client keys don't
  work on it directly and can't be used to bypass the queue.
- Path allowlist: generation endpoints, model list, health, props, tokenize and detokenize. Admin endpoints
  (loading and unloading models, slots, metrics) answer 403.
- Lockout: 10 wrong keys in 10 minutes locks that address out for the window. Only `TRUSTED_PROXIES` may name
  the real client address in a header.

**Long-chat archive** (optional; see below)

## Quick start

1. Put one long random key, and nothing else, in `keys/upstream.txt`: llama-server treats every line of that file as
   a valid key, comments included. For example: `openssl rand -hex 32 > keys/upstream.txt`.
   Run llama-server with it: `llama-server --api-key-file keys/upstream.txt --port 8081 ...`. If the gate runs in
   Docker, llama-server must listen where the container can reach it (`--host 0.0.0.0`). Firewall that port to
   Docker's networks: only the gate has the key, but nothing else needs to reach it.
2. Write `keys/clients.txt`, one `# name` line above each key:
   ```
   # alice
   sk-alice-0f3c...
   # automation
   sk-auto-77ab...
   ```
   The gate reloads it when it changes. Names are what `PRIORITY`, `RESTRICTED` and `ARCHIVE_KEYS` refer to and
   what the log shows.
3. `cp compose.example.yaml compose.yaml`, adjust, `mkdir -p archive && sudo chown 1000:1000 archive` (the container
   runs as UID 1000), `docker compose up -d --build`.
   Without Docker: `pip install -r requirements.txt`, set the same variables, `python3 gate.py`.
4. Point clients at `http://<host>:8090/v1` with their own key.

`GET /gate/status` (no key) returns the queue length, for dashboards.

## Settings

| Variable | Default | |
|---|---|---|
| `UPSTREAM` | `http://host.docker.internal:8081` | llama-server |
| `KEYS_FILE` / `UPSTREAM_KEY_FILE` | `/keys/server-keys.txt` / `/keys/upstream-key.txt` | client keys / llama-server's key |
| `HOST` / `PORT` | `0.0.0.0` / `8090` | |
| `MAX_ACTIVE` / `MAX_WAITING` | `1` / `8` | |
| `PRIORITY` / `HOLD` | none / `30` | key names; seconds |
| `DEFAULT_MODEL` | none | for requests that name no model |
| `AUTO_MODEL` / `RETURN_AFTER` | off / `0` | |
| `VISION_MODELS` / `VISION_FALLBACK` | none / off | comma lists |
| `RESTRICTED` | none | `model:name1\|name2;model2:name3` |
| `TRUSTED_PROXIES` | none | address prefixes, e.g. `172.` |
| `ARCHIVE_DB` | off | SQLite path; turns the archive on |
| `ARCHIVE_KEYS` | none | key names whose chats are archived |
| `ARCHIVE_LIMITS` / `ARCHIVE_DEFAULT_LIMIT` | none / `20000` | `model:tokens,...`: when to archive (gate estimate, see below) |
| `ARCHIVE_TOOL_MODELS` | none | models that get `archive_read` / `archive_search` |
| `EMBED_URL` | none | embedding server for meaning search; keywords only without it |
| `CLIENT_RULES` / `CLIENT_RULES_DIR` | none / `/rules` | `keyname:file.md,...`: rules appended to Cline's system prompt for that key (see below) |

## Long-chat archive

Chat apps send the whole history with every request. Once a chat passes its model's limit, the gate moves the
oldest messages into SQLite. The model then sees:

```
system prompt + archive index
first user message            (always kept: it usually states the task)
recent messages, unchanged
latest user message + excerpts from the archive that match it
```

The **index** has one line per ~2.5k tokens of archived messages: the message range, a one-line summary, and the
quoted opening words of the user's messages there.

```
• m19–22: Weekly backup 3:15 am Sunday; C++ compiler support table — user: “Decision: the weekly backup job runs at 3:15 am…”
• m91–94: APC BX1600 UPS ordered; PyTorch docs — user: “Also ordered an APC BX1600 UPS so the box…”
```

Models in `ARCHIVE_TOOL_MODELS` also get two tools:
- `archive_read(start, end)`: messages word for word, by the numbers in the index.
- `archive_search(query)`: keywords (SQLite FTS5) and meaning (embeddings), merged by reciprocal rank fusion.

The gate runs these calls itself, so clients never see them. A client's own tools still work: if the model calls
one, the request ends and the client gets that call as usual. After 4 lookup rounds the model is told to answer
with what it has.

Nothing is lost: the client still holds the full history, and archived text stays searchable. If a message in the
archived part is edited or regenerated, that chat's archive starts over.

### Designed for one cache slot on slow prompt processing

On a single GPU, llama-server usually keeps one cached prompt. Rereading a long chat after a cache miss is the
slow part: about 40 s for 22k tokens at 580 tokens/s. The design follows from that:

- **Cuts move in big steps.** When the history passes the limit, the gate archives down to half the limit. The
  start of the prompt then stays the same, and stays cached, until the history grows back to the limit.
- **The index lives in the system message**, which only changes at a cut.
- **Excerpts are chosen once per user message** and attached to it, then re-attached word for word on every
  following request. Agents (Cline and similar) send dozens of tool-result requests per user message; picking new
  excerpts for each one changed the middle of the prompt every time, and llama-server can't reuse a partial cache for
  models with sliding-window layers (gpt-oss), so every agent step reread the whole chat (60–80 s at 22k tokens).
  Between user messages the model uses `archive_read` / `archive_search`. The chosen excerpts are kept in memory for
  200 chats; after a gate restart the first request rereads once.
- **Summaries are written during the reply that makes the cut.** That reply rereads the prompt anyway. A
  background summary job would push the chat out of the cache slot, and the next message would pay for a full
  reread. The first cut of a long chat takes 30–40 s longer; later cuts less. On failure or timeout, the index
  keeps just the quoted openings.
- **The tools stay in the request on every round.** Some chat templates (gpt-oss) list tools at the top of the
  prompt, so removing them would invalidate the whole cache. In one test that made a reply take 93 s.

Token counts are estimated at 2.8 characters per token without calling the tokenizer. That's conservative: 85k
estimated ≈ 60k real for prose. Set `ARCHIVE_LIMITS` to about 65% of the context size.

### Rules for agents on client machines

`CLIENT_RULES=laptop:cline-windows.md` appends `CLIENT_RULES_DIR/cline-windows.md` to the system prompt of requests
made with the `laptop` key whose system prompt is Cline's. Use it to give a small model house rules (how to edit
files safely, what never to run, when to stop) without installing anything on that machine. The file is read on every
request, so edits apply at once; its text only changes when the file does, so the prompt cache stays valid. Mount
the folder read-only, e.g. `./rules:/rules:ro`.

### Embedding worker

`embed_worker.py` runs on the host next to llama-server. It needs llama-server and an embedding model; it's written
for [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF), and the gate's query prefix
matches it.

- A CPU llama-server answers the gate's query embeddings, about 0.2 s each. Set `EMBED_HOST=0.0.0.0` if the gate
  runs in Docker, and firewall the port.
- It embeds newly archived chunks in the background. Optionally, large batches run on a GPU: set `GPU_DEVICE`,
  plus `NVIDIA_GPU` / `IDLE_URL` to check free memory, or check that a ComfyUI queue is empty. The worker starts a
  GPU llama-server for the batch and stops it afterwards, so the GPU's main user gets its memory back.
- It deletes conversations not touched for `KEEP_DAYS` (90).

All settings are listed at the top of `embed_worker.py`. Example systemd unit:

```ini
[Service]
User=llm
Environment=ARCHIVE_DB=/srv/llm-gate/archive/chats.sqlite
Environment=LLAMA_SERVER=/opt/llama.cpp/build/bin/llama-server
Environment=EMBED_MODEL=/srv/models/Qwen3-Embedding-0.6B-Q8_0.gguf
Environment=EMBED_HOST=0.0.0.0
ExecStart=/usr/bin/python3 /srv/llm-gate/embed_worker.py
KillMode=control-group
Restart=on-failure
```

### Results

gpt-oss-20b on a Radeon Instinct MI25 (64k context, archive limit 44k), reasoning effort medium. Two ~85k-token
test chats, each with 8 planted facts and 11 questions:
- **A:** `tests/recall_test.py`, built from Python module docs.
- **B:** documentation pages with long titles, so the quoted openings in the index cut the titles short.

| | A | B |
|---|---|---|
| Excerpts only (no index) | 8/11 | 9/11 |
| Index, no tools | 10/11 | 9/11 |
| Index + tools | 10/11, 11/11 (two runs) | 11/11 |

- The index fixes most questions about position ("which section was in the same message as…") when the
  answer is in it: in A, the quoted openings contain the section titles.
- The tools matter when it isn't: in B, the titles are cut short.
- Without the index, the model made up "the first three sections I shared" in every run, confidently.
- With tools, that question was still hit-or-miss across tuning runs on B. Sometimes the model trusts the index
  without reading.

Costs:
- Lookups add 5–30 s to the answers that use them; most questions need none.
- The model sometimes searches the archive for something that is in its recent context.
- At reasoning effort low, it rarely looks anything up.

These are small tests on one setup: they show the behaviour, not a benchmark.

### Tested on

- **Generation:** Radeon Instinct MI25 (16 GB, Vulkan backend), llama.cpp b11804, router mode with one model loaded at
  a time. gpt-oss-20b (MXFP4) at 64k context: about 65 tokens/s generation and about 580 tokens/s prompt processing.
  That slow prompt processing is why the archive is built around the prompt cache. On a fast GPU, rereading a
  prompt costs much less, and summaries could just as well be written in the background.
- **Embeddings:** Qwen3-Embedding-0.6B (Q8_0). Queries run on 4 CPU threads, about 0.2 s each. Batches run on a
  GTX 1080 while it's idle: about 75k tokens in 70 s, against about 80 tokens/s on the CPU.

## Limitations

- Written for llama-server. Model switching and `RETURN_AFTER` need router mode (`/models`, `/models/load`).
  Other OpenAI-compatible servers should work for queueing and the archive, but that's untested.
- The archive identifies a chat by key name and first user message: two chats with the same opening message from
  the same key share an archive, and it starts over whenever they differ.
- Tool use for lookups has only been tested with gpt-oss-20b. Small models may ignore the tools; leave them out of
  `ARCHIVE_TOOL_MODELS` and they still get the index and excerpts.
- Archived chats are stored as plain text in SQLite. Protect the file like the chats themselves.

## Tests

```
python3 -m unittest discover tests        # archive logic, offline, a few seconds
python3 tests/recall_test.py --gate http://127.0.0.1:8090 --key KEY --model MODEL [--db archive/chats.sqlite --key-name NAME]
```

## License

MIT
