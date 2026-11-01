"""Conversation archive for llm-gate: long chats are trimmed instead of compressed.

Chat apps send the whole history with every request. Once a chat's history passes a model's limit, the oldest
messages (after the first user message, which usually holds the task) are moved to SQLite and the model gets:
  system prompt + an index of the archived part (one line per section, with its message numbers)
  first user message
  the recent messages, unchanged
  excerpts from the archive that match the latest message, appended to that message.
The gate also gives the model archive_read (exact messages by number, as the index points to) and archive_search.
Each index line quotes the opening words of the section's user messages; the gate has the model add a one-line
summary during the turn that made the cut (see gate.py).
The cut only moves in big steps (it archives down to half the limit), so llama-server can reuse its cached prompt
until the next cut. Search: FTS5 keywords always; meaning (embeddings, filled in by embed_worker.py)
when the chunk vectors and the query embedding server are available. Nothing is lost: the client still holds
the full history, and archived text stays searchable.
"""
import array
import hashlib
import json
import math
import re
import sqlite3
import time

CHARS_PER_TOKEN = 2.8  # conservative: code and JSON tokenize denser than prose
CHUNK_CHARS = 800
MAX_EXCERPTS = 12  # best matches only; the tail of the ranking is mostly noise
IMAGE_TOKENS = 1000  # an attached image, counted instead of its base64 text
SECTION_TOKENS = 2500  # archived messages are grouped into index sections of about this size, split at user messages
MAX_INDEX_LINES = 40  # longer indexes merge neighbouring sections
INDEX_TOKENS = 3000  # room kept for the index and tool definitions when planning a cut
READ_TOKENS = 4000  # most archive_read returns at once
STOP = set("the and for that this with you your are was were have has had not but what when where which who why how "
           "can could would should will just from they them then than there their about into also some more very "
           "does did done its it's i'm i've don't please thanks".split())

SCHEMA = """
create table if not exists convs(id text primary key, who text, cut integer not null, cut_hash text,
                                 created real, updated real);
create table if not exists chunks(id integer primary key, conv text not null, msg integer not null, part integer not null,
                                  role text, text text not null, vec blob, unique(conv, msg, part));
create index if not exists chunks_pending on chunks(id) where vec is null;
create table if not exists sections(conv text not null, start integer not null, "end" integer not null,
                                    opens text not null, summary text, primary key(conv, start));
create virtual table if not exists chunks_fts using fts5(text, content='chunks', content_rowid='id',
                                                         tokenize='porter unicode61');
"""


def connect(path):
    db = sqlite3.connect(path, timeout=10, isolation_level=None)
    db.execute("pragma journal_mode=wal")
    db.executescript(SCHEMA)
    return db


def delete_conv(db, conv, keep_conv=False):
    """Remove a conversation's archive. The FTS index is told what to forget with FTS5's 'delete' command: a plain
    DELETE on an external-content table can leave the words behind, pointing at whatever later reuses the row ids."""
    db.execute("insert into chunks_fts(chunks_fts, rowid, text) select 'delete', id, text from chunks where conv = ?", (conv,))
    db.execute("delete from chunks where conv = ?", (conv,))
    db.execute("delete from sections where conv = ?", (conv,))
    if not keep_conv:
        db.execute("delete from convs where id = ?", (conv,))


def text_of(m):
    """Plain text of a message: content parts, tool calls, tool results. Images become a marker."""
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(p.get("text", "") if p.get("type") == "text" else "[image]" for p in c if isinstance(p, dict))
    out = c or ""
    for tc in m.get("tool_calls") or []:
        f = tc.get("function") or {}
        out += f"\n[tool call {f.get('name')}: {f.get('arguments')}]"
    return out


def tokens(m):
    c = m.get("content")
    images = sum(1 for p in c if isinstance(p, dict) and p.get("type") != "text") if isinstance(c, list) else 0
    return len(text_of(m)) / CHARS_PER_TOKEN + images * IMAGE_TOKENS + 8


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_start(body, i):
    """A cut may start the kept part at a user message, or at an assistant message (tool results always follow the
    assistant message that called them, so this never separates a call from its result)."""
    return body[i].get("role") in ("user", "assistant")


def plan_cut(body, cut, limit, fixed):
    """New cut index into body (body[0] is pinned): archive oldest messages until the rest fits in half the limit."""
    kept = sum(tokens(m) for m in body[cut:])
    if fixed + kept <= limit:
        return cut
    target, new = limit / 2 - fixed, cut
    while new < len(body) - 1 and kept > target:
        kept -= tokens(body[new])
        new += 1
    # prefer starting at a user message; settle for an assistant message; never archive the latest message
    for want in (("user",), ("user", "assistant")):
        j = new
        while j < len(body) - 1 and body[j].get("role") not in want:
            j += 1
        if j < len(body) and body[j].get("role") in want:
            return j
    return cut


def chunks_of(text):
    """Paragraph-sized pieces (up to CHUNK_CHARS), so a one-line fact isn't drowned out by a pasted document
    around it when searching by meaning."""
    pieces = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        while len(para) > CHUNK_CHARS:  # long paragraph or code: cut at a sentence or line end where possible
            cut = max(para.rfind(". ", 0, CHUNK_CHARS), para.rfind("\n", 0, CHUNK_CHARS))
            cut = cut + 1 if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            pieces.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            pieces.append(para)
    out = []
    for p in pieces:  # join short neighbours back up to the chunk size
        if out and len(out[-1]) + len(p) + 2 <= CHUNK_CHARS // 2:
            out[-1] += "\n\n" + p
        else:
            out.append(p)
    return out or [""]


def make_sections(body, a, b):
    """Group body[a:b] into (start, end, opens) sections of about SECTION_TOKENS, each starting at a user message
    where possible. opens quotes the opening words of the section's user messages."""
    groups, size = [], 0
    for i in range(a, b):
        if not groups or size >= SECTION_TOKENS and body[i].get("role") == "user":
            groups.append([i, i])
            size = 0
        groups[-1][1] = i
        size += tokens(body[i])
    out = []
    for s, e in groups:
        heads = [" ".join(text_of(body[i]).split())[:70] for i in range(s, e + 1) if body[i].get("role") == "user"]
        heads = heads or [" ".join(text_of(body[s]).split())[:70]]
        out.append((s, e, " / ".join(f"“{h}…”" for h in heads)[:240]))
    return out


def fts_query(q):
    words = []
    for w in re.findall(r"[a-z0-9_]{3,}", q.lower()):
        if w not in STOP and w not in words:
            words.append(w)
    return " OR ".join(f'"{w}"' for w in words[:40])


def unpack(blob):
    a = array.array("f")
    a.frombytes(blob)
    return a


def search(db, conv, query, qvec, budget_tokens):
    """Best archive chunks for query: keyword and meaning rankings merged (reciprocal rank fusion)."""
    score = {}
    fq = fts_query(query)
    if fq:
        rows = db.execute("select c.id from chunks_fts f join chunks c on c.id = f.rowid "
                          "where chunks_fts match ? and c.conv = ? order by bm25(chunks_fts) limit 20", (fq, conv)).fetchall()
        for rank, (cid,) in enumerate(rows):
            score[cid] = score.get(cid, 0) + 1 / (60 + rank)
    if qvec is not None:
        sims = []
        for cid, blob in db.execute("select id, vec from chunks where conv = ? and vec is not null", (conv,)):
            v = unpack(blob)
            if len(v) == len(qvec):
                sims.append((sum(a * b for a, b in zip(v, qvec)), cid))
        for rank, (_, cid) in enumerate(sorted(sims, reverse=True)[:20]):
            score[cid] = score.get(cid, 0) + 1 / (60 + rank)
    picked, used = [], 0
    for cid in sorted(score, key=score.get, reverse=True)[:MAX_EXCERPTS]:
        msg, part, role, text = db.execute("select msg, part, role, text from chunks where id = ?", (cid,)).fetchone()
        cost = len(text) / CHARS_PER_TOKEN
        if used + cost > budget_tokens:
            continue
        picked.append((msg, part, role, text))
        used += cost
    return sorted(picked)


def query_text(body):
    """What to search for: the latest message, plus the latest user message if that was earlier (tool loops)."""
    last = text_of(body[-1])[-2000:]
    for m in reversed(body):
        if m.get("role") == "user":
            u = text_of(m)[-2000:]
            return last if u == last else u + "\n" + last
    return last


class Archive:
    def __init__(self, path, recall_tokens=3000):
        self.db = connect(path)
        self.recall_tokens = recall_tokens

    def prepare(self, payload, who, limit):
        """Trim payload['messages'] in place if needed. Returns (conv, query, info) when the archive is in use,
        else None; call recall() with the query embedding next."""
        self.db.execute("begin immediate")  # one commit for a cut's hundreds of rows, not one each
        try:
            out = self._prepare(payload, who, limit)
        except BaseException:
            self.db.execute("rollback")
            raise
        self.db.execute("commit")
        return out

    def _prepare(self, payload, who, limit):
        msgs = payload.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) for m in msgs):
            return None
        n_sys = 0
        while n_sys < len(msgs) and msgs[n_sys].get("role") in ("system", "developer"):
            n_sys += 1
        system, body = msgs[:n_sys], msgs[n_sys:]
        if len(body) < 3 or body[0].get("role") != "user":
            return None
        conv = digest([who, text_of(body[0])])[:32]
        row = self.db.execute("select cut, cut_hash from convs where id = ?", (conv,)).fetchone()
        cut = row[0] if row else 1
        if row and (cut >= len(body) or digest(body[:cut]) != row[1]):  # history edited or regenerated: start over
            delete_conv(self.db, conv, keep_conv=True)
            cut = 1
        fixed = sum(tokens(m) for m in system) + tokens(body[0]) + self.recall_tokens + INDEX_TOKENS
        new = plan_cut(body, cut, limit, fixed)
        now = time.time()
        if new != cut or not row:
            for i in range(cut, new):
                for part, text in enumerate(chunks_of(text_of(body[i]))):
                    cur = self.db.execute("insert or ignore into chunks(conv, msg, part, role, text) values (?,?,?,?,?)",
                                          (conv, i, part, body[i].get("role"), text))
                    if cur.rowcount:
                        self.db.execute("insert into chunks_fts(rowid, text) values (?, ?)", (cur.lastrowid, text))
            self.db.executemany('insert or replace into sections(conv, start, "end", opens) values (?,?,?,?)',
                                [(conv, s, e, t) for s, e, t in make_sections(body, cut, new)])
            self.db.execute("insert into convs(id, who, cut, cut_hash, created, updated) values (?,?,?,?,?,?) "
                            "on conflict(id) do update set cut = excluded.cut, cut_hash = excluded.cut_hash, "
                            "updated = excluded.updated", (conv, who, new, digest(body[:new]), now, now))
        else:
            self.db.execute("update convs set updated = ? where id = ?", (now, conv))
        if new <= 1:
            return None
        archived_tokens = sum(tokens(m) for m in body[1:new])
        kept = body[new:]
        bridge = [{"role": "assistant", "content": "(earlier messages archived)"}] if kept[0].get("role") == "user" else []
        payload["messages"] = system + [body[0]] + bridge + kept
        return conv, query_text(body), dict(archived=new - 1, archived_tokens=archived_tokens, new_cut=new != cut, cut=new)

    def index_lines(self, conv):
        rows = [[s, e, (summ + " — user: " if summ else "user: ") + opens] for s, e, opens, summ in self.db.execute(
            'select start, "end", opens, summary from sections where conv = ? order by start', (conv,))]
        while len(rows) > MAX_INDEX_LINES:  # merge neighbours until it fits
            rows = [[a[0], b[1], (a[2] + " | " + b[2])[:300]] if b else a for a, b in zip(rows[::2], rows[1::2] + [None])]
        return [f"• m{s + 1}–{e + 1}: {t}" for s, e, t in rows]

    def add_note(self, payload, conv, info, tools):
        """Tell the model what was archived, with the index. Goes into the system message, which only changes when
        the cut or the index does, so the cached prompt stays valid between cuts."""
        how = ("The index is only a rough guide. Before answering anything that depends on the archived messages (what "
               "was said, exact wording, names, numbers, order, or what was said where), call archive_read with the "
               "message range from this index, or archive_search if the index doesn't show where it is, and answer "
               "from what they return. Never guess archived details from the index alone. " if tools else "")
        note = (f"[Archive: messages 2–{info['cut']} of this conversation (~{info['archived_tokens'] / 1000:.0f}k tokens) "
                f"were moved out of your context to save space. Index of the archived messages:\n"
                + "\n".join(self.index_lines(conv)) + "\n" + how
                + "Excerpts that match the latest message are also attached to it as 'Archived excerpts'; treat "
                  "everything in the archive as part of this conversation.]")
        msgs = payload["messages"]
        if msgs and msgs[0].get("role") in ("system", "developer") and isinstance(msgs[0].get("content"), str):
            msgs[0] = dict(msgs[0], content=msgs[0]["content"] + "\n\n" + note)
        else:
            msgs.insert(0, {"role": "system", "content": note})

    def unsummarized(self, conv, user_chars=400, other_chars=200):
        """[(start, end, text)] for index sections without a summary yet: the start of every message in them."""
        out = []
        for s, e in self.db.execute('select start, "end" from sections where conv = ? and summary is null order by start', (conv,)).fetchall():
            msgs = {}
            for msg, role, t in self.db.execute(
                    "select msg, role, text from chunks where conv = ? and msg between ? and ? order by msg, part", (conv, s, e)):
                msgs.setdefault(msg, [role, ""])[1] += t + "\n"
            lines = []
            for role, t in msgs.values():
                n = user_chars if role == "user" else other_chars
                lines.append(f"({role}) " + (t[:n] + " …" if len(t) > n else t.strip()))
            out.append((s, e, "\n".join(lines)))
        return out

    def set_summaries(self, conv, summaries):
        self.db.executemany('update sections set summary = ? where conv = ? and start = ?',
                            [(t[:200], conv, s) for s, t in summaries.items()])

    def read(self, conv, start, end):
        """Archived messages start..end (message numbers as in the index), word for word, up to READ_TOKENS."""
        row = self.db.execute("select cut from convs where id = ?", (conv,)).fetchone()
        last = row[0] if row else 1  # messages 2..cut are archived
        start, end = max(2, int(start)), min(int(end), last, int(start) + 29)
        if start > last:
            return f"Message {start} is not archived: messages after {last} are in your context."
        out, used, msg_at = [], 0, start
        for msg, part, role, text in self.db.execute(
                "select msg, part, role, text from chunks where conv = ? and msg between ? and ? order by msg, part",
                (conv, start - 1, end - 1)):
            if used + len(text) / CHARS_PER_TOKEN > READ_TOKENS and out:
                out.append(f"[stopped at message {msg + 1} to save space: read from {msg + 1} on for the rest]")
                break
            out.append(f"--- message {msg + 1} ({role}){' cont.' if part else ''} ---\n{text}")
            used += len(text) / CHARS_PER_TOKEN
        return "\n\n".join(out) or f"No archived text for messages {start}–{end}."

    def search_text(self, conv, query, qvec):
        hits = search(self.db, conv, query, qvec, READ_TOKENS * 3 // 4)
        return "\n\n".join(f"--- message {msg + 1} ({role}){' cont.' if part else ''} ---\n{t}"
                             for msg, part, role, t in hits) or ("No matches in the archive. Only the earlier messages listed in the archive index "
                                                  "are archived; recent messages are in your context above.")

    def recall(self, payload, conv, query, qvec, tools=False):
        """Append matching archive excerpts to the latest message. Returns how many were added."""
        hits = search(self.db, conv, query, qvec, self.recall_tokens)
        last = payload["messages"][-1]
        if not hits or last.get("role") not in ("user", "tool"):
            return 0
        text = "\n\n".join(f"--- message {msg + 1} ({role}){' cont.' if part else ''} ---\n{t}" for msg, part, role, t in hits)
        end = ("[End of archived excerpts. They are search matches with gaps between them, not the conversation in "
               "order: for what came first, next or where, go by the archive index in the system message, and use "
               "archive_read or archive_search when these excerpts don't settle the answer.]" if tools else "[End of archived excerpts]")
        block = f"\n\n[Archived excerpts from earlier in this conversation (search matches only)]\n{text}\n{end}"
        c = last.get("content")
        if isinstance(c, list):
            last["content"] = c + [{"type": "text", "text": block}]
        else:
            last["content"] = (c or "") + block
        return len(hits)


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return array.array("f", (x / n for x in v))
