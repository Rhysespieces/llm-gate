"""Long-chat recall test through a running llm-gate with the archive on.

Builds a ~85k-token chat (estimated) from Python's own module docs (the same chat every run), with facts planted at
different depths, some changed later, and asks 11 questions: single facts, facts that changed or are spread over
the chat, and position questions ("which section was in the same message as..."). Compare runs with the archive
tools on and off (ARCHIVE_TOOL_MODELS), or against a gate without the archive.

  python3 tests/recall_test.py --gate http://127.0.0.1:8090 --key KEY --model gpt-oss-20b
  add --db archive/chats.sqlite --key-name NAME to delete the test conversation from the archive afterwards
"""
import argparse, hashlib, json, pydoc, random, re, sqlite3, sys, time, urllib.request, warnings

ap = argparse.ArgumentParser()
ap.add_argument("--gate", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--effort", default="medium", help="reasoning_effort for models that take it (gpt-oss)")
ap.add_argument("--db", help="archive database, to remove the test conversation afterwards")
ap.add_argument("--key-name", help="the key's name in KEYS_FILE (the archive files conversations under it)")
args = ap.parse_args()

MODULES = sorted("""argparse array ast asyncio base64 bisect calendar cmd codecs collections colorsys configparser contextlib
copy csv dataclasses datetime decimal difflib dis email enum filecmp fnmatch fractions ftplib functools getopt gettext
glob graphlib gzip hashlib heapq hmac html http imaplib inspect io ipaddress itertools json keyword linecache locale
logging lzma mailbox math mimetypes netrc numbers operator optparse os pathlib pickle pkgutil platform plistlib poplib
pprint profile pstats queue quopri random re reprlib sched secrets selectors shelve shlex shutil signal smtplib socket
sqlite3 ssl stat statistics string struct subprocess symtable tarfile tempfile textwrap threading timeit token tokenize
trace traceback types typing unittest urllib uuid warnings wave weakref zipfile zlib""".split())
random.seed(7)
random.shuffle(MODULES)
sections = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name in MODULES:
        try:
            text = pydoc.render_doc(__import__(name), renderer=pydoc.plaintext)
        except Exception:
            continue
        if len(text) > 3000:
            sections.append((f"Python module {name}", text[200:4200]))
        if len(sections) == 60:
            break

PLANT = {2: "By the way, the hint for my NAS admin password is 'purple elephant 1987'.",
         8: "Decision: the weekly backup job runs at 3:15 am every Sunday because the internet is idle then.",
         15: "I just bought a Samsung 990 Pro 2TB NVMe for the models drive.",
         20: "My dog is called Biscuit, she's a corgi.",
         30: "Change of plan on backups: move the weekly backup to 4:30 am Saturday, because the ZFS scrub now runs Sunday morning.",
         35: "The 92mm fan I bought is an Arctic P9 Max, 5000 rpm.",
         45: "Also ordered an APC BX1600 UPS so the box survives short power cuts.",
         52: "My router's IP is 192.168.1.1."}
title = lambda i: sections[i][0].removeprefix("Python module ")
QUESTIONS = [
    ("What was the hint for my NAS admin password?", ["purple elephant"], "fact"),
    ("What breed is my pet?", ["corgi"], "fact"),
    ("Which cooling fan model did I purchase?", ["P9"], "fact"),
    ("What's my router's address?", ["192.168.1.1"], "fact"),
    ("When does my weekly backup run now? Give the day and time.", ["4:30", "saturday"], "changed"),
    ("How did the backup schedule change over this chat, and why?", ["3:15", "4:30", "scrub"], "changed"),
    ("List every piece of hardware I said I bought or ordered in this chat.", ["990", "P9", "BX1600"], "changed"),
    ("What were the first three documentation sections I shared? Give their titles in order.", [title(0), title(1), title(2)], "position"),
    ("Which documentation section did I share in the same message where I told you my dog's name?", [title(20)], "position"),
    ("Which documentation section did I share in the message where I mentioned ordering the UPS?", [title(45)], "position"),
    ("Which documentation section did I share in the message where I first told you the backup schedule?", [title(8)], "position"),
]

hist = [{"role": "system", "content": "You are a helpful assistant for a home-lab owner."},
        {"role": "user", "content": "I'm going to share documentation sections and notes with you over this chat. Keep track of what I tell you."},
        {"role": "assistant", "content": "Sounds good, go ahead."}]
for turn, (name, text) in enumerate(sections):
    extra = PLANT[turn] + "\n\n" if turn in PLANT else ""
    hist.append({"role": "user", "content": f'{extra}Here is a section titled "{name}":\n\n{text}\n\nWhat is the key point?'})
    hist.append({"role": "assistant", "content": "Key point: " + " ".join(text[:300].split()) + "..."})
print(f"{args.model}: {len(hist)} messages, ~{sum(len(m['content']) for m in hist) / 2.8 / 1000:.0f}k tokens by the gate's estimate")


def ask(q):
    body = {"model": args.model, "messages": hist + [{"role": "user", "content": q}], "max_tokens": 3000, "stream": True,
            "chat_template_kwargs": {"reasoning_effort": args.effort}}
    req = urllib.request.Request(args.gate + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + args.key, "Content-Type": "application/json"})
    t, content, think = time.time(), "", ""
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if line.startswith("data: {"):
            d = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta", {})
            content += d.get("content") or ""
            think += d.get("reasoning_content") or ""
    return time.time() - t, content, think


wall, _, think = ask("Say ok.")
print(f"first request (archives the chat): {wall:.0f} s; gate notes: {[l for l in think.splitlines() if l[:1] in '📦📇']}")
score = {}
for q, want, kind in QUESTIONS:
    wall, ans, think = ask(q)
    norm = lambda s: re.sub(r"\s+", "", s.lower())
    hit = all(norm(w) in norm(ans) for w in want)
    score.setdefault(kind, [0, 0])
    score[kind][0] += hit
    score[kind][1] += 1
    looks = sum(l.startswith("🔎") for l in think.splitlines())
    print(f"{'OK  ' if hit else 'MISS'} {kind:8} {wall:5.1f} s {looks} lookups | {q} -> {' '.join(ans.split())[:140]}")
print(", ".join(f"{k} {a}/{n}" for k, (a, n) in score.items()))

if args.db and args.key_name:
    db = sqlite3.connect(args.db)
    sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
    from archive import delete_conv
    conv = hashlib.sha256(json.dumps([args.key_name, hist[1]["content"]], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
    delete_conv(db, conv)
    db.commit()
    print("test conversation removed from", args.db)
