"""Embeddings for llm-gate's long-chat archive (ARCHIVE_DB, see archive.py).

Runs a CPU llama-server with an embedding model (Qwen3-Embedding-0.6B: the gate's query prefix is written for it)
for the gate's query embeddings (~0.2 s each, no VRAM), and fills in vectors for newly archived chunks. Optionally,
big batches run on a GPU while it is idle (~10x faster): a second llama-server is started for the batch and stopped
again, so whatever normally uses that GPU gets its memory back. Also deletes conversations untouched for KEEP_DAYS.

Settings (environment):
  ARCHIVE_DB       same SQLite file as the gate's ARCHIVE_DB
  LLAMA_SERVER     path to the llama-server binary
  EMBED_MODEL      path to the embedding model (GGUF)
  EMBED_HOST/PORT  where the CPU server listens (the gate's EMBED_URL); 0.0.0.0 if the gate runs in Docker
  CPU_THREADS      threads for the CPU server (default 4)
  GPU_DEVICE       llama.cpp device for batches, e.g. CUDA0 or Vulkan0; empty = CPU only
  GPU_MIN_FREE_MB  only use the GPU with this much free memory (needs NVIDIA_GPU; default 2500)
  NVIDIA_GPU       nvidia-smi -i value (index or UUID) for the free-memory check; empty = skip the check
  IDLE_URL         a ComfyUI /queue URL: use the GPU only while that queue is empty; empty = skip the check
  KEEP_DAYS        delete conversations not touched for this long (default 90)
"""
import json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive import connect, delete_conv, normalize  # same schema as the gate

DB = os.environ.get('ARCHIVE_DB', 'archive/chats.sqlite')
SERVER = os.environ.get('LLAMA_SERVER', 'llama-server')
MODEL = os.environ.get('EMBED_MODEL', 'models/Qwen3-Embedding-0.6B-Q8_0.gguf')
HOST, PORT = os.environ.get('EMBED_HOST', '127.0.0.1'), int(os.environ.get('EMBED_PORT', '8098'))
CPU_THREADS = os.environ.get('CPU_THREADS', '4')
GPU_DEVICE = os.environ.get('GPU_DEVICE', '')
GPU_PORT = int(os.environ.get('GPU_PORT', '8096'))
GPU_MIN_FREE_MB = int(os.environ.get('GPU_MIN_FREE_MB', '2500'))
NVIDIA_GPU = os.environ.get('NVIDIA_GPU', '')
IDLE_URL = os.environ.get('IDLE_URL', '')
KEEP_DAYS = float(os.environ.get('KEEP_DAYS', '90'))
GPU_MIN_TOKENS = 1500  # smaller jobs aren't worth starting the GPU server for
CPU_URL, GPU_URL = f'http://127.0.0.1:{PORT}', f'http://127.0.0.1:{GPU_PORT}'
COMMON = ['-m', MODEL, '--embedding', '--pooling', 'last', '-c', '4096', '-b', '2048', '-ub', '2048', '-np', '1', '--no-webui']
CPU_ARGS = ['-dev', 'none', '-t', CPU_THREADS, '--host', HOST, '--port', str(PORT)]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def start(args, url):
    port = int(url.rsplit(':', 1)[1])
    with socket.socket() as s:  # another server on the port would answer our health checks
        if s.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'port {port} is already in use')
    p = subprocess.Popen([SERVER] + COMMON + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if p.poll() is not None:
            raise RuntimeError(f'embedding server exited ({p.returncode})')
        try:
            if json.load(urllib.request.urlopen(url + '/health', timeout=2)).get('status') == 'ok':
                return p
        except Exception:
            pass
        time.sleep(0.5)
    p.terminate()
    raise RuntimeError('embedding server did not start')


def embed(url, texts):
    req = urllib.request.Request(url + '/v1/embeddings', json.dumps({'input': texts}).encode(), {'Content-Type': 'application/json'})
    return [normalize(d['embedding']).tobytes() for d in json.load(urllib.request.urlopen(req, timeout=600))['data']]


def gpu_idle():
    if not IDLE_URL:
        return True
    try:
        q = json.load(urllib.request.urlopen(IDLE_URL, timeout=3))
        return not q.get('queue_running') and not q.get('queue_pending')
    except Exception:
        return True  # the GPU's usual user is down: the card is free


def gpu_free_mb():
    if not NVIDIA_GPU:
        return GPU_MIN_FREE_MB
    try:
        out = subprocess.run(['nvidia-smi', '-i', NVIDIA_GPU, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip())
    except Exception:
        return 0


def fill(db, url, rows, batch):
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vecs = embed(url, [t for _, t in part])
        db.executemany('update chunks set vec = ? where id = ?', [(v, cid) for v, (cid, _) in zip(vecs, part)])


def prune(db):
    old = [r[0] for r in db.execute('select id from convs where updated < ?', (time.time() - KEEP_DAYS * 86400,))]
    for conv in old:
        delete_conv(db, conv)
    if old:
        log(f'pruned {len(old)} conversations older than {KEEP_DAYS:g} days')


def main():
    Path(DB).parent.mkdir(parents=True, exist_ok=True)
    db = connect(DB)
    cpu = start(CPU_ARGS, CPU_URL)
    signal.signal(signal.SIGTERM, lambda *_: (cpu.terminate(), sys.exit(0)))
    log(f'CPU embedding server up on {HOST}:{PORT}')
    last_prune = 0
    while True:
        if cpu.poll() is not None:
            log('CPU embedding server died; restarting')
            try:
                cpu = start(CPU_ARGS, CPU_URL)
            except RuntimeError as e:
                log(f'restart failed: {e}; retrying in 30 s')
                time.sleep(30)
                continue
        if time.time() - last_prune > 86400:
            prune(db)
            last_prune = time.time()
        rows = db.execute('select id, text from chunks where vec is null order by id limit 2000').fetchall()
        if not rows:
            time.sleep(5)
            continue
        toks = sum(len(t) for _, t in rows) / 2.8
        t = time.time()
        free = gpu_free_mb() if GPU_DEVICE and toks >= GPU_MIN_TOKENS else 0
        if GPU_DEVICE and free >= GPU_MIN_FREE_MB and gpu_idle():
            gpu = None
            try:
                gpu = start(['-dev', GPU_DEVICE, '-ngl', '99', '--host', '127.0.0.1', '--port', str(GPU_PORT)], GPU_URL)
                fill(db, GPU_URL, rows, 16)
                where = GPU_DEVICE
            except Exception as e:
                log(f'GPU embedding failed ({e}); using the CPU')
                fill(db, CPU_URL, rows, 4)
                where = 'CPU after GPU failure'
            finally:
                if gpu:
                    gpu.terminate()
                    gpu.wait(30)
        else:
            fill(db, CPU_URL, rows, 4)
            where = 'CPU'
        log(f'embedded {len(rows)} chunks (~{toks / 1000:.1f}k tokens) on {where} in {time.time() - t:.1f} s')


if __name__ == '__main__':
    main()
