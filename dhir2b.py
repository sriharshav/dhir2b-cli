# Copyright 2026 Sriharsha Vardhan <sriharsha.vardhan@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Terminal chat for Dhīr 2B (zenteiq-ai/Dhir-2B) on a Mac / CPU / CUDA.

Built for Dhīr 2B only; there's no option to load another model.

Usage:
    dhir2b                 # interactive chat (Ctrl+C to quit)
    dhir2b --prompt "..."  # start the chat with this message
    dhir2b --examples      # run the built-in examples
    dhir2b --download      # download the model for offline use
    dhir2b --offline       # never touch the network (or export DHIR2B_OFFLINE=1)

In chat, Ctrl+C while the model is answering stops that answer;
Ctrl+C at the prompt (or Ctrl+D) quits. ↑/↓ recall earlier prompts.
Answers stream as rendered Markdown.

Model card: https://huggingface.co/zenteiq-ai/Dhir-2B
The Dhīr 2B model is by ZenteiQ AiTech Innovations (Apache 2.0) and is
downloaded separately; this script does not include its weights or code.
"""

__version__ = "0.2.3"
__author__ = "Sriharsha Vardhan <sriharsha.vardhan@gmail.com>"
__license__ = "Apache-2.0"

import argparse
import base64
import contextlib
import getpass
import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import termios
import time


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


# Offline mode: `--offline` for one run, or DHIR2B_OFFLINE=1 to make it the default.
# It must be decided before huggingface_hub/transformers are imported, because they
# read HF_HUB_OFFLINE at import time; that's why it's checked here and not in main().
OFFLINE = _truthy(os.environ.get("DHIR2B_OFFLINE")) or "--offline" in sys.argv[1:]
if OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
MODE = "offline" if OFFLINE else "online"

# Enables ↑/↓ history and line editing in input(). Prefer GNU readline: macOS's
# built-in libedit mishandles colour codes in the prompt.
try:
    import gnureadline as readline
except ImportError:
    try:
        import readline
    except ImportError:  # Python built without readline/libedit
        readline = None

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.spinner import Spinner
from rich.text import Text

# Harmless macOS-only notice PyTorch logs when transformers imports torch.distributed.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextStreamer

# dhir2b is built for Dhīr 2B only: the model, its revision, chat format and sampling
# settings are fixed here on purpose, and there's no option to load another model.
MODEL_ID = "zenteiq-ai/Dhir-2B"
# Pinned commit: the model runs custom code (trust_remote_code), so pin it to the
# version that was reviewed. Bump deliberately after checking modeling_dhir.py.
REVISION = "7cbe444cd1e45e25197cfa0f683b2cb575858169"

NOT_DOWNLOADED = (
    "Dhīr 2B isn't downloaded yet, and offline mode is on.\n"
    "Connect to the internet and run `dhir2b --download` once (~3.8 GB)."
)

# Sampling defaults from the repo's generation_config.json.
GEN_KWARGS = dict(
    do_sample=True,
    temperature=0.36,
    top_p=0.93,
    top_k=40,
    repetition_penalty=1.026,
)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download():
    """Download Dhīr 2B (pinned revision) so later runs can be fully offline."""
    if OFFLINE:
        sys.exit("Can't download in offline mode. Unset DHIR2B_OFFLINE / drop --offline and retry.")
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} (~3.8 GB, skips files you already have) ...", file=sys.stderr)
    path = snapshot_download(MODEL_ID, revision=REVISION)
    # Load the tokenizer once so transformers also caches the model's custom code.
    AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
    print(f"Done: {path}\nYou can now run `dhir2b --offline` (or set DHIR2B_OFFLINE=1).", file=sys.stderr)


def _human_size(n: float) -> str:
    """1536 -> '1.5 KB'."""
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _cached_revisions() -> tuple[object | None, list]:
    """(cache_info, revisions of Dhīr 2B in the local Hugging Face cache)."""
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
    except Exception:  # no cache directory yet
        return None, []
    for repo in cache.repos:
        if repo.repo_id == MODEL_ID and repo.repo_type == "model":
            return cache, sorted(repo.revisions, key=lambda r: r.last_modified, reverse=True)
    return cache, []


def model_info():
    """Show the pinned model revision, what's downloaded, and whether ZenteiQ has a newer one."""
    print(f"Model:     {MODEL_ID}")
    print(f"Uses:      revision {REVISION[:7]} (pinned in dhir2b {__version__})")
    _, revisions = _cached_revisions()
    if not revisions:
        print("Download:  not downloaded yet (run `dhir2b --download`)")
    for r in revisions:
        tag = "in use" if r.commit_hash == REVISION else "old, not used; remove with `dhir2b --clean-cache`"
        print(f"Download:  revision {r.commit_hash[:7]}  {_human_size(r.size_on_disk):>9}  ({tag})")
    if OFFLINE:
        print("Upstream:  not checked (offline mode)")
        return
    try:
        from huggingface_hub import HfApi

        latest = HfApi().model_info(MODEL_ID).sha
    except Exception as e:  # network or Hub error: report, don't crash
        print(f"Upstream:  couldn't check ({type(e).__name__})")
        return
    if latest == REVISION:
        print("Upstream:  up to date with ZenteiQ's latest revision")
    else:
        print(
            f"Upstream:  ZenteiQ has published a newer revision ({latest[:7]}).\n"
            "           dhir2b only uses reviewed revisions; a future dhir2b release will move to it.\n"
            "           See https://github.com/sriharshav/dhir2b-cli/releases"
        )


def clean_cache():
    """Delete downloaded Dhīr 2B revisions other than the one this dhir2b version uses."""
    cache, revisions = _cached_revisions()
    old = [r for r in revisions if r.commit_hash != REVISION]
    if not old:
        print("Nothing to clean: only the revision in use is downloaded.")
        return
    for r in old:
        print(f"Removing old revision {r.commit_hash[:7]}")
    strategy = cache.delete_revisions(*(r.commit_hash for r in old))
    strategy.execute()  # files shared with the revision in use are kept
    print(f"Freed {_human_size(strategy.expected_freed_size)}. Revision {REVISION[:7]} (in use) is untouched.")


REPO_URL = "https://github.com/sriharshav/dhir2b-cli"
_LATEST_RELEASE_API = "https://api.github.com/repos/sriharshav/dhir2b-cli/releases/latest"


def _parse_version(v: str) -> tuple[int, ...]:
    """'v0.2.10' -> (0, 2, 10). Non-numeric parts are ignored."""
    return tuple(int(p) for p in re.findall(r"\d+", v)[:3])


def _latest_release() -> tuple[str, str]:
    """(tag, release page URL) of the latest GitHub release."""
    import urllib.request

    # Python's default client header is enough for GitHub; nothing identifying is sent.
    req = urllib.request.Request(_LATEST_RELEASE_API, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.load(resp)
    return data["tag_name"], data["html_url"]


def _install_kind() -> str:
    """How this copy of dhir2b was installed: 'pipx', 'editable' or 'pip'."""
    if os.path.exists(os.path.join(sys.prefix, "pipx_metadata.json")):
        return "pipx"
    try:
        from importlib.metadata import distribution

        direct = json.loads(distribution("dhir2b-cli").read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            return "editable"
    except Exception:  # not installed as a package (e.g. `python dhir2b.py` from a clone)
        return "editable"
    return "pip"


def _update_command(kind: str, tag: str) -> list[str] | None:
    """The command that installs `tag` the same way dhir2b is installed now."""
    spec = f"git+{REPO_URL}.git@{tag}"
    if kind == "pipx":
        pipx = shutil.which("pipx")
        return [pipx, "install", "--force", spec] if pipx else None
    if kind == "pip":
        return [sys.executable, "-m", "pip", "install", "--upgrade", spec]
    return None  # editable: a git clone, updated with git pull


def update():
    """Upgrade dhir2b to the latest GitHub release, the same way it was installed."""
    if OFFLINE:
        sys.exit("Can't check for updates in offline mode. Unset DHIR2B_OFFLINE / drop --offline and retry.")
    try:
        tag, url = _latest_release()
    except Exception as e:
        sys.exit(f"Couldn't check for updates ({type(e).__name__}: {e}). See {REPO_URL}/releases")
    if _parse_version(tag) <= _parse_version(__version__):
        print(f"dhir2b {__version__} is up to date (latest release: {tag}).")
        return
    print(f"Updating dhir2b {__version__} -> {tag}  ({url})")
    kind = _install_kind()
    if kind == "editable":
        print("This is a development copy (git clone). Update it with:\n  git pull")
        return
    cmd = _update_command(kind, tag)
    if cmd is None:
        sys.exit(f"pipx not found on PATH. Run:\n  pipx install --force git+{REPO_URL}.git@{tag}")
    print("Running: " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Update failed (exit code {result.returncode}). Nothing was changed if pip/pipx stopped early.")
    print(
        f"\nUpdated to {tag}. If the release notes say the model changed, the next online run downloads it;\n"
        "afterwards, `dhir2b --clean-cache` frees the space used by the old one."
    )


def _purge_targets() -> list[str]:
    """Small files dhir2b leaves outside the model cache, for --purge."""
    from huggingface_hub import constants
    from transformers.utils import HF_MODULES_CACHE

    repo_folder = "models--" + MODEL_ID.replace("/", "--")
    paths = [
        os.path.join(constants.HF_HUB_CACHE, repo_folder),  # whatever remains of the model folder
        os.path.join(constants.HF_HUB_CACHE, ".locks", repo_folder),  # download lock files
        # transformers' copy of the model's custom code
        os.path.join(HF_MODULES_CACHE, "transformers_modules", "zenteiq_hyphen_ai", "Dhir_hyphen_2B"),
        HISTORY_FILE,  # prompt history
    ]
    return [p for p in paths if os.path.lexists(p)]


def _shared_blob_locks() -> list[str]:
    """Lock files next to the model's files in Hugging Face's shared blob store. huggingface_hub
    deletes the shared files themselves but leaves these empty .lock files behind."""
    from huggingface_hub import constants

    blobs = os.path.join(constants.HF_HUB_CACHE, "models--" + MODEL_ID.replace("/", "--"), "blobs")
    locks = []
    with contextlib.suppress(OSError):
        for name in os.listdir(blobs):
            link = os.path.join(blobs, name)
            if os.path.islink(link):
                locks.append(os.path.realpath(link) + ".lock")
    return locks


def _remove_empty_package_dir():
    """Remove transformers' zenteiq_hyphen_ai package folder if only empty boilerplate is left."""
    from transformers.utils import HF_MODULES_CACHE

    pkg = os.path.join(HF_MODULES_CACHE, "transformers_modules", "zenteiq_hyphen_ai")
    with contextlib.suppress(OSError):
        if set(os.listdir(pkg)) <= {"__init__.py", "__pycache__"}:
            shutil.rmtree(pkg)


def _path_size(path: str) -> int:
    if os.path.isfile(path) or os.path.islink(path):
        return os.lstat(path).st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(root, f)).st_size
    return total


def purge(assume_yes: bool = False):
    """Delete everything dhir2b downloaded or saved: the model, its cached code and prompt history."""
    cache, revisions = _cached_revisions()
    others = [p for p in _purge_targets() if not (revisions and p.endswith("models--" + MODEL_ID.replace("/", "--")))]
    if not revisions and not others:
        print("Nothing to purge: no Dhīr 2B model, cached model code or prompt history found.")
    else:
        print("This permanently deletes:")
        if revisions:
            # The model's files can live in Hugging Face's shared blob store; size and delete them
            # through huggingface_hub so the real data is removed, not just the links to it.
            size = sum(r.size_on_disk for r in revisions)
            print(f"  {_human_size(size):>9}  Dhīr 2B model ({len(revisions)} revision(s)) in the Hugging Face cache")
        for p in others:
            print(f"  {_human_size(_path_size(p)):>9}  {p}")
        print("You'd need to download the model again (~3.8 GB) to use dhir2b after this.")
        if not assume_yes:
            try:
                answer = input("Delete these? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in ("y", "yes"):
                print("Cancelled; nothing was deleted.")
                return
        shared_locks = _shared_blob_locks()
        if revisions:
            cache.delete_revisions(*(r.commit_hash for r in revisions)).execute()
        for p in _purge_targets() + [lk for lk in shared_locks if os.path.lexists(lk)]:
            if os.path.isdir(p) and not os.path.islink(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        _remove_empty_package_dir()
        print("Deleted.")
    print("\nTo remove the dhir2b program itself:")
    kind = _install_kind()
    if kind == "pipx":
        print("  pipx uninstall dhir2b-cli")
    elif kind == "pip":
        print(f"  {sys.executable} -m pip uninstall dhir2b-cli")
    else:
        print("  delete the cloned folder (and its .venv)")
    if os.environ.get("DHIR2B_OFFLINE"):
        print("and remove the `export DHIR2B_OFFLINE=...` line from your shell profile (e.g. ~/.zshrc).")


def load(device: str | None = None, warmup: bool = True):
    device = device or pick_device()
    mode = " (offline)" if OFFLINE else ""
    print(f"Loading {MODEL_ID} on {device}{mode} ...", file=sys.stderr)

    def _load(local_only: bool):
        common = dict(revision=REVISION, trust_remote_code=True, local_files_only=local_only)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, **common)
        # device_map puts weights straight on the GPU. Loading to CPU and then
        # calling .to(device) briefly holds two copies (2 x 3.8 GB), which swaps
        # heavily on 8 GB Macs and made startup ~5x slower.
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device, **common
        )
        return tok, mdl

    try:
        tokenizer, model = _load(local_only=True)  # cached: no network calls
    except OSError:
        if OFFLINE:
            sys.exit(NOT_DOWNLOADED)
        print("Not cached yet, downloading (~3.8 GB, one time) ...", file=sys.stderr)
        tokenizer, model = _load(local_only=False)
    model.eval()
    if warmup:
        # The first generation has a one-time ~15-20 s stall on Apple GPUs (memory
        # paging + kernel compilation). Pay it here so real answers stream at once.
        print("Warming up ...", file=sys.stderr)
        with torch.inference_mode():
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}], add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(model.device)
            model.generate(**ids, max_new_tokens=6, **GEN_KWARGS)
    return tokenizer, model


class _StopFlag(StoppingCriteria):
    """Stops generation when `.stop` is set (by the Ctrl+C handler)."""

    def __init__(self):
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop


CONSOLE = Console(highlight=False)
CODE_THEME = "monokai"  # any Pygments style; it paints its own background, so it reads on light and dark terminals

# Dhir often writes code without fences unless asked; this makes it fence and tag them.
SYSTEM_PROMPT = (
    "You are Dhir, a helpful assistant running in a terminal. Format replies in Markdown. "
    "Always put code in fenced code blocks with a language tag, e.g. ```python."
)

_FENCE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")


# Simple rules for untagged code blocks; first match wins. (Pygments' guess_lexer
# is unreliable on short snippets, e.g. it calls Python "scdoc".)
_LANG_RULES = [
    ("cpp", r"^\s*#include\s*<(iostream|vector|string)>|std::"),
    ("c", r"^\s*#include\s*[<\"]"),
    ("java", r"\bpublic\s+(static\s+)?(class|void)\b|System\.out\.print"),
    ("python", r"^\s*(def |class \w+.*:|import \w|from \S+ import|elif |print\()"),
    ("javascript", r"\bfunction\s*\w*\s*\(|=>|^\s*(const|let|var) \w+\s*=|console\.log"),
    ("sql", r"(?i)^\s*(select|insert|update|delete|create|alter|drop)\s"),
    ("html", r"(?i)^\s*<(!doctype|html|head|body|div|span|p|a|ul|table)\b"),
    ("json", r"^\s*[\[{]\s*\"?"),
    ("bash", r"^\s*(\$ |sudo |ls|cd |grep |find |echo |pip |git |curl |mkdir |rm |export |brew |cat |python3? )"),
]


def _guess_language(code: str) -> str:
    """Best-effort language name for an untagged code block ("text" if unsure)."""
    for lang, pattern in _LANG_RULES:
        if re.search(pattern, code, flags=re.M):
            return lang
    return "text"


def _tag_fences(md: str) -> str:
    """Add a language tag to fenced code blocks that lack one, so they get highlighted."""
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, fence, info = m.groups()
        j = i + 1  # find the closing fence (or end of text while still streaming)
        while j < len(lines) and not lines[j].strip().startswith(fence):
            j += 1
        if not info.strip():
            info = _guess_language("\n".join(lines[i + 1:j]))
        out.append(f"{indent}{fence}{info}")
        out.extend(lines[i + 1:j + 1])
        i = j + 1
    return "\n".join(out)


class _MarkdownStreamer(TextStreamer):
    """Streams the answer as rendered Markdown.

    Shows a spinner until the first word. Lines that have scrolled out of the
    live window are printed permanently; only the last LIVE_LINES lines are
    redrawn as text arrives. This keeps long answers clean even when they run
    past the top of the terminal.
    """

    LIVE_LINES = 8
    MIN_INTERVAL = 0.08  # seconds between redraws

    def __init__(self, tokenizer, console: Console = CONSOLE, **kw):
        super().__init__(tokenizer, **kw)
        self.console = console
        self.text = ""
        self.printed = 0  # number of rendered lines already printed permanently
        self._last = 0.0
        self.live = Live(
            Spinner("dots", text=Text("thinking", style="dim")),
            console=console, refresh_per_second=12, transient=True,
        )
        self.live.start()

    def _render_lines(self) -> list[str]:
        buf = io.StringIO()
        Console(
            file=buf, force_terminal=True, width=self.console.width,
            color_system=self.console.color_system, highlight=False,
        ).print(Markdown(_tag_fences(self.text), code_theme=CODE_THEME))
        return buf.getvalue().rstrip("\n").split("\n")

    def _refresh(self, final: bool = False):
        lines = self._render_lines()
        stable = len(lines) if final else max(len(lines) - self.LIVE_LINES, 0)
        if stable > self.printed:
            self.live.console.print(Text.from_ansi("\n".join(lines[self.printed:stable])))
            self.printed = stable
        self.live.update(Text.from_ansi("\n".join(lines[self.printed:])))

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if not text:
            return
        self.text += text
        now = time.monotonic()
        if now - self._last >= self.MIN_INTERVAL:
            self._last = now
            self._refresh()

    def close(self):
        if not self.live.is_started:
            return
        if self.text:
            self._refresh(final=True)
        self.live.update(Text(""))
        self.live.stop()


@contextlib.contextmanager
def _busy_terminal():
    """While the model answers: hide keystrokes and the cursor, then discard
    anything typed so it doesn't leak into the next prompt. Ctrl+C still works."""
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~(termios.ECHO | termios.ICANON)  # lflags: no echo, no line editing (ISIG kept)
    termios.tcsetattr(fd, termios.TCSANOW, new)
    sys.stdout.write("\033[?25l")  # hide cursor
    sys.stdout.flush()
    try:
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)  # drop keys typed while busy
        termios.tcsetattr(fd, termios.TCSANOW, old)
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()


class _TokenClock(StoppingCriteria):
    """Never stops generation; records when the first token arrived. generate() calls
    stopping criteria once per new token, so this works with or without streaming."""

    def __init__(self):
        self.start = time.perf_counter()
        self.first = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first is None:
            self.first = time.perf_counter()
        return False


def format_stats(stats: dict, mode: str | None = None, device: str | None = None) -> str:
    """One-line summary, e.g. '4.2 s · first token 0.5 s · 61 tokens · 16.4 tok/s · offline · mps'."""
    parts = [f"{stats['seconds']:.1f} s"]
    if stats.get("first_token_s") is not None:
        parts.append(f"first token {stats['first_token_s']:.1f} s")
    parts.append(f"{stats['tokens']} token{'' if stats['tokens'] == 1 else 's'}")
    if stats.get("tokens_per_s"):
        parts.append(f"{stats['tokens_per_s']:.1f} tok/s")
    parts += [p for p in (mode, device) if p]
    return " · ".join(parts)


@torch.inference_mode()
def chat(tokenizer, model, messages, tools=None, max_new_tokens=256, stream=False, stats=None, **kwargs) -> str:
    """Run one assistant turn. `messages` is an OpenAI-style list of {role, content}.

    Pass a dict as `stats` to get timing back: seconds, first_token_s, tokens,
    tokens_per_s (generation speed after the first token) and stopped.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    gen = {**GEN_KWARGS, **kwargs}
    if not gen.get("do_sample"):
        for k in ("temperature", "top_p", "top_k"):
            gen.pop(k, None)
    # While generating, Ctrl+C stops this answer cleanly instead of killing the program.
    flag = _StopFlag()
    clock = _TokenClock()
    prev = signal.signal(signal.SIGINT, lambda *_: setattr(flag, "stop", True))
    streamer = None
    try:
        if stream:
            # Rendered Markdown in a terminal; plain text when piped to a file.
            cls = _MarkdownStreamer if sys.stdout.isatty() else TextStreamer
            streamer = cls(tokenizer, skip_prompt=True, skip_special_tokens=True)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[clock, flag],
            **gen,
        )
    finally:
        signal.signal(signal.SIGINT, prev)
        if isinstance(streamer, _MarkdownStreamer):
            streamer.close()
    end = time.perf_counter()
    if flag.stop and stream:
        CONSOLE.print("[dim]\\[stopped][/dim]")
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    if stats is not None:
        n = int(new_tokens.shape[-1])
        first = None if clock.first is None else clock.first - clock.start
        decode_s = end - clock.first if clock.first is not None else 0.0
        stats.update(
            seconds=end - clock.start,
            first_token_s=first,
            tokens=n,
            tokens_per_s=(n - 1) / decode_s if n > 1 and decode_s > 0 else None,
            stopped=flag.stop,
        )
    # Keep special tokens when tools are in play so <tool_call> tags survive.
    return tokenizer.decode(new_tokens, skip_special_tokens=tools is None).strip()


def parse_tool_calls(text: str):
    """Extract [{"name": ..., "arguments": {...}}] from <tool_call>...</tool_call> blocks."""
    calls = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        try:
            calls.append(json.loads(raw))
        except json.JSONDecodeError:
            pass
    return calls


# ---------------------------------------------------------------- examples

def examples(tokenizer, model):
    def show(title, messages, **kw):
        print(f"\n=== {title} ===")
        print(chat(tokenizer, model, messages, **kw))

    # 1. Plain Hindi question
    show("Hindi Q&A", [{"role": "user", "content": "भारत की राजधानी के बारे में दो वाक्य लिखिए।"}])

    # 2. Translation
    show("Translation EN -> Telugu", [
        {"role": "user", "content": "Translate to Telugu: The meeting is moved to Monday at 10 AM."},
    ], do_sample=False)

    # 3. Document QA (RAG reader): answer only from the passage
    passage = (
        "बेंगलुरु कर्नाटक की राजधानी है। इसे भारत की सिलिकॉन वैली कहा जाता है क्योंकि "
        "यहाँ कई सूचना प्रौद्योगिकी कंपनियाँ स्थित हैं।"
    )
    show("Document QA", [
        {"role": "system", "content": "Answer only from the given passage. Be brief."},
        {"role": "user", "content": f"Passage: {passage}\n\nQuestion: बेंगलुरु को सिलिकॉन वैली क्यों कहा जाता है?"},
    ], do_sample=False)

    # 4. Tool calling
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    messages = [{"role": "user", "content": "What's the weather in Chennai right now?"}]
    reply = chat(tokenizer, model, messages, tools=tools, do_sample=False)
    calls = parse_tool_calls(reply)
    print("\n=== Tool calling ===")
    print("model ->", calls or reply)
    if calls:
        # Pretend we called the API, then send the result back for a final answer.
        messages += [
            {"role": "assistant", "content": "", "tool_calls": [{"function": c} for c in calls]},
            {"role": "tool", "content": json.dumps({"city": "Chennai", "temp_c": 31, "sky": "humid, partly cloudy"})},
        ]
        print("final ->", chat(tokenizer, model, messages, tools=tools, do_sample=False)
              .replace("<|end_of_turn|>", "").strip())


HELP = (
    "Dhīr 2B chat.\n"
    "  Ctrl+C       stop the current answer; at the prompt, quit\n"
    "  ↑ / ↓        previous / next prompt\n"
    "  /clear       clear the screen and start a new conversation\n"
    "  /reset       start a new conversation (keep the screen)\n"
    "  /copy        copy the last answer to the clipboard\n"
    "  /copy code   copy just the code blocks from the last answer\n"
    "  /help        show this help"
)


def _copy_to_clipboard(text: str) -> str:
    """Copy text to the system clipboard; returns the method used."""
    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"], ["clip.exe"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            return cmd[0]
    # No clipboard tool (e.g. over SSH): OSC 52 asks the terminal itself to copy.
    sys.stdout.write("\033]52;c;" + base64.b64encode(text.encode("utf-8")).decode() + "\a")
    sys.stdout.flush()
    return "terminal (OSC 52)"


def _code_blocks(md: str) -> str:
    """Contents of all fenced code blocks in md, joined by blank lines."""
    blocks = re.findall(r"^\s*(?:```+|~~~+)[^\n]*\n(.*?)^\s*(?:```+|~~~+)\s*$", md, flags=re.M | re.S)
    return "\n\n".join(b.rstrip("\n") for b in blocks)


def _copy_command(history: list[dict], code_only: bool = False):
    last = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
    if not last:
        CONSOLE.print("[dim](no answer to copy yet)[/dim]")
        return
    text = _code_blocks(last) if code_only else last
    if not text:
        CONSOLE.print("[dim](no code blocks in the last answer)[/dim]")
        return
    try:
        how = _copy_to_clipboard(text)
    except (OSError, subprocess.CalledProcessError) as e:
        CONSOLE.print(f"[red]copy failed:[/red] {escape(str(e))}")
        return
    what = "code" if code_only else "answer"
    CONSOLE.print(f"[dim](copied {what}: {len(text.splitlines())} lines, via {how})[/dim]")


HISTORY_FILE = os.environ.get("DHIR2B_HISTORY", os.path.expanduser("~/.dhir2b_history"))


def _setup_history():
    """↑/↓ recall previous prompts, kept across sessions in .dhir_history."""
    if readline is None:
        return
    with contextlib.suppress(OSError):
        readline.read_history_file(HISTORY_FILE)
    readline.set_history_length(1000)


def _save_history():
    if readline is not None:
        with contextlib.suppress(OSError):
            readline.write_history_file(HISTORY_FILE)


USER = getpass.getuser()  # login name of whoever runs the chat
# "<user>@dhir2b>" (like user@host) so it's clearly dhir2b's prompt, not the shell's.
PROMPT_LABEL = f"{USER}@dhir2b>"


def _prompt() -> str:
    """Coloured "<user>@dhir2b> " prompt. The \\001/\\002 markers tell readline the colour
    codes take no space, so ↑/↓ and line editing redraw at the right column."""
    if not sys.stdout.isatty():
        return f"{PROMPT_LABEL} "
    return f"\001\033[1;32m\002{PROMPT_LABEL}\001\033[0m\002 "


def _new_history() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def _banner(model):
    print(HELP)
    CONSOLE.print(f"[dim]Mode: [bold]{MODE}[/bold] · device: {model.device.type} · model: {MODEL_ID}[/dim]")


def interactive(tokenizer, model, first_prompt: str | None = None):
    _setup_history()
    _banner(model)
    history = _new_history()
    pending = first_prompt
    if pending and readline is not None:
        readline.add_history(pending.strip())
    try:
        _chat_loop(tokenizer, model, history, pending)
    finally:
        _save_history()


def _chat_loop(tokenizer, model, history, pending):
    while True:
        print()  # blank line before each prompt (kept out of input() so libedit redraws cleanly)
        if pending is not None:
            user, pending = pending.strip(), None
            CONSOLE.print(f"[bold green]{escape(PROMPT_LABEL)}[/bold green] {escape(user)}", highlight=False)
        else:
            try:
                user = input(_prompt()).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
        if user == "/clear":
            history = _new_history()
            print("\033[2J\033[3J\033[H", end="")  # clear screen + scrollback, cursor home
            _banner(model)
            continue
        if user == "/reset":
            history = _new_history()
            print("(new conversation)")
            continue
        if user in ("/copy", "/copy code"):
            _copy_command(history, code_only=user.endswith("code"))
            continue
        if user in ("/help", "/?"):
            _banner(model)
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        stats = {}
        try:
            with _busy_terminal():
                CONSOLE.print("[bold cyan]dhir>[/bold cyan]")
                reply = chat(tokenizer, model, history, max_new_tokens=1024, stream=True, stats=stats)
        except KeyboardInterrupt:
            # Ctrl+C landed just before generation began: drop this turn.
            history.pop()
            CONSOLE.print("[dim]\\[stopped][/dim]")
            continue
        history.append({"role": "assistant", "content": reply})
        CONSOLE.print(f"[dim]⏱ {format_stats(stats, MODE, model.device.type)}[/dim]")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(prog="dhir2b", description="Chat with Dhīr 2B (Ctrl+C to quit).")
    ap.add_argument("--prompt", help="first message; the chat stays open afterwards")
    ap.add_argument("--examples", action="store_true", help="run the built-in examples and exit")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"])
    ap.add_argument(
        "--offline", action="store_true",
        help="never use the network (or set DHIR2B_OFFLINE=1 to make this the default)",
    )
    ap.add_argument("--download", action="store_true", help="download the model for offline use, then exit")
    ap.add_argument(
        "--model-info", action="store_true",
        help="show the model revision in use, what's downloaded, and whether a newer one exists",
    )
    ap.add_argument("--clean-cache", action="store_true", help="delete old downloaded model revisions, then exit")
    ap.add_argument("--update", action="store_true", help="upgrade dhir2b to the latest release, then exit")
    ap.add_argument(
        "--purge", action="store_true",
        help="delete the downloaded model, its cached code and prompt history (asks first), then exit",
    )
    ap.add_argument("--yes", action="store_true", help="with --purge: don't ask for confirmation")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)

    if args.download:
        download()
        return
    if args.model_info:
        model_info()
        return
    if args.clean_cache:
        clean_cache()
        return
    if args.update:
        update()
        return
    if args.purge:
        purge(assume_yes=args.yes)
        return
    try:
        tok, mdl = load(args.device)
    except KeyboardInterrupt:
        sys.exit("\ncancelled")
    if args.examples:
        examples(tok, mdl)
    else:
        interactive(tok, mdl, args.prompt)


if __name__ == "__main__":
    main()
