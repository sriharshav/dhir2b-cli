# dhir2b-cli

[![CI](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Model: Dhīr 2B](https://img.shields.io/badge/%F0%9F%A4%97%20model-zenteiq--ai%2FDhir--2B-yellow)](https://huggingface.co/zenteiq-ai/Dhir-2B)

A terminal chat for **[Dhīr 2B](https://huggingface.co/zenteiq-ai/Dhir-2B)**, ZenteiQ's compact model pretrained from scratch in India for English and 10 Indian languages. It runs locally, works offline, and is tuned to run well on Apple Silicon Macs with 8 GB of RAM.

dhir2b is built for Dhīr 2B only. The model, its revision, chat format and sampling settings are fixed, and there's no option to load another model.

```
sriharshavardhan> Write a Python function to check if a number is prime
dhir>
 def is_prime(n):
     if n <= 1:
         return False
     ...
```

## Features

- **Streaming Markdown**: headings, bold and lists render as the answer streams, with syntax-highlighted code blocks.
- **Terminal-friendly**: Ctrl+C stops an answer, ↑/↓ recall earlier prompts (saved across sessions), and typing is blocked while the model answers.
- **`/copy`** copies the last answer to the clipboard; **`/copy code`** copies just its code.
- **Offline mode**: `dhir2b --offline` or `DHIR2B_OFFLINE=1` guarantees no network use; `dhir2b --download` prepares the model in advance.
- **Light on small Macs**: loads weights straight onto the GPU, avoiding a swap-heavy double copy.
- **Pinned model code**: Dhīr loads Python code from Hugging Face, so the model is pinned to a reviewed commit.

## Requirements

- macOS (Apple Silicon recommended) or Linux; NVIDIA GPUs are used when available, otherwise CPU
- Python 3.10+
- About 4 GB of free RAM and 4 GB of disk for the model

## Install

Install with [pipx](https://pipx.pypa.io), which gives dhir2b its own isolated environment and puts the `dhir2b` command on your PATH:

```bash
brew install pipx        # macOS; on Linux: python3 -m pip install --user pipx
pipx ensurepath          # once, so ~/.local/bin is on your PATH (open a new terminal after)
pipx install git+https://github.com/sriharshav/dhir2b-cli.git@v0.2.0
```

To upgrade later, run `pipx upgrade dhir2b-cli`; to remove it, run `pipx uninstall dhir2b-cli`.

> **Avoid plain `pip install` into your main Python.** dhir2b depends on PyTorch and Hugging Face Transformers, which pull in newer versions of common packages such as pydantic 2. Installing those into a Python you use for other things can break other tools (for example, ones that need pydantic 1). pipx or a virtual environment keeps them separate.

Or from a clone, in a virtual environment:

```bash
git clone https://github.com/sriharshav/dhir2b-cli.git
cd dhir2b-cli
python3 -m venv .venv
.venv/bin/pip install -e .
```

The first run downloads the model (~3.8 GB) to `~/.cache/huggingface`. To download it in advance, for example before going offline, run `dhir2b --download`.

## Usage

```bash
dhir2b                                    # chat
dhir2b --prompt "नमस्ते, आप कौन हैं?"       # chat, starting with this message
dhir2b --examples                         # Hindi Q&A, translation, document QA, tool calling
dhir2b --device cpu                       # force CPU (cuda | mps | cpu)
dhir2b --download                         # download the model for offline use
dhir2b --offline                          # never use the network
```

From a clone without installing: `python dhir2b.py`, which takes the same options.

### In the chat

| Key / command | What it does |
|---|---|
| Ctrl+C while answering | Stop that answer |
| Ctrl+C at the prompt, or Ctrl+D | Quit |
| ↑ / ↓ | Previous / next prompt (←/→, Ctrl+A/E, Ctrl+U edit the line) |
| `/copy` | Copy the last answer (raw Markdown) to the clipboard |
| `/copy code` | Copy just the code blocks from the last answer |
| `/clear` | Clear the screen and start a new conversation |
| `/reset` | Start a new conversation, keep the screen |
| `/help` | List the commands |

The prompt shows your login name. Prompt history is saved in `~/.dhir2b_history`; set `DHIR2B_HISTORY` to use another file. When output is piped to a file, answers are written as plain text.

## Offline use

Once the model is downloaded, dhir2b loads it from your disk. Offline mode goes further and guarantees that nothing touches the network.

1. **Download once** while you're online:

   ```bash
   dhir2b --download
   ```

2. **Run offline.** For a single run:

   ```bash
   dhir2b --offline
   ```

   To always run offline, add this to your shell profile (for example `~/.zshrc`):

   ```bash
   export DHIR2B_OFFLINE=1
   ```

The startup line shows `(offline)` when it's on. If the model isn't downloaded yet, dhir2b says so and exits instead of trying the network. To download again later, remove the setting for that run: `DHIR2B_OFFLINE=0 dhir2b --download`.

To set up a machine with no internet, copy `~/.cache/huggingface/hub/models--zenteiq-ai--Dhir-2B` from a machine that has run `dhir2b --download` into the same place, then use offline mode.

## Performance

On an 8 GB M1 MacBook, dhir2b reaches the prompt in roughly 20–60 s: about 5 s to load, plus a one-time warm-up while macOS pages the weights into GPU memory. The warm-up is slower when RAM is tight, so close heavy apps for faster starts. After that, it generates about 11 tokens/s, and the model stays loaded until you quit.

## Use from Python

```python
from dhir2b import load, chat

tok, model = load()
print(chat(tok, model, [{"role": "user", "content": "Translate to Kannada: Good morning"}]))
```

`chat()` also accepts `tools=[...]` for function calling; `parse_tool_calls()` extracts the calls. See `examples()` in [dhir2b.py](dhir2b.py).

## Customising

Constants at the top of [dhir2b.py](dhir2b.py):

- `CODE_THEME`: any [Pygments style](https://pygments.org/styles/) for code blocks (default `monokai`).
- `SYSTEM_PROMPT`: asks Dhīr to put code in fenced, language-tagged blocks. Without it, Dhīr often writes code as plain text.
- `GEN_KWARGS`: sampling settings, taken from the model's `generation_config.json`.

## Notes

- The model ships custom architecture code, so it loads with `trust_remote_code=True`. `REVISION` pins it to a reviewed commit; read the new `modeling_dhir.py` before bumping it.
- Thinking mode isn't supported by this model release. Answers are most reliable when you supply the source text (RAG).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Sriharsha Vardhan. Released under the [Apache License 2.0](LICENSE).

dhir2b-cli is an independent project, not affiliated with ZenteiQ. The Dhīr 2B model is developed by [ZenteiQ AiTech Innovations](https://www.zenteiq.ai) with support from the IndiaAI Mission, MeitY, and is released separately under Apache 2.0 on [Hugging Face](https://huggingface.co/zenteiq-ai/Dhir-2B). This repository doesn't include the model's weights or code; they're downloaded from Hugging Face on first run.
