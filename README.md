# dhir2b-cli

[![CI](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Model: Dhīr 2B](https://img.shields.io/badge/%F0%9F%A4%97%20model-zenteiq--ai%2FDhir--2B-yellow)](https://huggingface.co/zenteiq-ai/Dhir-2B)

A terminal chat for **[Dhīr 2B](https://huggingface.co/zenteiq-ai/Dhir-2B)**, ZenteiQ's compact model pretrained from scratch in India for English and 10 Indian languages. It runs locally, works offline, and is tuned to run well on Apple Silicon Macs with 8 GB of RAM.

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
- **Private and offline**: after the first download, nothing leaves your machine.
- **Light on small Macs**: loads weights straight onto the GPU, avoiding a swap-heavy double copy.
- **Pinned model code**: Dhīr loads Python code from Hugging Face, so the model is pinned to a reviewed commit.

## Requirements

- macOS (Apple Silicon recommended) or Linux; NVIDIA GPUs are used when available, otherwise CPU
- Python 3.10+
- About 4 GB of free RAM and 4 GB of disk for the model

## Install

```bash
pip install git+https://github.com/sriharshav/dhir2b-cli.git
```

Or from a clone:

```bash
git clone https://github.com/sriharshav/dhir2b-cli.git
cd dhir2b-cli
python3 -m venv .venv
.venv/bin/pip install -e .
```

The first run downloads the model (~3.8 GB) to `~/.cache/huggingface`.

## Usage

```bash
dhir2b                                    # chat
dhir2b --prompt "नमस्ते, आप कौन हैं?"       # chat, starting with this message
dhir2b --examples                         # Hindi Q&A, translation, document QA, tool calling
dhir2b --device cpu                       # force CPU (cuda | mps | cpu)
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

After the first download, dhir2b loads only from the local cache and makes no network calls. To guarantee it (for example on a plane):

```bash
export HF_HUB_OFFLINE=1
```

To set up a machine with no internet, copy `~/.cache/huggingface/hub/models--zenteiq-ai--Dhir-2B` and `~/.cache/huggingface/modules/transformers_modules` from a machine that has run dhir2b.

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
- For less memory and faster startup, ZenteiQ also publishes a 4-bit GGUF build ([zenteiq-ai/Dhir-2B-GGUF](https://huggingface.co/zenteiq-ai/Dhir-2B-GGUF), 1.3 GB) for its own branch of llama.cpp.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Sriharsha Vardhan. Released under the [Apache License 2.0](LICENSE).

dhir2b-cli is an independent project, not affiliated with ZenteiQ. The Dhīr 2B model is developed by [ZenteiQ AiTech Innovations](https://www.zenteiq.ai) with support from the IndiaAI Mission, MeitY, and is released separately under Apache 2.0 on [Hugging Face](https://huggingface.co/zenteiq-ai/Dhir-2B). This repository doesn't include the model's weights or code; they're downloaded from Hugging Face on first run.
