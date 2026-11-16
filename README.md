# Dhīr 2B – getting started

[zenteiq-ai/Dhir-2B](https://huggingface.co/zenteiq-ai/Dhir-2B): a 2B model from ZenteiQ, pretrained from scratch, for English + 10 Indian languages (16k context, Apache 2.0).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads the weights (~3.8 GB) to `~/.cache/huggingface`. After that, everything runs offline.

## Run

```bash
.venv/bin/python dhir2b.py                              # chat
.venv/bin/python dhir2b.py --prompt "नमस्ते, आप कौन हैं?"   # chat, starting with this message
.venv/bin/python dhir2b.py --examples                   # Hindi Q&A, translation, document QA, tool calling
.venv/bin/python dhir2b.py --device cpu                 # force CPU
```

In chat:
- **Ctrl+C while it's answering** stops that answer.
- **Ctrl+C at the `you>` prompt** (or Ctrl+D) quits.
- `/clear` clears the screen and starts a new conversation.
- `/reset` starts a new conversation but keeps the screen.
- `/copy` copies the last answer (raw Markdown) to the clipboard; `/copy code` copies just its code blocks.
- `/help` lists the commands.

- **↑ / ↓** step through earlier prompts, including those from previous sessions (saved in `.dhir_history`). ←/→, Ctrl+A/E and Ctrl+U edit the line.

The prompt shows your login name (e.g. `sriharshavardhan>`). Answers stream as rendered Markdown (headings, bold, lists) with syntax-highlighted code blocks (Monokai theme; change `CODE_THEME` in `dhir2b.py` to any Pygments style). A short system prompt (`SYSTEM_PROMPT`) asks Dhir to put code in fenced, language-tagged blocks; without it, Dhir often writes code as plain text. While it's answering, a spinner shows until the first word arrives. Typing is blocked, and anything you type is discarded. When output is piped to a file, it's written as plain text.

## Offline

Once the weights are downloaded, the script loads only from the local cache and makes no network calls. To guarantee that (for example on a plane), set:

```bash
export HF_HUB_OFFLINE=1
```

To move it to an offline machine, copy `~/.cache/huggingface/hub/models--zenteiq-ai--Dhir-2B` and `~/.cache/huggingface/modules/transformers_modules` along with this folder.

## Startup time

On an 8 GB M1 the model reaches the prompt in roughly 20–60 s: ~5 s to load, plus a one-time warm-up while macOS pages the weights into GPU memory. That part is slower when RAM is tight, so close heavy apps for faster starts. After that, it generates ~11 tokens/s, and the chat stays loaded until you quit.

## Use from your own code

```python
from dhir2b import load, chat

tok, model = load()
print(chat(tok, model, [{"role": "user", "content": "Summarise in Kannada: ..."}]))
```

## Notes

- The model ships custom architecture code, so it loads with `trust_remote_code=True`. `dhir2b.py` pins the repo to a reviewed commit (`REVISION`); check `modeling_dhir.py` again before you bump it.
- Sampling defaults come from the repo's `generation_config.json`. Pass `do_sample=False` for deterministic output (translation, QA).
- Thinking mode is not supported. Answers are most reliable when you supply the source text (RAG).
- For less memory and much faster startup, use the 4-bit GGUF build (`zenteiq-ai/Dhir-2B-GGUF`, 1.3 GB). It needs ZenteiQ's `dhir-support` branch of llama.cpp until upstream support lands.

## License

Copyright 2026 Sriharsha Vardhan. This code is released under the [Apache License 2.0](LICENSE).

The Dhīr 2B model is developed by [ZenteiQ AiTech Innovations](https://www.zenteiq.ai) and released separately under Apache 2.0 on [Hugging Face](https://huggingface.co/zenteiq-ai/Dhir-2B). This repository doesn't include its weights or code; they're downloaded from Hugging Face on first run.
