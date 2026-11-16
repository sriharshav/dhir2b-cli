# dhir2b-cli

[![CI](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/sriharshav/dhir2b-cli/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
[![Model: Dhīr 2B](https://img.shields.io/badge/%F0%9F%A4%97%20model-zenteiq--ai%2FDhir--2B-yellow)](https://huggingface.co/zenteiq-ai/Dhir-2B)

A terminal chat for **[Dhīr 2B](https://huggingface.co/zenteiq-ai/Dhir-2B)**, ZenteiQ's compact model pretrained from scratch in India for English and 10 Indian languages. It runs locally, works offline, and is tuned to run well on Apple Silicon Macs with 8 GB of RAM.

dhir2b is built for Dhīr 2B only. The model, its revision, chat format and sampling settings are fixed, and there's no option to load another model.

```
sriharshavardhan@dhir2b> Write a Python function to check if a number is prime
dhir>
 def is_prime(n):
     if n <= 1:
         return False
     ...
```

## About Dhīr 2B

Dhīr 2B is a small language model that [ZenteiQ](https://www.zenteiq.ai) designed and pretrained from scratch in India, under the IndiaAI Mission. It's small enough to run on a laptop, offline, while staying strong at Indian languages.

| | |
|---|---|
| **Size** | 2.03B parameters: 3.8 GB in BF16, about 1.1 GB at 4-bit. Large models often have 70B or more, so this one runs on ordinary laptops. |
| **Dense** | All parameters are used for every token, unlike [Mixture-of-Experts](https://huggingface.co/blog/moe) models that activate only part of the network. That keeps memory use simple and predictable. (Its sibling, [Dhīr 5B](https://huggingface.co/zenteiq-ai/Dhir-5B-A1.7B), is a Mixture-of-Experts model.) |
| **Languages** | English plus 10 Indian languages: Hindi, Bengali, Gujarati, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil and Telugu. |
| **Indic tokenizer** | Its own [tokenizer](https://huggingface.co/learn/llm-course/chapter6/1), with 131K tokens designed for Indian scripts. Indian-language text uses fewer tokens, so it's faster and fits more into the context window. |
| **Context** | 16K tokens, enough for long documents in a single prompt. |
| **Text only** | It reads and writes text, code and structured data such as JSON, but not images or audio. |
| **Instruction-tuned** | Trained to follow instructions with supervised fine-tuning, then [GRPO](https://arxiv.org/abs/2402.03300) reinforcement learning. |
| **Tool calling** | It can reply with a structured call to your tool or API instead of plain text. See `examples()` in [dhir2b.py](dhir2b.py) and the [Transformers guide](https://huggingface.co/docs/transformers/en/chat_extras). |
| **Efficient attention** | [Grouped-query attention](https://arxiv.org/abs/2305.13245) (fewer key/value heads) and [rotary position embeddings](https://arxiv.org/abs/2104.09864) keep memory low in long conversations. |
| **Licence** | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), so commercial use is allowed. |

**Good at:** translation between Indian languages and English, answering questions about a document you provide (the "reader" in [RAG](https://huggingface.co/learn/llm-course/chapter1/4)), summarising, sentiment, and short Python functions. ZenteiQ reports it as best in class among open models up to 3.2B on translation and Indian-language document QA. Those figures come from ZenteiQ's own evaluation; see the [model card](https://huggingface.co/zenteiq-ai/Dhir-2B).

**Keep in mind:** it has no step-by-step "thinking" mode, its maths and coding are basic, and like any small model it can state wrong facts confidently. It's most reliable when you give it the source text. Running offline keeps your data on your machine, but it's still worth checking important answers.

**Learn more:** [model card and benchmarks](https://huggingface.co/zenteiq-ai/Dhir-2B) · [AIKosh listing](https://aikosh.indiaai.gov.in/home/models/details/dhir_2b.html) · [ZenteiQ](https://www.zenteiq.ai)

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
pipx install git+https://github.com/sriharshav/dhir2b-cli.git@v0.2.3
```

To upgrade later, run `dhir2b --update` (see [Upgrading](#upgrading)). To remove it, see [Uninstalling](#uninstalling).

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
dhir2b --model-info                       # model revision in use, what's downloaded, newer one available?
dhir2b --clean-cache                      # delete old downloaded model revisions
dhir2b --update                           # upgrade dhir2b to the latest release
dhir2b --purge                            # delete the model, cached model code and history (asks first)
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

The prompt is your login name followed by `@dhir2b`, for example `sriharshavardhan@dhir2b>`, so it's clearly dhir2b's prompt and not your shell's. The startup banner shows the mode (`online` or `offline`) and device, and after each answer a dim line shows how long it took, for example:

```
⏱ 4.2 s · first token 0.5 s · 61 tokens · 16.4 tok/s · offline · mps
```

 Prompt history is saved in `~/.dhir2b_history`; set `DHIR2B_HISTORY` to use another file. When output is piped to a file, answers are written as plain text.

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

## Upgrading

### Upgrading dhir2b

```bash
dhir2b --update
```

It checks the [Releases page](https://github.com/sriharshav/dhir2b-cli/releases) for a newer version and upgrades the same way you installed it (pipx, or pip in a virtual environment). If you're up to date, it says so and changes nothing. Your downloaded model and prompt history are kept. It needs the network, so it won't run in offline mode.

`dhir2b --version` shows the version you have.

**Upgrading by hand.** dhir2b 0.2.2 and earlier don't have `--update`, so upgrade from those once with:

```bash
pipx install --force git+https://github.com/sriharshav/dhir2b-cli.git@v0.2.3
```

> `pipx upgrade dhir2b-cli` doesn't work here. pipx remembers the exact tag you installed and reinstalls that same version. dhir2b isn't on PyPI, so there's no package index for pipx to check. `dhir2b --update` handles this for you.

**From a clone,** `--update` tells you to run `git pull` instead of changing your working copy. An editable install (`pip install -e .`) picks up the new code straight away; rerun `.venv/bin/pip install -e .` if the release notes mention new dependencies.

### Model updates

dhir2b uses one specific, reviewed revision of Dhīr 2B (see [Notes](#notes)), so a new model from ZenteiQ doesn't reach you automatically. Instead:

1. **Check:** `dhir2b --model-info` shows the revision in use and whether ZenteiQ has published a newer one.
2. **Wait for a dhir2b release.** After the new model code is reviewed, a new dhir2b version moves to it, and the release notes say so.
3. **Upgrade dhir2b** with `dhir2b --update`. The next online run downloads the new model (~3.8 GB). To do it in advance, for example before going offline, run `dhir2b --download`.
4. **Free the space** used by the old model:

   ```bash
   dhir2b --clean-cache
   ```

   It deletes only Dhīr 2B revisions that your dhir2b version no longer uses.

> Don't use `hf cache prune` for this. Hugging Face's tool treats dhir2b's pinned model as unreferenced and deletes the model in use too, which means a 3.8 GB download again.

## Uninstalling

Removing dhir2b completely takes two commands, run in this order:

```bash
dhir2b --purge              # 1. delete the model (~3.8 GB), its cached code and your prompt history
pipx uninstall dhir2b-cli   # 2. remove the program
```

`--purge` lists exactly what it will delete, with sizes, and asks before deleting anything (add `--yes` to skip the question). It removes the model through Hugging Face's own cache API, so the real files are deleted, including any in Hugging Face's shared blob store, not just the links to them. Your other Hugging Face models aren't touched. At the end it prints the right uninstall command for how you installed dhir2b.

Then, if they apply to you:

- **Offline setting:** if you added `export DHIR2B_OFFLINE=1` to your shell profile (for example `~/.zshrc`), remove that line.
- **pipx itself:** if you installed pipx only for dhir2b, remove it with `brew uninstall pipx`, and delete the `export PATH="$PATH:$HOME/.local/bin"` line that `pipx ensurepath` added to your shell profile.
- **A clone:** delete the cloned folder, which includes its `.venv`.

**Already uninstalled the program first?** Then delete these by hand. The paths are the defaults; they move if you set `HF_HOME`:

```bash
rm -rf ~/.cache/huggingface/hub/models--zenteiq-ai--Dhir-2B
rm -rf ~/.cache/huggingface/hub/.locks/models--zenteiq-ai--Dhir-2B
rm -rf ~/.cache/huggingface/modules/transformers_modules/zenteiq_hyphen_ai
rm -f ~/.dhir2b_history
```

Newer versions of huggingface_hub keep large model files in a shared store (`~/.cache/huggingface/hub/blobs/`), so deleting the model folder by hand can leave the 3.8 GB there. If no other Hugging Face tools use your cache, deleting `~/.cache/huggingface` entirely is the simplest way to be sure. Otherwise, reinstall dhir2b and run `dhir2b --purge`.

## Performance

On an 8 GB M1 MacBook, dhir2b reaches the prompt in roughly 20–60 s: about 5 s to load, plus a one-time warm-up while macOS pages the weights into GPU memory. The warm-up is slower when RAM is tight, so close heavy apps for faster starts. After that, it generates about 11 tokens/s, and the model stays loaded until you quit.

## Use from Python

```python
from dhir2b import load, chat

tok, model = load()
print(chat(tok, model, [{"role": "user", "content": "Translate to Kannada: Good morning"}]))
```

Pass `stats={}` to `chat()` to get timing back (`seconds`, `first_token_s`, `tokens`, `tokens_per_s`, `stopped`), and `format_stats()` turns it into the line shown in the chat. `chat()` also accepts `tools=[...]` for function calling; `parse_tool_calls()` extracts the calls. See `examples()` in [dhir2b.py](dhir2b.py).

## Customising

Constants at the top of [dhir2b.py](dhir2b.py):

- `CODE_THEME`: any [Pygments style](https://pygments.org/styles/) for code blocks (default `monokai`).
- `SYSTEM_PROMPT`: asks Dhīr to put code in fenced, language-tagged blocks. Without it, Dhīr often writes code as plain text.
- `GEN_KWARGS`: sampling settings, taken from the model's `generation_config.json`.

## Notes

- The model ships custom architecture code, so it loads with `trust_remote_code=True`. `REVISION` pins it to a reviewed commit, so the code that runs can't change without a dhir2b release. See [Model updates](#model-updates).
- Thinking mode isn't supported by this model release. Answers are most reliable when you supply the source text (RAG).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

Copyright 2026 Sriharsha Vardhan. Released under the [Apache License 2.0](LICENSE).

dhir2b-cli is an independent project, not affiliated with ZenteiQ. The Dhīr 2B model is developed by [ZenteiQ AiTech Innovations](https://www.zenteiq.ai) with support from the IndiaAI Mission, MeitY, and is released separately under Apache 2.0 on [Hugging Face](https://huggingface.co/zenteiq-ai/Dhir-2B). This repository doesn't include the model's weights or code; they're downloaded from Hugging Face on first run.
