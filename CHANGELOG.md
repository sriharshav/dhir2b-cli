# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-15

### Added

- Interactive terminal chat for [Dhīr 2B](https://huggingface.co/zenteiq-ai/Dhir-2B), installable as the `dhir2b` command.
- Answers stream as rendered Markdown with syntax-highlighted code blocks.
- Ctrl+C stops an answer; Ctrl+C at the prompt quits. Typing is blocked while the model answers.
- ↑/↓ prompt history, kept across sessions in `~/.dhir2b_history`.
- `/copy`, `/copy code`, `/clear`, `/reset` and `/help` commands.
- `--prompt`, `--examples`, `--device` and `--version` options.
- Loads weights straight onto the GPU and warms up at start; runs fully offline after the first download.
- The model revision is pinned because the model loads custom code.

[0.1.0]: https://github.com/sriharshav/dhir2b-cli/releases/tag/v0.1.0
