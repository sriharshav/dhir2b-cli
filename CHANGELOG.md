# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `dhir2b --purge` deletes everything dhir2b downloaded or saved: the model (through huggingface_hub, so files in its shared blob store are really removed), its cached code and prompt history. It lists what it will delete with sizes and asks first (`--yes` skips the question), then prints the right uninstall command.
- README: an Uninstalling section covering a complete removal.
- README: an "About Dhīr 2B" section: what the model is, its characteristics and capabilities, and its limits, with links to learn more.

### Changed

- The chat prompt is now `<user>@dhir2b>`, so it's clearly dhir2b's prompt and not the shell's.

## [0.2.3] - 2026-11-15

### Added

- `dhir2b --update` upgrades dhir2b to the latest GitHub release, the same way it was installed. pipx installs are reinstalled with the new tag (which `pipx upgrade` can't do), pip-in-venv installs are upgraded in place, and development clones are told to `git pull`. It reports when you're up to date and refuses in offline mode.

## [0.2.2] - 2026-11-15

### Added

- `--model-info` shows the model revision in use, what's downloaded, and whether ZenteiQ has published a newer revision.
- `--clean-cache` deletes old downloaded Dhīr 2B revisions and never the one in use. (Hugging Face's `hf cache prune` would delete the pinned model too.)

### Fixed

- The README said `pipx upgrade dhir2b-cli` upgrades dhir2b. It doesn't for tag-based installs; the new Upgrading section gives the working command and explains how model updates arrive.

## [0.2.1] - 2026-11-15

### Added

- After each answer, a line shows the response time, time to first token, token count, speed (tokens/s), mode and device.
- The startup banner shows the mode (`online` / `offline`) and device.
- `chat(..., stats={})` returns timing, and `format_stats()` formats it.

### Changed

- README recommends installing with pipx, and warns that plain `pip install` into a shared Python can break other tools (dhir2b's dependencies pull in pydantic 2).

## [0.2.0] - 2026-11-15

### Added

- Offline mode: `--offline` for one run, or `DHIR2B_OFFLINE=1` to make it the default. It guarantees no network use, and if the model isn't downloaded yet it says so and exits instead of trying the network.
- `--download` fetches the model in advance for offline use.

### Changed

- The startup line shows `(offline)` when offline mode is on.
- Documentation states that dhir2b supports Dhīr 2B only.
- Builds with hatchling instead of setuptools.

### Fixed

- `pip install git+https://github.com/sriharshav/dhir2b-cli.git` failed with Homebrew's Python (`project.license must be valid exactly by one definition`). Homebrew's old setuptools/packaging couldn't read the `license = "Apache-2.0"` string form, so the licence now uses the table form older tools understand. A CI job builds against those old versions to keep it working.

## [0.1.0] - 2026-11-15 [YANKED]

Withdrawn: installing it from git failed on Homebrew's Python (fixed in 0.2.0). Its GitHub release and tag were removed; use 0.2.0 instead.

### Added

- Interactive terminal chat for [Dhīr 2B](https://huggingface.co/zenteiq-ai/Dhir-2B), installable as the `dhir2b` command.
- Answers stream as rendered Markdown with syntax-highlighted code blocks.
- Ctrl+C stops an answer; Ctrl+C at the prompt quits. Typing is blocked while the model answers.
- ↑/↓ prompt history, kept across sessions in `~/.dhir2b_history`.
- `/copy`, `/copy code`, `/clear`, `/reset` and `/help` commands.
- `--prompt`, `--examples`, `--device` and `--version` options.
- Loads weights straight onto the GPU and warms up at start; runs fully offline after the first download.
- The model revision is pinned because the model loads custom code.

[Unreleased]: https://github.com/sriharshav/dhir2b-cli/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/sriharshav/dhir2b-cli/releases/tag/v0.2.3
[0.2.2]: https://github.com/sriharshav/dhir2b-cli/releases/tag/v0.2.2
[0.2.1]: https://github.com/sriharshav/dhir2b-cli/releases/tag/v0.2.1
[0.2.0]: https://github.com/sriharshav/dhir2b-cli/releases/tag/v0.2.0
