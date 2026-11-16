# Contributing

Thanks for your interest in dhir2b-cli! Bug reports, ideas and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/sriharshav/dhir2b-cli.git
cd dhir2b-cli
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]" pytest
```

The first `dhir2b` run downloads the model (~3.8 GB).

## Before opening a pull request

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q tests
```

CI runs the same checks on Linux and macOS. It doesn't download the model, so if you change how the chat behaves, please also try it by hand in a terminal (`dhir2b`) and say what you tested in the PR.

## Guidelines

- Keep it a single, readable file (`dhir2b.py`) unless there's a good reason not to.
- Add a test in `tests/` for helpers that don't need the model.
- Update `README.md` and `CHANGELOG.md` when behaviour changes.
- The model runs custom code from Hugging Face, so it's pinned to a reviewed revision (`REVISION`). If you bump it, read the new `modeling_dhir.py` and `configuration_dhir.py` first, and say in the PR that you did.

## Licence

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).
