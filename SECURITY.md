# Security

## Reporting a vulnerability

Please don't open a public issue for security problems. Report them privately through GitHub's [private vulnerability reporting](https://github.com/sriharshav/dhir2b-cli/security/advisories/new). I'll reply as soon as I can.

## Notes

- **Custom model code.** Dhīr 2B ships its architecture as Python code on Hugging Face, which runs when the model loads (`trust_remote_code=True`). dhir2b-cli pins the model to a specific, reviewed commit (`REVISION` in `dhir2b.py`), so that code can't change without a release of this project.
- **Offline use.** After the first download, dhir2b-cli loads only from the local cache and makes no network calls.
- **Model output.** Like any language model, Dhīr can produce incorrect or unsafe text. Review its output before acting on it, especially code you plan to run.
