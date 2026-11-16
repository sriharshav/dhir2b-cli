# Copyright 2026 Sriharsha Vardhan <sriharsha.vardhan@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for dhir2b's pure helpers. They don't need the model weights."""

import pytest

import dhir2b


@pytest.mark.parametrize(
    "code, lang",
    [
        ("def f(x):\n    return x * 2", "python"),
        ("from os import path", "python"),
        ("function add(a, b) { return a + b; }", "javascript"),
        ("const f = (a) => a + 1;", "javascript"),
        ("SELECT * FROM users LIMIT 5;", "sql"),
        ("#include <stdio.h>\nint main() { return 0; }", "c"),
        ("#include <iostream>\nint main() { std::cout << 1; }", "cpp"),
        ("public class A {}", "java"),
        ("ls -la | grep py", "bash"),
        ('{"a": 1}', "json"),
        ("just some words here", "text"),
    ],
)
def test_guess_language(code, lang):
    assert dhir2b._guess_language(code) == lang


def test_tag_fences_adds_language_only_when_missing():
    md = "hi\n```\ndef f():\n    return 1\n```\nbye\n```bash\nls\n```"
    out = dhir2b._tag_fences(md)
    assert "```python\ndef f():" in out
    assert "```bash\nls" in out
    assert out.count("```") == 4


def test_tag_fences_handles_unclosed_fence_while_streaming():
    out = dhir2b._tag_fences("Here:\n```\nimport os\nos.getcwd(")
    assert out.startswith("Here:\n```python\nimport os")


def test_code_blocks_extracts_all_blocks():
    md = "Text\n\n```python\ndef f():\n    return 1\n```\n\nand\n\n```bash\nls -la\n```\nDone."
    assert dhir2b._code_blocks(md) == "def f():\n    return 1\n\nls -la"


def test_code_blocks_empty_when_no_code():
    assert dhir2b._code_blocks("no code here") == ""


def test_parse_tool_calls():
    text = 'ok <tool_call>{"name": "get_weather", "arguments": {"city": "Chennai"}}</tool_call>'
    assert dhir2b.parse_tool_calls(text) == [{"name": "get_weather", "arguments": {"city": "Chennai"}}]


def test_parse_tool_calls_ignores_bad_json():
    assert dhir2b.parse_tool_calls("<tool_call>{not json}</tool_call>") == []


def test_copy_command_without_answer(capsys):
    dhir2b._copy_command([{"role": "system", "content": "x"}])
    assert "no answer to copy yet" in capsys.readouterr().out


def test_copy_command_copies_code(monkeypatch):
    copied = {}
    monkeypatch.setattr(dhir2b, "_copy_to_clipboard", lambda t: copied.setdefault("text", t) and "test")
    history = [{"role": "assistant", "content": "See:\n```python\nprint(1)\n```"}]
    dhir2b._copy_command(history, code_only=True)
    assert copied["text"] == "print(1)"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_truthy_accepts_on_values(value):
    assert dhir2b._truthy(value)


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe"])
def test_truthy_rejects_other_values(value):
    assert not dhir2b._truthy(value)


def test_model_is_fixed_to_dhir_2b():
    assert dhir2b.MODEL_ID == "zenteiq-ai/Dhir-2B"
    assert len(dhir2b.REVISION) == 40  # a full commit hash, not a moving branch name


def test_download_refuses_in_offline_mode(monkeypatch):
    monkeypatch.setattr(dhir2b, "OFFLINE", True)
    with pytest.raises(SystemExit, match="offline mode"):
        dhir2b.download()


def test_format_stats_full_line():
    stats = {"seconds": 4.23, "first_token_s": 0.51, "tokens": 61, "tokens_per_s": 16.44, "stopped": False}
    assert dhir2b.format_stats(stats, "offline", "mps") == (
        "4.2 s · first token 0.5 s · 61 tokens · 16.4 tok/s · offline · mps"
    )


def test_format_stats_single_token_and_no_mode():
    stats = {"seconds": 0.3, "first_token_s": 0.3, "tokens": 1, "tokens_per_s": None, "stopped": True}
    assert dhir2b.format_stats(stats) == "0.3 s · first token 0.3 s · 1 token"


def test_token_clock_records_first_token_only():
    clock = dhir2b._TokenClock()
    assert clock(None, None) is False
    first = clock.first
    assert first is not None and first >= clock.start
    clock(None, None)
    assert clock.first == first


@pytest.mark.parametrize("n, text", [(847, "847 bytes"), (1536, "1.5 KB"), (4.1 * 1024**3, "4.1 GB")])
def test_human_size(n, text):
    assert dhir2b._human_size(n) == text


class _FakeRev:
    def __init__(self, commit_hash, size):
        self.commit_hash, self.size_on_disk = commit_hash, size


class _FakeStrategy:
    expected_freed_size = 2048

    def __init__(self, hashes):
        self.hashes = hashes
        self.executed = False

    def execute(self):
        self.executed = True


class _FakeCache:
    def __init__(self):
        self.strategy = None

    def delete_revisions(self, *hashes):
        self.strategy = _FakeStrategy(hashes)
        return self.strategy


def test_clean_cache_removes_only_old_revisions(monkeypatch, capsys):
    cache = _FakeCache()
    revs = [_FakeRev(dhir2b.REVISION, 4 * 1024**3), _FakeRev("a" * 40, 1000), _FakeRev("b" * 40, 2000)]
    monkeypatch.setattr(dhir2b, "_cached_revisions", lambda: (cache, revs))
    dhir2b.clean_cache()
    assert cache.strategy.hashes == ("a" * 40, "b" * 40)  # never the pinned revision
    assert cache.strategy.executed
    assert "Freed 2.0 KB" in capsys.readouterr().out


def test_clean_cache_with_nothing_old(monkeypatch, capsys):
    monkeypatch.setattr(dhir2b, "_cached_revisions", lambda: (_FakeCache(), [_FakeRev(dhir2b.REVISION, 1)]))
    dhir2b.clean_cache()
    assert "Nothing to clean" in capsys.readouterr().out


def test_model_info_reports_newer_upstream(monkeypatch, capsys):
    import huggingface_hub

    class FakeApi:
        def model_info(self, repo_id):
            return type("Info", (), {"sha": "c" * 40})()

    monkeypatch.setattr(dhir2b, "OFFLINE", False)
    monkeypatch.setattr(dhir2b, "_cached_revisions", lambda: (None, [_FakeRev(dhir2b.REVISION, 4 * 1024**3)]))
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    dhir2b.model_info()
    out = capsys.readouterr().out
    assert "newer revision (ccccccc)" in out
    assert "4.0 GB" in out and "(in use)" in out


def test_model_info_offline_skips_network(monkeypatch, capsys):
    monkeypatch.setattr(dhir2b, "OFFLINE", True)
    monkeypatch.setattr(dhir2b, "_cached_revisions", lambda: (None, []))
    dhir2b.model_info()
    out = capsys.readouterr().out
    assert "not downloaded yet" in out and "not checked (offline mode)" in out
