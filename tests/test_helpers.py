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
