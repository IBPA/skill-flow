"""Tests for SKILL.md section splitter."""

from __future__ import annotations

from skill_flow.corpus.splitter import split_skill_sections


def test_happy_path() -> None:
    md = (
        "---\n"
        "name: example\n"
        "description: Does the thing.\n"
        "---\n"
        "\n"
        "Some prose explaining how to do the thing.\n"
        "\n"
        "```python\n"
        "def hello():\n"
        "    return 1\n"
        "```\n"
        "\n"
        "More prose.\n"
    )
    yaml, prose, code = split_skill_sections(md)
    assert "name: example" in yaml
    assert "description: Does the thing." in yaml
    assert "Some prose explaining how to do the thing." in prose
    assert "More prose." in prose
    assert "```" not in prose
    assert "def hello()" in code
    assert "return 1" in code


def test_no_frontmatter() -> None:
    md = "Just some prose, no fences here."
    yaml, prose, code = split_skill_sections(md)
    assert yaml == ""
    assert prose == "Just some prose, no fences here."
    assert code == ""


def test_no_code_blocks() -> None:
    md = "---\nname: x\n---\n\nProse only.\n"
    yaml, prose, code = split_skill_sections(md)
    assert yaml == "name: x"
    assert prose == "Prose only."
    assert code == ""


def test_multiple_code_blocks_with_langs() -> None:
    md = (
        "---\nname: multi\n---\n\n"
        "Intro.\n\n"
        "```python\nprint('a')\n```\n\n"
        "Middle.\n\n"
        "```bash\nls -la\n```\n\n"
        "```\nno lang tag\n```\n"
    )
    yaml, prose, code = split_skill_sections(md)
    assert yaml == "name: multi"
    assert "Intro." in prose
    assert "Middle." in prose
    assert "```" not in prose
    # All three blocks present, in document order.
    assert code.index("print('a')") < code.index("ls -la") < code.index("no lang tag")


def test_unterminated_frontmatter_falls_back() -> None:
    # Opening ``---`` but no closing fence — treat whole input as body.
    md = "---\nname: broken\nno closing fence\n\nProse.\n"
    yaml, prose, code = split_skill_sections(md)
    assert yaml == ""
    assert "Prose." in prose
    assert code == ""


def test_empty_input() -> None:
    yaml, prose, code = split_skill_sections("")
    assert yaml == ""
    assert prose == ""
    assert code == ""


def test_only_code() -> None:
    md = "```python\nx = 1\n```\n"
    yaml, prose, code = split_skill_sections(md)
    assert yaml == ""
    assert prose == ""
    assert "x = 1" in code


def test_whitespace_in_frontmatter_fences() -> None:
    # Trailing spaces after the fence marker are tolerated.
    md = "---  \nname: ws\n---  \n\nBody.\n"
    yaml, _, _ = split_skill_sections(md)
    assert yaml == "name: ws"
