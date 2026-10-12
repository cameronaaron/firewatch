import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_python_docstring_atop_non_python_files():
    # A leading """...""" line makes .env, YAML and Markdown files invalid.
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    offenders = [
        name
        for name in tracked
        if not name.endswith(".py")
        and (ROOT / name).is_file()
        and (ROOT / name).read_bytes().startswith(b'"""')
    ]
    assert offenders == []


def test_external_page_assets_carry_integrity_hashes():
    import re

    html = (ROOT / "app" / "static" / "index.html").read_text()
    scripts = re.findall(r"<script\b[^>]*\bsrc=\"https?://[^>]*>", html)
    # Google Fonts varies its CSS by browser, so it cannot be hashed; CSS
    # cannot run script. Every other external stylesheet is fixed content.
    styles = [
        tag
        for tag in re.findall(r"<link\b[^>]*\bhref=\"https?://[^>]*>", html)
        if 'rel="stylesheet"' in tag and "fonts.googleapis.com" not in tag
    ]
    assert scripts and styles, "expected the CDN-hosted map library"
    for tag in scripts + styles:
        assert re.search(r'integrity="sha(256|384|512)-', tag), tag


def test_env_example_names_only_real_settings():
    # Settings forbids unknown keys, so a stale example breaks startup.
    from app.core.config import Settings

    keys = {
        line.split("=", 1)[0].strip()
        for line in (ROOT / ".env.example").read_text().splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    assert keys and keys <= set(Settings.model_fields)


def test_page_links_to_its_source():
    # AGPL-3.0 section 13: users of the hosted app must be offered the source.
    html = (ROOT / "app" / "static" / "index.html").read_text()
    assert 'href="https://github.com/cameronaaron/firewatch"' in html
