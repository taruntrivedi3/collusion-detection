"""Render report.md to report.pdf: Markdown -> HTML (math typeset by KaTeX) -> PDF (WeasyPrint).

Usage: python build_pdf.py [--katex DIR] [--html]

Needs: pip install weasyprint markdown-it-py mdit-py-plugins; Node.js with the katex package
(npm install katex, here or globally; or pass --katex DIR). --html also writes report.html for inspection.
Page: US Letter, 1-inch margins, 10.5 pt Computer Modern (KaTeX_Main) body text, page numbers.
"""
import argparse
import html
import json
import os
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

HERE = Path(__file__).resolve().parent

CSS = """
@font-face { font-family: Body; src: url(KATEX/fonts/KaTeX_Main-Regular.woff2); }
@font-face { font-family: Body; font-weight: bold; src: url(KATEX/fonts/KaTeX_Main-Bold.woff2); }
@font-face { font-family: Body; font-style: italic; src: url(KATEX/fonts/KaTeX_Main-Italic.woff2); }
@font-face { font-family: Body; font-weight: bold; font-style: italic; src: url(KATEX/fonts/KaTeX_Main-BoldItalic.woff2); }
@page { size: Letter; margin: 1in;
        @top-center { content: "AI Collusion Research Sprint, October 2026  ·  Apart Research";
                      font-family: Body; font-size: 8.5pt; color: #666; }
        @bottom-center { content: counter(page); font-family: Body; font-size: 9pt; color: #444; } }
@page :first { @top-center { content: none; } }
html { font-size: 10.5pt; }
body { font-family: Body, "KaTeX_Math", "DejaVu Sans", serif; line-height: 1.32; color: #000; margin: 0;
       text-align: justify; hyphens: auto; }
h1 { font-size: 16pt; text-align: center; line-height: 1.2; margin: 0 0 10pt; hyphens: none;
     border-top: 1pt solid #000; border-bottom: 1pt solid #000; padding: 7pt 0; }
h1 ~ p.front { text-align: center; margin: 2pt 0; }
.authors { display: flex; justify-content: center; gap: 48pt; text-align: center; margin: 4pt 0 6pt; }
p.with { text-align: center; margin: 4pt 0 2pt; }
.fn { float: footnote; font-size: 8.5pt; text-align: left; }
::footnote-call { font-size: 7pt; vertical-align: super; line-height: 0; }
::footnote-marker { font-size: 7pt; vertical-align: super; }
.abstract { margin: 4pt 0.4in 6pt; font-size: 10pt; }
.abstract h2 { text-align: center; font-size: 11pt; margin: 6pt 0 3pt; }
h2.appendix-start { break-before: page; }
h2 { font-size: 12.5pt; margin: 13pt 0 4pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 5pt 0; orphans: 3; widows: 3; }
ul, ol { margin: 4pt 0; padding-left: 16pt; }
li { margin: 1.5pt 0; text-align: left; }
li > p { margin: 1.5pt 0; }
hr { border: 0; border-top: 0.5pt solid #888; margin: 10pt 0; }
code { font-family: "Source Code Pro", "DejaVu Sans Mono", monospace; font-size: 0.85em; hyphens: none;
       overflow-wrap: anywhere; }
pre { background: #f5f5f5; padding: 6pt; font-size: 0.85em; break-inside: avoid; white-space: pre-wrap; }
a { color: #1a4fa0; text-decoration: none; }
table { border-collapse: collapse; margin: 4pt auto 9pt; font-size: 9.5pt; break-inside: avoid;
        text-align: left; hyphens: none; }
thead { border-top: 1pt solid #000; border-bottom: 0.5pt solid #000; }
tbody { border-bottom: 1pt solid #000; }
th, td { padding: 2pt 6pt; vertical-align: top; }
th { font-weight: bold; }
td:nth-child(-n+2) { white-space: nowrap; }
p.figure { text-align: center; margin: 8pt 0 2pt; break-inside: avoid; break-after: avoid; }
p.figure img { max-width: 100%; max-height: 4in; }
p.figure img[src*="fig1_cancellation"] { max-height: 2.5in; }
p.caption { font-size: 9.5pt; break-after: avoid; }
.katex-display { margin: 7pt 0; text-align: center; }
.katex { font-size: 1.05em; text-indent: 0; }
"""


def find_katex(arg):
    candidates = [arg, str(HERE / "node_modules" / "katex")]
    try:
        candidates.append(os.path.join(subprocess.check_output(["npm", "root", "-g"], text=True).strip(), "katex"))
    except (OSError, subprocess.CalledProcessError):
        pass
    for c in candidates:
        if c and (Path(c) / "dist" / "katex.min.css").exists():
            return Path(c)
    raise SystemExit("KaTeX not found: npm install katex (here or globally), or pass --katex DIR")


def render_math(items, katex_dir):
    """items: list of (tex, display). Returns KaTeX HTML for each, rendered in Node."""
    if not items:
        return []
    js = ("const katex = require(process.argv[1]); let s = ''; process.stdin.on('data', d => s += d);"
          "process.stdin.on('end', () => { const out = JSON.parse(s).map(([t, d]) => katex.renderToString(t,"
          " {displayMode: d, throwOnError: true, output: 'html'})); process.stdout.write(JSON.stringify(out)); });")
    res = subprocess.run(["node", "-e", js, str(katex_dir)], input=json.dumps(items), capture_output=True, text=True)
    if res.returncode:
        raise SystemExit("KaTeX failed:\n" + res.stderr)
    return json.loads(res.stdout)


def to_html(md_text, katex_dir):
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    # allow_space / allow_digits off: "$80 and $100" stays text, "$S_t$" is math (pandoc's rule)
    md.use(dollarmath_plugin, allow_space=False, allow_digits=False, double_inline=True)
    tokens = md.parse(md_text)

    items = []

    def is_math_fence(t):  # ```math blocks, which GitHub renders as display math (also inside lists)
        return t.type == "fence" and t.info.strip() == "math"

    def collect(toks):
        for t in toks:
            if t.type.startswith("math") or is_math_fence(t):
                items.append((t.content.strip(), t.type != "math_inline"))
            if t.children:
                collect(t.children)

    collect(tokens)
    rendered = iter(render_math(items, katex_dir))
    for kind in ("math_inline", "math_inline_double", "math_block", "math_block_label"):
        md.add_render_rule(kind, lambda self, toks, idx, opts, env: next(rendered))
    default_fence = md.renderer.rules["fence"]
    md.add_render_rule("fence", lambda self, toks, idx, opts, env: next(rendered) if is_math_fence(toks[idx])
                       else default_fence(toks, idx, opts, env))
    body = md.renderer.render(tokens, md.options, {})

    body = re.sub(r"<p>(<img [^>]*>)</p>", r'<p class="figure">\1</p>', body)
    # WeasyPrint ignores <ol start>; set the counter explicitly so a list resumed after an equation continues
    body = re.sub(r'<ol start="(\d+)">', lambda m: f'<ol start="{m[1]}" style="counter-reset: list-item {int(m[1]) - 1}">', body)
    body = re.sub(r"<p>(<strong>(?:Figure|Table) [A-Z]?\d+\.)", r'<p class="caption">\1', body)
    body = body.replace("<h2>Appendix A.", '<h2 class="appendix-start">Appendix A.', 1)
    head, sep, rest = body.partition("<hr")  # title block: everything before the first rule
    head = re.sub(r"<p>", '<p class="front">', head)
    return head + sep + rest, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "report.md"))
    ap.add_argument("--output", default=None)
    ap.add_argument("--katex", default=os.environ.get("KATEX_DIR"))
    ap.add_argument("--html", action="store_true")
    args = ap.parse_args()

    from weasyprint import HTML  # imported here so --help works without it

    src = Path(args.input).resolve()
    out = Path(args.output or src.with_suffix(".pdf")).resolve()
    katex_dir = find_katex(args.katex)
    dist = (katex_dir / "dist").as_uri()

    text = src.read_text()
    body, n_math = to_html(text, katex_dir)
    title = html.escape(next((l[2:] for l in text.splitlines() if l.startswith("# ")), src.stem))
    page = (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>'
            f'<link rel="stylesheet" href="{dist}/katex.min.css">'
            f"<style>{CSS.replace('KATEX', dist)}</style></head><body>{body}</body></html>")
    if args.html:
        out.with_suffix(".html").write_text(page)
    HTML(string=page, base_url=str(src.parent) + "/").write_pdf(out)
    print(f"wrote {out} ({n_math} math expressions)")


if __name__ == "__main__":
    main()
