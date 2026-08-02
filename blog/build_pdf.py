#!/usr/bin/env python3
"""Render a MAD blog post markdown file to PDF.

Strips the MyST/Jekyll YAML frontmatter, renders the body with Python-Markdown
(tables + fenced code + syntax highlighting), and lays it out with WeasyPrint.
Frontmatter title/author/date are promoted into a cover header, and image paths
resolve relative to the markdown file's directory.

Usage:
    python3 blog/build_pdf.py blog/kimi-k3-day0-mad-automation.md
    python3 blog/build_pdf.py blog/foo.md -o /tmp/foo.pdf
"""
import argparse
import os
import re
import sys

import markdown
import yaml
from weasyprint import HTML

# Rendered at ~11pt on Letter; the code font is sized down so the widest fenced
# block in the K3 post (the --additional-context JSON) does not wrap mid-token.
CSS = """
@page {
    size: Letter;
    margin: 0.85in 0.75in 0.9in 0.75in;
    @bottom-center {
        content: counter(page);
        font-family: "DejaVu Sans", sans-serif;
        font-size: 8.5pt;
        color: #777;
    }
}
body {
    font-family: "DejaVu Serif", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
}
h1, h2, h3, h4 {
    font-family: "DejaVu Sans", Helvetica, sans-serif;
    color: #0b0b0b;
    line-height: 1.25;
    page-break-after: avoid;
}
h1 { font-size: 19pt; margin: 0 0 0.15em; }
h2 { font-size: 14pt; margin: 1.5em 0 0.4em; border-bottom: 1px solid #ddd; padding-bottom: 0.15em; }
h3 { font-size: 11.5pt; margin: 1.2em 0 0.3em; }
h4 { font-size: 10.5pt; margin: 1em 0 0.3em; }
p { margin: 0.5em 0; orphans: 2; widows: 2; }
a { color: #0b5cad; text-decoration: none; }
code, pre {
    font-family: "DejaVu Sans Mono", monospace;
}
code { font-size: 8.2pt; background: #f4f4f4; padding: 0.08em 0.25em; border-radius: 2px; }
pre {
    font-size: 7.8pt;
    line-height: 1.35;
    background: #f7f7f7;
    border: 1px solid #e2e2e2;
    border-radius: 3px;
    padding: 0.55em 0.7em;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.7em 0;
    font-size: 8.4pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #ccc;
    padding: 0.32em 0.45em;
    text-align: left;
    vertical-align: top;
}
th { background: #ededed; font-family: "DejaVu Sans", sans-serif; font-weight: bold; }
img { max-width: 100%; display: block; margin: 0.8em auto; }
blockquote {
    margin: 0.7em 0 0.7em 1em;
    padding-left: 0.8em;
    border-left: 3px solid #ccc;
    color: #444;
    font-style: italic;
}
hr { border: none; border-top: 1px solid #ddd; margin: 1.4em 0; }
ul, ol { margin: 0.45em 0; padding-left: 1.4em; }
li { margin: 0.22em 0; }
.cover-meta {
    font-family: "DejaVu Sans", sans-serif;
    font-size: 9pt;
    color: #555;
    margin: 0 0 1.4em;
    padding-bottom: 0.6em;
    border-bottom: 2px solid #d0d0d0;
}
/* Figure/table captions: the posts write these as a plain paragraph starting
   with "Figure N:" / "Table N:" immediately after the image or table. */
p.caption {
    font-size: 8.8pt;
    color: #555;
    font-style: italic;
    margin-top: 0.15em;
}
"""

CAPTION_RE = re.compile(r"^<p>((?:Figure|Table)\s+\d+[:.].*?)</p>", re.S | re.M)


def split_frontmatter(text):
    """Return (frontmatter_dict, body). Frontmatter is optional."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].lstrip("\n")


def strip_html_comments(text):
    """Drop <!-- ... --> blocks (license header, author TODOs) from the body."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def tag_captions(html):
    """Add class="caption" to paragraphs that begin with Figure N:/Table N:."""
    return CAPTION_RE.sub(r'<p class="caption">\1</p>', html)


def build(md_path, out_path=None):
    md_path = os.path.abspath(md_path)
    if out_path is None:
        out_path = os.path.splitext(md_path)[0] + ".pdf"

    with open(md_path, encoding="utf-8") as f:
        raw = f.read()

    meta, body = split_frontmatter(raw)
    body = strip_html_comments(body)

    html_body = markdown.markdown(
        body,
        extensions=["tables", "fenced_code", "codehilite", "attr_list", "sane_lists"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )
    html_body = tag_captions(html_body)

    header_bits = []
    if meta.get("author"):
        header_bits.append(str(meta["author"]))
    if meta.get("date"):
        header_bits.append(str(meta["date"]))
    header = ""
    if header_bits:
        header = '<p class="cover-meta">{}</p>'.format(" &middot; ".join(header_bits))

    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{meta.get('blog_title', os.path.basename(md_path))}</title>"
        f"<style>{CSS}</style></head><body>{header}{html_body}</body></html>"
    )

    # base_url anchors relative image paths (images/foo.png) to the md's directory.
    HTML(string=doc, base_url=os.path.dirname(md_path) + os.sep).write_pdf(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("markdown", help="path to the blog post .md")
    ap.add_argument("-o", "--output", help="output PDF path (default: alongside the .md)")
    args = ap.parse_args()

    if not os.path.isfile(args.markdown):
        sys.exit(f"error: no such file: {args.markdown}")

    out = build(args.markdown, args.output)
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")


if __name__ == "__main__":
    main()
