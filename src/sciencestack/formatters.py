"""Compact output formatters for CLI.

Design principle: minimize tokens while preserving all information an agent needs.
Strip image URLs, pagination metadata, version strings, UUIDs.
"""

from __future__ import annotations

import json
import re
import textwrap


def format_json(data: dict) -> str:
    """Full JSON passthrough."""
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def format_search(data: dict) -> str:
    papers = data.get("data", [])
    if not papers:
        return "No results found."
    lines = []
    for p in papers:
        year = p.get("published", "")[:4]
        cites = p.get("citationCount")
        cite_str = f"  [{cites} cites]" if cites else ""
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        lines.append(f"{p.get('arxivId', '?')}  {p.get('title', '?')}  ({year}){cite_str}")
        lines.append(f"  {authors}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def format_overview(data: dict, filter_key: str | None = None) -> str:
    """Format paper overview. filter_key: sections|equations|tables|figures|theorems|algorithms"""
    d = data.get("data", data)

    if filter_key == "sections":
        return _format_toc(d)
    elif filter_key == "equations":
        return _format_equation_list(d)
    elif filter_key == "tables":
        return _format_table_list(d)
    elif filter_key == "figures":
        return _format_figure_list(d)
    elif filter_key == "theorems":
        return _format_mathenv_list(d)
    elif filter_key == "algorithms":
        return _format_algorithm_list(d)
    else:
        return _format_overview_compact(d)


def _format_overview_compact(d: dict) -> str:
    lines = []
    title = d.get("title", "?")
    arxiv = d.get("arxivId", "?")
    year = d.get("published", "")[:4]

    # Authors
    authors = d.get("authors", [])
    author_names = [a.get("name", "?") if isinstance(a, dict) else str(a) for a in authors[:5]]
    if len(authors) > 5:
        author_names.append("et al.")

    lines.append(f"{title}")
    lines.append(f"{arxiv} ({year})  {', '.join(author_names)}")
    lines.append("")

    # Summary
    summary = d.get("aiSummary")
    if summary:
        lines.append(summary)
        lines.append("")

    # Counts
    counts = []
    for key, label in [
        ("toc", "sections"),
        ("figures", "figures"),
        ("tables", "tables"),
        ("mathEnvs", "theorems/proofs"),
        ("algorithms", "algorithms"),
    ]:
        items = d.get(key, [])
        if items:
            counts.append(f"{len(items)} {label}")
    if counts:
        lines.append("Contains: " + ", ".join(counts))

    return "\n".join(lines)


def _format_toc(d: dict) -> str:
    toc = d.get("toc", [])
    if not toc:
        return "No sections found."
    lines = [f"{d.get('title', '?')} ({d.get('arxivId', '?')}) — Table of Contents", ""]
    for entry in toc:
        indent = "  " * (entry.get("depth", 1) - 1)
        lines.append(f"{indent}{entry.get('nodeId', '?')}  {entry.get('title', '?')}")
    return "\n".join(lines)


def _format_equation_list(d: dict) -> str:
    """Equations aren't in the overview response. Direct user to nodes command."""
    return (
        f"Equations are not included in overview. To fetch equations:\n\n"
        f"  sciencestack nodes {d.get('arxivId', '<id>')} --type equation --format latex"
    )


def _format_table_list(d: dict) -> str:
    tables = d.get("tables", [])
    if not tables:
        return "No tables found."
    lines = [f"{d.get('title', '?')} ({d.get('arxivId', '?')}) — {len(tables)} tables", ""]
    for t in tables:
        caption = t.get("caption", "")
        # Truncate long captions
        if len(caption) > 100:
            caption = caption[:97] + "..."
        lines.append(f"{t.get('nodeId', '?')}  {caption}")
    return "\n".join(lines)


def _format_figure_list(d: dict) -> str:
    figs = d.get("figures", [])
    if not figs:
        return "No figures found."
    lines = [f"{d.get('title', '?')} ({d.get('arxivId', '?')}) — {len(figs)} figures", ""]
    for f in figs:
        caption = f.get("caption", "")
        if len(caption) > 100:
            caption = caption[:97] + "..."
        # Deliberately skip imageUrls — that's the token savings
        lines.append(f"{f.get('nodeId', '?')}  {caption}")
    return "\n".join(lines)


def _format_mathenv_list(d: dict) -> str:
    envs = d.get("mathEnvs", [])
    if not envs:
        return "No theorems/proofs found."
    lines = [f"{d.get('title', '?')} ({d.get('arxivId', '?')}) — {len(envs)} math environments", ""]
    for e in envs:
        num = e.get("number", "")
        title = e.get("title", "")
        type_str = e.get("type", "?")
        parts = [e.get("nodeId", "?"), type_str]
        if num:
            parts.append(f"#{num}")
        if title:
            parts.append(title)
        lines.append("  ".join(parts))
    return "\n".join(lines)


def _format_algorithm_list(d: dict) -> str:
    algs = d.get("algorithms", [])
    if not algs:
        return "No algorithms found."
    lines = [f"{d.get('title', '?')} ({d.get('arxivId', '?')}) — {len(algs)} algorithms", ""]
    for a in algs:
        caption = a.get("caption", "")
        lines.append(f"{a.get('nodeId', '?')}  {caption}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def format_nodes(data: dict, api_format: str) -> str:
    """Format node content. Handles both string data (markdown/latex) and array data (raw)."""
    content = data.get("data", "")

    # When format=markdown or latex, the API returns a string directly
    if isinstance(content, str):
        if api_format == "latex":
            return _format_latex_equations(content)
        return content

    # When format=raw, it's an array of node objects
    if isinstance(content, list):
        lines = []
        for node in content:
            node_id = node.get("nodeId", "?")
            node_type = node.get("type", "?")
            lines.append(f"[{node_id}] ({node_type})")
            node_content = node.get("content")
            if isinstance(node_content, str):
                lines.append(node_content)
            elif isinstance(node_content, (list, dict)):
                lines.append(json.dumps(node_content, indent=2))
            lines.append("")
        return "\n".join(lines)

    return str(content)


def _format_latex_equations(text: str) -> str:
    """Parse a blob of LaTeX equations into labeled blocks.

    The API returns all equations concatenated. We split on \\begin{equation}
    and label each with its \\label if present, or a sequential ID.
    """
    # Try to split on equation environments
    blocks = re.split(r'(?=\\begin\{equation\})', text)
    blocks = [b.strip() for b in blocks if b.strip()]

    if len(blocks) <= 1:
        # Might be $$ delimited or a single block
        return text

    lines = []
    eq_num = 0
    for block in blocks:
        eq_num += 1
        # Extract label if present
        label_match = re.search(r'\\label\{([^}]+)\}', block)
        label = label_match.group(1) if label_match else f"eq:{eq_num}"

        # Strip environment wrappers and label for compact output
        inner = block
        inner = re.sub(r'\\begin\{equation\}', '', inner)
        inner = re.sub(r'\\end\{equation\}', '', inner)
        inner = re.sub(r'\\label\{[^}]+\}', '', inner)
        inner = inner.strip()

        if inner:
            lines.append(f"[{label}]")
            lines.append(inner)
            lines.append("")

    return "\n".join(lines) if lines else text


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def format_references(data: dict) -> str:
    refs = data.get("data", [])
    if not refs:
        return "No references found."
    lines = []
    for r in refs:
        cite_key = r.get("citeKey", "?")
        enrichment = r.get("enrichment", {}) or {}
        s2 = enrichment.get("semanticScholar") or {}
        title = s2.get("title", "")
        year = (s2.get("publicationDate") or "")[:4]
        arxiv = (enrichment.get("externalIds") or {}).get("ArXiv", "")
        # Also check the direct arxivId field
        if not arxiv:
            arxiv = r.get("arxivId", "")

        parts = [cite_key]
        if title:
            parts.append(title)
        if year:
            parts.append(f"({year})")
        if arxiv:
            parts.append(f"[{arxiv}]")
        lines.append("  ".join(parts))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def format_citations(data: dict) -> str:
    cites = data.get("data", [])
    if not cites:
        return "No citations found."
    lines = [f"{len(cites)} citing papers", ""]
    for c in cites:
        arxiv = c.get("arxivId", "?")
        title = c.get("title", "?")
        year = (c.get("published") or "")[:4]
        cite_count = c.get("citationCount", 0)
        lines.append(f"{arxiv}  {title}  ({year})  [{cite_count} cites]")
    return "\n".join(lines)
