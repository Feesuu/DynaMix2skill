from __future__ import annotations

import fnmatch
import html
import json
import os
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_READ_CHARS = 4000
MAX_GREP_MATCHES = 20
MAX_GLOB_MATCHES = 50
MAX_ORACLE_PAGE_CHARS = 24000
MAX_ORACLE_CONTEXT_CHARS = 80000
_MONTH_TO_NUMBER = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find candidate local document files by filename or relative-path glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a local text document excerpt by path and line window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a local text document for a literal pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
]


def resolve_docs_roots(data_dirs: list[str] | tuple[str, ...] | str | None) -> list[str]:
    candidates: list[str] = []
    if data_dirs:
        if isinstance(data_dirs, str):
            candidates.extend(part.strip() for chunk in data_dirs.split(os.pathsep) for part in chunk.split(",") if part.strip())
        else:
            candidates.extend(str(item).strip() for item in data_dirs if str(item).strip())
    env_value = os.environ.get("OFFICEQA_DOCS_DIR", "").strip()
    if env_value:
        candidates.extend(part.strip() for chunk in env_value.split(os.pathsep) for part in chunk.split(",") if part.strip())

    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        path = Path(candidate).expanduser()
        transformed = path / "transformed"
        if transformed.is_dir():
            path = transformed
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            roots.append(resolved)
    if not roots:
        raise FileNotFoundError("OfficeQA docs directory not found. Pass --docs-dir or set OFFICEQA_DOCS_DIR.")
    return roots


def resolve_candidate_files(source_files: list[str], roots: list[str]) -> list[str]:
    if not source_files:
        return []
    root_paths = [Path(root).resolve() for root in roots]
    by_rel: dict[str, list[str]] = {}
    by_name: dict[str, list[str]] = {}
    for root in root_paths:
        for dirpath, _, filenames in os.walk(root):
            for filename in sorted(filenames):
                full_path = Path(dirpath, filename).resolve()
                try:
                    rel = full_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                full = str(full_path)
                by_rel.setdefault(rel, []).append(full)
                by_name.setdefault(filename, []).append(full)

    resolved: list[str] = []
    seen: set[str] = set()
    for source_file in source_files:
        text = str(source_file or "").strip()
        if not text:
            continue
        matches: list[str] = []
        raw = Path(text).expanduser()
        if raw.is_absolute():
            try:
                candidate = raw.resolve()
            except OSError:
                candidate = raw
            if candidate.is_file() and _is_under_any_root(candidate, root_paths):
                matches = [str(candidate)]
        else:
            rel = text.replace("\\", "/").lstrip("./")
            exact_matches = by_rel.get(rel, [])
            if exact_matches:
                matches = exact_matches
            else:
                # Fall back to basename only when it is globally unambiguous.
                name_matches = by_name.get(raw.name, [])
                matches = name_matches if len(name_matches) == 1 else []
        for full in matches:
            if full not in seen:
                seen.add(full)
                resolved.append(full)
    return resolved


def build_oracle_parsed_pages_context(
    source_files: list[str],
    source_docs: list[str],
    docs_roots: list[str],
    *,
    max_page_chars: int = MAX_ORACLE_PAGE_CHARS,
    max_total_chars: int = MAX_ORACLE_CONTEXT_CHARS,
    evidence_note: str = "Treat it as primary document evidence and combine it with local document tool evidence when useful.",
) -> str:
    refs = _iter_oracle_refs(source_files, source_docs)
    if not refs:
        return ""

    blocks: list[str] = []
    total_chars = 0
    seen_pages: set[tuple[str, int]] = set()
    for source_file, page_number, source_doc in refs:
        json_path = _locate_parsed_json(source_file, docs_roots)
        if json_path is None:
            continue
        page_key = (str(json_path), page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        page_text = _render_parsed_page(str(json_path), page_number)
        if not page_text:
            continue
        if len(page_text) > max_page_chars:
            omitted = len(page_text) - max_page_chars
            page_text = page_text[:max_page_chars].rstrip() + f"\n\n[... {omitted} characters omitted from this parsed page ...]"
        block = f"### {source_file} page {page_number}\nSource URL: {source_doc}\n\n{page_text}"
        if total_chars + len(block) > max_total_chars:
            remaining = max_total_chars - total_chars
            if remaining <= 0:
                break
            blocks.append(block[:remaining].rstrip() + "\n\n[... oracle parsed page context truncated ...]")
            break
        blocks.append(block)
        total_chars += len(block)
    if not blocks:
        return ""
    return (
        "The following content is pre-parsed from the oracle OfficeQA source page(s). "
        f"{evidence_note.strip()}\n\n"
        + "\n\n".join(blocks)
    )


def run_tool(name: str, arguments: dict, *, allowed_roots: list[str], allowed_paths: list[str]) -> tuple[str, str]:
    if name == "glob":
        pattern = str(arguments.get("pattern") or "*")
        if not allowed_paths:
            return f"glob(pattern={pattern!r})", "[no matches: candidate file allowlist is empty]"
        matches: list[str] = []
        for root in allowed_roots:
            for dirpath, _, filenames in os.walk(root):
                for filename in sorted(filenames):
                    full = os.path.join(dirpath, filename)
                    if allowed_paths and str(Path(full).resolve()) not in set(allowed_paths):
                        continue
                    rel = os.path.relpath(full, root)
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(filename, pattern):
                        matches.append(full)
                    if len(matches) >= MAX_GLOB_MATCHES:
                        break
                if len(matches) >= MAX_GLOB_MATCHES:
                    break
        return f"glob(pattern={pattern!r})", "\n".join(matches) if matches else "[no matches]"

    if name == "read":
        path = str(arguments.get("path") or "")
        if not path:
            return "read(path='')", "[read error: missing path]"
        if not _is_allowed(path, allowed_roots, allowed_paths):
            return f"read(path={path!r})", "[read error: path not allowed]"
        start = max(int(arguments.get("start") or 1), 1)
        limit = max(int(arguments.get("limit") or 80), 1)
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        excerpt = "".join(lines[start - 1:start - 1 + limit])
        return f"read(path={path!r}, start={start}, limit={limit})", excerpt[:MAX_READ_CHARS] or "[empty file]"

    if name == "grep":
        pattern = str(arguments.get("pattern") or "").lower()
        path = str(arguments.get("path") or "")
        if not pattern or not path:
            return f"grep(pattern={pattern!r}, path={path!r})", "[grep error: missing pattern or path]"
        if not _is_allowed(path, allowed_roots, allowed_paths):
            return f"grep(pattern={pattern!r}, path={path!r})", "[grep error: path not allowed]"
        matches: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if pattern in line.lower():
                    matches.append(f"{idx}: {line.rstrip()}")
                if len(matches) >= MAX_GREP_MATCHES:
                    break
        return f"grep(pattern={pattern!r}, path={path!r})", "\n".join(matches) if matches else "[no matches]"

    return name, f"[tool error: unknown tool {name}]"


def _iter_oracle_refs(source_files: list[str], source_docs: list[str]) -> list[tuple[str, int, str]]:
    refs: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    if not source_files or not source_docs:
        return refs
    for index, source_doc in enumerate(source_docs):
        page_number = _extract_page_number(source_doc)
        if page_number is None:
            continue
        source_file = _source_file_from_source_doc(source_doc)
        if source_file:
            pass
        elif index < len(source_files):
            source_file = source_files[index]
        elif len(source_files) == 1:
            source_file = source_files[0]
        else:
            continue
        key = (source_file, page_number, source_doc)
        if key in seen:
            continue
        seen.add(key)
        refs.append(key)
    return refs


def _source_file_from_source_doc(source_doc: str) -> str:
    text = str(source_doc or "").strip().lower()
    if not text:
        return ""
    match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{4})(?:-\d+)?",
        text,
    )
    if not match:
        return ""
    month = _MONTH_TO_NUMBER.get(match.group(1), "")
    year = match.group(2)
    return f"treasury_bulletin_{year}_{month}.txt" if month else ""


def _extract_page_number(source_doc: str) -> int | None:
    text = str(source_doc or "").strip()
    if not text:
        return None
    parsed = urlparse(text)
    query = parse_qs(parsed.query)
    for key in ("page", "pagenum", "page_id"):
        for raw_value in query.get(key, []):
            try:
                return int(str(raw_value).strip())
            except ValueError:
                continue
    match = re.search(r"(?:[?&]|^)page=(\d+)", text)
    return int(match.group(1)) if match else None


def _parsed_root_candidates(docs_roots: list[str]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in docs_roots:
        path = Path(root).expanduser()
        for candidate in (
            path,
            path.parent,
            path / "treasury_bulletins_parsed",
            path.parent / "treasury_bulletins_parsed",
        ):
            marker = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if marker in seen:
                continue
            seen.add(marker)
            candidates.append(candidate)
    return candidates


def _locate_parsed_json(source_file: str, docs_roots: list[str]) -> Path | None:
    source_path = Path(str(source_file).strip())
    stem = source_path.stem if source_path.suffix else source_path.name
    if not stem:
        return None
    candidate_names = [stem + ".json"]
    if source_path.suffix == ".json":
        candidate_names.insert(0, source_path.name)
    for root in _parsed_root_candidates(docs_roots):
        for name in candidate_names:
            path = root / "jsons" / name
            if path.is_file():
                return path
    return None


class _TableMarkdownParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict[str, object]]] = []
        self._row: list[dict[str, object]] | None = None
        self._cell: list[str] | None = None
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.lower()
        if normalized == "tr":
            self._row = []
        elif normalized in {"td", "th"} and self._row is not None:
            attrs_dict = {key.lower(): value for key, value in attrs if value is not None}
            self._cell = []
            self._cell_rowspan = max(_safe_positive_int(attrs_dict.get("rowspan"), 1), 1)
            self._cell_colspan = max(_safe_positive_int(attrs_dict.get("colspan"), 1), 1)
        elif normalized == "br" and self._cell is not None:
            self._cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized in {"td", "th"} and self._cell is not None and self._row is not None:
            cell = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self._row.append({
                "text": cell,
                "rowspan": self._cell_rowspan,
                "colspan": self._cell_colspan,
            })
            self._cell = None
        elif normalized == "tr" and self._row is not None:
            if any(str(cell.get("text") or "") for cell in self._row):
                self.rows.append(self._row)
            self._row = None
            self._cell = None


def _html_table_to_markdown(raw_html: str) -> str:
    parser = _TableMarkdownParser()
    try:
        parser.feed(raw_html)
    except Exception:  # noqa: BLE001
        parser.rows = []
    if not parser.rows:
        text = re.sub(r"(?is)<[^>]+>", " ", raw_html)
        return re.sub(r"\s+", " ", html.unescape(text)).strip()
    rows = _expand_table_spans(parser.rows)
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(cell) for cell in rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return str(value).replace("\n", " ").replace("|", "\\|").strip()


def _expand_table_spans(rows: list[list[dict[str, object]]]) -> list[list[str]]:
    expanded: list[list[str]] = []
    pending: dict[int, tuple[int, str]] = {}
    for row in rows:
        rendered: list[str] = []
        col = 0

        def fill_pending() -> None:
            nonlocal col
            while col in pending:
                remaining, value = pending[col]
                rendered.append(value)
                if remaining <= 1:
                    del pending[col]
                else:
                    pending[col] = (remaining - 1, value)
                col += 1

        for cell in row:
            fill_pending()
            value = str(cell.get("text") or "")
            rowspan = max(_safe_positive_int(cell.get("rowspan"), 1), 1)
            colspan = max(_safe_positive_int(cell.get("colspan"), 1), 1)
            for offset in range(colspan):
                rendered.append(value)
                if rowspan > 1:
                    pending[col + offset] = (rowspan - 1, value)
            col += colspan
        fill_pending()
        expanded.append(rendered)
    return expanded


def _safe_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _render_parsed_content(content: str) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    if "<table" in text.lower():
        return _html_table_to_markdown(text)
    text = html.unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _element_page_ids(element: dict) -> set[int]:
    page_ids: set[int] = set()
    bbox = element.get("bbox")
    if not isinstance(bbox, list):
        return page_ids
    for box in bbox:
        if not isinstance(box, dict):
            continue
        try:
            page_ids.add(int(box.get("page_id")))
        except (TypeError, ValueError):
            continue
    return page_ids


@lru_cache(maxsize=256)
def _load_parsed_elements(json_path: str) -> tuple[dict, ...]:
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ()
    document = payload.get("document") if isinstance(payload, dict) else {}
    elements = document.get("elements") if isinstance(document, dict) else []
    if not isinstance(elements, list):
        return ()
    return tuple(element for element in elements if isinstance(element, dict))


@lru_cache(maxsize=2048)
def _render_parsed_page(json_path: str, page_number: int) -> str:
    rendered: list[str] = []
    for element in _load_parsed_elements(json_path):
        if page_number not in _element_page_ids(element):
            continue
        content = element.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        section = _render_parsed_content(content)
        if section:
            rendered.append(section)
    return "\n\n".join(rendered).strip()


def _is_allowed(path: str, roots: list[str], allowed_paths: list[str]) -> bool:
    resolved_path = _resolve_doc_path(path, roots)
    if resolved_path is None:
        return False
    if not allowed_paths:
        return False
    return resolved_path in {str(Path(path).resolve()) for path in allowed_paths}


def _is_under_any_root(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _resolve_doc_path(path: str, roots: list[str]) -> str | None:
    raw = Path(str(path).strip()).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    for root in roots:
        candidates.append(Path(root, raw))
        candidates.append(Path(root, raw.name))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if any(str(resolved) == root or str(resolved).startswith(root + os.sep) for root in roots):
            return str(resolved)
    return None
