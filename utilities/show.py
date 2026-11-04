"""Look at any data file or folder in this project as a table in the terminal.

    python utilities/show.py data/sample/checking/checking_20250101_20250401.csv   # one file
    python utilities/show.py data/sample/checking/                                 # whole folder
    python utilities/show.py data/warehouse/silver/transactions --rows 200
    python utilities/show.py data/warehouse/quality/report.json

Or paste a path into PATH below and run ``python utilities/show.py`` with no arguments.

Rules
* A path to a file reads only that file; other files in its folder are ignored.
* A path to a folder reads every .csv, .parquet or .json file under it (including sub-folders
  such as Spark's ``_load_date=...`` partitions). The folder must hold one kind only.
* CSVs are read as raw text, exactly as they are on disk (no number or date guessing, blank
  cells stay blank), which is what you want when inspecting an export. ``--infer`` types them.
* Spark's partition folders become columns (``_load_date=2026-10-24`` -> a ``_load_date``
  column), and files with different columns are combined with blanks where a column is absent.
* When more than one file is read, a ``_file`` column says where each row came from.

Needs pandas and pyarrow (``pip install -e ".[dashboard]"``). It does not start Spark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Paste a path between the quotes to use it when no command-line path is given.
PATH = ""

DEFAULT_ROWS = 50
KINDS = {".csv": "csv", ".parquet": "parquet", ".json": "json"}


class ShowError(Exception):
    """A problem with the path or its contents, phrased for the person at the terminal."""


# --------------------------------------------------------------------------- finding files


def find_files(path: str | Path) -> tuple[str, list[Path]]:
    """Return ``(kind, files)`` for a file or folder; raise ShowError if nothing sensible."""
    p = Path(path).expanduser()
    if not p.exists():
        raise ShowError(f"path does not exist: {p}")
    if p.is_file():
        kind = KINDS.get(p.suffix.lower())
        if kind is None:
            raise ShowError(f"unsupported file type {p.suffix!r}; use .csv, .parquet or .json")
        return kind, [p]

    found: dict[str, list[Path]] = {}
    for f in sorted(p.rglob("*")):
        relative = f.relative_to(p).parts
        hidden = any(part.startswith((".", "_")) and "=" not in part for part in relative)
        if f.is_file() and not hidden and f.suffix.lower() in KINDS:
            found.setdefault(KINDS[f.suffix.lower()], []).append(f)
    if not found:
        raise ShowError(f"no .csv, .parquet or .json files under {p}")
    if len(found) > 1:
        counts = ", ".join(f"{len(v)} {k}" for k, v in sorted(found.items()))
        raise ShowError(
            f"{p} holds more than one kind of file ({counts}); point at a sub-folder or a file"
        )
    kind, files = next(iter(found.items()))
    return kind, files


# --------------------------------------------------------------------------- loading


def _partition_values(file: Path, root: Path) -> dict[str, str]:
    """Hive-style ``key=value`` folder names between ``root`` and the file."""
    values = {}
    for part in file.relative_to(root).parts[:-1]:
        if "=" in part:
            key, _, value = part.partition("=")
            values[key] = value
    return values


def _json_frame(data):
    import pandas as pd

    if isinstance(data, list):
        if all(isinstance(x, dict) for x in data):
            return pd.json_normalize(data)
        return pd.DataFrame({"value": data})
    if isinstance(data, dict):
        tables = {k: v for k, v in data.items()
                  if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)}
        if tables:  # e.g. a report with a list of "checks": show the list as the table
            return pd.json_normalize(tables[max(tables, key=lambda k: len(tables[k]))])
        return pd.json_normalize([data])
    return pd.DataFrame({"value": [data]})


def _read_one(file: Path, kind: str, infer: bool):
    import pandas as pd

    if kind == "csv":
        if infer:
            return pd.read_csv(file)
        return pd.read_csv(file, dtype=str, keep_default_na=False, na_filter=False)
    if kind == "parquet":
        import pyarrow.parquet as pq

        return pq.read_table(file).to_pandas()
    try:
        return _json_frame(json.loads(file.read_text()))
    except json.JSONDecodeError as exc:
        raise ShowError(f"{file.name} is not valid JSON: {exc}") from exc


def load_frame(path: str | Path, infer: bool = False):
    """Load a file or folder into one DataFrame. Returns ``(frame, kind, files)``."""
    import pandas as pd

    kind, files = find_files(path)
    root = Path(path).expanduser()
    base = root if root.is_dir() else root.parent
    frames = []
    for f in files:
        try:
            df = _read_one(f, kind, infer)
        except ShowError:
            raise
        except Exception as exc:  # unreadable file: say which one
            raise ShowError(f"could not read {f}: {exc}") from exc
        for key, value in _partition_values(f, base).items():
            if key not in df.columns:
                df[key] = value
        if len(files) > 1:
            df.insert(0, "_file", str(f.relative_to(base)))
        frames.append(df)
    frame = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True, sort=False)
    if kind == "csv" and not infer and len(files) > 1:
        frame = frame.fillna("")   # a column missing from one file's layout shows as blank
    return frame, kind, files


# --------------------------------------------------------------------------- showing


def render(frame, kind: str, files: list[Path], rows: int | None, columns: list[str] | None,
           shown_path: str) -> str:
    import pandas as pd

    if columns:
        unknown = [c for c in columns if c not in frame.columns]
        if unknown:
            raise ShowError(
                f"no such column(s): {', '.join(unknown)}. Columns are: {', '.join(frame.columns)}"
            )
        frame = frame[columns]
    total = len(frame)
    shown = frame if rows is None else frame.head(rows)

    lines = [f"{shown_path}  [{kind}, {len(files)} file{'s' if len(files) != 1 else ''}]"]
    lines += [f"  {f}" for f in files[:5]]
    if len(files) > 5:
        lines.append(f"  ... and {len(files) - 5} more")
    lines.append(f"{total:,} rows x {len(frame.columns)} columns")
    lines.append("columns: " + ", ".join(f"{c} ({t})" for c, t in frame.dtypes.astype(str).items()))
    lines.append("")
    with pd.option_context("display.max_colwidth", 100):
        lines.append(shown.to_string(max_cols=None, max_colwidth=100) if total else "(no rows)")
    if len(shown) < total:
        lines.append(f"\nshowing the first {len(shown):,} of {total:,} rows (use --rows N or --all)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Show a CSV / Parquet / JSON file or folder as a table.")
    p.add_argument("path", nargs="?", help="file or folder (default: PATH set in this file)")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"rows to show (default {DEFAULT_ROWS})")
    p.add_argument("--all", action="store_true", help="show every row")
    p.add_argument("--cols", help="comma-separated columns to show")
    p.add_argument("--infer", action="store_true", help="let pandas guess CSV column types")
    args = p.parse_args(argv)

    path = args.path or PATH
    if not path:
        print("No path given. Pass one as an argument or paste it into PATH in this file.",
              file=sys.stderr)
        return 2
    if args.rows < 1:
        print("--rows must be at least 1", file=sys.stderr)
        return 2
    try:
        frame, kind, files = load_frame(path, infer=args.infer)
        columns = [c.strip() for c in args.cols.split(",") if c.strip()] if args.cols else None
        print(render(frame, kind, files, None if args.all else args.rows, columns, path))
    except ShowError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:   # e.g. piped into `head`: not an error
        sys.stderr.close()
        return 0
    except ImportError as exc:
        print(f"error: {exc}. Install the extras: pip install -e \".[dashboard]\"", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
