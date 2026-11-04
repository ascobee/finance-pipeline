"""utilities/show.py: which files get read, and how they are combined. Needs pandas."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_SPEC = importlib.util.spec_from_file_location(
    "show", Path(__file__).resolve().parent.parent / "utilities" / "show.py")
show = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(show)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run_main(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = show.main(list(argv))
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- which files


def test_a_file_path_reads_only_that_file(tmp_path):
    write(tmp_path / "a.csv", "x,y\n1,2\n")
    write(tmp_path / "b.csv", "x,y\n3,4\n5,6\n")
    frame, kind, files = show.load_frame(tmp_path / "a.csv")
    assert kind == "csv" and files == [tmp_path / "a.csv"]
    assert len(frame) == 1 and "_file" not in frame.columns


def test_a_folder_reads_every_file_and_labels_the_source(tmp_path):
    write(tmp_path / "a.csv", "x,y\n1,2\n")
    write(tmp_path / "sub" / "b.csv", "x,y\n3,4\n5,6\n")
    frame, _, files = show.load_frame(tmp_path)
    assert len(files) == 2 and len(frame) == 3
    assert list(frame["_file"]) == ["a.csv", str(Path("sub") / "b.csv"), str(Path("sub") / "b.csv")]
    assert frame.columns[0] == "_file"


def test_hidden_and_marker_files_are_ignored(tmp_path):
    write(tmp_path / "a.csv", "x\n1\n")
    for name in ("_SUCCESS", ".a.csv.crc", "_hidden.csv", ".DS_Store"):
        write(tmp_path / name, "junk")
    _, files = show.find_files(tmp_path)
    assert [f.name for f in files] == ["a.csv"]


def test_a_folder_with_two_kinds_of_file_is_an_error(tmp_path):
    write(tmp_path / "a.csv", "x\n1\n")
    write(tmp_path / "b.json", "[]")
    with pytest.raises(show.ShowError, match="more than one kind"):
        show.find_files(tmp_path)


def test_empty_folder_missing_path_and_unsupported_file_are_errors(tmp_path):
    with pytest.raises(show.ShowError, match="no .csv"):
        show.find_files(tmp_path)
    with pytest.raises(show.ShowError, match="does not exist"):
        show.find_files(tmp_path / "nope")
    with pytest.raises(show.ShowError, match="unsupported"):
        show.find_files(write(tmp_path / "notes.txt", "hi"))


# ---------------------------------------------------------------- csv contents


def test_csv_is_read_as_raw_text_with_blanks_preserved(tmp_path):
    write(tmp_path / "a.csv", "id,amount,note\n007,1.50,\n008,,hello\n")
    frame, _, _ = show.load_frame(tmp_path / "a.csv")
    assert list(frame["id"]) == ["007", "008"]          # leading zeros survive
    assert list(frame["amount"]) == ["1.50", ""]        # blank is blank, not NaN
    assert list(frame["note"]) == ["", "hello"]


def test_infer_types_the_columns(tmp_path):
    write(tmp_path / "a.csv", "id,amount\n7,1.5\n8,2.5\n")
    frame, _, _ = show.load_frame(tmp_path / "a.csv", infer=True)
    assert frame["amount"].sum() == 4.0


def test_files_with_different_layouts_combine_with_blanks(tmp_path):
    write(tmp_path / "old.csv", "date,amount\n2025-01-01,5.00\n")
    write(tmp_path / "new.csv", "date,debit,credit\n2025-02-01,6.00,\n")
    frame, _, _ = show.load_frame(tmp_path)
    assert {"amount", "debit", "credit"} <= set(frame.columns)
    by_file = frame.set_index("_file")
    assert by_file.loc["old.csv", "debit"] == "" and by_file.loc["new.csv", "amount"] == ""


# ---------------------------------------------------------------- parquet


def test_parquet_folder_with_spark_style_partitions_and_markers(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for day, rows in (("2026-10-23", [{"a": "1"}]), ("2026-10-24", [{"a": "2", "extra": "x"}])):
        d = tmp_path / f"_load_date={day}"
        d.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), d / "part-0.parquet")
        (d / ".part-0.parquet.crc").write_text("junk")
    write(tmp_path / "_SUCCESS", "")
    frame, kind, files = show.load_frame(tmp_path)
    assert kind == "parquet" and len(files) == 2 and len(frame) == 2
    assert sorted(frame["_load_date"]) == ["2026-10-23", "2026-10-24"]
    assert frame["extra"].isna().sum() == 1             # column present in only one file


# ---------------------------------------------------------------- json


def test_json_list_of_records(tmp_path):
    write(tmp_path / "a.json", json.dumps([{"k": 1}, {"k": 2}]))
    frame, kind, _ = show.load_frame(tmp_path / "a.json")
    assert kind == "json" and list(frame["k"]) == [1, 2]


def test_json_object_with_a_list_of_records_shows_that_list(tmp_path):
    write(tmp_path / "report.json", json.dumps(
        {"ok": True, "checks": [{"name": "a", "status": "pass"}, {"name": "b", "status": "skip"}]}))
    frame, _, _ = show.load_frame(tmp_path / "report.json")
    assert list(frame["name"]) == ["a", "b"]


def test_plain_json_object_is_one_row_and_bad_json_is_an_error(tmp_path):
    write(tmp_path / "s.json", json.dumps({"month": "2025-03", "text": "hi", "meta": {"n": 1}}))
    frame, _, _ = show.load_frame(tmp_path / "s.json")
    assert len(frame) == 1 and frame["meta.n"].iloc[0] == 1
    write(tmp_path / "bad.json", "{oops")
    with pytest.raises(show.ShowError, match="not valid JSON"):
        show.load_frame(tmp_path / "bad.json")


# ---------------------------------------------------------------- command line


def numbered_csv(tmp_path, n=120):
    return write(tmp_path / "n.csv", "n\n" + "".join(f"{i}\n" for i in range(n)))


def test_default_row_cap_is_50_and_says_so(tmp_path):
    code, out, _ = run_main(str(numbered_csv(tmp_path)))
    assert code == 0 and "120 rows x 1 columns" in out
    assert "showing the first 50 of 120 rows" in out and "\n49 " in out and "\n50 " not in out


def test_rows_and_all_flags(tmp_path):
    path = str(numbered_csv(tmp_path))
    _, out, _ = run_main(path, "--rows", "3")
    assert "showing the first 3 of 120" in out
    _, out, _ = run_main(path, "--all")
    assert "showing the first" not in out and "\n119 " in out


def test_cols_selects_columns_and_rejects_unknown_ones(tmp_path):
    path = str(write(tmp_path / "a.csv", "x,y,z\n1,2,3\n"))
    _, out, _ = run_main(path, "--cols", "x,z")
    assert "columns: x (str), z (str)" in out or "columns: x (object), z (object)" in out
    code, _, err = run_main(path, "--cols", "nope")
    assert code == 1 and "no such column" in err and "x, y, z" in err


def test_no_path_and_empty_default_is_a_usage_error(monkeypatch):
    monkeypatch.setattr(show, "PATH", "")
    code, _, err = run_main()
    assert code == 2 and "No path given" in err


def test_the_PATH_constant_is_used_when_no_argument_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(show, "PATH", str(write(tmp_path / "a.csv", "x\n1\n")))
    code, out, _ = run_main()
    assert code == 0 and "1 rows x 1 columns" in out


def test_errors_exit_1_with_a_message(tmp_path):
    code, _, err = run_main(str(tmp_path / "missing"))
    assert code == 1 and err.startswith("error: path does not exist")
