# finance-pipeline

A personal finance data pipeline built with PySpark: messy bank exports go in, a clean
deduplicated transaction ledger comes out, and (next milestone) an LLM categorizes merchants.
The repo ships only **synthetic** data; real statements stay outside version control.

## Architecture

```
raw CSVs ──► bronze ──► silver ──► [LLM categorization] ──► gold
 (2 banks,   raw strings,  common schema,  distinct merchants     monthly rollups,
 different   append-only,  stable IDs,     only, cached,          recurring charges,
 layouts)    lineage       deduped         validated output       anomalies
```

| Layer | Status | What it does |
|-------|--------|--------------|
| Bronze | built | Lands CSVs as raw strings, sanitizes column names, tags `_source_file` / `_load_date`, skips files it already loaded |
| Silver | built | Parses both layouts into one schema, quarantines bad rows, assigns deterministic `txn_id`, dedupes overlapping exports, normalizes merchant names, flags transfers |
| Categorization | built | Corrections, then rules, then cache, then LLM on distinct merchants only; Pydantic-validated output; review queue |
| Gold | built | Monthly summary, spend by category with month-over-month change, recurring charges, anomalies, and a browsable transactions table |
| Dashboard and monthly summary | built | Local Streamlit app over the gold Parquet; LLM-written monthly summary whose numbers are checked against the data |

## Quick start

```bash
pip install -e ".[dev]"          # needs Java 11+ for Spark

python -m finpipe.synthetic --out data/sample --seed 42     # regenerate sample data
python -m finpipe.pipeline --raw data/sample --warehouse data/warehouse
pytest

# categorize merchants with the Anthropic API (cheapest model by default)
export ANTHROPIC_API_KEY=...                       # never commit this
python -m finpipe.pipeline --categorize
python -m finpipe.evaluate run                     # score the model on the golden set
```

Override the model with `--model` or the `FINPIPE_MODEL` environment variable.

```bash
pip install -e ".[dashboard]"                       # Streamlit, PyArrow, pandas
python -m finpipe.pipeline --categorize --gold      # build the gold tables
python -m finpipe.summary --month 2026-09           # verified monthly summary (add --template-only for no API call)
streamlit run dashboard/app.py                      # http://localhost:8501
```

The sample data has two sources with different CSV layouts and sign conventions, export
windows that overlap by a month (so 432 exported rows describe 369 real transactions), and
same-day identical purchases. `data/sample/truth/` holds the ground truth the tests check
against, plus a raw-description to merchant/category mapping that seeds the labeled set for
measuring the LLM step.

## Design decisions

- **Integer cents, never floats.** Amounts are parsed through `decimal(18,2)`.
- **Sign convention:** negative is money out, positive is money in, for every source. The
  card export shows purchases as positive numbers, so its parser flips the sign. Card
  exports come in two layouts, picked by column names: a signed `Amount` (flipped) or separate
  positive `Debit` and `Credit` columns (`credit - debit`). Exactly one side must be filled in;
  rows with both, neither, or an unreadable value are quarantined rather than guessed at.
  Files with different layouts can share a folder: each file is read on its own and unioned by
  column name, because reading them together would match columns by the first file's header.
  A `Card No.` column is cut to its last four digits at ingest and never reaches silver.
- **Deterministic IDs.** `txn_id = sha256(source | date | cents | description_key | occurrence)`.
  Reloading a file or loading overlapping exports yields identical IDs.
- **Occurrence index is per source file.** Numbering identical rows across the union would
  count the overlapping copies as separate transactions and silently disable dedupe. A
  pure-Python model of the algorithm on the sample data gives 369 rows with per-file
  numbering and 432 (no dedupe at all) with global numbering.
- **IDs hash the raw description, not the normalized merchant**, so improving normalization
  rules never re-keys history.
- **Bronze never fails.** Values stay raw strings; anything unparseable is caught in silver
  and written to `quarantine/` instead of being dropped.
- **Latest file wins** when the same transaction appears in more than one export.
- **Transfers are flagged** (`is_transfer`) so card payments do not double count as spending.
- **Merchant normalization before the LLM.** On the sample data 294 distinct raw descriptions
  collapse to 28 merchant keys, which is what bounds LLM calls.

## Categorization

Each distinct merchant is resolved in this order, first match wins:

1. **Corrections**: your own `data/private/corrections.csv` (`merchant,category`), git-ignored.
2. **Rules**: merchants whose every row is a transfer are `Transfer` without asking the LLM.
3. **Cache**: earlier LLM answers in `warehouse/categorization/cache.jsonl`, so reruns are free.
4. **LLM**: batched forced tool call, so the answer has a fixed JSON shape.

LLM output is treated as untrusted. Every item is validated against the `Category` enum with
Pydantic; bad or missing items are retried, then fall back to `Other` flagged for review
(fallbacks are not cached, so the next run retries them). Answers under 0.7 confidence are kept
but written to `warehouse/categorization/review_queue.csv`; fix them by adding a line to the
corrections file. Cache writes happen per batch, so a crash keeps the work already paid for.
Bank descriptions are untrusted text too, so they are sent as JSON data with a system prompt
that says to ignore instructions inside them.

`python -m finpipe.evaluate run` scores the model alone (no corrections, no rules, separate
cache) against a golden set and prints accuracy, accuracy among unflagged answers, per-category
results, the errors, and token usage. The bundled golden set is synthetic and easy; for a real
number, label ~100 of your own merchants into `data/private/` and pass `--golden`.

## Gold tables

Written to `warehouse/gold/`, rebuilt from silver on every run (no state):

| Table | Contents |
|-------|----------|
| `monthly_summary` | income, spend, net, savings rate, transfer net, partial-month flag |
| `monthly_category_spend` | month x category spend, previous month, change in cents and percent |
| `recurring_charges` | merchants on a weekly/biweekly/monthly/quarterly/yearly cadence, typical amount, whether the amount is stable, next expected date |
| `anomalies` | charges far above that merchant's norm |
| `transactions` | every transaction: date, cleaned merchant, category, amount, type (spend / income / transfer), source, and the bank's raw description. Same classification as the other tables, so it nets to them |

Decisions: transfers are excluded from spend and income (reported separately so totals still
reconcile to silver); **refunds net against the spend of their category**; uncategorized
merchants show as `Uncategorized`; month axes are densified so month-over-month always compares
adjacent calendar months; the first and last months are flagged when the data only partly
covers them. Anomalies use the median and median absolute deviation per merchant, because one
huge charge inflates a mean and standard deviation enough to hide itself. Defaults are
conservative (robust z >= 6, at least 2x the typical charge and $50 above it, merchants with
at least 6 charges); see `gold_config.py`. `tests/reference_gold.py` is an independent
pure-Python implementation that the Spark tests compare against.

## Data quality

`python -m finpipe.quality` (or `--check` on the pipeline) verifies the warehouse against the
original exports and exits 1 if any error-level check fails, so an orchestrator can stop. The
report is also saved to `warehouse/quality/report.json`. Facts are gathered in two small steps
(a pure-Python scan of the raw CSVs, a few Spark aggregations over the warehouse) and the
rules are a separate pure-Python function, so every rule is unit-tested with crafted inputs.

| Layer | Check | Severity |
|-------|-------|----------|
| source | every raw file reached bronze, with the same row count | error |
| source | card exports with debit/credit columns: every row has exactly one side filled in (this export has no balance column, so this is the sign check available for it) | warn |
| source | **running-balance continuity**: each balance equals the previous balance plus the row's amount (verifies amounts, signs and missing rows using the bank's own numbers; handles newest-first files) | error |
| silver | rows accounted for: bronze = silver + quarantined + duplicates collapsed | error |
| silver | `txn_id` unique; required fields non-null; quarantine rate under 1% | error |
| silver | dates plausible (a day/month mix-up shows as future dates) | error |
| silver | anything quarantined; absurd or zero amounts; missing source; stale data (45 days) | warn |
| categorization | every merchant has a category | error |
| categorization | fallbacks; review queue over 25 | warn |
| gold | income - spend + transfers equals the silver total; the two spend tables agree | error |
| gold | the transactions table has silver's row count and the same income, spend and transfer totals as the monthly summary (skipped for a warehouse built before the table existed) | error |

A check that cannot run (raw directory not given, a layer not built yet) is reported as
`skip`, never as a silent pass. Thresholds live in `QualityConfig`. I chose plain checks over
Great Expectations to avoid a heavy dependency for a dozen rules; they would map onto GE
expectations if the project grows.

## Dashboard and monthly summary

`dashboard/app.py` is a local, read-only Streamlit app (Overview, Categories, Recurring, Transactions, Needs
attention). It reads the gold Parquet directly with PyArrow, so no Spark session starts, and
all data preparation lives in `dashboard_data.py`, which is unit-tested without a browser.

The **Transactions** tab lists the transactions of the month picked in the sidebar (or every
month with "All months"), newest first, with filters for category and source, a "Hide
transfers" box, a text search over the merchant and the bank's raw description, and a CSV
download. Totals use the Overview's conventions and are checked against the monthly summary
(a warning appears if they ever disagree). Text cells in the CSV that start like a spreadsheet
formula are prefixed with an apostrophe so opening the file in Excel cannot run them. The
transactions stay on this machine; only aggregates are ever sent to the API.

The monthly summary treats the LLM as a **writer, not a calculator**. `summary.py` computes
every figure in plain code and pre-formats it ("$4,031.00", "12.5%"). The model only turns
that facts bundle into prose. `verify_numbers` then rejects any draft containing a number that
is not in the facts (whole-dollar and one-decimal roundings are allowed); the model is told
which numbers were wrong and retries, and after three failures a deterministic template is
used. So a summary always exists and never states an invented figure. Results are cached by a
hash of facts + model + prompt version, so reruns cost nothing, and a template fallback is
never cached as if it were an LLM answer.

Privacy: only aggregate facts go to the API, never individual transactions. Merchant names
appear only for the top recurring charges and anomalies; `--no-merchants` removes them.

## Inspecting data

`utilities/show.py` prints any CSV, Parquet or JSON file, or a whole folder of them, as a table
in the terminal. It uses pandas and PyArrow (no Spark), so it starts instantly.

```bash
python utilities/show.py data/sample/checking/checking_20250101_20250401.csv   # one file only
python utilities/show.py data/warehouse/silver/transactions --rows 200         # whole folder
python utilities/show.py data/warehouse/bronze/credit_card --cols _file,description,debit,credit
```

Paste a path into `PATH` at the top of the file to run it with no arguments. CSVs are shown
as raw text (leading zeros and blank cells intact; `--infer` guesses types), Spark partition
folders such as `_load_date=...` become columns, and a `_file` column appears when several
files are combined. Flags: `--rows N` (default 50), `--all`, `--cols a,b`, `--infer`.

## Known limitations

- Spark is overkill for personal-scale data. The jobs are written as if volume were far
  larger: window functions instead of `collect()`, built-in column expressions for hashing
  instead of Python UDFs (the one UDF, merchant normalization, is the first thing to replace
  with a broadcast alias join at scale).
- Occurrence numbering assumes an export never splits a calendar day. Banks export on day
  boundaries, but a file that cuts a day mid-way could re-key one of two identical purchases.
- Merchant normalization is heuristic (alias table plus suffix stripping) and will mangle
  some names. It only needs to be consistent; the LLM step and corrections table handle the rest.
- Parquet today. Delta or Iceberg (time travel, merge) is a planned upgrade.

## Layout

```
src/finpipe/
  ingest.py      bronze: read, sanitize, lineage, idempotent append
  silver.py      parsers, quarantine, deterministic IDs, dedupe
  pipeline.py    raw -> bronze -> silver entry point
  normalize.py   merchant-name cleanup (pure Python)
  ids.py         transaction identity (pure Python, mirrors the Spark expression)
  categories.py  category taxonomy the LLM output is validated against
  synthetic.py   synthetic bank exports + ground truth
  categorize.py  resolution order, validation, retries, cache, corrections, review queue
  llm.py         Anthropic API backend (forced tool call, token accounting)
  enrich.py      Spark glue: categorize silver's merchants, broadcast-join categories back
  evaluate.py    accuracy evaluation against a golden set
  gold.py        gold tables (summary, category spend, recurring charges, anomalies, transactions)
  gold_config.py thresholds and cadence bands for gold (pure Python)
  quality.py     data quality checks: raw scan, warehouse metrics, pure-Python rules, report
  gold_io.py     read gold Parquet with PyArrow (no Spark)
  dashboard_data.py  dashboard view-models: pure functions over gold rows
  summary.py     facts -> verified LLM summary, cache, template fallback, CLI
dashboard/app.py Streamlit UI (thin; uses dashboard_data)
utilities/show.py print any CSV/Parquet/JSON file or folder as a table
tests/           pure-Python tests and Spark tests (skipped if pyspark is missing)
data/golden/     synthetic golden set (real labeled data belongs in data/private/)
```
