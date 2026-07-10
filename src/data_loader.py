"""Load and validate the EduPro Excel workbook.

Exposes :func:`load_data` which returns the four sheets as a dict of
``DataFrame`` objects after validating schema, row counts, null-freeness and
the transaction date range declared in :mod:`config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config


@dataclass
class ValidationReport:
    """Outcome of schema/row/null/date validation for the workbook."""

    ok: bool
    messages: list[str]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        status = "PASS" if self.ok else "FAIL"
        body = "\n".join(f"  - {m}" for m in self.messages)
        return f"[Validation {status}]\n{body}"


def load_data(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every sheet of the workbook into a dict keyed by sheet name.

    Parameters
    ----------
    path:
        Optional override for the workbook location. Defaults to
        :data:`config.RAW_XLSX`.
    """
    path = path or config.RAW_XLSX
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the workbook there and retry."
        )
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    # Parse the transaction date column once, up front.
    if config.DATE_COLUMN in sheets.get("Transactions", pd.DataFrame()).columns:
        sheets["Transactions"][config.DATE_COLUMN] = pd.to_datetime(
            sheets["Transactions"][config.DATE_COLUMN]
        )
    return sheets


def validate(sheets: dict[str, pd.DataFrame]) -> ValidationReport:
    """Validate the loaded sheets against the expected schema in config."""
    messages: list[str] = []
    ok = True

    for name, spec in config.SHEETS.items():
        if name not in sheets:
            messages.append(f"MISSING SHEET: {name}")
            ok = False
            continue
        df = sheets[name]

        # Row count
        if len(df) != spec["rows"]:
            messages.append(
                f"{name}: row count {len(df)} != expected {spec['rows']}"
            )
            ok = False
        else:
            messages.append(f"{name}: {len(df)} rows OK")

        # Columns
        missing = [c for c in spec["columns"] if c not in df.columns]
        if missing:
            messages.append(f"{name}: missing columns {missing}")
            ok = False

        # Nulls
        n_null = int(df.isnull().sum().sum())
        if n_null:
            messages.append(f"{name}: {n_null} null cells (expected 0)")
            ok = False

    # Date range
    tx = sheets.get("Transactions")
    if tx is not None and config.DATE_COLUMN in tx.columns:
        dmin = tx[config.DATE_COLUMN].min()
        dmax = tx[config.DATE_COLUMN].max()
        lo = pd.Timestamp(config.DATE_MIN)
        hi = pd.Timestamp(config.DATE_MAX)
        if dmin < lo or dmax > hi:
            messages.append(
                f"Transactions: dates {dmin.date()}..{dmax.date()} outside "
                f"{config.DATE_MIN}..{config.DATE_MAX}"
            )
            ok = False
        else:
            messages.append(
                f"Transactions: dates {dmin.date()}..{dmax.date()} within range OK"
            )

    return ValidationReport(ok=ok, messages=messages)


def null_report(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a tidy per-sheet null-count summary."""
    rows = []
    for name, df in sheets.items():
        rows.append(
            {
                "sheet": name,
                "rows": len(df),
                "cols": df.shape[1],
                "null_cells": int(df.isnull().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point: load, validate and print a short report."""
    sheets = load_data()
    print("=" * 70)
    print("EduPro dataset — load & validation report")
    print("=" * 70)
    for name, df in sheets.items():
        print(f"\n{name}: shape={df.shape}")
        print(df.head(3).to_string())

    print("\n" + null_report(sheets).to_string(index=False))

    report = validate(sheets)
    print("\n" + str(report))
    if not report.ok:
        raise SystemExit("Validation FAILED — see messages above.")
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
