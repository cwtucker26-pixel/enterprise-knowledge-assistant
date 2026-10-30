"""Load the structured CSVs into SQLite for the SQL tool (not part of the vector pipeline)."""
import sqlite3

import pandas as pd

from src.config import RAW_DIR, SQLITE_PATH

TABLES = ["users", "products", "transactions"]


def build_sqlite() -> dict[str, int]:
    """Rebuild acme.db from scratch each run. Returns rows loaded per table."""
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQLITE_PATH.unlink(missing_ok=True)
    counts = {}
    with sqlite3.connect(SQLITE_PATH) as conn:
        for table in TABLES:
            df = pd.read_csv(RAW_DIR / "structured" / f"{table}.csv")
            if "active" in df.columns:  # "True"/"False" -> 1/0 (SQLite has no boolean)
                df["active"] = df["active"].astype(str).str.lower().eq("true").astype(int)
            df.to_sql(table, conn, index=False)
            counts[table] = len(df)
    return counts
