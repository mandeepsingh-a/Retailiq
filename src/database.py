"""
database.py
===========
Loads the cleaned transaction table into SQLite and exposes a thin
query interface. SQLite is used so the whole project runs with zero
external infra (no server to spin up) while still exercising real SQL —
window functions, CTEs, and cohort-style analysis — instead of hiding
everything behind pandas.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "retailiq.db"
CLEAN_CSV = Path(__file__).resolve().parent.parent / "data" / "processed" / "transactions_clean.csv"
CUSTOMERS_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "customers_raw.csv"


def build_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    df = pd.read_csv(CLEAN_CSV, parse_dates=["order_date"])
    customers = pd.read_csv(CUSTOMERS_CSV, parse_dates=["signup_date"])

    conn = sqlite3.connect(db_path)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    customers.to_sql("customers", conn, if_exists="replace", index=False)

    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_txn_customer ON transactions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(order_date);
        CREATE INDEX IF NOT EXISTS idx_cust_id ON customers(customer_id);
    """)
    conn.commit()
    return conn


def run_query_file(conn: sqlite3.Connection, sql_path: Path) -> dict[str, pd.DataFrame]:
    """Run every named statement in a .sql file (split on '-- name:' markers)
    and return a dict of {query_name: DataFrame}."""
    text = sql_path.read_text()
    blocks = [b for b in text.split("-- name:") if b.strip()]
    results = {}
    for block in blocks:
        name, _, query = block.partition("\n")
        name = name.strip()
        # Skip the file's own header block (not a valid single-word query name).
        if not name or " " in name or len(name.split()) != 1:
            continue
        results[name] = pd.read_sql_query(query, conn)
    return results


if __name__ == "__main__":
    conn = build_database()
    sql_path = Path(__file__).resolve().parent.parent / "sql" / "analysis_queries.sql"
    results = run_query_file(conn, sql_path)
    for name, res_df in results.items():
        print(f"\n--- {name} ---")
        print(res_df.head(8).to_string(index=False))
    conn.close()
