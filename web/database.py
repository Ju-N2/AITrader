from pathlib import Path
import json
import sqlite3

ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "aitrader.db"


def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    connection = sqlite3.connect(
        f"file:{DB_PATH.as_posix()}?mode=ro",
        uri=True,
        timeout=5,
    )
    connection.row_factory = sqlite3.Row
    return connection


def get_health():
    with get_connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }

    return {
        "status": "ok",
        "database": str(DB_PATH),
        "tables": sorted(tables),
    }


def get_latest_candle(
    exchange="binance_spot",
    symbol="BTC/USDT",
    timeframe="1m",
):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM candles
            WHERE exchange = ?
              AND symbol = ?
              AND timeframe = ?
            ORDER BY timestamp_ms DESC
            LIMIT 1
            """,
            (exchange, symbol, timeframe),
        ).fetchone()

    return dict(row) if row else None


def get_candles(
    exchange="binance_spot",
    symbol="BTC/USDT",
    timeframe="1m",
    limit=200,
):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM (
                SELECT *
                FROM candles
                WHERE exchange = ?
                  AND symbol = ?
                  AND timeframe = ?
                ORDER BY timestamp_ms DESC
                LIMIT ?
            )
            ORDER BY timestamp_ms ASC
            """,
            (exchange, symbol, timeframe, limit),
        ).fetchall()

    return [dict(row) for row in rows]


def get_positions(open_only=True):
    query = "SELECT * FROM positions"
    if open_only:
        query += " WHERE status = 'open'"
    query += " ORDER BY opened_at DESC"

    with get_connection() as connection:
        rows = connection.execute(query).fetchall()

    return [dict(row) for row in rows]


def get_portfolio_summary():
    with get_connection() as connection:
        position_count = connection.execute(
            "SELECT COUNT(*) FROM positions WHERE status = 'open'"
        ).fetchone()[0]

        realized_pnl = connection.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0)
            FROM position_events
            WHERE operation = 'exit'
            """
        ).fetchone()[0]

    return {
        "open_positions": position_count,
        "realized_pnl": float(realized_pnl),
        "balance": None,
        "total_equity": None,
    }



AI_ANALYSIS_PATH = ROOT_DIR / "data" / "latest_ai_analysis.json"


def get_latest_ai_analysis():
    if not AI_ANALYSIS_PATH.exists():
        return None

    try:
        with AI_ANALYSIS_PATH.open("r", encoding="utf-8") as file:
            result = json.load(file)

        if not isinstance(result, dict):
            return None

        return result

    except (json.JSONDecodeError, OSError):
        return None
