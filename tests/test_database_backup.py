import sqlite3

import pytest

from aitrader.data.backup import backup_database
from aitrader.data.storage import SQLiteCandleRepository


def test_repository_uses_wal_journal_mode(tmp_path):
    database = tmp_path / "aitrader.db"
    SQLiteCandleRepository(database)

    with sqlite3.connect(database) as connection:
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_backup_is_readable_and_retention_is_enforced(tmp_path):
    database = tmp_path / "aitrader.db"
    SQLiteCandleRepository(database)
    backup_dir = tmp_path / "backups"

    for index in range(3):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS backup_probe (value INTEGER)"
            )
            connection.execute(
                "INSERT INTO backup_probe (value) VALUES (?)",
                (index,),
            )

        backup_database(database, backup_dir, keep=2)

    backups = sorted(backup_dir.glob("aitrader-*.sqlite3"))
    assert len(backups) == 2

    with sqlite3.connect(backups[-1]) as connection:
        values = connection.execute(
            "SELECT value FROM backup_probe ORDER BY value"
        ).fetchall()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()

    assert values == [(0,), (1,), (2,)]
    assert integrity == ("ok",)


def test_backup_captures_committed_wal_data_during_active_write(tmp_path):
    database = tmp_path / "aitrader.db"
    SQLiteCandleRepository(database)
    writer = sqlite3.connect(database, timeout=5)
    writer.execute("CREATE TABLE backup_probe (value INTEGER)")
    writer.execute("INSERT INTO backup_probe VALUES (1)")
    writer.commit()

    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT INTO backup_probe VALUES (2)")
    backup_path = backup_database(database, tmp_path / "backups")

    try:
        with sqlite3.connect(backup_path) as backup:
            values = backup.execute(
                "SELECT value FROM backup_probe ORDER BY value"
            ).fetchall()
        assert values == [(1,)]
    finally:
        writer.rollback()
        writer.close()


def test_read_only_dashboard_can_read_during_wal_write(tmp_path):
    database = tmp_path / "aitrader.db"
    SQLiteCandleRepository(database)
    writer = sqlite3.connect(database, timeout=5)
    reader = sqlite3.connect(
        f"file:{database.as_posix()}?mode=ro",
        uri=True,
        timeout=5,
    )

    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            "binance_spot",
            "BTC/USDT",
            "1m",
            1_000,
            100,
            101,
            99,
            100,
            1,
        ))

        count = reader.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        assert count == 0

        writer.commit()
        count = reader.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        assert count == 1
    finally:
        writer.rollback()
        reader.close()
        writer.close()


@pytest.mark.parametrize("keep", [0, -1, True, 1.5])
def test_backup_rejects_invalid_retention(tmp_path, keep):
    database = tmp_path / "aitrader.db"
    SQLiteCandleRepository(database)

    with pytest.raises(ValueError, match="keep"):
        backup_database(database, tmp_path / "backups", keep=keep)
