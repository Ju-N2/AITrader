import os
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

BACKUP_PREFIX = "aitrader-"
BACKUP_SUFFIX = ".sqlite3"


def backup_database(
    source_path: str | Path = "data/aitrader.db",
    destination_dir: str | Path = "data/backups",
    keep: int = 7,
) -> Path:
    source_path = Path(source_path)
    destination_dir = Path(destination_dir)

    if not source_path.is_file():
        raise FileNotFoundError("AITrader database does not exist")

    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("keep must be a positive integer")

    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination_path = (
        destination_dir / f"{BACKUP_PREFIX}{timestamp}{BACKUP_SUFFIX}"
    )

    temporary_path = None
    source = None
    destination = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination_dir,
            prefix=".aitrader-backup-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        source = sqlite3.connect(source_path, timeout=30)
        destination = sqlite3.connect(temporary_path, timeout=30)
        source.backup(destination)

        integrity = destination.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise sqlite3.DatabaseError("Backup integrity check failed")

        destination.close()
        destination = None
        source.close()
        source = None

        with temporary_path.open("rb") as backup_file:
            os.fsync(backup_file.fileno())

        os.replace(temporary_path, destination_path)
        temporary_path = None

        directory_fd = os.open(destination_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        backups = sorted(
            destination_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}"),
            key=lambda path: path.name,
            reverse=True,
        )

        for old_backup in backups[keep:]:
            old_backup.unlink()

        return destination_path

    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
