import argparse
import sqlite3

from aitrader.data.backup import backup_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a consistent SQLite backup of AITrader data"
    )
    parser.add_argument("--source", default="data/aitrader.db")
    parser.add_argument("--destination-dir", default="data/backups")
    parser.add_argument("--keep", type=int, default=7)
    args = parser.parse_args()

    try:
        result = backup_database(
            source_path=args.source,
            destination_dir=args.destination_dir,
            keep=args.keep,
        )
    except (OSError, sqlite3.Error, ValueError) as error:
        parser.error(str(error))

    print(result)


if __name__ == "__main__":
    main()
