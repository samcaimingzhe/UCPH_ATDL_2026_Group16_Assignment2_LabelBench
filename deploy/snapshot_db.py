"""NEW helper: publish a stopped worker's SQLite database as a single file."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

source, target = map(lambda p: Path(p).resolve(), sys.argv[1:])
if not source.is_file():
    raise SystemExit(f"No database to publish: {source}")
target.parent.mkdir(parents=True, exist_ok=True)
if any(Path(str(target) + suffix).exists() for suffix in ("-wal", "-shm")):
    raise SystemExit("Destination has SQLite sidecars; do not overwrite an active database.")
fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
os.close(fd)
try:
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src:
        with sqlite3.connect(temporary) as dest:
            src.backup(dest)
            assert dest.execute("PRAGMA quick_check").fetchone() == ("ok",)
            assert dest.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print("Published database snapshot:", target)
