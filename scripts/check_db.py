#!/usr/bin/env python3
"""Show the number of recorded training rounds in a LabelBench database."""
import argparse
from pathlib import Path
import sqlite3
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, type=Path, help='Path to the SQLite database')
    args = parser.parse_args()
    path = args.db.expanduser().resolve()
    if not path.is_file():
        parser.error(f'Database not found: {path}')
    try:
        with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=30) as db:
            total = db.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]
            rows = db.execute('''
                SELECT strategy, seed, phase, COUNT(*)
                FROM metrics
                GROUP BY strategy, seed, phase
                ORDER BY phase, strategy, seed
            ''').fetchall()
    except sqlite3.Error as exc:
        print(f'Error reading database: {exc}', file=sys.stderr)
        return 1
    finally:
        if 'db' in locals():
            db.close()
    print(f'Database: {path}')
    print(f'Total recorded rounds: {total}')
    print(f'Total method/seed/phase groups: {len(rows)}')
    print('\nstrategy | seed | phase | recorded_rounds')
    for row in rows:
        print(' | '.join(map(str, row)))
    print('\nOnly saved records are counted; unfinished rounds and internal epochs are excluded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
