#!/usr/bin/env python3
"""Show recorded training rounds in a Figure 1 or Figure 5 database."""
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
            columns = {row[1] for row in db.execute('PRAGMA table_info(metrics)')}
            if not {'strategy', 'seed', 'round'}.issubset(columns):
                raise sqlite3.DatabaseError('metrics table is missing required columns')
            has_phase = 'phase' in columns
            group_columns = 'strategy, seed, phase' if has_phase else 'strategy, seed'
            order_columns = 'phase, strategy, seed' if has_phase else 'strategy, seed'
            total = db.execute('SELECT COUNT(*) FROM metrics').fetchone()[0]
            rows = db.execute(
                f'SELECT {group_columns}, COUNT(*) FROM metrics '
                f'GROUP BY {group_columns} ORDER BY {order_columns}'
            ).fetchall()
    except sqlite3.Error as exc:
        print(f'Error reading database: {exc}', file=sys.stderr)
        return 1
    print(f'Database: {path}')
    print(f'Total recorded rounds: {total}')
    print(f'Total method/seed{"/phase" if has_phase else ""} groups: {len(rows)}')
    print(f'\nstrategy | seed{" | phase" if has_phase else ""} | recorded_rounds')
    for row in rows:
        print(' | '.join(map(str, row)))
    print('\nOnly saved records are counted; unfinished rounds and internal epochs are excluded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
