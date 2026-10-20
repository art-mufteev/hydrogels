#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import openpyxl

_MONO_COL = re.compile(r'^(?:Φ|Phi|PHI|phi)_?([A-Za-z][A-Za-z0-9_]*)$')
_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
_SUM_TOL = 1e-6

Row = dict[str, Any]
Candidate = dict[str, Any]


def _to_float(x: Any) -> Optional[float]:
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return None


def parse_number_pair(s: Any) -> tuple[Optional[float], Optional[float]]:
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return float(s), None
    s = str(s).strip()
    if s in ('', '/', '-', 'N/A', 'na', 'NA'):
        return None, None
    for sep in ('±', '+/-', '+-'):
        if sep in s:
            head, _, tail = s.partition(sep)
            return _to_float(head), _to_float(tail) if tail.strip() else None
    return _to_float(s), None


def normalize_composition(raw: dict[str, float]) -> dict[str, float]:
    comp = {k: float(v) for k, v in raw.items() if v is not None and float(v) > 0}
    if not comp:
        raise ValueError('composition is empty after dropping zeros')
    total = sum(comp.values())
    if total <= 0:
        raise ValueError(f'composition sum is non-positive: {total}')
    norm = {k: v / total for k, v in comp.items()}
    s = sum(norm.values())
    if abs(s - 1.0) > 1e-12:
        big = max(norm, key=lambda k: norm[k])
        norm[big] += 1.0 - s
    return norm


def sanitize_ml(s: Any) -> str:
    s = str(s).strip()
    if s in ('', '/', '-'):
        return 'NA'
    s = s.replace('*', 'S').replace('_', '-')
    s = re.sub(r'[^A-Za-z0-9-]', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s or 'NA'


def read_table(path: Path) -> list[Row]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        rows = wb.active.iter_rows(values_only=True)
        header_raw = next(rows, None)
        if header_raw is None:
            raise ValueError(f'{path}: sheet is empty')
        header = [str(c).strip() if c is not None else '' for c in header_raw]
        out: list[Row] = []
        for r in rows:
            if all(c is None or str(c).strip() == '' for c in r):
                continue
            out.append(dict(zip(header, r)))
        return out
    finally:
        wb.close()


def _row_get(row: Row, key: str) -> Any:
    key_lower = key.lower()
    for k, v in row.items():
        if k is None:
            continue
        k_str = str(k).strip().lower()
        if k_str == key_lower:
            return v
        if k_str.startswith(key_lower):
            rest = k_str[len(key_lower):]
            if rest and rest[0] in ' ([':
                return v
    return None


def _extract_composition(row: Row) -> dict[str, float]:
    raw: dict[str, float] = {}
    for k, v in row.items():
        if k is None:
            continue
        m = _MONO_COL.match(str(k).strip())
        if not m:
            continue
        val, _ = parse_number_pair(v)
        if val is not None:
            raw[m.group(1)] = val
    return raw


def _build_candidate(name: str, row: Row, source: str) -> Candidate:
    raw_comp = _extract_composition(row)
    if not raw_comp:
        raise ValueError('no Φ columns')
    comp = normalize_composition(raw_comp)
    q_val, q_std = parse_number_pair(_row_get(row, 'Q'))
    if q_val is None or q_val <= 0:
        raise ValueError(f'Q missing or non-positive: {_row_get(row, "Q")!r}')
    cand: Candidate = {
        'name': name,
        'source': source,
        'composition': comp,
        'Q': q_val,
        'Q_source': 'experimental',
    }
    if q_std is not None:
        cand['Q_std'] = q_std
    exp: dict[str, float] = {}
    fa_max, _ = parse_number_pair(_row_get(row, 'Fa_max'))
    fa_ave, fa_std = parse_number_pair(_row_get(row, 'Fa_ave'))
    if fa_max is not None:
        exp['Fa_max'] = fa_max
    if fa_ave is not None:
        exp['Fa_ave'] = fa_ave
        if fa_std is not None:
            exp['Fa_ave_std'] = fa_std
    if exp:
        cand['exp'] = exp
    return cand


def build_from_table1(path: Path, source: str) -> dict[str, Candidate]:
    pool: dict[str, Candidate] = {}
    skipped = 0
    for row in read_table(path):
        name = row.get('No.')
        if name is None or not str(name).strip():
            continue
        name = str(name).strip()
        if not _NAME_RE.match(name):
            continue
        try:
            pool[name] = _build_candidate(name, row, source)
        except (ValueError, TypeError):
            skipped += 1
    if skipped:
        print(f'  [table1] skipped {skipped} malformed row(s)', file=sys.stderr)
    return pool


def build_from_table2(path: Path, source: str) -> dict[str, Candidate]:
    pool: dict[str, Candidate] = {}
    cur_round: Optional[str] = None
    cur_ml: Optional[str] = None
    cur_rank: Optional[str] = None
    skipped = 0
    for row in read_table(path):
        r, m, rk = row.get('Round'), row.get('ML'), row.get('Rank')
        if r is not None and str(r).strip():
            cur_round = str(r).strip()
        if m is not None and str(m).strip():
            cur_ml = str(m).strip()
        if rk is not None and str(rk).strip():
            cur_rank = str(rk).strip()
        no = row.get('No.')
        if no is None or not str(no).strip():
            continue
        no = str(no).strip()
        if cur_round is None or cur_ml is None:
            continue
        if cur_rank and cur_rank.upper() == 'PRED':
            continue
        ml_san = sanitize_ml(cur_ml)
        rank_san = sanitize_ml(cur_rank) if cur_rank else 'NA'
        name = f'ML-R{cur_round}-{ml_san}-{rank_san}{no}'
        try:
            pool[name] = _build_candidate(name, row, source)
        except (ValueError, TypeError):
            skipped += 1
    if skipped:
        print(f'  [table2] skipped {skipped} malformed row(s)', file=sys.stderr)
    return pool


def lookup(pool: dict[str, Any], name: str) -> Optional[str]:
    if name in pool:
        return name
    lower = name.lower()
    for k in pool:
        if k.lower() == lower:
            return k
    norm = re.sub(r'[-_ ]', '', name).lower()
    for k in pool:
        if re.sub(r'[-_ ]', '', k).lower() == norm:
            return k
    return None


def load_wishlist(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.append(line)
    return out


def write_output(out_dir: Path, selected: list[Candidate]) -> None:
    sys_dir = out_dir / 'systems'
    sys_dir.mkdir(parents=True, exist_ok=True)
    idx_lines: list[str] = []
    for cand in selected:
        (sys_dir / f'{cand["name"]}.json').write_text(
            json.dumps(cand, indent=2, ensure_ascii=False), encoding='utf-8')
        idx_lines.append(json.dumps({
            'name': cand['name'],
            'source': cand['source'],
            'file': f'systems/{cand["name"]}.json',
        }, ensure_ascii=False))
    (out_dir / 'index.jsonl').write_text(
        '\n'.join(idx_lines) + '\n', encoding='utf-8')


def coverage_table(selected: list[Candidate]) -> str:
    monomers = ('HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm')
    hdr = f'{"#":>2}  {"name":<24} {"Q":>5}  ' + \
          ' '.join(f'{m:>6}' for m in monomers) + '  Fa_max  Fa_ave'
    lines = [hdr, '-' * len(hdr)]
    for i, c in enumerate(selected, 1):
        comp = c['composition']
        cells = ' '.join(f'{comp.get(m, 0):>6.3f}' for m in monomers)
        fa_max = c.get('exp', {}).get('Fa_max')
        fa_ave = c.get('exp', {}).get('Fa_ave')
        fa_max_s = f'{fa_max:>6.2f}' if isinstance(fa_max, (int, float)) else '     —'
        fa_ave_s = f'{fa_ave:>7.2f}' if isinstance(fa_ave, (int, float)) else '      —'
        lines.append(
            f'{i:>2}  {c["name"]:<24} {c["Q"]:>5.2f}  {cells}  {fa_max_s}  {fa_ave_s}')
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Build requested candidate JSONs from MOESM5/MOESM6.')
    p.add_argument('--table1', type=Path, help='MOESM5 .xlsx')
    p.add_argument('--table2', type=Path, help='MOESM6 .xlsx')
    p.add_argument('--wishlist', type=Path, help='names, one per line')
    p.add_argument('--out', type=Path, help='output directory')
    p.add_argument('--source1', default='article')
    p.add_argument('--source2', default='article-ml')
    p.add_argument('--list-pool', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()

    if not args.table1 and not args.table2:
        p.error('at least one of --table1 / --table2 is required')

    pool1: dict[str, Candidate] = {}
    pool2: dict[str, Candidate] = {}
    try:
        if args.table1:
            print(f'[table1] reading {args.table1}', file=sys.stderr)
            pool1 = build_from_table1(args.table1, args.source1)
            print(f'[table1] {len(pool1)} candidates', file=sys.stderr)
        if args.table2:
            print(f'[table2] reading {args.table2}', file=sys.stderr)
            pool2 = build_from_table2(args.table2, args.source2)
            print(f'[table2] {len(pool2)} candidates (PRED skipped)',
                  file=sys.stderr)
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR reading Excel: {e}', file=sys.stderr)
        sys.exit(1)

    pool: dict[str, Candidate] = {**pool1, **pool2}
    collisions = set(pool1) & set(pool2)
    if collisions:
        print(f'WARN: {len(collisions)} collisions (table1 kept): '
              f'{sorted(collisions)[:5]}', file=sys.stderr)

    if args.list_pool:
        for name in sorted(pool1):
            c = pool1[name]
            fa = c.get('exp', {}).get('Fa_max', '—')
            print(f'{name:<24}\tQ={c["Q"]:.2f}\tFa_max={fa}')
        for name in sorted(pool2):
            c = pool2[name]
            fa = c.get('exp', {}).get('Fa_max', '—')
            print(f'{name:<24}\tQ={c["Q"]:.2f}\tFa_max={fa}')
        return

    if not args.wishlist or not args.out:
        p.error('--wishlist and --out are required unless --list-pool')

    try:
        names = load_wishlist(args.wishlist)
    except FileNotFoundError:
        print(f'ERROR: wishlist not found: {args.wishlist}', file=sys.stderr)
        sys.exit(1)
    if not names:
        print(f'ERROR: wishlist is empty: {args.wishlist}', file=sys.stderr)
        sys.exit(1)

    selected: list[Candidate] = []
    seen: set[str] = set()
    missing: list[str] = []
    for n in names:
        resolved = lookup(pool, n)
        if resolved is None:
            missing.append(n)
            continue
        if resolved in seen:
            print(f'ERROR: duplicate (after normalization): {n} → {resolved}',
                  file=sys.stderr)
            sys.exit(1)
        seen.add(resolved)
        if resolved != n:
            print(f'  resolved: {n} → {resolved}', file=sys.stderr)
        selected.append(pool[resolved])

    if missing:
        print(f'\nERROR: {len(missing)} name(s) not found in pool:',
              file=sys.stderr)
        for n in missing:
            print(f'  {n}', file=sys.stderr)
        print('\nRun with --list-pool to see all available names.',
              file=sys.stderr)
        sys.exit(1)

    for c in selected:
        s = sum(c['composition'].values())
        if abs(s - 1.0) > _SUM_TOL:
            print(f'ERROR: {c["name"]}: composition sums to {s:.8f}',
                  file=sys.stderr)
            sys.exit(1)

    print('\n' + coverage_table(selected), file=sys.stderr)

    if args.validate_only:
        print(f'\nOK: {len(selected)} candidates validated (no files written).',
              file=sys.stderr)
        return

    write_output(args.out, selected)
    print(f'\nWrote {len(selected)} systems to {args.out / "systems"}',
          file=sys.stderr)
    print(f'Index: {args.out / "index.jsonl"}', file=sys.stderr)


if __name__ == '__main__':
    main()
