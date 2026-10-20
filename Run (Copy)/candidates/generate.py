#!/usr/bin/env python3
"""
generate.py — build candidates/systems/*.json + index.jsonl from an Excel table.

Reads columns:
  No.          candidate name (G-001, G-002, ...)
  Φ<MONO>      mole fractions. Accepted prefixes: Φ, Phi, PHI, phi
               (with optional underscore). Examples: ΦHEA, Φ_HEA,
               PhiBA, PHI_CBEA, phiATAC.
  Q            swelling ratio, "0.42±0.04" or "0.42" or "/"
  Fa_max       experimental adhesion, optional
  Fa_ave       experimental adhesion, "138.41±7.36" or "/"

Output per candidate:
  {
    "name": "G-042",
    "source": "article",
    "composition": {"HEA": 0.141, "BA": 0.591, ...},
    "Q": 0.42,
    "Q_source": "experimental",
    "exp": {"Fa_max": 146.64, "Fa_ave": 138.41}
  }

Composition values are renormalized (no rounding). Monomer keys with
f == 0 are omitted. `name` must match [A-Za-z0-9_.-]+ (same regex as
orchestrator.load_index).

This generator owns candidates/index.jsonl: existing entries with the
same --source are replaced; entries with other sources are preserved.
A backup (index.jsonl.bak) is written before modification.

Usage:
  python generate.py table.xlsx --out . --source article
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import openpyxl


_MONO_COL = re.compile(
    r'^(?:Φ|Phi|PHI|phi)_?([A-Za-z][A-Za-z0-9_]*)$'
)
_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


# =============================================================================
# Parsing helpers
# =============================================================================

def parse_number_pair(s):
    """
    Parse '138.41±7.36', '138.41±', '±7.36', '138.41', 138.41, None.
    Returns (value, std_or_None). Either element may be None.
    """
    if s is None:
        return None, None
    if isinstance(s, (int, float)):
        return float(s), None

    s = str(s).strip()
    if s in ('', '/', '-', 'N/A', 'na', 'NA'):
        return None, None

    for sep in ('±', '+/-', '+-'):
        if sep in s:
            parts = s.split(sep, 1)
            v = sd = None
            try:
                v = float(parts[0].strip())
            except ValueError:
                pass
            if len(parts) > 1 and parts[1].strip():
                try:
                    sd = float(parts[1].strip())
                except ValueError:
                    pass
            return v, sd

    try:
        return float(s), None
    except ValueError:
        return None, None


def normalize_composition(raw: dict):
    """
    Drop zero/None entries, renormalize, then absorb the residual
    float error into the largest element. In practice the result
    satisfies abs(sum - 1.0) < 1e-12, which is well inside the 1e-6
    tolerance enforced by pipeline.validate_candidate.
    """
    comp = {k: float(v) for k, v in raw.items() if v is not None and v > 0}
    if not comp:
        raise ValueError('composition is empty after dropping zeros')

    total = sum(comp.values())
    if total <= 0:
        raise ValueError(f'composition sum is non-positive: {total}')

    norm = {k: v / total for k, v in comp.items()}

    s = sum(norm.values())
    if abs(s - 1.0) > 1e-12:
        big = max(norm, key=norm.get)
        norm[big] += (1.0 - s)

    return norm


# =============================================================================
# Row processing
# =============================================================================
def _row_get(row: dict, key: str):
    """
    Case-insensitive prefix lookup. Header "Fa_max (kPa)" matches key
    "Fa_max" because the remainder starts with a space. Header "Q_swell"
    does NOT match key "Q" (remainder starts with "_").
    """
    key_lower = key.lower()
    for k in row:
        if k is None:
            continue
        k_str = str(k).strip()
        k_lower = k_str.lower()
        if k_lower == key_lower:
            return row[k]
        if k_lower.startswith(key_lower):
            rest = k_str[len(key):]
            if rest and rest[0] in ' ([':
                return row[k]
    return None


def build_candidate(row: dict, source: str, source_row: int):
    name = row.get('No.')
    if name is None or not str(name).strip():
        raise ValueError(f'row {source_row}: empty "No."')
    name = str(name).strip()

    if not _NAME_RE.match(name):
        raise ValueError(
            f'row {source_row}: name "{name}" must match {_NAME_RE.pattern}'
        )

    raw_comp = {}
    for k, v in row.items():
        if k is None:
            continue
        m = _MONO_COL.match(str(k).strip())
        if not m:
            continue
        monomer = m.group(1)
        val, _ = parse_number_pair(v)
        if val is not None:
            raw_comp[monomer] = val

    if not raw_comp:
        raise ValueError(
            f'{name}: no Φ columns found '
            f'(header keys: {sorted(str(k) for k in row.keys() if k)})'
        )

    composition = normalize_composition(raw_comp)

    Q, Q_std = parse_number_pair(_row_get(row, 'Q'))
    if Q is None or Q <= 0:
        raise ValueError(
            f'{name}: Q missing or non-positive: '
            f'{_row_get(row, "Q")!r} '
            f'(header keys: {sorted(str(k) for k in row.keys() if k)})'
        )

    exp = {}
    Fa_max, _      = parse_number_pair(_row_get(row, 'Fa_max'))
    Fa_ave, Fa_std = parse_number_pair(_row_get(row, 'Fa_ave'))
    if Fa_max is not None:
        exp['Fa_max'] = Fa_max
    if Fa_ave is not None:
        exp['Fa_ave'] = Fa_ave
        if Fa_std is not None:
            exp['Fa_ave_std'] = Fa_std

    out = {
        'name':        name,
        'source':      source,
        'composition': composition,
        'Q':           Q,
        'Q_source':    'experimental',
    }
    if Q_std is not None:
        out['Q_std'] = Q_std
    if exp:
        out['exp'] = exp
    return out


# =============================================================================
# Excel reading
# =============================================================================

def read_table(path: Path) -> list:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        ws = wb.active
        rows = ws.iter_rows(values_only=True)
        try:
            header_raw = next(rows)
        except StopIteration:
            raise ValueError(f'{path}: sheet is empty') from None

        header = [str(c).strip() if c is not None else '' for c in header_raw]

        out = []
        for r in rows:
            if all(c is None or str(c).strip() == '' for c in r):
                continue
            out.append(dict(zip(header, r)))
        return out
    finally:
        wb.close()


# =============================================================================
# Index merge
# =============================================================================

def load_index_entries(path: Path) -> list:
    """Read existing index.jsonl entries (name, source, file)."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_index(path: Path, new_entries: list, source: str):
    """
    Preserve existing entries with source != `source`; replace entries
    with source == `source`. Write a backup first.
    """
    old = load_index_entries(path)
    kept = [e for e in old if e.get('source') != source]
    replaced = len(old) - len(kept)

    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + '.bak'))

    all_entries = kept + new_entries
    path.write_text(
        '\n'.join(json.dumps(r, ensure_ascii=False) for r in all_entries) + '\n',
        encoding='utf-8',
    )
    return replaced, len(all_entries)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('xlsx', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=str, required=True)
    args = p.parse_args()

    if not args.source.strip():
        print('ERROR: --source must be non-empty', file=sys.stderr)
        sys.exit(1)

    out_dir = args.out / 'systems'
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out / 'index.jsonl'

    try:
        rows = read_table(args.xlsx)
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'Read {len(rows)} data rows from {args.xlsx}', file=sys.stderr)

    index_lines = []
    seen_names  = set()
    ok, skipped = 0, 0

    for i, row in enumerate(rows, 2):
        try:
            cand = build_candidate(row, args.source, source_row=i)
        except ValueError as e:
            print(f'  SKIP: {e}', file=sys.stderr)
            skipped += 1
            continue

        if cand['name'] in seen_names:
            print(f'  SKIP: duplicate name "{cand["name"]}" at row {i}',
                  file=sys.stderr)
            skipped += 1
            continue
        seen_names.add(cand['name'])

        cand_path = out_dir / f'{cand["name"]}.json'
        cand_path.write_text(
            json.dumps(cand, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        index_lines.append({
            'name':   cand['name'],
            'source': args.source,
            'file':   f'systems/{cand["name"]}.json',
        })
        ok += 1

    # Warn about orphan .json files in out_dir that are not in this run
    new_names = {f'{e["name"]}.json' for e in index_lines}
    existing  = {p.name for p in out_dir.glob('*.json')}
    orphans   = existing - new_names
    if orphans:
        shown = sorted(orphans)[:5]
        print(f'WARNING: {len(orphans)} stale .json in {out_dir} '
              f'(not in this run): {shown}{"..." if len(orphans) > 5 else ""}',
              file=sys.stderr)

    replaced, total = write_index(index_path, index_lines, args.source)

    print(f'\nWrote {ok} systems to {out_dir}', file=sys.stderr)
    print(f'Index: {total} entries ({replaced} replaced, '
          f'{len(index_lines)} new) in {index_path}', file=sys.stderr)
    if skipped:
        print(f'Skipped {skipped} rows', file=sys.stderr)

    print(json.dumps({
        'ok':       ok,
        'skipped':  skipped,
        'index':    str(index_path.resolve()),
        'total':    total,
    }))


if __name__ == '__main__':
    main()