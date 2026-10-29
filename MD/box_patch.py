#!/usr/bin/env python3
import re, sys; from pathlib import Path
_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
_BOX_RE = re.compile(rf'^(\s*)({_NUM})\s+({_NUM})(\s+zlo\s+zhi\s*)$', re.MULTILINE)
def patch_data_box(path: Path, zlo: float, zhi: float, dry_run: bool = False):
    txt = Path(path).read_text(encoding='utf-8')
    matches = list(_BOX_RE.finditer(txt))
    n = len(matches)
    if n == 0: raise SystemExit(f'{path}: no "zlo zhi" line found. File not modified.')
    if n > 1: raise SystemExit(f'{path}: expected 1 "zlo zhi" line, found {n}. File not modified.')
    old = matches[0].group(0).strip(); new_line = f'{zlo} {zhi} zlo zhi'
    if dry_run: print(f'{path}: would change "{old}" -> "{new_line}"'); return 1
    new_txt = _BOX_RE.sub(rf'\g<1>{zlo} {zhi}\g<4>', txt, count=1)
    tmp = path.with_suffix(path.suffix + '.tmp'); tmp.write_text(new_txt, encoding='utf-8'); tmp.replace(path)
    return 1
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('data', type=Path, help='LAMMPS data file (e.g. setup.data)'); p.add_argument('--zlo', type=float, default=-160.0); p.add_argument('--zhi', type=float, default=160.0); p.add_argument('--dry-run', action='store_true', help='show intended change, do not write'); args = p.parse_args()
    n = patch_data_box(args.data, args.zlo, args.zhi, dry_run=args.dry_run); print(f'{args.data}: {n} substitution(s) for zlo zhi')