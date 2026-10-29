#!/usr/bin/env python3
import argparse, json, re, sys; from pathlib import Path
_COUNT_RE = re.compile(r'^\s*(\d+)\s+(atoms?|bonds?|angles?|dihedrals?|impropers?|atom\s+types|bond\s+types|angle\s+types|dihedral\s+types|improper\s+types)\s*(?:#.*)?$', re.MULTILINE | re.IGNORECASE)
_SECTION_RE = re.compile(r'^\s*((?:Atom|Bond|Angle|Dihedral|Improper)\s+Type\s+Labels|PairIJ\s+Coeffs|Pair\s+Coeffs|Bond\s+Coeffs|Angle\s+Coeffs|Dihedral\s+Coeffs|Improper\s+Coeffs|BondBond\s+Coeffs|BondAngle\s+Coeffs|MiddleBondTorsion\s+Coeffs|EndBondTorsion\s+Coeffs|AngleTorsion\s+Coeffs|AngleAngleTorsion\s+Coeffs|BondBond13\s+Coeffs|AngleAngle\s+Coeffs|Masses|Atoms|Velocities|Ellipsoids|Lines|Triangles|Bodies|Bonds|Angles|Dihedrals|Impropers)\b', re.IGNORECASE)
_TYPE_KEYS = [('atom_types','extra/atom/types'),('bond_types','extra/bond/types'),('angle_types','extra/angle/types'),('dihedral_types','extra/dihedral/types'),('improper_types','extra/improper/types')]
def _key(label: str) -> str: return re.sub(r'\s+', '_', label.strip().lower())
def _read_header(path: Path) -> str:
    lines = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if _SECTION_RE.match(line): break
            lines.append(line)
    return ''.join(lines)
def read_data_counts(path: Path) -> dict:
    txt = _read_header(Path(path))
    counts = {}
    for m in _COUNT_RE.finditer(txt):
        n = int(m.group(1)); k = _key(m.group(2))
        if k in counts and counts[k] != n: raise ValueError(f'Duplicate section "{k}": {counts[k]} vs {n}')
        counts[k] = n
    if not counts: raise ValueError(f'{path}: no count lines found in header')
    return counts
def format_extra_flags(counts: dict, delta: dict | None = None) -> str:
    delta = delta or {}
    mapping = [
        ('atom_types',     'extra/atom/types',     'atom'),
        ('bond_types',     'extra/bond/types',     'bond'),
        ('angle_types',    'extra/angle/types',    'angle'),
        ('dihedral_types', 'extra/dihedral/types', 'dihedral'),
        ('improper_types', 'extra/improper/types', 'improper'),
    ]
    lines = []
    for key, flag, short in mapping:
        n = counts.get(key, 0) + delta.get(short, 0)
        if n > 0:
            lines.append(f'{flag} {n}')
    return ' &\n'.join(lines)
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('data', type=Path, help='LAMMPS data file'); args = p.parse_args()
    try: counts = read_data_counts(args.data)
    except (ValueError, FileNotFoundError) as e: print(f'error: {e}', file=sys.stderr); sys.exit(1)
    print(json.dumps(counts, indent=2, sort_keys=True)); print(); print('read_data flags:'); print(format_extra_flags(counts))