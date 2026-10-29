#!/usr/bin/env python3
"""
test_emc_monomers.py — quick PCFF/EMC parameterization check per monomer.

Builds a short homopolymer (REPEAT_UNITS × CHAINS) via the same
templates/build.emc.j2 used by pipeline.py, runs EMC, and detects
missing-force-field errors. Runtime: ~30-90 sec per monomer.

Usage:
    python3 test_emc_monomers.py                      # 5 critical (HEA/BA/CBEA/ATAC/PEA)
    python3 test_emc_monomers.py --all                # all monomers with emc_chemistry
    python3 test_emc_monomers.py --only M16 M05 M28   # specific
    python3 test_emc_monomers.py --jobs 8             # parallel
    python3 test_emc_monomers.py --clean              # wipe old work dir first
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from jinja2 import Template

HERE = Path(__file__).resolve().parent
TPL = HERE / 'templates'

REPEAT_UNITS = 5
CHAINS = 2
BOX = 40.0
TIMEOUT = 600

# Fallback names — pick whichever exists in registry
CRITICAL_FALLBACKS = [
    ('HEA', 'M16'),
    ('BA', 'M05'),
    ('CBEA', 'M28'),
    ('ATAC', 'M35'),
    ('PEA', 'M45'),
]


def render_build_emc(out_path: Path, short: str, ec: str) -> None:
    ctx = {
        'name': f'test_{short}',
        'seed': 12345,
        'N_chains': CHAINS,
        'monomers': [{'short': short, 'emc_chemistry': ec, 'nrepeat': REPEAT_UNITS}],
        'Lx': BOX, 'Ly': BOX, 'lzz': BOX,
    }
    tpl = Template((TPL / 'build.emc.j2').read_text(encoding='utf-8'))
    out_path.write_text(tpl.render(**ctx), encoding='utf-8')


def test_monomer(short: str, entry: dict, work_root: Path,
                 emc_cmd: list[str], timeout: int) -> dict:
    ec = entry.get('emc_chemistry')
    name = entry.get('name', short)

    if ec is None:
        return {'short': short, 'name': name,
                'status': 'SKIP', 'reason': 'no emc_chemistry'}

    work = work_root / short
    work.mkdir(parents=True, exist_ok=True)

    try:
        render_build_emc(work / 'build.emc', short, ec)
    except Exception as e:
        return {'short': short, 'name': name,
                'status': 'FAIL', 'reason': f'render: {e}'}

    try:
        res = subprocess.run(emc_cmd, cwd=work,
                             capture_output=True, text=True,
                             timeout=timeout, check=False)
    except FileNotFoundError:
        return {'short': short, 'name': name,
                'status': 'FAIL', 'reason': f'{emc_cmd[0]} not found'}
    except subprocess.TimeoutExpired:
        return {'short': short, 'name': name,
                'status': 'TIMEOUT', 'reason': f'>{timeout}s'}

    (work / 'emc.stdout.log').write_text(res.stdout, encoding='utf-8')
    (work / 'emc.stderr.log').write_text(res.stderr, encoding='utf-8')
    log = res.stdout + '\n' + res.stderr

    failures: list[str] = []
    if res.returncode != 0:
        failures.append(f'exit={res.returncode}')

    # Fatal / suspicious patterns from EMC
    for marker in ('Missing force field parameters',
                   'increment pair',
                   'Invalid atom type',
                   'Program aborted',
                   'no parameters'):
        if marker in log:
            for line in log.splitlines():
                if marker in line:
                    failures.append(line.strip()[:100])
                    break
            else:
                failures.append(marker)

    # Output files must exist and have plausible size
    for fname in ('setup.data', 'setup.params'):
        p = work / fname
        if not p.exists() or p.stat().st_size < 500:
            failures.append(f'{fname} missing/small')

    if failures:
        return {'short': short, 'name': name, 'status': 'FAIL',
                'reason': ' | '.join(failures[:2]),
                'log': str(work / 'emc.stdout.log')}

    # Success — extract atom count
    atoms = '?'
    try:
        sys.path.insert(0, str(HERE))
        from read_data_counts import read_data_counts
        c = read_data_counts(work / 'setup.data')
        atoms = str(c.get('atoms', '?'))
    except Exception:
        pass

    return {'short': short, 'name': name, 'status': 'OK',
            'reason': f'atoms={atoms}',
            'log': str(work / 'emc.stdout.log')}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--registry', type=Path, default=HERE / 'monomers.yaml')
    ap.add_argument('--work', type=Path, default=HERE / 'monomer_tests')
    ap.add_argument('--emc-cmd', default='emc_linux_x86_64 build.emc')
    ap.add_argument('--only', nargs='+', default=None)
    ap.add_argument('--all', action='store_true',
                    help='test every monomer with emc_chemistry set')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--timeout', type=int, default=TIMEOUT)
    ap.add_argument('--clean', action='store_true',
                    help='wipe work dir before running')
    args = ap.parse_args()

    if not args.registry.exists():
        print(f'ERROR: {args.registry} not found', file=sys.stderr)
        sys.exit(1)
    registry = yaml.safe_load(args.registry.read_text(encoding='utf-8'))['monomers']

    if args.only:
        to_test = args.only
    elif args.all:
        to_test = sorted(registry.keys())
    else:
        to_test = []
        for a, b in CRITICAL_FALLBACKS:
            for cand in (a, b):
                if cand in registry:
                    to_test.append(cand)
                    break

    to_test = [s for s in to_test if s in registry]
    if not to_test:
        print('ERROR: no monomers to test (check registry / --only / --all)',
              file=sys.stderr)
        sys.exit(1)

    if args.clean and args.work.exists():
        import shutil
        shutil.rmtree(args.work)

    args.work.mkdir(parents=True, exist_ok=True)
    emc_cmd = args.emc_cmd.split()

    print(f'Testing {len(to_test)} monomers in {args.work}', file=sys.stderr)
    print(f'  repeat={REPEAT_UNITS} chains={CHAINS} box={BOX} '
          f'timeout={args.timeout}s jobs={args.jobs}', file=sys.stderr)
    print(file=sys.stderr)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(test_monomer, s, registry[s],
                             args.work, emc_cmd, args.timeout): s
                   for s in to_test}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"  {r['status']:<8} {r['short']:<6} "
                  f"{r['name'][:48]:<48} {r['reason']}",
                  file=sys.stderr)

    print(file=sys.stderr)
    ok = sum(1 for r in results if r['status'] == 'OK')
    fail = sum(1 for r in results if r['status'] == 'FAIL')
    skip = sum(1 for r in results if r['status'] == 'SKIP')
    tmo = sum(1 for r in results if r['status'] == 'TIMEOUT')
    print(f'=== OK={ok}  FAIL={fail}  SKIP={skip}  TIMEOUT={tmo} ===',
          file=sys.stderr)

    out_json = args.work / 'results.json'
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'Full results: {out_json}', file=sys.stderr)

    if fail or tmo:
        sys.exit(1)


if __name__ == '__main__':
    main()