#!/usr/bin/env python3
import argparse, json, shlex, shutil, subprocess, sys; from pathlib import Path; import yaml; from jinja2 import Template
from sizing import compute_sizing; from box_patch import patch_data_box; from read_data_counts import read_data_counts, format_extra_flags
HERE = Path(__file__).resolve().parent; TPL = HERE / 'templates'
REQUIRED_TEMPLATES = ['build.emc.j2', 'in.comp.j2', 'in.equil.j2', 'in.smd.j2']
REQUIRED_CFG_KEYS = ['N', 'C0', 'C_out', 'rho_solution', 'K_pack',
                     'Lx', 'Ly', 'Lz',
                     'Z_film_half', 'Z_box_half', 'Z_final_half',
                     'SMD_velocities']
_POSITIVE_FLOAT_KEYS = ['C0', 'rho_solution', 'K_pack',
                        'Lx', 'Ly', 'Lz',
                        'Z_film_half', 'Z_box_half', 'Z_final_half']
def validate_config(cfg: dict):
    missing = [k for k in REQUIRED_CFG_KEYS if k not in cfg]
    if missing: raise KeyError(f'config.yaml missing keys: {missing}')
    if not isinstance(cfg['N'], int) or cfg['N'] <= 0: raise ValueError(f"N must be a positive integer, got {cfg['N']!r}")
    for key in _POSITIVE_FLOAT_KEYS:
        v = cfg[key]
        if not isinstance(v, (int, float)) or v <= 0: raise ValueError(f"{key} must be positive, got {v!r}")
    if cfg['C_out'] < 0: raise ValueError(f"C_out must be non-negative, got {cfg['C_out']}")
    if cfg['Z_film_half'] > cfg['Z_box_half']: raise ValueError(f"Z_film_half ({cfg['Z_film_half']}) > Z_box_half ({cfg['Z_box_half']}): film does not fit in box")
    if cfg['Z_final_half'] > cfg['Z_box_half']: raise ValueError(f"Z_final_half ({cfg['Z_final_half']}) > Z_box_half ({cfg['Z_box_half']}): soft-compress would expand")
    fp = cfg.get('frac_poly', 0.5)
    if not isinstance(fp, (int, float)) or not (0.0 <= fp <= 1.0): raise ValueError(f"frac_poly must be in [0, 1], got {fp!r}")
    smd = cfg['SMD_velocities']
    if not isinstance(smd, list) or not smd: raise ValueError(f"SMD_velocities must be a non-empty list, got {smd!r}")
def validate_registry(registry: dict, path: Path):
    if not isinstance(registry, dict) or not registry: raise ValueError(f'{path}: expected non-empty dict with monomers')
def validate_candidate(candidate: dict, registry: dict):
    for key in ('name', 'composition', 'Q'):
        if key not in candidate: raise ValueError(f'candidate missing "{key}"')
    if not isinstance(candidate['Q'], (int, float)) or candidate['Q'] <= 0: raise ValueError(f"Q must be a positive number, got {candidate['Q']!r}")
    comp = candidate['composition']
    if not isinstance(comp, dict) or not comp: raise ValueError('composition must be a non-empty dict')
    for s, f in comp.items():
        if not isinstance(f, (int, float)): raise ValueError(f"composition['{s}'] is not a number: {f!r}")
        if f < 0: raise ValueError(f"composition['{s}'] is negative: {f}")
    total = sum(comp.values())
    if abs(total - 1.0) > 1e-6: raise ValueError(f"{candidate['name']}: composition sums to {total}, expected 1.0")
    missing = [s for s in comp if s not in registry]
    if missing: raise KeyError(f"{candidate['name']}: monomers not in registry: {missing}")
    no_emc = [s for s in comp if registry[s].get('emc_chemistry') is None]
    if no_emc: raise ValueError(f"{candidate['name']}: monomers missing emc_chemistry (fill MANUAL_EMC in build_registry.py): {no_emc}")
    for s in comp:
        for key in ('smiles', 'M', 'charge'):
            if registry[s].get(key) is None: raise KeyError(f"{candidate['name']}: registry[{s}] has no '{key}'")
def validate_templates():
    missing = [t for t in REQUIRED_TEMPLATES if not (TPL / t).exists()]
    if missing: raise FileNotFoundError(f'Missing templates in {TPL}: {missing}')
    for asset in ('Silico_system.data', 'water.mol'):
        if not (TPL / asset).exists():
            raise FileNotFoundError(f'Missing asset in {TPL}: {asset}')
def render(tpl_name: str, out_path: Path, ctx: dict):
    tpl = Template((TPL / tpl_name).read_text(encoding='utf-8')); out_path.write_text(tpl.render(**ctx), encoding='utf-8')
def nrepeat(composition: dict, N: int) -> dict:
    shorts = list(composition.keys()); counts = [round(composition[s] * N) for s in shorts]; counts[-1] = N - sum(counts[:-1])
    if counts[-1] < 0: raise ValueError(f'nrepeat rounding underflow for {shorts[-1]}: {counts}')
    for s, c, f in zip(shorts, counts, composition.values()):
        if f > 0 and c == 0: raise ValueError(f'nrepeat for {s} rounded to 0 with f={f}; check composition granularity (N={N})')
    return {s: c for s, c in zip(shorts, counts)}
def monomers_for_template(composition: dict, N: int, registry: dict) -> list:
    nrep = nrepeat(composition, N)
    return [{'short': s, 'emc_chemistry': registry[s]['emc_chemistry'], 'nrepeat': nrep[s]} for s in composition]
def frac_split(total: int, frac: float):
    a = int(total * frac + 0.5); return a, total - a

def compute_bath(Lx, Ly, Z_bath, C_NaCl, rho_solution):
    NA = 6.022e23
    M_Na, M_Cl, M_H2O = 22.990, 35.453, 18.015
    V_A3  = Lx * Ly * Z_bath
    V_cm3 = V_A3 * 1e-24
    V_L   = V_cm3 * 1e-3
    n_NaCl      = C_NaCl * V_L
    m_NaCl      = n_NaCl * (M_Na + M_Cl)
    m_solution  = V_cm3 * rho_solution
    m_water     = m_solution - m_NaCl
    n_water     = m_water / M_H2O
    return {'N_Na':    round(n_NaCl * NA), 'N_Cl':    round(n_NaCl * NA), 'N_water': round(n_water * NA),}
def run_cmd(cmd: list, cwd: Path, stage: str):
    if not cmd: raise ValueError(f'[{stage}] empty command')
    try: subprocess.run(cmd, cwd=cwd, check=True)
    except FileNotFoundError as e: raise RuntimeError(f'[{stage}] command not found: {cmd[0]}') from e
    except subprocess.CalledProcessError as e: raise RuntimeError(f'[{stage}] command failed with exit code {e.returncode}: {" ".join(cmd)} (cwd={cwd})') from e
def log(name: str, stage: str, msg: str): print(f'[{name}] {stage}: {msg}', file=sys.stderr, flush=True)
def dump_json(path: Path, obj: dict): path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
def build_system(run_dir: Path, candidate: dict, seed: int, cfg: dict, registry: dict, emc_cmd: list = None, lmp_cmd: list = None) -> Path:
    name = candidate['name']; log(name, 'validate', 'checking config, candidate, templates')
    validate_config(cfg); validate_candidate(candidate, registry); validate_templates()
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    composition = candidate['composition']; Q = candidate['Q']; local_cfg = {**cfg, 'Q': Q}
    log(name, 'sizing', 'computing V_w, chain and ion counts')
    sizing = compute_sizing(composition, registry, local_cfg); dump_json(run_dir / 'sizing.json', sizing)
    log(name, 'sizing', f"N_chains={sizing['N_chains']} N_water={sizing['N_water']} N_Na={sizing['N_Na']} N_Cl={sizing['N_Cl']}")
    log(name, 'emc', 'rendering build.emc')
    emc_dir = run_dir / 'emc'; emc_dir.mkdir(exist_ok=True)
    render('build.emc.j2', emc_dir / 'build.emc', {'name': name, 'seed': seed, 'N_chains': sizing['N_chains'], 'monomers': monomers_for_template(composition, cfg['N'], registry), 'Lx': cfg['Lx'], 'Ly': cfg['Ly'], 'lzz': cfg['Lz']})
    log(name, 'emc', 'running EMC')
    emc_cmd = emc_cmd or ['emc_linux_x86_64', 'build.emc']; run_cmd(emc_cmd, emc_dir, stage='emc')
    for fname in ('setup.data', 'setup.params'):
        if not (emc_dir / fname).exists(): raise FileNotFoundError(f'EMC did not produce {emc_dir / fname}')
    log(name, 'box_patch', f'setting zlo zhi to {-cfg["Z_box_half"]} {cfg["Z_box_half"]}')
    try: patch_data_box(emc_dir / 'setup.data', zlo=-cfg['Z_box_half'], zhi=cfg['Z_box_half'])
    except SystemExit as e: raise RuntimeError(f'[{name}] box_patch failed: {e}') from None
    log(name, 'compress', 'rendering in.comp')
    comp_dir = run_dir / 'compress'; comp_dir.mkdir(exist_ok=True)
    shutil.copy(emc_dir / 'setup.data', comp_dir / 'setup.data'); shutil.copy(emc_dir / 'setup.params', comp_dir / 'setup.params')
    render('in.comp.j2', comp_dir / 'in.comp', {'name': name, 'Z_film_half': cfg['Z_film_half'], 'Z_box_half': cfg['Z_box_half'], 'Z_final_half': cfg['Z_final_half']})
    log(name, 'compress', 'running LAMMPS (pair_style soft)')
    lmp_cmd = lmp_cmd or [
    'lmp',
    '-k', 'on', 'g', '1', '-sf', 'kk',
    '-pk', 'kokkos', 'newton', 'off', 'comm', 'device',
    '-in', 'in.comp',
    '-log', 'log.lammps',
    '-screen', 'screen.txt',
    ]
    run_cmd(lmp_cmd, comp_dir, stage='compress')
    polymer_data = comp_dir / f'Polymer_{name}.data'
    if not polymer_data.exists(): raise FileNotFoundError(f'Expected {polymer_data} after compress, but it is missing. Check templates/in.comp.j2 -- the write_data line must produce exactly this filename.')
    log(name, 'counts', f'parsing {polymer_data.name}')
    counts = read_data_counts(polymer_data)
    extra  = format_extra_flags(counts)
    log(name, 'counts', f'atoms={counts.get("atoms")} '
                        f'atom_types={counts["atom_types"]} '
                        f'bond_types={counts["bond_types"]}')
    log(name, 'md', 'rendering in.equil, in.smd')
    md_dir = run_dir / 'md'; md_dir.mkdir(exist_ok=True)
    shutil.copy(polymer_data, md_dir / polymer_data.name)
    dump_json(md_dir / 'counts.json', counts)

    for fname in ('Silico_system.data', 'water.mol'):
        src = TPL / fname
        if not src.exists():
            raise FileNotFoundError(f'{src} not found -- put {fname} in {TPL}/')
        shutil.copy(src, md_dir / fname)
        log(name, 'md', f'copied {fname} -> md/')
    N_water_poly = sizing['N_water']
    N_Na_poly    = sizing['N_Na']
    N_Cl_poly    = sizing['N_Cl']
    Z_bath = cfg.get('Z_bath', 50.0)
    bath = compute_bath(cfg['Lx'], cfg['Ly'], Z_bath, cfg['C_out'], cfg['rho_solution'])
    N_water_bath = bath['N_water']
    N_Na_bath    = bath['N_Na']
    N_Cl_bath    = bath['N_Cl']
    log(name, 'md', f'poly: H2O={N_water_poly} Na={N_Na_poly} Cl={N_Cl_poly}')
    log(name, 'md', f'bath: H2O={N_water_bath} Na={N_Na_bath} Cl={N_Cl_bath}')
    md_ctx = {'name': name, 'seed': seed, 'Q': Q, 'N_chains': sizing['N_chains'], 'extra': extra, 'N_water_poly': N_water_poly, 'N_water_bath': N_water_bath, 'N_Na_poly': N_Na_poly, 'N_Na_bath': N_Na_bath, 'N_Cl_poly': N_Cl_poly, 'N_Cl_bath': N_Cl_bath, 'SMD_velocities': cfg['SMD_velocities']}
    render('in.equil.j2', md_dir / 'in.equil', md_ctx); render('in.smd.j2', md_dir / 'in.smd', md_ctx)
    dump_json(run_dir / '.done', {'name': name, 'seed': seed, 'N_chains': sizing['N_chains'], 'N_monomers': sizing['N_monomers'], 'N_water': sizing['N_water'], 'N_Na': sizing['N_Na'], 'N_Cl': sizing['N_Cl']})
    log(name, 'done', f'wrote {run_dir / ".done"}')
    return run_dir
if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('candidate', type=Path, help='JSON file with a single candidate'); p.add_argument('--seed', type=int, required=True, help='random seed (333333 / 666666 / 999999)'); p.add_argument('--out', type=Path, default=Path('runs'), help='output directory (default: runs/)'); p.add_argument('--config', type=Path, default=HERE / 'config.yaml'); p.add_argument('--registry', type=Path, default=HERE / 'monomers.yaml'); p.add_argument('--emc-cmd', type=str, default=None, help='override EMC command (default: emc_linux_x86_64 build.emc)'); p.add_argument('--lmp-cmd', type=str, default=None, help='override LAMMPS command (default: lmp -in in.comp)'); args = p.parse_args()
    try:
        cfg = yaml.safe_load(args.config.read_text(encoding='utf-8')); registry = yaml.safe_load(args.registry.read_text(encoding='utf-8'))['monomers']; candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
    except (FileNotFoundError, KeyError, yaml.YAMLError, json.JSONDecodeError) as e: print(f'ERROR reading inputs: {e}', file=sys.stderr); sys.exit(1)
    try: validate_registry(registry, args.registry)
    except ValueError as e: print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
    emc_cmd = shlex.split(args.emc_cmd) if args.emc_cmd else None; lmp_cmd = shlex.split(args.lmp_cmd) if args.lmp_cmd else None
    run_dir = args.out / f"{candidate['name']}_s{args.seed}"
    try: build_system(run_dir, candidate, args.seed, cfg, registry, emc_cmd=emc_cmd, lmp_cmd=lmp_cmd)
    except (Exception, SystemExit) as e: print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
    print(f'Built: {run_dir}')