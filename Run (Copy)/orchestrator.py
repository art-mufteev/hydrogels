#!/usr/bin/env python3
import argparse, json, os, re, subprocess, sys; from datetime import datetime, timezone; from pathlib import Path; import yaml
HERE = Path(__file__).resolve().parent
SEEDS = [333333, 666666, 999999]
_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
REQUIRED_MD_FILES = ('in.equil', 'in.smd')
def now_iso() -> str: return datetime.now(timezone.utc).isoformat(timespec='seconds')
def append_state(path: Path, record: dict):
    line = (json.dumps(record) + '\n').encode('utf-8'); fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try: os.write(fd, line)
    finally: os.close(fd)
def load_state(path: Path) -> dict:
    state = {}
    if not path.exists(): return state
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except json.JSONDecodeError: print(f'WARNING: {path}:{i}: malformed JSON, skipping', file=sys.stderr); continue
        key = (rec.get('name'), rec.get('seed'))
        if key[0] is None or key[1] is None: continue
        state[key] = rec
    return state
def load_index(path: Path) -> list:
    out = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line: continue
        try: rec = json.loads(line)
        except json.JSONDecodeError as e: raise ValueError(f'{path}:{i}: invalid JSON: {e}') from e
        for key in ('name', 'file'):
            if key not in rec: raise ValueError(f'{path}:{i}: missing "{key}"')
        if not _NAME_RE.match(rec['name']): raise ValueError(f'{path}:{i}: name "{rec["name"]}" must match {_NAME_RE.pattern}')
        out.append(rec)
    return out
def resolve_candidate_path(candidates_dir: Path, rel: str) -> Path:
    base = candidates_dir.resolve(); p = (candidates_dir / rel).resolve()
    if not str(p).startswith(str(base) + os.sep): raise ValueError(f'candidate file escapes candidates dir: {rel}')
    return p
def is_built(run_dir: Path) -> bool:
    marker = run_dir / '.done'
    if not marker.exists(): return False
    try: json.loads(marker.read_text(encoding='utf-8')); return True
    except (json.JSONDecodeError, OSError): print(f'WARNING: {marker} is malformed, will rebuild', file=sys.stderr); return False
def _bash_dquote_escape(s: str) -> str: return s.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
def render_sbatch(name: str, seed: int, cfg: dict) -> str:
    ctr = cfg['container_image']; md = cfg['md_cmd']; md_q = _bash_dquote_escape(md)
    return f"""#!/bin/bash
#SBATCH --job-name={name}_s{seed}
#SBATCH --partition={cfg['partition']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cfg.get('cpus_per_task', 8)}
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

srun --mpi=pmi2 \\
     --container-image={ctr} \\
     --container-mounts="$PWD:/work" \\
     --container-workdir=/work \\
     bash -lc "set -e; {md_q} -in in.equil"
"""
def submit_md(run_dir: Path, name: str, seed: int, cfg: dict) -> str:
    md_dir = run_dir / 'md'; md_dir.mkdir(parents=True, exist_ok=True)
    required = list(REQUIRED_MD_FILES) + [f'Polymer_{name}.data']
    missing = [f for f in required if not (md_dir / f).exists()]
    if missing: raise RuntimeError(f'{md_dir}: missing {missing}; run_dir may be partially built -- rerun pipeline')
    sbatch_path = md_dir / 'job.sbatch'; sbatch_path.write_text(render_sbatch(name, seed, cfg), encoding='utf-8')
    try: res = subprocess.run(['sbatch', '--parsable', str(sbatch_path)], cwd=md_dir, check=True, capture_output=True, text=True)
    except FileNotFoundError as e: raise RuntimeError('sbatch not found in PATH') from e
    except subprocess.CalledProcessError as e: raise RuntimeError(f'sbatch failed (rc={e.returncode}): {e.stderr.strip() if e.stderr else "(no stderr)"}') from e
    jobid = res.stdout.strip()
    if not jobid: raise RuntimeError(f'sbatch returned empty jobid for {sbatch_path}')
    return jobid
def run_pipeline(candidate_path: Path, seed: int, out_dir: Path, config: Path, registry: Path, emc_cmd: str | None, compress_cmd: str | None):
    cmd = [sys.executable, str(HERE / 'pipeline.py'), str(candidate_path), '--seed', str(seed), '--out', str(out_dir), '--config', str(config), '--registry', str(registry)]
    if emc_cmd: cmd += ['--emc-cmd', emc_cmd]
    if compress_cmd: cmd += ['--lmp-cmd', compress_cmd]
    subprocess.run(cmd, check=True)
def process_one(entry: dict, seed: int, cfg: dict, args, state: dict, state_path: Path) -> str:
    name = entry['name']; key = (name, seed); run_dir = args.out / f'{name}_s{seed}'; built = is_built(run_dir)
    prev = state.get(key, {}); md_submitted = prev.get('event') == 'submitted_md'
    if md_submitted and built: return 'skipped'
    if md_submitted and not built: print(f'[{name} s{seed}] WARN: state says submitted but .done missing -- will rebuild', file=sys.stderr, flush=True)
    if not built:
        if args.no_build: return 'skipped'
        try: candidate_path = resolve_candidate_path(args.candidates, entry['file'])
        except ValueError as e: append_state(state_path, {'ts': now_iso(), 'name': name, 'seed': seed, 'event': 'failed_build', 'error': str(e)}); return 'failed'
        print(f'[{name} s{seed}] building', file=sys.stderr, flush=True)
        try: run_pipeline(candidate_path, seed, args.out, args.config, args.registry, args.emc_cmd, args.compress_cmd)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            rc = getattr(e, 'returncode', None); append_state(state_path, {'ts': now_iso(), 'name': name, 'seed': seed, 'event': 'failed_build', 'rc': rc, 'error': str(e)}); return 'failed'
        append_state(state_path, {'ts': now_iso(), 'name': name, 'seed': seed, 'event': 'built', 'run_dir': str(run_dir)})
    if args.no_md: return 'built'
    print(f'[{name} s{seed}] submitting MD', file=sys.stderr, flush=True)
    try: jobid = submit_md(run_dir, name, seed, cfg)
    except (RuntimeError, FileNotFoundError) as e: append_state(state_path, {'ts': now_iso(), 'name': name, 'seed': seed, 'event': 'failed_submit', 'error': str(e)}); return 'failed'
    append_state(state_path, {'ts': now_iso(), 'name': name, 'seed': seed, 'event': 'submitted_md', 'jobid': jobid, 'run_dir': str(run_dir)}); return 'submitted'
def main():
    p = argparse.ArgumentParser(); p.add_argument('--candidates', type=Path, default=HERE / 'candidates'); p.add_argument('--index', type=Path, default=None); p.add_argument('--out', type=Path, default=HERE / 'runs'); p.add_argument('--state', type=Path, default=None); p.add_argument('--config', type=Path, default=HERE / 'config.yaml'); p.add_argument('--registry', type=Path, default=HERE / 'monomers.yaml'); p.add_argument('--slurm-config', type=Path, default=HERE / 'slurm.yaml'); p.add_argument('--seeds', type=int, nargs='+', default=SEEDS); p.add_argument('--source', type=str, default=None); p.add_argument('--name', type=str, nargs='+', default=None); p.add_argument('--no-build', action='store_true'); p.add_argument('--no-md', action='store_true'); p.add_argument('--emc-cmd', type=str, default=None, help='EMC command for pipeline (default: emc_linux_x86_64 build.emc)'); p.add_argument('--compress-cmd', type=str, default=None, help='LAMMPS command for the COMPRESS stage in pipeline (default: lmp -in in.comp). The MD (equil+SMD) command lives in slurm.yaml as `md_cmd`.'); args = p.parse_args()
    if args.no_build and args.no_md: print('WARNING: --no-build and --no-md are both set; nothing to do', file=sys.stderr)
    if args.index is None: args.index = args.candidates / 'index.jsonl'
    if args.state is None: args.state = args.out / 'state.jsonl'
    args.out.mkdir(parents=True, exist_ok=True)
    try: cfg = yaml.safe_load(args.config.read_text(encoding='utf-8')); slurm_cfg = yaml.safe_load(args.slurm_config.read_text(encoding='utf-8'))
    except (FileNotFoundError, yaml.YAMLError) as e: print(f'ERROR reading config: {e}', file=sys.stderr); sys.exit(1)
    for cname, obj in (('config.yaml', cfg), ('slurm.yaml', slurm_cfg)):
        if not isinstance(obj, dict): print(f'ERROR: {cname} must be a mapping, got {type(obj).__name__}', file=sys.stderr); sys.exit(1)
    overlap = set(cfg) & set(slurm_cfg)
    if overlap: print(f'WARNING: keys in both config.yaml and slurm.yaml: {sorted(overlap)} (slurm.yaml wins)', file=sys.stderr)
    merged = {**cfg, **slurm_cfg}
    for key in ('container_image', 'partition', 'md_cmd'):
        if key not in merged: print(f'ERROR: slurm.yaml must define "{key}"', file=sys.stderr); sys.exit(1)
    try: index = load_index(args.index)
    except (FileNotFoundError, ValueError) as e: print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
    state = load_state(args.state)
    total_index = len(index)
    if args.source: index = [e for e in index if e.get('source') == args.source]
    if args.name:
        wanted = set(args.name); index = [e for e in index if e['name'] in wanted]
    suffix = ''
    if len(index) != total_index: suffix = f' (filtered from {total_index})'
    print(f'Candidates: {len(index)}{suffix}  Seeds: {len(args.seeds)}', file=sys.stderr)
    counts = {'skipped': 0, 'built': 0, 'submitted': 0, 'failed': 0}
    for entry in index:
        for seed in args.seeds:
            try: status = process_one(entry, seed, merged, args, state, args.state)
            except Exception as e: print(f'[{entry.get("name","?")} s{seed}] ERROR: {e}', file=sys.stderr, flush=True); status = 'failed'
            counts.setdefault(status, 0); counts[status] += 1
    print(); print(f'skipped:   {counts["skipped"]}'); print(f'built:     {counts["built"]}'); print(f'submitted: {counts["submitted"]}'); print(f'failed:    {counts["failed"]}')
if __name__ == '__main__': main()