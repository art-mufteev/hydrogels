#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Template

HERE = Path(__file__).resolve().parent
TPL = HERE / 'templates'

try:
    from pipeline import compute_bath
    from read_data_counts import read_data_counts, format_extra_flags
except ImportError as e:
    print(f'WARNING: import failed: {e}', file=sys.stderr)
    compute_bath = None
    format_extra_flags = None

SEEDS_DEFAULT = [333333, 666666, 999999]
_NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
REQUIRED_MD_FILES = ('in.equil', 'in.smd')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def sha256_of(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def render_template(tpl_name: str, out_path: Path, ctx: dict[str, Any]) -> None:
    tpl = Template((TPL / tpl_name).read_text(encoding='utf-8'))
    out_path.write_text(tpl.render(**ctx), encoding='utf-8')


def _bash_dquote_escape(s: str) -> str:
    return (s.replace('\\', '\\\\').replace('"', '\\"')
             .replace('$', '\\$').replace('`', '\\`'))


class Logger:
    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        line = {'ts': now_iso(), 'level': level, 'event': event, **kwargs}
        s = json.dumps(line, default=str)
        print(s, file=sys.stderr, flush=True)
        if self.log_path:
            with self.log_path.open('a', encoding='utf-8') as f:
                f.write(s + '\n')

    def info(self, event: str, **kw: Any) -> None: self._emit('INFO', event, **kw)
    def warn(self, event: str, **kw: Any) -> None: self._emit('WARN', event, **kw)
    def error(self, event: str, **kw: Any) -> None: self._emit('ERROR', event, **kw)
    def progress(self, name: str, seed: int, stage: str, **kw: Any) -> None:
        self.info('progress', name=name, seed=seed, stage=stage, **kw)


def append_state(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, default=str) + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def load_state(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    state: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return state
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        name, seed = rec.get('name'), rec.get('seed')
        if name is None or seed is None:
            continue
        state[(name, seed)] = rec
    return state


def load_index(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for k in ('name', 'file'):
            if k not in rec:
                raise ValueError(f'{path}:{i}: missing "{k}"')
        if not _NAME_RE.match(rec['name']):
            raise ValueError(f'{path}:{i}: bad name "{rec["name"]}"')
        out.append(rec)
    return out


def resolve_candidate_path(candidates_dir: Path, rel: str) -> Path:
    base = candidates_dir.resolve()
    p = (candidates_dir / rel).resolve()
    if not str(p).startswith(str(base) + os.sep):
        raise ValueError(f'candidate file escapes: {rel}')
    return p


def run_dir_for(out_root: Path, name: str, seed: int) -> Path:
    return out_root / name / f'{name}_s{seed}'


def is_built(run_dir: Path) -> bool:
    marker = run_dir / '.done'
    if not marker.exists():
        return False
    try:
        json.loads(marker.read_text(encoding='utf-8'))
        return True
    except (json.JSONDecodeError, OSError):
        return False


def write_manifest(run_dir: Path, name: str, seed: int, cfg: dict[str, Any],
                   event: str, extra: Optional[dict[str, Any]] = None) -> None:
    md_dir = run_dir / 'md'
    try:
        git_commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE,
            capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        git_commit = None
    manifest: dict[str, Any] = {
        'name': name,
        'seed': seed,
        'event': event,
        'timestamp': now_iso(),
        'git_commit': git_commit,
        'lammps_version': cfg.get('lammps_version'),
        'emc_version': cfg.get('emc_version'),
        'md_cmd': cfg.get('md_cmd'),
        'slurm': {
            'partition': cfg.get('partition'),
            'time': cfg.get('time'),
            'mem': cfg.get('mem'),
            'cpus_per_task': cfg.get('cpus_per_task'),
        },
        'input_sha256': {
            'in.equil': sha256_of(md_dir / 'in.equil'),
            'in.smd': sha256_of(md_dir / 'in.smd'),
            f'Polymer_{name}_{seed}.data':
                sha256_of(md_dir / f'Polymer_{name}_{seed}.data'),
            'Silico_system.data': sha256_of(md_dir / 'Silico_system.data'),
            'water.mol': sha256_of(md_dir / 'water.mol'),
            'counts.json': sha256_of(md_dir / 'counts.json'),
            'sizing.json': sha256_of(run_dir / 'sizing.json'),
        },
    }
    sizing_p = run_dir / 'sizing.json'
    if sizing_p.exists():
        try:
            manifest['sizing'] = json.loads(sizing_p.read_text())
        except json.JSONDecodeError:
            pass
    if extra:
        manifest.update(extra)
    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8')


def run_build(candidate_path: Path, seed: int, out_root: Path,
              config: Path, registry: Path,
              emc_cmd: Optional[str], compress_cmd: Optional[str],
              logger: Logger, dry_run: bool = False) -> bool:
    if dry_run:
        logger.info('dry_run_build', candidate=str(candidate_path), seed=seed)
        return True
    cmd = [sys.executable, str(HERE / 'pipeline.py'), str(candidate_path),
           '--seed', str(seed), '--out', str(out_root),
           '--config', str(config), '--registry', str(registry)]
    if emc_cmd:
        cmd += ['--emc-cmd', emc_cmd]
    if compress_cmd:
        cmd += ['--lmp-cmd', compress_cmd]
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def render_sbatch(name: str, seed: int, cfg: dict[str, Any],
                  dependency: Optional[str] = None) -> str:
    ctr = cfg['container_image']
    md_q = _bash_dquote_escape(cfg['md_cmd'])
    dep = f'#SBATCH --dependency={dependency}\n' if dependency else ''
    extra = cfg.get('extra_sbatch', '') or ''
    if extra and not extra.endswith('\n'):
        extra += '\n'
    return f"""#!/bin/bash
#SBATCH --job-name={name}_s{seed}
#SBATCH --partition={cfg['partition']}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cfg.get('cpus_per_task', 8)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={cfg.get('time', '48:00:00')}
#SBATCH --mem={cfg.get('mem', '32G')}
{extra}{dep}#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

srun --mpi={cfg.get('mpi', 'pmi2')} \\
     --container-image={ctr} \\
     --container-mounts="$PWD:/work" \\
     --container-workdir=/work \\
     bash -lc "set -e; {md_q} -in in.equil -log log.lammps -screen screen.txt"
"""


def submit_md(run_dir: Path, name: str, seed: int, cfg: dict[str, Any],
              dependency: Optional[str] = None,
              dry_run: bool = False) -> Optional[str]:
    md_dir = run_dir / 'md'
    if not md_dir.exists():
        raise RuntimeError(f'{md_dir}: missing; run --mode build first')
    required = [*REQUIRED_MD_FILES, f'Polymer_{name}_{seed}.data']
    missing = [f for f in required if not (md_dir / f).exists()]
    if missing:
        raise RuntimeError(f'{md_dir}: missing {missing}')
    sbatch_path = md_dir / 'job.sbatch'
    sbatch_path.write_text(render_sbatch(name, seed, cfg, dependency),
                           encoding='utf-8')
    if dry_run:
        return 'DRY_RUN'
    try:
        res = subprocess.run(
            ['sbatch', '--parsable', str(sbatch_path)],
            cwd=md_dir, check=True, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError('sbatch not found — use --mode local-md on workstation')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'sbatch failed rc={e.returncode}: {(e.stderr or "").strip()}')
    jobid = res.stdout.strip()
    if not jobid:
        raise RuntimeError(f'sbatch returned empty jobid for {sbatch_path}')
    return jobid


def run_md_local(run_dir: Path, name: str, seed: int, cfg: dict[str, Any],
                 logger: Logger, dry_run: bool = False) -> int:
    md_dir = run_dir / 'md'
    if not md_dir.exists():
        raise RuntimeError(f'{md_dir}: missing')
    required = [*REQUIRED_MD_FILES, f'Polymer_{name}_{seed}.data']
    missing = [f for f in required if not (md_dir / f).exists()]
    if missing:
        raise RuntimeError(f'{md_dir}: missing {missing}')
    cmd = shlex.split(cfg['md_cmd']) + [
        '-in', 'in.equil', '-log', 'log.lammps', '-screen', 'screen.txt'
    ]
    logger.info('local_md_start', name=name, seed=seed, cmd=' '.join(cmd))
    if dry_run:
        return 0
    rc = subprocess.run(cmd, cwd=md_dir).returncode
    logger.info('local_md_end', name=name, seed=seed, rc=rc)
    return rc


def rerender_md(run_dir: Path, candidate: dict[str, Any], seed: int,
                cfg: dict[str, Any], logger: Logger) -> None:
    md_dir = run_dir / 'md'
    sizing_p = run_dir / 'sizing.json'
    counts_p = md_dir / 'counts.json'
    if not sizing_p.exists():
        raise RuntimeError(f'{sizing_p}: missing; run --mode build first')
    if not counts_p.exists():
        raise RuntimeError(f'{counts_p}: missing; run --mode build first')
    if format_extra_flags is None or compute_bath is None:
        raise RuntimeError('pipeline/read_data_counts unavailable')
    sizing = json.loads(sizing_p.read_text())
    counts = json.loads(counts_p.read_text())
    extra = format_extra_flags(counts)
    bath = compute_bath(cfg['Lx'], cfg['Ly'], cfg.get('Z_bath', 50.0),
                        cfg['C_out'], cfg['rho_solution'])
    md_ctx = {
        'name': candidate['name'], 'seed': seed, 'Q': candidate['Q'],
        'N_chains': sizing['N_chains'], 'extra': extra,
        'N_water_poly': sizing['N_water'], 'N_water_bath': bath['N_water'],
        'N_Na_poly': sizing['N_Na'], 'N_Na_bath': bath['N_Na'],
        'N_Cl_poly': sizing['N_Cl'], 'N_Cl_bath': bath['N_Cl'],
        'SMD_velocities': cfg['SMD_velocities'],
    }
    before = {f: sha256_of(md_dir / f) for f in ('in.equil', 'in.smd')}
    render_template('in.equil.j2', md_dir / 'in.equil', md_ctx)
    render_template('in.smd.j2', md_dir / 'in.smd', md_ctx)
    after = {f: sha256_of(md_dir / f) for f in ('in.equil', 'in.smd')}
    changed = [f for f in before if before[f] != after[f]]
    logger.info('rerender', name=candidate['name'], seed=seed, changed=changed or 'none')
    write_manifest(run_dir, candidate['name'], seed, cfg, event='rerendered')


def poll_job(jobid: str) -> dict[str, Any]:
    if not jobid or not jobid.isdigit():
        return {'state': 'UNKNOWN'}
    try:
        res = subprocess.run(
            ['sacct', '-j', jobid,
             '--format=JobID,State,ExitCode,Elapsed,MaxRSS',
             '--noheader', '--parsable2', '--allocations'],
            capture_output=True, text=True, check=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return {'state': 'UNKNOWN'}
    for line in res.stdout.strip().splitlines():
        parts = line.split('|')
        if len(parts) >= 3 and parts[0].strip() == jobid:
            return {
                'state': parts[1].strip().rstrip('+').split()[0],
                'exit_code': parts[2].strip(),
                'elapsed': parts[3].strip() if len(parts) > 3 else '',
                'max_rss': parts[4].strip() if len(parts) > 4 else '',
            }
    return {'state': 'UNKNOWN'}


def validate_md_output(run_dir: Path, n_velocities: int) -> dict[str, Any]:
    md_dir = run_dir / 'md'
    log = md_dir / 'log.lammps'
    result: dict[str, Any] = {'ok': True, 'errors': [], 'n_metrics': 0}
    if not log.exists():
        result['ok'] = False
        result['errors'].append('log.lammps missing')
        return result
    text = log.read_text(errors='ignore')
    for marker in ('ERROR', 'Lost atoms', 'Inconsistent image flags',
                   'Bond atoms missing', 'nan', 'NaN'):
        if marker in text:
            result['ok'] = False
            result['errors'].append(marker)
    if 'Total wall time' not in text:
        result['ok'] = False
        result['errors'].append('did not finish')
    metrics = sorted(md_dir.glob('metrics_smd_*.txt'))
    result['n_metrics'] = len(metrics)
    if result['n_metrics'] < n_velocities:
        result['ok'] = False
        result['errors'].append(f'only {result["n_metrics"]}/{n_velocities} SMD')
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['build', 'submit', 'local-md', 'rerender-md',
                            'poll', 'validate', 'all'])
    p.add_argument('--candidates', type=Path, default=HERE / 'candidates')
    p.add_argument('--index', type=Path, default=None)
    p.add_argument('--out', type=Path, default=HERE / 'runs')
    p.add_argument('--state', type=Path, default=None)
    p.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    p.add_argument('--registry', type=Path, default=HERE / 'monomers.yaml')
    p.add_argument('--slurm-config', type=Path, default=HERE / 'slurm.yaml')
    p.add_argument('--seeds', type=int, nargs='+', default=SEEDS_DEFAULT)
    p.add_argument('--source', type=str, default=None)
    p.add_argument('--name', type=str, nargs='+', default=None)
    p.add_argument('--emc-cmd', type=str, default=None)
    p.add_argument('--compress-cmd', type=str, default=None)
    p.add_argument('--dependency', action='store_true')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--log-file', type=Path, default=None)
    args = p.parse_args()

    if args.index is None:
        args.index = args.candidates / 'index.jsonl'
    if args.state is None:
        args.state = args.out / 'state.jsonl'
    logger = Logger(args.log_file or (args.out / 'orchestrator.log'))
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        slurm_cfg = yaml.safe_load(args.slurm_config.read_text(encoding='utf-8'))
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error('config_load_failed', error=str(e)); sys.exit(1)
    if not isinstance(cfg, dict) or not isinstance(slurm_cfg, dict):
        logger.error('config_not_mapping'); sys.exit(1)
    merged: dict[str, Any] = {**cfg, **slurm_cfg}
    if args.mode in ('submit', 'all'):
        for k in ('partition', 'container_image', 'md_cmd'):
            if k not in merged:
                logger.error('slurm_yaml_missing_key', key=k); sys.exit(1)
    if args.mode == 'local-md' and 'md_cmd' not in merged:
        logger.error('md_cmd_missing'); sys.exit(1)

    try:
        index = load_index(args.index)
    except (FileNotFoundError, ValueError) as e:
        logger.error('index_load_failed', error=str(e)); sys.exit(1)
    state = load_state(args.state)

    if args.source:
        index = [e for e in index if e.get('source') == args.source]
    if args.name:
        wanted = set(args.name)
        index = [e for e in index if e['name'] in wanted]
    if args.retry_failed:
        index = [e for e in index
                 if any(state.get((e['name'], s), {}).get('event', '')
                        .startswith('failed_')
                        for s in args.seeds)]

    logger.info('start', mode=args.mode, n_candidates=len(index),
                n_seeds=len(args.seeds),
                total_runs=len(index) * len(args.seeds),
                dry_run=args.dry_run)

    counts = {'built': 0, 'skipped_build': 0, 'failed_build': 0,
              'submitted': 0, 'skipped_submit': 0, 'failed_submit': 0,
              'local_md_ok': 0, 'local_md_fail': 0,
              'rerendered': 0, 'failed_rerender': 0,
              'polled': 0, 'validated_ok': 0, 'validated_fail': 0}

    prev_jobid: Optional[str] = None
    for entry in index:
        name = entry['name']
        try:
            candidate_path = resolve_candidate_path(args.candidates, entry['file'])
            candidate = json.loads(candidate_path.read_text())
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error('candidate_load_failed', name=name, error=str(e))
            continue

        for seed in args.seeds:
            key = (name, seed)
            run_dir = run_dir_for(args.out, name, seed)
            prev = state.get(key, {})
            logger.progress(name, seed, 'start', last_event=prev.get('event'))

            if args.mode in ('build', 'all'):
                if is_built(run_dir):
                    logger.progress(name, seed, 'build_skip', reason='exists')
                    counts['skipped_build'] += 1
                else:
                    logger.progress(name, seed, 'build_start')
                    ok = run_build(candidate_path, seed, args.out,
                                   args.config, args.registry,
                                   args.emc_cmd, args.compress_cmd,
                                   logger, dry_run=args.dry_run)
                    if ok:
                        if not args.dry_run:
                            write_manifest(run_dir, name, seed, merged, event='built')
                            append_state(args.state, {
                                'ts': now_iso(), 'name': name, 'seed': seed,
                                'event': 'built', 'run_dir': str(run_dir)})
                            state[key] = {'event': 'built'}
                            counts['built'] += 1
                        logger.progress(name, seed, 'build_ok')
                    else:
                        if not args.dry_run:
                            append_state(args.state, {
                                'ts': now_iso(), 'name': name, 'seed': seed,
                                'event': 'failed_build'})
                            state[key] = {'event': 'failed_build'}
                            counts['failed_build'] += 1
                        logger.error('build_failed', name=name, seed=seed)
                        continue

            if args.mode in ('submit', 'all'):
                if not is_built(run_dir):
                    logger.warn('submit_skip_not_built', name=name, seed=seed)
                    counts['skipped_submit'] += 1
                    continue
                dep = (f'afterany:{prev_jobid}'
                       if args.dependency and prev_jobid else None)
                try:
                    jobid = submit_md(run_dir, name, seed, merged,
                                      dependency=dep, dry_run=args.dry_run)
                    if not args.dry_run:
                        write_manifest(run_dir, name, seed, merged,
                                       event='submitted', extra={'jobid': jobid})
                        append_state(args.state, {
                            'ts': now_iso(), 'name': name, 'seed': seed,
                            'event': 'submitted_md', 'jobid': jobid,
                            'run_dir': str(run_dir)})
                        state[key] = {'event': 'submitted_md', 'jobid': jobid}
                        counts['submitted'] += 1
                        if args.dependency:
                            prev_jobid = jobid
                except RuntimeError as e:
                    if not args.dry_run:
                        append_state(args.state, {
                            'ts': now_iso(), 'name': name, 'seed': seed,
                            'event': 'failed_submit', 'error': str(e)})
                        state[key] = {'event': 'failed_submit'}
                        counts['failed_submit'] += 1
                    logger.error('submit_failed', name=name, seed=seed, error=str(e))

            if args.mode == 'local-md':
                if not is_built(run_dir):
                    logger.warn('local_md_skip_not_built', name=name, seed=seed)
                    continue
                try:
                    rc = run_md_local(run_dir, name, seed, merged,
                                      logger, dry_run=args.dry_run)
                    if not args.dry_run:
                        ev = 'local_md_done' if rc == 0 else 'failed_local_md'
                        append_state(args.state, {
                            'ts': now_iso(), 'name': name, 'seed': seed,
                            'event': ev, 'rc': rc, 'run_dir': str(run_dir)})
                        state[key] = {'event': ev, 'rc': rc}
                        counts['local_md_ok' if rc == 0 else 'local_md_fail'] += 1
                        write_manifest(run_dir, name, seed, merged, event=ev)
                except RuntimeError as e:
                    counts['local_md_fail'] += 1
                    logger.error('local_md_failed', name=name, seed=seed, error=str(e))

            if args.mode == 'rerender-md':
                if not is_built(run_dir):
                    logger.warn('rerender_skip_not_built', name=name, seed=seed)
                    continue
                try:
                    rerender_md(run_dir, candidate, seed, merged, logger)
                    counts['rerendered'] += 1
                except RuntimeError as e:
                    counts['failed_rerender'] += 1
                    logger.error('rerender_failed', name=name, seed=seed, error=str(e))

            if args.mode == 'poll':
                jobid = state.get(key, {}).get('jobid')
                if not jobid:
                    continue
                info = poll_job(jobid)
                if info['state'] in ('COMPLETED', 'FAILED', 'TIMEOUT',
                                     'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY'):
                    new_event = ('completed' if info['state'] == 'COMPLETED'
                                 else f'failed_{info["state"].lower()}')
                    append_state(args.state, {
                        'ts': now_iso(), 'name': name, 'seed': seed,
                        'event': new_event, 'jobid': jobid, **info})
                    state[key] = {'event': new_event, 'jobid': jobid, **info}
                counts['polled'] += 1

            if args.mode == 'validate':
                if not is_built(run_dir):
                    continue
                n_vel = len(merged.get('SMD_velocities', []))
                res = validate_md_output(run_dir, n_vel)
                append_state(args.state, {
                    'ts': now_iso(), 'name': name, 'seed': seed,
                    'event': 'validated' if res['ok'] else 'failed_validation',
                    **res})
                if res['ok']:
                    counts['validated_ok'] += 1
                else:
                    counts['validated_fail'] += 1
                    logger.error('validation_failed', name=name, seed=seed,
                                 errors=res['errors'])

    logger.info('summary', **counts)
    print('\n=== Summary ===', file=sys.stderr)
    for k, v in counts.items():
        if v:
            print(f'  {k:20s} {v}', file=sys.stderr)


if __name__ == '__main__':
    main()