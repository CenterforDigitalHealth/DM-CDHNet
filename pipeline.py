"""One local entry point for training, policy-aware evaluation and generation."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from utils import sha256, write_json

ROOT = Path(__file__).resolve().parent
ENTRYPOINTS = {'train': 'train.py', 'evaluate': 'evaluate.py', 'calibrate': 'calibrate.py',
              'generate': 'generate.py', 'prepare-temporal': 'prepare_temporal.py',
              'generate-temporal': 'generate_temporal.py', 'report': 'report.py'}


def command(script, arguments):
    env = dict(os.environ)
    # Avoid BLAS thread oversubscription; callers can opt into their own settings.
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        env.setdefault(name, '1')
    subprocess.run([sys.executable, str(ROOT / script), *map(str, arguments)],
                   cwd=ROOT, env=env, check=True)


def load_run_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    allowed = {'train_data', 'validation_data', 'test_data', 'labels', 'model_config',
               'checkpoint', 'output_dir', 'device', 'correction', 'synthetic', 'evaluation', 'generation'}
    if set(config) - allowed:
        raise ValueError(f'Unknown pipeline settings: {sorted(set(config) - allowed)}')
    required = ['validation_data', 'test_data', 'labels', 'output_dir']
    required += ['train_data', 'model_config'] if not config.get('checkpoint') else ['checkpoint']
    for name in required:
        if not config.get(name):
            raise ValueError(f'Missing pipeline setting: {name}')
    for name in ['train_data', 'validation_data', 'test_data', 'labels', 'model_config', 'checkpoint', 'output_dir']:
        if config.get(name):
            config[name] = str((path.parent / config[name]).resolve())
            if name != 'output_dir' and not Path(config[name]).is_file():
                raise FileNotFoundError(f'{name}: {config[name]}')
    if config.get('correction', 'native') not in ['native', 'final']:
        raise ValueError('correction must be native or final')
    if not config.get('checkpoint') and config.get('correction') == 'final':
        raise ValueError('Published correction requires the published checkpoint; new training uses native inference')
    seen = {}
    for role in ['train_data', 'validation_data', 'test_data']:
        if config.get(role):
            digest = sha256(config[role])
            if digest in seen:
                raise ValueError(f'{role} and {seen[digest]} have identical contents; validation/calibration and test inputs must be separate')
            seen[digest] = role
    return config


def options(settings, allowed):
    unknown = set(settings) - set(allowed)
    if unknown:
        raise ValueError(f'Unknown stage settings: {sorted(unknown)}')
    return [value for key, item in settings.items() for value in ['--' + key.replace('_', '-'), str(item)]]


def run_pipeline(path):
    config = load_run_config(path)
    evaluation = options(config.get('evaluation', {}), ['mode', 'quantiles', 'batch_size', 'max_patients', 'minimum_history_days', 'max_transitions'])
    generation = options(config.get('generation', {}), ['patients', 'replicates', 'horizon_days', 'minimum_history_days', 'max_events', 'seed'])
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    record = dict(status='RUNNING', synthetic=config.get('synthetic', False), config=config, stages=[],
                  policy_sha256=sha256(ROOT / 'evaluation_policy.json'))
    write_json(out / 'pipeline.json', record)

    def stage(name, script, arguments):
        command(script, arguments)
        record['stages'].append(name)
        write_json(out / 'pipeline.json', record)

    try:
        device = config.get('device', 'cpu')
        checkpoint = config.get('checkpoint')
        if not checkpoint:
            stage('training', 'train.py', ['--config', config['model_config'], '--train-data', config['train_data'],
                  '--val-data', config['validation_data'], '--out', out/'training', '--device', device])
            checkpoint = str(out / 'training/best.pt')
        shared = ['--checkpoint', checkpoint, '--device', device, '--labels', config['labels'],
                  '--correction', config.get('correction', 'native')]
        stage('validation', 'evaluate.py', shared + ['--data', config['validation_data'], '--out', out/'validation'] + evaluation)
        stage('calibration', 'calibrate.py', ['--evaluation', out/'validation', '--out', out/'calibration.json'])
        stage('evaluation', 'evaluate.py', shared + ['--data', config['test_data'], '--out', out/'evaluation',
              '--calibration', out/'calibration.json'] + evaluation)
        stage('generation', 'generate.py', shared + ['--data', config['test_data'], '--out', out/'generation'] + generation)
        stage('report', 'report.py', ['--run', out])
        record.update(status='PASS', checkpoint_sha256=sha256(checkpoint))
    except Exception as error:
        record.update(status='FAILED', error=str(error))
        raise
    finally:
        write_json(out / 'pipeline.json', record)
    print(f'Pipeline PASS: {out}')


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ENTRYPOINTS:
        command(ENTRYPOINTS[sys.argv[1]], sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__, epilog='Stage commands: ' + ', '.join(ENTRYPOINTS))
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='Train (or load checkpoint), evaluate, calibrate, generate, report')
    run.add_argument('--config', required=True)
    smoke = sub.add_parser('smoke', help='Tiny CPU synthetic end-to-end check')
    smoke.add_argument('--out', required=True)
    doctor = sub.add_parser('doctor', help='Check local dependencies, model sources and configured inputs')
    doctor.add_argument('--config')
    args = parser.parse_args()
    if args.command == 'run':
        run_pipeline(args.config)
    elif args.command == 'smoke':
        from demo import create_demo
        run_pipeline(create_demo(args.out))
    else:
        expected = json.loads((ROOT / 'docs/checkpoint_provenance.json').read_text())['checkpoint_source_hashes']
        missing = [name for name in ['numpy', 'torch', 'sklearn', 'pandas', 'matplotlib'] if importlib.util.find_spec(name) is None]
        changed = [name for name, digest in expected.items() if sha256(ROOT / name) != digest]
        if args.config:
            load_run_config(args.config)
        print(json.dumps(dict(status='PASS' if not missing and not changed else 'FAIL',
                              project_root=str(ROOT), missing_packages=missing,
                              changed_frozen_sources=changed, config_checked=bool(args.config)), indent=2))
        if missing or changed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
