"""Download, inspect, train and evaluate the pinned official LeWM baseline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'vendor' / 'le-wm'
TASKS = {
    'pusht': ('pusht', 'pusht_expert_train', 30),
    'reacher': ('dmc', 'dmc/reacher_random', 10),
    'cube': ('ogb', 'ogbench/cube_single_expert', 10),
}


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def data_root():
    value = os.environ.get('STABLEWM_HOME')
    if not value:
        raise SystemExit('Set STABLEWM_HOME to your large data disk first.')
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_provenance(dataset):
    dataset = Path(dataset)
    manifest = dataset.with_suffix('.provenance.json')
    record = json.loads(manifest.read_text()) if manifest.is_file() else {}
    return dict(record, dataset_bytes=dataset.stat().st_size)


def uncertainty_artifact_paths(base, settings, settings_dir):
    checkpoints = base / 'checkpoints'
    paths = {'base_model_config': (checkpoints / settings['base_checkpoint']).with_name('config.json')}
    for key in ('training_bank', 'prior_stats', 'diagnostic_prior_stats', 'action_bank'):
        if key in settings:
            paths[key] = settings_dir / settings[key]
    for index, checkpoint in enumerate(settings.get('predictor_checkpoints', [])):
        weights = checkpoints / checkpoint
        paths[f'predictor_{index}_weights'] = weights
        paths[f'predictor_{index}_config'] = weights.with_name('config.json')
    return paths


def verify_uncertainty_artifacts(base, settings, settings_dir, *, required=False):
    expected = settings.get('artifact_sha256')
    if not expected:
        if required:
            raise ValueError('Refit score artifacts with content hashes before hybrid threshold calibration or gating.')
        return
    paths = uncertainty_artifact_paths(base, settings, settings_dir)
    if set(paths) != set(expected):
        raise ValueError('Fitted uncertainty artifact paths do not match the recorded content hashes.')
    for name, path in paths.items():
        if not path.is_file() or sha256(path) != expected[name]:
            raise ValueError(f'Fitted uncertainty artifact content has changed: {name}.')


def extract_archive(archive, target, is_tar):
    import zstandard
    target.mkdir(parents=True, exist_ok=True)
    with archive.open('rb') as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
            if is_tar:
                with tarfile.open(fileobj=stream, mode='r|') as contents:
                    for member in contents:
                        dest = (target / member.name).resolve()
                        if not dest.is_relative_to(target.resolve()):
                            raise ValueError('Archive path escapes extraction directory')
                        if member.isdir():
                            dest.mkdir(parents=True, exist_ok=True)
                        elif member.isfile():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            with contents.extractfile(member) as src, dest.open('wb') as dst:
                                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
                        else:
                            raise ValueError('Unexpected link or special file in archive')
            else:
                with (target / archive.name.removesuffix('.zst')).open('wb') as out:
                    shutil.copyfileobj(stream, out, 8 * 1024 * 1024)


def download(task):
    from huggingface_hub import hf_hub_download
    base = data_root()
    repo = f'quentinll/lewm-{task}'
    metadata = next(x for x in read_json('dataset-manifest.json') if x['task'] == task)
    revision = next(x['revision'] for x in read_json('dataset-revisions.json') if x['task'] == task)
    entry = next(x for x in metadata['files'] if x['path'].endswith('.zst'))
    unpacked = base / 'extracted' / task
    marker = unpacked / '.complete'
    if marker.exists() and marker.read_text().strip() != revision:
        raise ValueError(f'Extracted dataset revision differs from the pinned revision: {marker}.')
    archive_dir = base / 'archives' / task
    archive = Path(hf_hub_download(repo, entry['path'], repo_type='dataset',
                   revision=revision, local_dir=archive_dir))
    if archive.stat().st_size != entry['size']:
        raise ValueError('Dataset download size does not match manifest')
    expected_hash = entry.get('lfs', {}).get('oid')
    if expected_hash and sha256(archive) != expected_hash:
        raise ValueError('Dataset SHA256 does not match official manifest')
    if not marker.exists():
        extract_archive(archive, unpacked, archive.name.endswith('.tar.zst'))
        marker.write_text(revision)
    files = list(unpacked.rglob('*.h5'))
    expected = TASKS[task][1]
    matching = [f for f in files if f.stem == Path(expected).name]
    if len(matching) == 1:
        dataset = matching[0]
    elif len(files) == 1:
        dataset = files[0]
    else:
        raise ValueError(f'Cannot identify {expected}; found {files}')
    destination = base / 'datasets' / f'{expected}.h5'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.symlink_to(dataset)
    elif destination.resolve() != dataset.resolve():
        raise FileExistsError(f'Dataset destination already exists: {destination}')
    destination.with_suffix('.provenance.json').write_text(json.dumps({
        'dataset_source': repo, 'dataset_revision': revision, 'archive_sha256': expected_hash,
        'archive_bytes': entry['size'],
    }, indent=2))
    download_checkpoint(task, base)
    print(f'{task}: dataset and released checkpoint ready at {base}')


def download_checkpoint(task, base):
    from huggingface_hub import hf_hub_download
    repo = f'quentinll/lewm-{task}'
    model = next(x for x in read_json('model-manifest.json') if x['task'] == task)
    for name in ('config.json', 'weights.pt'):
        hf_hub_download(repo, name, revision=model['revision'],
                        local_dir=base / 'checkpoints' / 'released' / task)


def doctor():
    import torch
    import stable_worldmodel as swm
    import hydra
    import h5py
    import dm_control
    import ogbench
    import stable_worldmodel.solver
    import stable_worldmodel.wm.lewm
    assert hasattr(swm.data, 'HDF5Dataset')
    print('Python:', sys.version)
    print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Install a GPU-compatible PyTorch build.')
    print('GPU:', torch.cuda.get_device_name())
    print('GPU GB:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    print('Imports passed; simulator rendering and dataset checks still require a smoke run.')


def eval_checkpoint(args):
    if getattr(args, 'checkpoint', None):
        return args.checkpoint
    if args.retrained:
        return f'retrained/{args.task}/seed-{args.training_seed}/weights_epoch_10.pt'
    return f'released/{args.task}/weights.pt'


def command(args, base, output):
    group, dataset, iterations = TASKS[args.task]
    common = [sys.executable, str(ROOT / 'seed_entry.py')]
    if args.action == 'train':
        holdout = getattr(args, 'action_holdout', None)
        prefix = f'restricted/{holdout}' if holdout else 'retrained'
        smoke_prefix = f'smoke-restricted/{holdout}' if holdout else 'smoke'
        split_seed = getattr(args, 'split_seed', 42)
        if holdout and split_seed != 42:
            prefix += f'/split-{split_seed}'
            smoke_prefix += f'/split-{split_seed}'
        sigreg_weight = getattr(args, 'sigreg_weight', None)
        if sigreg_weight is not None:
            prefix += f'/sigreg-{sigreg_weight:g}'
            smoke_prefix += f'/sigreg-{sigreg_weight:g}'
        run_subdir = f'{args.task}/seed-{args.seed}'
        if holdout or sigreg_weight is not None:
            run_subdir = f'{prefix}/{run_subdir}'
        cmd = common + [str(SOURCE / 'train.py'), f'data={group}',
            f'data.dataset.name={dataset}.h5', 'trainer.max_epochs=10',
            f'seed={args.seed}', f'subdir={run_subdir}',
            f'output_model_name={prefix}/{args.task}/seed-{args.seed}',
            'wandb.enabled=false']
        if sigreg_weight is not None:
            cmd.append(f'loss.sigreg.weight={sigreg_weight}')
        if args.smoke:
            cmd += ['trainer.max_epochs=1', '+trainer.limit_train_batches=2',
                    '+trainer.limit_val_batches=1', 'loader.batch_size=2',
                    f'output_model_name={smoke_prefix}/{args.task}/seed-{args.seed}',
                    f'subdir={smoke_prefix}/{args.task}/seed-{args.seed}']
    else:
        policy = eval_checkpoint(args)
        if not (base / 'checkpoints' / policy).is_file():
            raise FileNotFoundError(base / 'checkpoints' / policy)
        cmd = common + [str(SOURCE / 'eval.py'), f'--config-name={args.task}',
            f'policy={policy}', f'seed={args.seed}', f'solver.n_steps={iterations}',
            '+cache_dir=null', f'eval.num_eval={2 if args.smoke else 50}',
            'output.filename=results.txt']
        if args.smoke:
            cmd += ['solver.n_steps=2', 'solver.num_samples=8', 'solver.topk=4']
        if getattr(args, 'num_eval', None) is not None:
            cmd.append(f'eval.num_eval={args.num_eval}')
    return cmd


def run(args):
    base = data_root()
    dataset = base / 'datasets' / f'{TASKS[args.task][1]}.h5'
    if not dataset.is_file():
        raise FileNotFoundError(f'Download and extract the dataset first: {dataset}')
    label = 'retrained' if args.retrained else 'released'
    run_name = f'{args.action}-{args.task}-{label}-train{args.training_seed}-seed{args.seed}'
    if args.action_holdout:
        run_name += f'-action-id-{args.action_holdout}'
        if args.split_seed != 42:
            run_name += f'-split{args.split_seed}'
    if args.sigreg_weight is not None:
        run_name += f'-sigreg-{args.sigreg_weight:g}'
    if args.checkpoint:
        run_name += f'-checkpoint-{hashlib.sha256(args.checkpoint.encode()).hexdigest()[:12]}'
    if args.uncertainty_config:
        settings = json.loads(args.uncertainty_config.read_text())
        checkpoint = eval_checkpoint(args)
        if settings.get('base_checkpoint', checkpoint) != checkpoint:
            raise ValueError('The uncertainty artifacts were fitted with a different base checkpoint.')
        if settings.get('base_checkpoint_sha256') and sha256(base / 'checkpoints' / checkpoint) != settings['base_checkpoint_sha256']:
            raise ValueError('The fitted base checkpoint content has changed.')
        verify_uncertainty_artifacts(base, settings, args.uncertainty_config.parent,
                                     required=bool(args.gate or (args.threshold_calibration and settings.get('observation_scale'))))
        run_name += f'-uncertainty-{args.uncertainty_config.stem}'
        if args.beta is not None:
            settings['beta'] = args.beta
            run_name += f'-beta-{args.beta:g}'
        evaluation_split = 'threshold' if args.threshold_calibration else ('validation' if args.calibration else 'test')
        if evaluation_split != 'test' and f'{evaluation_split}_episode_ids' not in settings:
            raise ValueError('Refit artifacts with separate selection and threshold episode partitions.')
        settings.update(evaluation_split=evaluation_split,
                        visual_tint=args.visual_tint)
        if args.visual_family:
            settings['visual_shift'] = {
                'family': args.visual_family, 'severity': args.visual_severity,
                'mode': args.visual_mode, 'seed': args.visual_seed,
            }
            run_name += f'-{args.visual_family}-{args.visual_severity:g}-{args.visual_mode}-visualseed{args.visual_seed}'
        if args.gate:
            settings['gate'] = json.loads(args.gate.read_text())
            run_name += f'-gated-{args.gate.stem}'
        if 'action_severity_edges' in settings:
            settings['action_band'] = args.action_band if args.action_band is not None else 0
        elif args.action_band is not None:
            raise ValueError('--action-band requires artifacts fitted from restricted training')
        if args.action_band is not None:
            run_name += f'-action-band-{args.action_band}'
        if args.calibration:
            run_name += '-calibration'
        if args.threshold_calibration:
            run_name += '-threshold-calibration'
        if args.visual_tint:
            run_name += f'-tint-{args.visual_tint}'
    if args.smoke:
        run_name = 'smoke-' + run_name
    if args.num_eval:
        run_name += f'-cases{args.num_eval}'
    output = base / 'results' / run_name
    output.mkdir(parents=True, exist_ok=False)
    cmd = command(args, base, output)
    env = dict(os.environ, STABLEWM_HOME=str(base), LEWM_SEED=str(args.seed), LEWM_RESULTS_DIR=str(output),
               MUJOCO_GL='egl', PYTHONUNBUFFERED='1')
    env.pop('LEWM_UNCERTAINTY', None)
    env.pop('LEWM_ACTION_HOLDOUT', None)
    if args.action_holdout:
        env['LEWM_ACTION_HOLDOUT'] = str(args.action_holdout)
        env['LEWM_SPLIT_SEED'] = str(args.split_seed)
        env['LEWM_SMOKE_TRAINING'] = '1' if args.smoke else '0'
        env['LEWM_DATASET_PROVENANCE'] = json.dumps(dataset_provenance(dataset))
    if args.uncertainty_config:
        for name in ('training_bank', 'prior_stats', 'action_bank'):
            if name in settings:
                settings[name] = str((args.uncertainty_config.parent / settings[name]).resolve())
        env['LEWM_UNCERTAINTY'] = json.dumps(settings)
        (output / 'uncertainty.json').write_text(json.dumps(settings, indent=2))
    (output / 'command.json').write_text(json.dumps(cmd, indent=2))
    with (output / 'run.log').open('w') as log:
        process = subprocess.Popen(cmd, cwd=SOURCE, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()
        code = process.wait()
    (output / 'status.json').write_text(json.dumps({'exit_code': code, 'smoke': args.smoke}))
    if code:
        raise SystemExit(code)
    print(f'Saved run to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['doctor', 'download', 'train', 'eval'])
    parser.add_argument('--task', choices=list(TASKS))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training-seed', type=int, default=3072)
    parser.add_argument('--retrained', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--uncertainty-config', type=Path)
    parser.add_argument('--beta', type=float, help='Override the uncertainty penalty for a validation sweep')
    parser.add_argument('--checkpoint', help='Weights path in the stable-worldmodel checkpoint cache')
    parser.add_argument('--action-holdout', type=float, help='Train on clips below this training action quantile')
    parser.add_argument('--split-seed', type=int, default=42, help='Freeze episode partitions across training seeds')
    parser.add_argument('--sigreg-weight', type=float, help='Override the upstream SIGReg coefficient (default: 0.09)')
    parser.add_argument('--num-eval', type=int, help='Number of evaluation cases for an uncertainty run')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--calibration', action='store_true', help='Evaluate ID penalty-selection episodes')
    calibration.add_argument('--threshold-calibration', action='store_true', help='Evaluate separate ID threshold episodes')
    parser.add_argument('--action-band', type=int, choices=(0, 1, 2, 3),
                        help='Select cases by recorded action band, with 0 in the training range')
    parser.add_argument('--visual-tint', type=float, default=0.0, choices=(0.0, 0.15, 0.35, 0.6))
    parser.add_argument('--visual-family', choices=('tint', 'distractor', 'occlusion'))
    parser.add_argument('--visual-severity', type=float)
    parser.add_argument('--visual-mode', choices=('observation', 'both', 'goal'), default='both')
    parser.add_argument('--visual-seed', type=int, default=42)
    parser.add_argument('--gate', type=Path, help='Frozen initial-plan threshold produced by report_episode_metrics.py')
    args = parser.parse_args()
    if args.action_holdout is not None and (args.action != 'train' or not 0 < args.action_holdout < 1):
        parser.error('--action-holdout requires train and a quantile strictly between zero and one')
    if args.checkpoint and args.action != 'eval':
        parser.error('--checkpoint is only supported for eval')
    if args.sigreg_weight is not None and (
        args.action != 'train' or not math.isfinite(args.sigreg_weight) or args.sigreg_weight < 0
    ):
        parser.error('--sigreg-weight requires train and a finite, nonnegative weight')
    if args.uncertainty_config and args.action != 'eval':
        parser.error('--uncertainty-config is only supported for eval')
    if args.beta is not None and (not args.calibration or not math.isfinite(args.beta) or args.beta < 0):
        parser.error('--beta requires calibration and a finite, nonnegative weight')
    shifted = args.visual_family or args.visual_tint
    validating = args.calibration or args.threshold_calibration
    if (validating or shifted or args.gate or args.num_eval is not None or args.action_band is not None) and (
        not args.uncertainty_config or args.action != 'eval'
    ):
        parser.error('Calibration and shift flags require evaluation with an uncertainty configuration')
    if args.num_eval is not None and args.num_eval < 1:
        parser.error('--num-eval must be positive')
    if args.visual_family:
        from visual_shift import LEVELS
        if args.visual_tint or args.visual_severity not in LEVELS[args.visual_family]:
            parser.error('Choose a preregistered --visual-severity for --visual-family')
    elif args.visual_severity is not None or args.visual_mode != 'both' or args.visual_seed != 42:
        parser.error('Visual severity, mode, and seed require --visual-family')
    if validating and (shifted or args.gate):
        parser.error('Calibration requires clean visuals and ungated execution')
    if args.action_band and (validating or shifted):
        parser.error('Higher action bands must use test episodes with clean visuals')
    if args.action == 'doctor':
        doctor()
    elif not args.task:
        parser.error('--task is required')
    elif args.action == 'download':
        download(args.task)
    else:
        run(args)


if __name__ == '__main__':
    main()
