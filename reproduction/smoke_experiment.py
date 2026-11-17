"""Run the LeWM benchmark plumbing on small simulator-generated fixtures."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run import ROOT, SOURCE, TASKS, download_checkpoint


def generate_fixture(task, root):
    import h5py
    import numpy as np
    from omegaconf import OmegaConf
    from stable_worldmodel import World
    from stable_worldmodel.policy import RandomPolicy

    name = TASKS[task][1]
    target = root / 'datasets' / f'{name}.h5'
    target.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(SOURCE / 'config' / 'eval' / f'{task}.yaml')
    settings = OmegaConf.to_container(cfg.world, resolve=True)
    settings.update(num_envs=2, max_episode_steps=36)
    world = World(**settings, image_shape=(224, 224))
    try:
        world.envs.action_space.seed(42)
        world.set_policy(RandomPolicy())
        world.collect(path=target, episodes=6, seed=42, format='hdf5')
    finally:
        world.envs.close()
    with h5py.File(target, 'r+') as data:
        lengths = data['ep_len'][:]
        if 'episode_idx' not in data and 'ep_idx' not in data:
            data['ep_idx'] = np.repeat(np.arange(len(lengths)), lengths)
        if 'step_idx' not in data:
            data['step_idx'] = np.concatenate([np.arange(length) for length in lengths])
        for group in [key for key in data if isinstance(data[key], h5py.Group)]:
            leaves = []
            data[group].visititems(lambda name, item: leaves.append(name) if isinstance(item, h5py.Dataset) else None)
            for key in leaves:
                name = f"{group}_{key.replace('/', '_')}"
                if name not in data:
                    data[name] = data[f'{group}/{key}'][:]
            del data[group]


def smoke(task, root, existing_checkpoint):
    root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, STABLEWM_HOME=str(root), MUJOCO_GL='egl',
               SDL_VIDEODRIVER='dummy', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')

    def run(script, *args):
        subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                       cwd=ROOT, env=env, check=True)

    run('run.py', 'doctor')
    os.environ.update({key: env[key] for key in ('MUJOCO_GL', 'SDL_VIDEODRIVER')})
    generate_fixture(task, root)
    if existing_checkpoint:
        destination = root / 'checkpoints' / 'released' / task
        destination.parent.mkdir(parents=True)
        destination.symlink_to(existing_checkpoint.resolve(), target_is_directory=True)
    else:
        download_checkpoint(task, root)
    run('run.py', 'eval', '--task', task, '--smoke')
    run('run.py', 'train', '--task', task, '--seed', 3072, '--smoke')
    run('fit_uncertainty.py', '--task', task, '--name', 'smoke', '--epochs', 1,
        '--batch-size', 4, '--max-clips', 8, '--bank-size', 12, '--validation-clips', 12)
    artifacts = root / 'uncertainty' / task / 'smoke'
    run('evaluate_uncertainty.py', '--task', task, '--artifacts', artifacts,
        '--probe-clips', 12, '--test-clips', 10, '--batch-size', 4, '--output', root / 'offline.json')
    settings = artifacts / 'prior.json'
    run('run.py', 'eval', '--task', task, '--smoke', '--uncertainty-config', settings, '--calibration')
    run('run.py', 'eval', '--task', task, '--smoke', '--uncertainty-config', settings, '--visual-tint', 0.35)
    cases = root / 'results'
    base = f'smoke-eval-{task}-released-train3072-seed42-uncertainty-prior'
    validation = cases / f'{base}-calibration' / 'metrics.json'
    tinted = cases / f'{base}-tint-0.35' / 'metrics.json'
    run('report_episode_metrics.py', '--calibration', validation, '--test', tinted,
        '--output', root / 'episodes.json')
    zero = json.loads(settings.read_text())
    zero['beta'] = 0
    zero_settings = artifacts / 'prior-zero.json'
    zero_settings.write_text(json.dumps(zero))
    run('run.py', 'eval', '--task', task, '--smoke', '--uncertainty-config', zero_settings, '--visual-tint', 0.35)
    baseline = cases / f'{base}-zero-tint-0.35' / 'metrics.json'
    run('compare_planning.py', '--pair', baseline, tinted, '--output', root / 'paired.json')
    print(f'All smoke stages completed for {task}. Outputs: {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, default='pusht')
    parser.add_argument('--output', type=Path, required=True, help='New directory for generated fixture and reports')
    parser.add_argument('--existing-checkpoint', type=Path,
                        help='Reuse a local released checkpoint directory containing config.json and weights.pt')
    args = parser.parse_args()
    if args.existing_checkpoint and not all((args.existing_checkpoint / name).is_file()
                                            for name in ('config.json', 'weights.pt')):
        parser.error('--existing-checkpoint must contain config.json and weights.pt')
    smoke(args.task, args.output.resolve(), args.existing_checkpoint)
