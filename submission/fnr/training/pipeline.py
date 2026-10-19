"""Reconstruct all model artifacts from the public 2016–2022 raw tables."""
from ..paths import config_path
from pathlib import Path
import argparse
import shutil
from threadpoolctl import threadpool_limits

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', '--out-dir', dest='model_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    from .frozen import main as train_frozen
    from .refinement import train as train_new
    from .artifacts import manifest
    root = args.model_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Choose a new empty model directory')
    train_frozen(['--data-dir', str(args.data_dir.resolve()), '--model-dir', str(root / 'frozen_base')])
    with threadpool_limits(limits=1):
        train_new(args.data_dir.resolve(), root / 'frozen_base', root)
    shutil.copyfile(config_path('ranking_policy.json'), root / 'ranking_policy.json')
    manifest(root)
if __name__ == '__main__':
    main()
