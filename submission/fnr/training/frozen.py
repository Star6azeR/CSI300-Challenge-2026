"""Offline reproduction of the frozen base, factor heads and statistical prior."""
from ..paths import config_path
from pathlib import Path
import argparse
from threadpoolctl import threadpool_limits
from .base import main as train_base
from .factor import main as train_auxiliary
from .peer_ensemble import main as train_peer_ensemble
from .statistics import train as train_correction
from .artifacts import manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', '--out-dir', dest='model_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    cli = ['--data-dir', str(args.data_dir.resolve()), '--model-dir', str(args.model_dir.resolve())]
    with threadpool_limits(limits=2):
        train_base(cli)
        train_auxiliary(cli)
        train_peer_ensemble(cli)
        train_correction(args.data_dir.resolve(), args.model_dir.resolve() / 'statistical_correction', config_path('statistical_recipe.json'))
    manifest(args.model_dir.resolve())
if __name__ == '__main__':
    main()
