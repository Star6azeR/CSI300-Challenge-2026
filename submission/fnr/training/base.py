"""Rebuild the four frozen components from supplied monthly raw tables."""
from ..paths import config_path
from pathlib import Path
import argparse
import json
from .base_training import prepare_training, fit_preprocessing, training_contract, train_components
from .ple import deep_learner
from .export import export
from .artifacts import sanitize, write, manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--model-dir', '--out-dir', dest='model_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.model_dir.exists() and any(args.model_dir.iterdir()):
        raise FileExistsError('Refusing to overwrite model output')
    recipe = json.loads(config_path('training_recipe.json').read_text())
    train, columns = prepare_training(args.data_dir)
    if columns != recipe['ensemble']['features']:
        raise ValueError('Raw rebuilt feature order differs')
    prep = fit_preprocessing(train, columns)
    train_components(train, columns, prep, args.model_dir, training_contract(train, columns, prep))
    for name, cfg in recipe['ple_configs'].items():
        fitted = deep_learner.fit(train, cfg, year=2023, columns=columns)
        source = args.model_dir.parent / (args.model_dir.name + '_' + name + '_torch')
        if source.exists():
            raise FileExistsError('Temporary training model path exists')
        deep_learner.save(fitted, source)
        export(source, args.model_dir / name)
    write(args.model_dir / 'ensemble.json', recipe['ensemble'])
    for path in args.model_dir.rglob('*.json'):
        if path.name == 'manifest.json':
            continue
        write(path, sanitize(json.loads(path.read_text())))
    manifest(args.model_dir)
    print(json.dumps({'components': ['lgbm', 'tabm', 'ple_rank', 'ple_residual'], 'training_completed': True}))
if __name__ == '__main__':
    main()
