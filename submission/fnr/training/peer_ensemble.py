"""Offline reconstruction of two peer heads from the permitted raw tables."""
from ..paths import config_path
from pathlib import Path
import argparse
import json
import shutil
import traceback
from . import peer as worker
from .artifacts import manifest, sha, write

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    args = p.parse_args(argv)
    root = args.model_dir.resolve()
    source = root / 'factor'
    initial = root / 'factor_initial'
    pending = root / 'factor_peer_pending'
    if not (source / 'numpy_model').is_dir():
        raise ValueError('Train the initial factor model first')
    if initial.exists() or pending.exists():
        raise FileExistsError('Preserve existing peer artifacts')
    recipe = json.loads(config_path('peer_ensemble_recipe.json').read_text())
    if len(recipe['members']) != 2 or [m['weight'] for m in recipe['members']] != [0.475, 0.525]:
        raise ValueError('Expected fixed 47.5/52.5 weighted members')
    pending.mkdir()
    completed = []
    try:
        # Each call fits one small head; source remains the same initial teacher.
        for i, member in enumerate(recipe['members']):
            cfg = member['config']
            if cfg['seed'] != 42 or cfg['epochs'] != 16 or cfg['neighbors'] != 24:
                raise ValueError('Unexpected training configuration')
            stage = pending / 'training_members' / f'head{i}'
            stage.mkdir(parents=True)
            completion = worker.train(root, args.data_dir, cfg, stage)
            completed.append(completion)
            shutil.copytree(stage / 'numpy_model/adapter', pending / 'numpy_model' / member['directory'])
            if i == 0:
                shutil.copytree(stage / 'numpy_model/teacher', pending / 'numpy_model/teacher')
                shutil.copyfile(stage / 'preprocessing.json', pending / 'preprocessing.json')
            else:
                if sha(stage / 'preprocessing.json') != sha(pending / 'preprocessing.json'):
                    raise AssertionError('Teacher preprocessing changed')
                if completed[0]['contract']['prepared_array_sha256'] != completion['contract']['prepared_array_sha256']:
                    raise AssertionError('Head training inputs differ')
            # Keep contracts and curves, with only one runtime copy of weights.
            shutil.rmtree(stage / 'numpy_model')
        runtime = {k: v for k, v in recipe.items() if k != 'members'}
        runtime['members'] = [{k: v for k, v in m.items() if k != 'config'} for m in recipe['members']]
        write(pending / 'numpy_model/ensemble.json', runtime)
        write(pending / 'config.json', recipe)
        write(pending / 'training_contract.json', completed[0]['contract'])
        write(pending / 'training_completion.json', dict(candidate='RH50', heads=completed, total_adapter_parameters=4098, teacher_parameters_frozen=True))
        source.rename(initial)
        try:
            pending.rename(source)
        except BaseException:
            initial.rename(source)
            raise
        manifest(root)
    except BaseException:
        failure = pending if pending.exists() else source
        (failure / 'peer_training_failure.txt').write_text(traceback.format_exc())
        raise
    print(json.dumps(dict(training_completed=True, heads=2, total_adapter_parameters=4098)), flush=True)
if __name__ == '__main__':
    main()
