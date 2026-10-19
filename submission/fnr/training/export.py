"""Serialize trusted existing TabM weights to NPZ; never execute model predictions."""
import argparse
import hashlib
import json
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def export(source, destination):
    # Torch is required only for reading the trusted weights-only source artifact.
    import numpy as np
    import torch
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Destination must be separate from the source model')
    metadata = json.loads((source / 'model_meta.json').read_text(encoding='utf-8'))
    artifact = (source / metadata['artifact']).resolve()
    if artifact.parent != source or sha256(artifact) != metadata['artifact_sha256']:
        raise ValueError('Source model checksum/path mismatch')
    if metadata['family'] != 'torch_tabm' or metadata['artifact_format'] != 'torch_state_dict_weights_only':
        raise ValueError('Expected trusted Torch TabM state dict')
    embedding = metadata['architecture']['num_embeddings']
    if embedding is None or embedding['kind'] != 'piecewise_linear' or embedding['version'] != 'B' or (embedding['activation'] is not False):
        raise ValueError('Expected PLE version B with activation=False')
    payload = torch.load(artifact, map_location='cpu', weights_only=True)
    if set(payload) != {'state_dict'}:
        raise ValueError('Unexpected Torch checkpoint structure')
    state = {key: tensor.detach().cpu().numpy().copy() for key, tensor in payload['state_dict'].items()}
    if any((value.dtype not in (np.dtype(np.float32), np.dtype(np.bool_)) for value in state.values())):
        raise ValueError('Unexpected Torch tensor dtype')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Refusing to overwrite nonempty destination')
    destination.mkdir(parents=True, exist_ok=True)
    exported = destination / 'model.npz'
    np.savez(exported, **state)
    converted = dict(metadata)
    converted['artifact'] = exported.name
    converted['artifact_sha256'] = sha256(exported)
    converted['artifact_format'] = 'numpy_npz_tensor_state_dict'
    converted['numpy_export'] = {'source_artifact_sha256': metadata['artifact_sha256'], 'source_metadata_sha256': sha256(source / 'model_meta.json'), 'source_model_directory': str(source), 'operation': 'weights_only_deserialization_and_exact_tensor_npz_serialization', 'prediction_or_training_executed': False, 'runtime_file': 'fnr/inference/kernels.py', 'runtime_sha256': sha256(Path(__file__).resolve().parents[1] / 'inference/kernels.py')}
    (destination / 'model_meta.json').write_text(json.dumps(converted, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Loader validates names/shapes/dtypes/masks/checksums only; no forward call.
    from ..inference.kernels import load_tabm
    load_tabm(destination)
    report = {'source': str(source), 'destination': str(destination), 'artifact_sha256': converted['artifact_sha256'], 'tensor_count': len(state), 'artifact_bytes': exported.stat().st_size, 'static_loader_validation': 'passed', 'prediction_executed': False, 'tensor_shapes': {k: list(v.shape) for k, v in state.items()}}
    (destination / 'export_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Existing trusted model directory')
    parser.add_argument('--destination', required=True, help='New, empty NPZ model directory')
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), indent=2))
if __name__ == '__main__':
    main()
