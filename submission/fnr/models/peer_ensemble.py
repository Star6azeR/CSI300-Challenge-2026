"""Two bounded peer heads sharing one frozen factor encoding and neighborhood."""
from pathlib import Path
import json
import numpy as np
from . import peer as single

def load_numpy(path):
    path = Path(path)
    spec = json.loads((path / 'ensemble.json').read_text())
    if spec['score_arithmetic'] != 'float32_heads_float64_average_float32_output':
        raise ValueError('Unsupported ensemble arithmetic')
    members = [(float(m['weight']), single.load_adapter(path / m['directory'])) for m in spec['members']]
    if len(members) != 2 or [w for w, _ in members] != [0.475, 0.525]:
        raise ValueError('Expected fixed 47.5/52.5 weighted heads')
    configs = [a[0]['config'] for _, a in members]
    if any((c['neighborhood'] != 'factor_barra' or c['neighbors'] != 24 for c in configs)):
        raise ValueError('Mismatched neighborhood configuration')
    return (single.core.load_numpy(path / 'teacher'), members)

def predict_numpy(path, x, mask, history=None):
    teacher, members = load_numpy(path) if not isinstance(path, tuple) else path
    result = []
    for i in range(len(x)):
        nodes, baseline, graphs = single.encode(teacher, x[i:i + 1], mask[i:i + 1])
        indices, weights = graphs['factor_barra']
        features = single.features(nodes, indices, weights, 'factor_barra')
        scores = []
        for weight, adapter in members:
            head, _ = single.apply_adapter(adapter, features[None], mask[i:i + 1], baseline[None])
            scores.append(weight * head.astype(np.float64))
        result.append((scores[0] + scores[1]).astype(np.float32)[0])
    return {'score': np.asarray(result, np.float32)}
