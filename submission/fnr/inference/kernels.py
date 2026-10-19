"""CPU numerical inference for fixed LightGBM and TabM tensors."""
from pathlib import Path
import hashlib
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def neural_raw_matrix(frame, columns, dtypes):
    if not frame.columns.is_unique or set(columns).difference(frame.columns):
        raise ValueError('Missing or duplicate neural input columns')
    values = frame.loc[:, columns].astype(dtypes).to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values

def neural_matrix(frame, metadata):
    values = neural_raw_matrix(frame, metadata['features'], metadata['feature_dtypes'])
    prep = metadata['preprocessing']
    if prep['clip_bounds'] is not None:
        values = np.clip(values, prep['clip_bounds']['lower'], prep['clip_bounds']['upper'])
    fill = np.array([prep['fillna'][c] for c in metadata['features']], dtype=float)
    values = np.where(np.isfinite(values), values, fill)
    if prep['standardize']:
        values = (values - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite float32 neural matrix after frozen preprocessing')
    return result

def load_lgbm(directory):
    directory = Path(directory).resolve()
    meta = read_json(directory / 'model_meta.json')
    artifact = (directory / meta['artifact']).resolve()
    if artifact.parent != directory or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('LightGBM model checksum/path mismatch')
    if meta['family'] != 'lgbm':
        raise ValueError('Incorrect LightGBM component metadata')
    model = lgb.Booster(model_file=str(artifact))
    if model.feature_name() != meta['features']:
        raise ValueError('LightGBM feature order mismatch')
    return (model, meta)

def lgbm_matrix(frame, metadata):
    columns = metadata['features']
    if set(columns).difference(frame.columns) or not frame.columns.is_unique:
        raise ValueError('Missing or duplicate LightGBM input columns')
    matrix = frame.loc[:, columns].astype(metadata['feature_dtypes']).replace([np.inf, -np.inf], np.nan)
    prep = metadata['preprocessing']
    if prep.get('clip_bounds') is not None:
        bounds = prep['clip_bounds']
        matrix = matrix.clip(lower=pd.Series(bounds['lower'], index=columns), upper=pd.Series(bounds['upper'], index=columns), axis=1)
    if prep['fillna'] is not None:
        matrix = matrix.fillna(prep['fillna'])
    if prep.get('standardize'):
        matrix = (matrix - np.asarray(prep['mean'])) / np.asarray(prep['scale'])
    return matrix

def predict_lgbm(model, frame, metadata):
    with threadpool_limits(limits=1):
        result = np.asarray(model.predict(lgbm_matrix(frame, metadata), num_threads=1), dtype=np.float64)
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError('Invalid LightGBM prediction')
    return result

def load_tabm(directory):
    directory = Path(directory).resolve()
    meta = read_json(directory / 'model_meta.json')
    artifact = (directory / meta['artifact']).resolve()
    if artifact.parent != directory or sha256(artifact) != meta['artifact_sha256']:
        raise ValueError('TabM model checksum/path mismatch')
    architecture = meta['architecture']
    embedding = architecture['num_embeddings']
    if meta['family'] != 'torch_tabm' or meta['artifact_format'] != 'numpy_npz_tensor_state_dict' or architecture['activation'] != 'ReLU' or (architecture['arch_type'] != 'tabm') or (architecture['n_blocks'] != 2) or (architecture['input_dim'] != len(meta['features'])):
        raise ValueError('Unsupported frozen TabM architecture')
    width, hidden, members = (len(meta['features']), architecture['d_block'], architecture['k'])
    shapes = {'output.weight': (members, hidden, 1), 'output.bias': (members, 1)}
    optional = {}
    if embedding is not None:
        if embedding.get('kind') != 'piecewise_linear' or embedding.get('version') != 'B' or embedding.get('activation') is not False:
            raise ValueError('Unsupported frozen numerical embedding')
        dimension = int(embedding['d_embedding'])
        bins = max((len(edges) - 1 for edges in meta['numerical_embedding']['bins']))
        if dimension <= 0 or bins <= 0:
            raise ValueError('Invalid embedding dimension/bin count')
        shapes.update({'num_module.linear0.weight': (width, dimension), 'num_module.linear0.bias': (width, dimension), 'num_module.impl.weight': (width, bins), 'num_module.impl.bias': (width, bins), 'num_module.linear.weight': (width, bins, dimension)})
        optional = {'num_module.impl.single_bin_mask': (width,), 'num_module.impl.mask': (width, bins)}
        width *= dimension
    for block, n_in in ((0, width), (1, hidden)):
        prefix = 'backbone.blocks.' + str(block) + '.0.'
        shapes.update({prefix + 'weight': (hidden, n_in), prefix + 'r': (members, n_in), prefix + 's': (members, hidden), prefix + 'bias': (members, hidden)})
    with np.load(artifact, allow_pickle=False) as arrays:
        state = {name: arrays[name].copy() for name in arrays.files}
    if set(state).difference(shapes).difference(optional) or set(shapes).difference(state):
        raise ValueError('Unexpected TabM tensor names')
    for name, array in state.items():
        shape = shapes.get(name, optional.get(name))
        dtype = np.bool_ if name in optional else np.float32
        if array.shape != shape or array.dtype != dtype or (not np.isfinite(array).all()):
            raise ValueError('Invalid TabM tensor: ' + name)
        array.setflags(write=False)
    if embedding is not None:
        # Optional official buffers are omitted from state_dict when registered as None.
        counts = np.asarray([len(edges) - 1 for edges in meta['numerical_embedding']['bins']])
        if counts.shape != (len(meta['features']),):
            raise ValueError('Embedding feature/bin count mismatch')
        single = state.get('num_module.impl.single_bin_mask')
        if single is None and np.any(counts == 1) or (single is not None and (not np.array_equal(single, counts == 1))):
            raise ValueError('Embedding single-bin mask mismatch')
        mask = state.get('num_module.impl.mask')
        expected = np.arange(bins)[None, :] < counts[:, None] - 1
        expected[:, -1] = True
        if mask is None and (not np.all(counts == bins)) or (mask is not None and (not np.array_equal(mask, expected))):
            raise ValueError('Embedding padding mask mismatch')
    return (state, meta)

def piecewise_linear_embeddings(state, values):
    """Official rtdl 0.0.12 version B, activation=False, including edge extrapolation."""
    raw = values[..., None]
    encoded = raw * state['num_module.impl.weight'] + state['num_module.impl.bias']
    if encoded.shape[-1] > 1:
        np.minimum(encoded[..., :1], np.float32(1), out=encoded[..., :1])
        np.clip(encoded[..., 1:-1], np.float32(0), np.float32(1), out=encoded[..., 1:-1])
        last = encoded[..., -1:]
        single = state.get('num_module.impl.single_bin_mask')
        if single is None:
            np.maximum(last, np.float32(0), out=last)
        else:
            last[...] = np.where(single[None, :, None], last, np.maximum(last, np.float32(0)))
    # The official impl stores the last valid bin in the last channel and zero
    # coefficients for interior padding. The mask buffer is not applied in forward.
    projected = np.swapaxes(encoded, 0, 1) @ state['num_module.linear.weight']
    projected = np.swapaxes(projected, 0, 1)
    linear = raw * state['num_module.linear0.weight'] + state['num_module.linear0.bias']
    return np.ascontiguousarray((linear + projected).reshape(len(values), -1), dtype=np.float32)

def tabm_forward(state, values):
    if 'num_module.linear.weight' in state:
        values = piecewise_linear_embeddings(state, values)
    x = values[:, None, :]
    for block in (0, 1):
        prefix = 'backbone.blocks.' + str(block) + '.0.'
        x = x * state[prefix + 'r']
        x = x @ state[prefix + 'weight'].T
        x = x * state[prefix + 's']
        x = x + state[prefix + 'bias']
        np.maximum(x, np.float32(0), out=x)
    x = np.swapaxes(x, 0, 1) @ state['output.weight']
    x = np.swapaxes(x, 0, 1) + state['output.bias']
    return x.squeeze(-1).mean(axis=1, dtype=np.float32)

def predict_tabm(state, frame, metadata, prepared=None):
    parameters = metadata['parameters']
    kernel = int(parameters['inference_kernel_batch_size'])
    requested = int(parameters['predict_batch_size'])
    chunk = max(kernel, requested // kernel * kernel)
    result = np.empty(len(frame), dtype=np.float64)
    with threadpool_limits(limits=1):
        for start in range(0, len(frame), chunk):
            matrix = neural_matrix(frame.iloc[start:start + chunk], metadata) if prepared is None else prepared[start:start + chunk]
            for offset in range(0, len(matrix), kernel):
                actual = matrix[offset:offset + kernel]
                padded = np.zeros((kernel, len(metadata['features'])), dtype=np.float32)
                padded[:len(actual)] = actual
                values = tabm_forward(state, padded)
                result[start + offset:start + offset + len(actual)] = values[:len(actual)]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite TabM prediction')
    return np.round(result, parameters['raw_score_round_decimals'])
