"""Prepare complete offline factor and neighborhood arrays."""
from pathlib import Path
import json
import numpy as np
from .peer import array_hash
from .artifacts import sha
from .data import DailyData, KEY
from .base_training import read_public_tables, h1_labels
from ..features.daily import build_features
from ..models import peer as model

def prepare_extended(data_dir, source, cfg):
    pre = json.loads((source / 'preprocessing.json').read_text())
    columns = pre['features']
    if len(columns) != 347 or len(set(columns)) != 347:
        raise ValueError('The frozen teacher requires 347 ordered features')
    tables = read_public_tables(data_dir)
    full, _ = build_features(tables['Factor33_winsor'], tables['Barra_factor'], tables['MarketData'])
    labels = h1_labels(tables['MarketData'], tables['Barra_factor'])
    selected = full['日期'].ge(20160101) & full['日期'].lt(20230101)
    frame = full.loc[selected, KEY + columns].sort_values(KEY).reset_index(drop=True)
    # Match the research treatment: selected raw feature is zeroed before the
    # original frozen mean/std transform, not after standardization.
    frame['is_st'] = 0.0
    legal = labels.loc[labels['日期'].ge(20160101) & labels['日期'].lt(20230101) & labels.exit_date.lt(20230101), KEY + ['entry_date', 'exit_date', 'future_return']]
    if legal.empty or not legal.exit_date.lt(20230101).all():
        raise ValueError('No legal pre-2023 H1 labels')
    if not (legal.entry_date.gt(legal['日期']).all() and legal.exit_date.gt(legal.entry_date).all()):
        raise ValueError('Invalid H1 label endpoints')
    frame = frame.merge(legal, on=KEY, how='left', validate='one_to_one')
    if frame.empty or frame.duplicated(KEY).any():
        raise ValueError('Empty or duplicated asof feature rows')
    contract = dict(split='train', year=2023, start=20160101, cutoff_exclusive=20230101, label_exit_max=int(legal.exit_date.max()), asof_rows=len(frame), valid_labels=int(np.isfinite(frame.future_return).sum()), is_st='raw zero', H=1, label='adjusted O(t+2)/O(t+1)-1', input_lookback=61, membership='asof only, independent of labels', teacher_sha256={str(p.relative_to(source / 'numpy_model')): sha(p) for p in sorted((source / 'numpy_model').rglob('*')) if p.is_file()}, preprocessing_sha256=sha(source / 'preprocessing.json'))
    data = DailyData(frame, pre, history_length=0)
    del tables, full, labels, legal, frame
    teacher = model.core.load_numpy(source / 'numpy_model')
    mode = cfg['neighborhood']
    graph_key = 'barra' if mode == 'none' else mode
    n, width = data.mask.shape
    nodes = np.zeros((n, width, 42), np.float32)
    scores = np.zeros((n, width), np.float32)
    indices = np.zeros((n, width, 24), np.int16)
    weights = np.zeros(indices.shape, np.float32)
    for i in range(n):
        nodes[i], scores[i], graphs = model.encode(teacher, data.x[i:i + 1], data.mask[i:i + 1])
        indices[i], weights[i] = graphs[graph_key]
    codes = np.full(data.mask.shape, '', dtype='<U32')
    for i, values in enumerate(data.codes):
        codes[i, :len(values)] = values
    packed = dict(nodes=nodes, codes=codes, mask=data.mask, label_mask=data.label_mask, returns=data.raw_return, y_rank=data.y_rank, teacher=scores, days=data.days)
    contract['prepared_array_sha256'] = {key: array_hash(value) for key, value in packed.items()}
    contract['prepared_array_sha256'].update(nodes=array_hash(nodes), neighbors=array_hash(indices), weights=array_hash(weights))
    del data, teacher
    features = model.features_many(nodes, indices, weights, mode)
    del nodes
    return (pre, contract, packed, features, indices, weights)
