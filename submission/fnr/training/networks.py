"""Torch network construction used only by the offline training entry point."""
import torch
import tabm

def build_tabm(metadata):
    architecture = metadata['architecture']
    if architecture['family'] != 'torch_tabm' or architecture['num_embeddings'] is not None or architecture['input_dim'] != len(metadata['features']):
        raise ValueError('Submission supports ordinary TabM with its frozen feature schema only')
    return tabm.TabM.make(n_num_features=architecture['input_dim'], cat_cardinalities=None, d_out=1, num_embeddings=None, n_blocks=architecture['n_blocks'], d_block=architecture['d_block'], dropout=architecture['dropout'], k=architecture['k'], arch_type=architecture['arch_type'], activation=architecture['activation'], start_scaling_init=architecture['start_scaling_init']).to(device='cpu', dtype=torch.float32)
