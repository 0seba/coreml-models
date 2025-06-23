import numpy as np


def compute_default_rope_parameters(hdim, rope_freq_constant):
    inv_freq = 1.0 / (
        rope_freq_constant ** ((np.arange(0, hdim, 2, dtype=np.float32)) / hdim)
    )
    attention_factor = 1.0
    return inv_freq, attention_factor


def compute_rope_embedding(inv_freq, attention_scaling, length):
    pos_index = np.arange(length, dtype=np.float32)
    pos_index_theta = np.einsum("i,j->ij", pos_index, inv_freq)
    emb = np.concatenate((pos_index_theta, pos_index_theta), axis=-1)
    cos_emb = np.cos(emb)
    sin_emb = np.sin(emb)

    cos_emb = cos_emb * attention_scaling
    sin_emb = sin_emb * attention_scaling

    return sin_emb, cos_emb
