import json
import struct
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np
from coremltools._deps.kmeans1d import cluster
from multiprocessing import Pool
import atexit
from itertools import repeat
from tqdm import tqdm

LUT_GROUP_SIZE = 4
NUM_WORKERS = 4  # number of E+P cores


def clusterize(arr, bits):
    clustered = cluster(arr.reshape(-1), k=2**bits)
    wq, lut = clustered.clusters, clustered.centroids
    return np.array(wq, dtype=np.uint8).reshape(arr.shape), np.array(
        lut, dtype=np.float16
    ).reshape(1, 2**bits, 1)


def quantize_tensor(tensors, wname: str, shape, numel_thresh, configs, rest, pool):
    w = tensors.get_tensor(wname).float().half().numpy()
    if np.prod(w.shape) < numel_thresh:
        return {wname: w}
        # print(f"Skipping weight {wname} with {np.prod(w.shape)} elements")

    if wname in configs:
        bits = configs.get(wname)
    elif np.any([wname.startswith(k) for k in configs.keys()]):
        matching_keys = [k for k in configs.keys() if wname.startswith(k)]
        assert (
            len(matching_keys) == 1
        ), f"Weight {wname} matched with more than 1 configurations: {matching_keys}"
        bits = configs[matching_keys[0]]
    else:
        bits = rest

    if bits is None:
        # print(
        #     f"Skipping weight {wname} with {np.prod(w.shape)} elements from configuration"
        # )
        return {wname: w}

    # print("Quantizing", wname, bits, w.shape)

    per_channel_scale = np.max(np.abs(w), axis=1, keepdims=True)
    per_channel_scale[per_channel_scale == 0] = 1
    w /= per_channel_scale

    arrs = np.split(w, indices_or_sections=w.shape[0] // LUT_GROUP_SIZE)
    # outindices, outlut = [], []
    # for arr in arrs:
    #     indices, lut = clusterize(arr, bits)
    #     outindices.append(indices)
    #     outlut.append(lut)
    # clustered = cluster(arr.reshape(-1), k=bits)
    # wq, lut = clustered.clusters, clustered.centroids
    # # print(wq.shape)
    # outws.append(np.array(wq, dtype=np.uint8).reshape(arr.shape))
    # outlut.append(np.array(lut, dtype=np.float16).reshape(6, 1))
    # outws = np.concatenate(outws, axis=0)
    # lut = np.stack(outlut, axis=0)

    indices, lut = zip(*pool.starmap(clusterize, zip(arrs, repeat(bits))))
    outws = np.concatenate(indices, axis=0)
    lut = np.stack(lut, axis=0)

    return {
        wname + ".weight": outws,
        wname + ".scales": per_channel_scale.astype(np.float16),
        wname + ".lut": lut,
    }


def main(filename, bits, include_original_emb, outfile):
    configs = {
        "model.embed_tokens.weight": None,
        "model.layers.0": None,
        "model.layers.23": None,
    }
    numel_thresh = 10_000
    rest = bits

    out_tensors = {}

    pool = Pool(NUM_WORKERS)
    atexit.register(pool.terminate)
    # torch instead of numpy because of bf16

    with open(filename, "rb") as f:
        length_of_header = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(length_of_header)
        header = json.loads(header_data)

    with safe_open(filename, framework="torch") as tensors:
        wname: str
        for wname in tqdm(list(sorted(tensors.keys()))):
            # print("W:", outws.shape, ", LUT:", lut.shape, ", S:", per_channel_scale.shape)
            shape = header[wname]
            out_tensors.update(
                quantize_tensor(
                    tensors, wname, shape, numel_thresh, configs, rest, pool
                )
            )

        if include_original_emb:
            out_tensors["model.embed_tokens.weight.original"] = (
                tensors.get_tensor("model.embed_tokens.weight").float().half().numpy()
            )

    save_file(out_tensors, outfile)


if __name__ == "__main__":
    tfile = "/Users/sebastianamenabar/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/model.safetensors"
    bits = 4
    include_original_emb = False
    outfile = "quantized_4bit_2.safetensors"
    main(tfile, bits, include_original_emb, outfile)
