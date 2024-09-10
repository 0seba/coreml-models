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
import argparse


# LUT_GROUP_SIZE = 4
# NUM_WORKERS = 6  # number of E+P cores


def clusterize(arr, bits):
    clustered = cluster(arr.reshape(-1), k=2**bits)
    wq, lut = clustered.clusters, clustered.centroids
    return np.array(wq, dtype=np.uint8).reshape(arr.shape), np.array(
        lut, dtype=np.float16
    ).reshape(1, 2**bits, 1)


def quantize_tensor(
    tensors, wname: str, shape, numel_thresh, configs, rest, pool, lut_group_size
):
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
        print(
            f"Skipping weight {wname} with {np.prod(w.shape)} elements from configuration"
        )
        return {wname: w}

    # print("Quantizing", wname, bits, w.shape)

    per_channel_scale = np.max(np.abs(w), axis=1, keepdims=True)
    per_channel_scale[per_channel_scale == 0] = 1
    w /= per_channel_scale

    arrs = np.split(w, indices_or_sections=w.shape[0] // lut_group_size)
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


def main(filename, bits, include_original_emb, outfile, lut_group_size, num_workers):
    numel_thresh = 10_000
    rest = bits

    out_tensors = {}

    pool = Pool(num_workers)
    atexit.register(pool.terminate)
    # torch instead of numpy because of bf16

    with open(filename, "rb") as f:
        length_of_header = struct.unpack("<Q", f.read(8))[0]
        header_data = f.read(length_of_header)
        header = json.loads(header_data)

    layers = filter(lambda x: x.startswith("model.layers."), header)
    last_layer_index = max(map(lambda x: int(x.split(".", 3)[-2]), layers))
    print("Model layers:", last_layer_index)

    configs = {
        "model.embed_tokens.weight": None,
        "model.layers.0": None,
        f"model.layers.{last_layer_index}": None,
    }

    with safe_open(filename, framework="torch") as tensors:
        wname: str
        for wname in tqdm(list(sorted(tensors.keys()))):
            # print("W:", outws.shape, ", LUT:", lut.shape, ", S:", per_channel_scale.shape)
            shape = header[wname]
            out_tensors.update(
                quantize_tensor(
                    tensors,
                    wname,
                    shape,
                    numel_thresh,
                    configs,
                    rest,
                    pool,
                    lut_group_size,
                )
            )

        if include_original_emb:
            out_tensors["model.embed_tokens.weight.original"] = (
                tensors.get_tensor("model.embed_tokens.weight").float().half().numpy()
            )

    print("\n".join(list(sorted(out_tensors.keys()))))
    save_file(out_tensors, outfile)


if __name__ == "__main__":
    include_original_emb = False
    parser = argparse.ArgumentParser(
        description="Quantize tensors from a safetensors file."
    )
    parser.add_argument("tfile", help="Path to the input safetensors file")
    parser.add_argument(
        "--bits", type=int, required=True, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--include_original_emb", action="store_true", help="Include original embedding"
    )
    parser.add_argument("--outfile", required=True, help="Output file name")
    parser.add_argument(
        "--lut_group_size", type=int, required=True, help="LUT group size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of worker processes"
    )

    args = parser.parse_args()

    # if not args.outfile:
    #     args.outfile = f"output_{args.bits}B.safetensors"

    main(
        args.tfile,
        args.bits,
        args.include_original_emb,
        args.outfile,
        args.lut_group_size,
        args.num_workers,
    )
    # bits = 8
    # main(tfile, bits, include_original_emb, outfile)

    # outfile = f"Qwen2-0.5B-{bits}B.safetensors"
