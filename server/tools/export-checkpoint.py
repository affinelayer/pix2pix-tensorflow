import argparse
import os
import tempfile
import subprocess as sp
import json
import struct
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def log_quantize(data, mu, bins):
    # mu-law encoding
    scale = np.max(np.abs(data))
    norm_data = data / scale
    log_data = np.sign(data) * np.log(1 + mu * np.abs(norm_data)) / np.log(1 + mu)

    _counts, edges = np.histogram(log_data, bins=bins)
    log_points = (edges[:-1] + edges[1:]) / 2
    return np.sign(log_points) * (1 / mu) * ((1 + mu)**np.abs(log_points) - 1) * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--output_file", required=True, help="where to write output")
    args = parser.parse_args()

    model_path = None
    with open(os.path.join(args.checkpoint, "checkpoint")) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            key, _sep, val = line.partition(": ")
            val = val[1:-1]  # remove quotes
            if key == "model_checkpoint_path":
                model_path = val

    if model_path is None:
        raise Exception("failed to find model path")

    checkpoint_file = os.path.join(args.checkpoint, model_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = ["python", "-u", os.path.join(SCRIPT_DIR, "dump_checkpoints/dump_checkpoint_vars.py"), "--model_type", "tensorflow", "--output_dir", tmp_dir, "--checkpoint_file", checkpoint_file]
        sp.check_call(cmd)

        with open(os.path.join(tmp_dir, "manifest.json")) as f:
            manifest = json.loads(f.read())

        names = []
        for key in manifest.keys():
            if not key.startswith("generator") or "Adam" in key or "_loss" in key or "_train" in key or "_moving_" in key:
                continue
            names.append(key)
        names = sorted(names)

        arrays = []
        for name in names:
            value = manifest[name]
            with open(os.path.join(tmp_dir, value["filename"]), "rb") as f:
                arr = np.frombuffer(f.read(), dtype=np.float32).copy().reshape(value["shape"])
                arrays.append(arr)

    shapes = []
    for name, arr in zip(names, arrays):
        shapes.append(dict(
            name=name,
            shape=arr.shape,
        ))

    flat = np.hstack([arr.reshape(-1) for arr in arrays])

    start = time.time()
    index = log_quantize(flat, mu=255, bins=256).astype(np.float32)
    print("index found in %0.2fs" % (time.time() - start))

    print("quantizing")
    encoded = np.zeros(flat.shape, dtype=np.uint8)
    elem_count = 0
    for i, x in enumerate(flat):
        distances = np.abs(index - x)
        nearest = np.argmin(distances)
        encoded[i] = nearest
        elem_count += 1
        if elem_count % 1000000 == 0:
            print("rate", int(elem_count / (time.time() - start)))

    with open(args.output_file, "wb") as f:
        def write(name, buf):
            print("%s bytes %d" % (name, len(buf)))
            f.write(struct.pack(">L", len(buf)))
            f.write(buf)

        write("shape", json.dumps(shapes).encode("utf8"))
        write("index", index.tobytes())
        write("encoded", encoded.tobytes())

main()