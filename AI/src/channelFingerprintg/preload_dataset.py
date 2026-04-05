"""Pre-load HuggingFace and Sionna datasets into a local HDF5 cache.

Usage:
    python preload_dataset.py                  # defaults to Sinusoid
    python preload_dataset.py --signal_type deltaPulse

The cache file is written to /home/Research/POWDER/Data/{signal_type}_dataset.h5
and stores raw complex IQ as float32 (I, Q) channels with per-sample metadata
so the training/test pipelines can skip remote fetches and generation.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DatasetHandler import DatasetHandler
from SionnaDataGenerator import generate_multi_scenario_data, NODE_NAME_TO_ID

HOME_DIR = "/home/Research/POWDER/"
DATA_DIR = os.path.join(HOME_DIR, "Data")

SOURCE_OTA_LAB = 0
SOURCE_OTA_DENSE = 1
SOURCE_SIONNA = 2
SOURCE_NAMES = {
    str(SOURCE_OTA_LAB): "OTA-Lab",
    str(SOURCE_OTA_DENSE): "OTA-Dense",
    str(SOURCE_SIONNA): "Sionna",
}

NODE_IDS = {
    "Sinusoid": {
        "OTA-Lab": [
            [1, 2, 3], [1, 4, 5], [1, 4, 8], [2, 4, 3],
            [4, 2, 5], [4, 2, 8], [4, 8, 5], [5, 7, 8],
            [5, 8, 7], [8, 4, 1], [8, 5, 1],
        ],
        "OTA-Dense": [
            [1, 2, 3], [1, 2, 5], [1, 3, 2], [4, 3, 5],
        ],
    },
    "deltaPulse": {
        "OTA-Lab": [
            [5, 8, 2], [5, 8, 3], [5, 8, 6], [5, 8, 7],
            [6, 7, 2], [6, 7, 3], [6, 7, 4], [6, 7, 5], [6, 7, 8],
        ],
        "OTA-Dense": [
            [1, 2, 3], [1, 2, 4], [1, 3, 2], [1, 3, 4],
            [1, 4, 2], [1, 4, 3], [2, 3, 1], [2, 3, 4],
            [2, 4, 1], [2, 4, 3], [3, 4, 1], [3, 4, 2],
        ],
    },
    "WiFi": {
        "OTA-Lab": [
            [5, 6, 2], [5, 6, 3], [5, 6, 4], [5, 6, 7], [5, 6, 8],
            [5, 7, 2], [5, 7, 3], [5, 7, 4], [5, 7, 6], [5, 7, 8],
            [5, 8, 2], [5, 8, 3], [5, 8, 6], [5, 8, 7], [6, 7, 2],
        ],
        "OTA-Dense": [],
    },
}

SIONNA_SCENARIOS = [
    ["EBC", "Guest House", "Moran"],
    ["EBC", "Guest House", "USTAR"],
    ["EBC", "Moran", "Guest House"],
    ["EBC", "Moran", "USTAR"],
    ["EBC", "USTAR", "Guest House"],
    ["EBC", "USTAR", "Moran"],
    ["Guest House", "Moran", "EBC"],
    ["Guest House", "Moran", "USTAR"],
    ["Guest House", "USTAR", "Moran"],
    ["Guest House", "USTAR", "EBC"],
    ["USTAR", "Moran", "EBC"],
    ["USTAR", "Moran", "Guest House"],
]

SIONNA_CONFIG = {
    "num_probes": 100,
    "num_bins": 1024,
    "backend": "gnuradio",
    "noise_voltage": 0.01,
    "seed": 42,
}

CANONICAL_SAMPLES = {
    "Sinusoid": 8192,
    "deltaPulse": 8192 * 2,
    "WiFi": 128,
}

SAMPLE_SOURCE_FOR_SIGNAL = {
    "Sinusoid": "iq",
    "deltaPulse": "iq",
    "WiFi": "iq",
}


def _configure_iq_source(dataset, signal_type, sample_source="iq"):
    """Ensure DatasetHandler has I/Q columns for complex conversion."""
    df = dataset.dataFrame
    source = str(sample_source).strip().lower()
    if source == "auto":
        source = "chan_est_samples" if signal_type == "WiFi" else "iq"
    col_map = {
        "chan_est_samples": ("chan_est_samples_I", "chan_est_samples_Q"),
        "iq": ("I", "Q"),
        "csi": ("csi_I", "csi_Q"),
    }
    i_col, q_col = col_map.get(source, ("I", "Q"))
    if source == "iq" and "I" not in df.columns and "iq_I" in df.columns:
        i_col, q_col = "iq_I", "iq_Q"
    df = df.copy()
    df["I"] = df[i_col]
    df["Q"] = df[q_col]
    dataset.dataFrame = df


def _load_hf_band(signal_type, band, node_ids_list, sample_source="iq"):
    """Load one deployment band (OTA-Lab or OTA-Dense) from HuggingFace.

    Returns (data_complex, labels, rx, tx, alice_arr, bob_arr, eve_arr, scenario_idx_arr).
    Data is NOT shuffled; scenarios appear in node_ids_list order.
    """
    if not node_ids_list:
        return None, None, None, None, None, None, None, None

    dataset_name = "Key-Generation"
    config_prefix = f"{signal_type}-Powder-{band}-Nodes"
    repo_name = "CAAI-FAU"

    dataset = None
    for idx, nids in enumerate(node_ids_list):
        cfg = config_prefix + "-" + "".join(str(n) for n in nids)
        print(f"  [{band}] Loading config: {cfg}")
        if idx == 0:
            dataset = DatasetHandler(dataset_name, cfg, repo_name)
        else:
            dataset.add_dataset(dataset_name, cfg, repo_name)

    _configure_iq_source(dataset, signal_type, sample_source)
    data, labels, rx, tx = dataset.load_data(shuffle=False)
    data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])

    n_total = data.shape[0]
    samples_per_scenario = n_total // len(node_ids_list)

    alice_arr = np.empty(n_total, dtype=np.int32)
    bob_arr = np.empty(n_total, dtype=np.int32)
    eve_arr = np.empty(n_total, dtype=np.int32)
    scenario_idx_arr = np.empty(n_total, dtype=np.int32)

    for sc_i, nids in enumerate(node_ids_list):
        start = sc_i * samples_per_scenario
        end = start + samples_per_scenario
        alice_arr[start:end] = nids[0]
        bob_arr[start:end] = nids[1]
        eve_arr[start:end] = nids[2]
        scenario_idx_arr[start:end] = sc_i

    return data, labels, rx, tx, alice_arr, bob_arr, eve_arr, scenario_idx_arr


def _generate_sionna(signal_type, scenarios, canonical_len):
    """Generate Sionna ray-tracing synthetic data.

    Returns (data_complex, labels, rx, tx, alice_arr, bob_arr, eve_arr, scenario_idx_arr).
    """
    sc = SIONNA_CONFIG
    num_samples = canonical_len

    print(f"  [Sionna] Generating {len(scenarios)} scenarios, {sc['num_probes']} probes each ...")
    data, labels, rx, tx = generate_multi_scenario_data(
        scenarios=scenarios,
        num_probes=sc["num_probes"],
        signal_type=signal_type,
        num_samples=num_samples,
        backend=sc["backend"],
        noise_voltage=sc["noise_voltage"],
        seed=sc["seed"],
        num_bins=sc["num_bins"],
    )

    if data.shape[1] > canonical_len:
        data = data[:, -canonical_len:]
    elif data.shape[1] < canonical_len:
        pad = canonical_len - data.shape[1]
        data = np.pad(data, ((0, 0), (pad, 0)), mode="constant", constant_values=0)

    n_total = data.shape[0]
    samples_per_scenario = sc["num_probes"] * 4

    alice_arr = np.empty(n_total, dtype=np.int32)
    bob_arr = np.empty(n_total, dtype=np.int32)
    eve_arr = np.empty(n_total, dtype=np.int32)
    scenario_idx_arr = np.empty(n_total, dtype=np.int32)

    for sc_i, (alice, bob, eve) in enumerate(scenarios):
        a_id = NODE_NAME_TO_ID[alice] if isinstance(alice, str) else alice
        b_id = NODE_NAME_TO_ID[bob] if isinstance(bob, str) else bob
        e_id = NODE_NAME_TO_ID[eve] if isinstance(eve, str) else eve
        start = sc_i * samples_per_scenario
        end = start + samples_per_scenario
        alice_arr[start:end] = a_id
        bob_arr[start:end] = b_id
        eve_arr[start:end] = e_id
        scenario_idx_arr[start:end] = sc_i

    return data, labels, rx, tx, alice_arr, bob_arr, eve_arr, scenario_idx_arr


def preload(signal_type):
    """Fetch all data sources and write the unified HDF5 cache."""
    canonical_len = CANONICAL_SAMPLES.get(signal_type, 8192)
    sample_source = SAMPLE_SOURCE_FOR_SIGNAL.get(signal_type, "iq")
    node_ids = NODE_IDS.get(signal_type, {"OTA-Lab": [], "OTA-Dense": []})

    all_data = []
    all_labels = []
    all_rx = []
    all_tx = []
    all_source = []
    all_alice = []
    all_bob = []
    all_eve = []
    all_scenario = []

    global_scenario_offset = 0

    # OTA-Lab
    lab_ids = node_ids.get("OTA-Lab", [])
    if lab_ids:
        print(f"Loading OTA-Lab ({len(lab_ids)} scenarios) ...")
        d, l, r, t, a, b, e, si = _load_hf_band(signal_type, "OTA-Lab", lab_ids, sample_source)
        d = np.array([ex[-canonical_len:] for ex in d])
        si = si + global_scenario_offset
        global_scenario_offset += len(lab_ids)
        all_data.append(d)
        all_labels.append(l)
        all_rx.append(r)
        all_tx.append(t)
        all_source.append(np.full(len(d), SOURCE_OTA_LAB, dtype=np.uint8))
        all_alice.append(a)
        all_bob.append(b)
        all_eve.append(e)
        all_scenario.append(si)
        print(f"  OTA-Lab: {len(d)} samples loaded")

    # OTA-Dense
    dense_ids = node_ids.get("OTA-Dense", [])
    if dense_ids:
        print(f"Loading OTA-Dense ({len(dense_ids)} scenarios) ...")
        d, l, r, t, a, b, e, si = _load_hf_band(signal_type, "OTA-Dense", dense_ids, sample_source)
        d = np.array([ex[-canonical_len:] for ex in d])
        si = si + global_scenario_offset
        global_scenario_offset += len(dense_ids)
        all_data.append(d)
        all_labels.append(l)
        all_rx.append(r)
        all_tx.append(t)
        all_source.append(np.full(len(d), SOURCE_OTA_DENSE, dtype=np.uint8))
        all_alice.append(a)
        all_bob.append(b)
        all_eve.append(e)
        all_scenario.append(si)
        print(f"  OTA-Dense: {len(d)} samples loaded")

    # Sionna synthetic
    if signal_type in ("Sinusoid", "deltaPulse"):
        print(f"Generating Sionna data ({len(SIONNA_SCENARIOS)} scenarios) ...")
        d, l, r, t, a, b, e, si = _generate_sionna(signal_type, SIONNA_SCENARIOS, canonical_len)
        si = si + global_scenario_offset
        global_scenario_offset += len(SIONNA_SCENARIOS)
        all_data.append(d)
        all_labels.append(l)
        all_rx.append(r)
        all_tx.append(t)
        all_source.append(np.full(len(d), SOURCE_SIONNA, dtype=np.uint8))
        all_alice.append(a)
        all_bob.append(b)
        all_eve.append(e)
        all_scenario.append(si)
        print(f"  Sionna: {len(d)} samples generated")

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int32)
    rx = np.concatenate(all_rx, axis=0).astype(np.int32)
    tx = np.concatenate(all_tx, axis=0).astype(np.int32)
    source = np.concatenate(all_source, axis=0)
    alice = np.concatenate(all_alice, axis=0)
    bob = np.concatenate(all_bob, axis=0)
    eve = np.concatenate(all_eve, axis=0)
    scenario_index = np.concatenate(all_scenario, axis=0)

    # Store I/Q as float32 (N, num_samples, 2) to avoid complex dtype HDF5 issues
    iq_data = np.empty((data.shape[0], data.shape[1], 2), dtype=np.float32)
    iq_data[:, :, 0] = data.real.astype(np.float32)
    iq_data[:, :, 1] = data.imag.astype(np.float32)
    del data

    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, f"{signal_type}_dataset.h5")
    print(f"\nWriting {output_path} ...")
    print(f"  Samples: {iq_data.shape[0]}")
    print(f"  IQ length: {iq_data.shape[1]}")
    print(f"  IQ array size: {iq_data.nbytes / 1024**2:.1f} MB (uncompressed)")

    t0 = time.time()
    with h5py.File(output_path, "w") as f:
        f.create_dataset("iq_data", data=iq_data, compression="gzip", compression_opts=4,
                         chunks=(min(64, iq_data.shape[0]), iq_data.shape[1], 2))
        f.create_dataset("labels", data=labels)
        f.create_dataset("rx", data=rx)
        f.create_dataset("tx", data=tx)
        f.create_dataset("source", data=source)
        f.create_dataset("alice", data=alice)
        f.create_dataset("bob", data=bob)
        f.create_dataset("eve", data=eve)
        f.create_dataset("scenario_index", data=scenario_index)

        f.attrs["signal_type"] = signal_type
        f.attrs["num_samples"] = int(iq_data.shape[1])
        f.attrs["sample_source"] = sample_source
        f.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["source_names"] = json.dumps(SOURCE_NAMES)

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Written in {elapsed:.1f}s, file size: {file_size_mb:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-load dataset cache to HDF5")
    parser.add_argument("--signal_type", default="Sinusoid",
                        choices=["Sinusoid", "deltaPulse", "WiFi"],
                        help="Signal type to cache (default: Sinusoid)")
    args = parser.parse_args()
    preload(args.signal_type)
