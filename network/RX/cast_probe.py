#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CaST probe detection and channel estimation helpers for PHY-Key-Generation."""

from pathlib import Path

import numpy as np

CAST_SEQUENCE_DIR = Path(__file__).resolve().parents[1] / "TX" / "cast_sequences"
CAST_SEQUENCE_CSV_STEMS = {
    "ga128": "ga128_bpsk",
    "glfsr": "glfsr_bpsk",
    "ls1": "ls1_bpsk",
    "ls1_all": "ls1all_bpsk",
    "gold": "gold_bpsk",
}
CAST_SEQUENCE_ALIASES = {
    "cast": "glfsr",
    "cast_glfsr": "glfsr",
    "ga128_bpsk": "ga128",
    "glfsr_bpsk": "glfsr",
    "ls1_bpsk": "ls1",
    "ls1all": "ls1_all",
    "ls1all_bpsk": "ls1_all",
    "ls1_all_bpsk": "ls1_all",
    "gold_bpsk": "gold",
}


def normalize_sequence_name(sequence):
    key = str(sequence or "glfsr").strip().lower()
    return CAST_SEQUENCE_ALIASES.get(key, key)


def load_cast_sequence(sequence="cast"):
    canonical_name = normalize_sequence_name(sequence)
    csv_stem = CAST_SEQUENCE_CSV_STEMS.get(canonical_name)
    if csv_stem is None:
        raise ValueError(f"Unsupported CaST probe sequence: {sequence!r}")
    csv_path = CAST_SEQUENCE_DIR / f"{csv_stem}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CaST probe sequence not found: {csv_path}")
    values = np.genfromtxt(csv_path, delimiter=",")
    return np.asarray(values, dtype=np.complex64).reshape(-1)


def _pdp_db(taps, floor_db=-120.0):
    floor_linear = 10.0 ** (float(floor_db) / 20.0)
    magnitude = np.maximum(np.abs(np.asarray(taps, dtype=np.complex64)), floor_linear)
    return (20.0 * np.log10(magnitude)).astype(np.float32)


def _normalized_valid_correlation(rx, tx):
    tx_energy = float(np.vdot(tx, tx).real) + 1e-12
    corr = np.convolve(rx, np.conj(tx[::-1]), mode="valid")
    window_energy = np.convolve(np.abs(rx) ** 2, np.ones(tx.size, dtype=np.float32), mode="valid")
    norm = np.abs(corr) / np.sqrt(np.maximum(window_energy * tx_energy, 1e-12))
    return corr.astype(np.complex64), norm.astype(np.float32), tx_energy


def estimate_cast_probe_channel(
    rx_samples,
    sequence="cast",
    num_taps=128,
    detection_threshold=0.05,
    estimation_window_repetitions=4,
    pre_periods=1,
):
    """Detect a repeated CaST probe in RX IQ and estimate a matched-filter CIR."""
    rx = np.asarray(rx_samples, dtype=np.complex64).reshape(-1)
    tx = load_cast_sequence(sequence)
    tap_count = max(1, int(num_taps))
    period = int(tx.size)

    empty_taps = np.zeros(tap_count, dtype=np.complex64)
    if rx.size < tx.size:
        return {
            "detected": False,
            "iq": rx,
            "probe": tx,
            "taps": empty_taps,
            "cir": empty_taps,
            "pdp_db": _pdp_db(empty_taps),
            "metadata": {
                "reason": "rx_shorter_than_probe",
                "sequence": normalize_sequence_name(sequence),
                "sequence_length": period,
                "rx_samples": int(rx.size),
                "num_taps": tap_count,
            },
        }

    valid_corr, norm_corr, tx_energy = _normalized_valid_correlation(rx, tx)
    peak_index = int(np.argmax(norm_corr))
    normalized_peak = float(norm_corr[peak_index])
    matched_peak = valid_corr[peak_index] / tx_energy
    detected = normalized_peak >= float(detection_threshold)

    window_start = max(0, peak_index - int(pre_periods) * period)
    min_window_len = tx.size + tap_count - 1
    requested_window_len = max(min_window_len, int(estimation_window_repetitions) * period)
    window_stop = min(rx.size, window_start + requested_window_len)
    rx_window = rx[window_start:window_stop]

    full_corr = np.convolve(rx_window, np.conj(tx[::-1]), mode="full") / tx_energy
    full_peak_index = int(np.argmax(np.abs(full_corr))) if full_corr.size else 0
    taps = full_corr[full_peak_index:full_peak_index + tap_count].astype(np.complex64)
    if taps.size < tap_count:
        taps = np.pad(taps, (0, tap_count - taps.size)).astype(np.complex64)

    # The frame phase is useful for tracking a continuous repeated probe stream.
    frame_phase = int(peak_index % period)
    first_detected_frame = int(peak_index - ((peak_index - frame_phase) // period) * period)

    return {
        "detected": bool(detected),
        "iq": rx,
        "probe": tx,
        "taps": taps,
        "cir": taps,
        "pdp_db": _pdp_db(taps),
        "metadata": {
            "sequence": normalize_sequence_name(sequence),
            "sequence_length": period,
            "rx_samples": int(rx.size),
            "num_taps": tap_count,
            "detection_threshold": float(detection_threshold),
            "normalized_peak": normalized_peak,
            "correlation_peak_real": float(np.real(matched_peak)),
            "correlation_peak_imag": float(np.imag(matched_peak)),
            "pathloss_db": float(20.0 * np.log10(max(abs(matched_peak), 1e-12))),
            "peak_index": peak_index,
            "frame_phase": frame_phase,
            "first_detected_frame": first_detected_frame,
            "window_start": int(window_start),
            "window_stop": int(window_stop),
            "full_correlation_peak_index": full_peak_index,
            "estimation_window_repetitions": int(estimation_window_repetitions),
        },
    }
