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


def canonical_sequence_metadata_name(sequence):
    canonical_name = normalize_sequence_name(sequence)
    return CAST_SEQUENCE_CSV_STEMS.get(canonical_name, canonical_name)


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


def _matched_filter_taps(rx_segment, tx, tap_count, tx_energy):
    segment = np.asarray(rx_segment, dtype=np.complex64).reshape(-1)
    required_len = tx.size + tap_count - 1
    if segment.size < required_len:
        segment = np.pad(segment, (0, required_len - segment.size))
    else:
        segment = segment[:required_len]

    matched = np.convolve(segment, np.conj(tx[::-1]), mode="full") / tx_energy
    start = tx.size - 1
    taps = matched[start:start + tap_count].astype(np.complex64)
    if taps.size < tap_count:
        taps = np.pad(taps, (0, tap_count - taps.size)).astype(np.complex64)
    return taps


def _detected_frame_starts(norm_corr, peak_index, frame_len, detection_threshold, requested_repetitions):
    max_start = int(norm_corr.size - 1)
    requested = max(1, int(requested_repetitions))
    if frame_len <= 0 or max_start < 0:
        return []

    first_start = int(peak_index)
    while first_start - frame_len >= 0:
        first_start -= frame_len

    starts = []
    candidate = first_start
    relaxed_threshold = 0.5 * float(detection_threshold)
    while candidate <= max_start:
        if candidate == int(peak_index) or float(norm_corr[candidate]) >= relaxed_threshold:
            starts.append(int(candidate))
            if len(starts) >= requested:
                break
        candidate += frame_len

    if int(peak_index) not in starts and 0 <= int(peak_index) <= max_start:
        starts.append(int(peak_index))
        starts = sorted(starts)[:requested]
    return starts


def estimate_cast_probe_channel(
    rx_samples,
    sequence="cast",
    num_taps=128,
    detection_threshold=0.05,
    estimation_window_repetitions=4,
    num_repetitions=None,
    guard_len=0,
    sample_rate_hz=1e6,
    estimation_mode="matched_filter",
    pre_periods=1,
):
    """Detect a repeated CaST probe in RX IQ and estimate a matched-filter CIR."""
    rx = np.asarray(rx_samples, dtype=np.complex64).reshape(-1)
    tx = load_cast_sequence(sequence)
    tap_count = max(1, int(num_taps))
    period = int(tx.size)
    guard_samples = max(0, int(guard_len))
    repetitions = max(1, int(num_repetitions if num_repetitions is not None else estimation_window_repetitions))
    frame_len = period + guard_samples
    sample_rate = float(sample_rate_hz)
    tap_delays_s = (np.arange(tap_count, dtype=np.float32) / sample_rate).astype(np.float32)
    mode = str(estimation_mode or "matched_filter").strip().lower()
    if mode != "matched_filter":
        raise ValueError(f"Unsupported online CaST estimation_mode: {estimation_mode!r}")

    empty_taps = np.zeros(tap_count, dtype=np.complex64)
    if rx.size < tx.size:
        return {
            "detected": False,
            "iq": rx,
            "probe": tx,
            "taps": empty_taps,
            "cir": empty_taps,
            "pdp_db": _pdp_db(empty_taps),
            "tap_delays_s": tap_delays_s,
            "metadata": {
                "reason": "rx_shorter_than_probe",
                "sequence": normalize_sequence_name(sequence),
                "sequence_name": canonical_sequence_metadata_name(sequence),
                "sequence_length": period,
                "rx_samples": int(rx.size),
                "num_taps": tap_count,
                "guard_len": guard_samples,
                "frame_len": frame_len,
                "num_repetitions": repetitions,
                "sample_rate_hz": sample_rate,
                "estimation_mode": mode,
            },
        }

    valid_corr, norm_corr, tx_energy = _normalized_valid_correlation(rx, tx)
    peak_index = int(np.argmax(norm_corr))
    normalized_peak = float(norm_corr[peak_index])
    matched_peak = valid_corr[peak_index] / tx_energy
    detected = normalized_peak >= float(detection_threshold)

    frame_starts = _detected_frame_starts(
        norm_corr=norm_corr,
        peak_index=peak_index,
        frame_len=frame_len,
        detection_threshold=detection_threshold,
        requested_repetitions=repetitions,
    )
    if not frame_starts:
        frame_starts = [peak_index]

    response_len = tx.size + tap_count - 1
    per_probe_taps = [
        _matched_filter_taps(rx[start:start + response_len], tx, tap_count, tx_energy)
        for start in frame_starts
    ]
    taps = np.mean(np.stack(per_probe_taps, axis=0), axis=0).astype(np.complex64)

    window_start = int(frame_starts[0])
    window_stop = int(min(rx.size, frame_starts[-1] + response_len))
    rx_window = rx[window_start:window_stop]
    full_corr = np.convolve(rx_window, np.conj(tx[::-1]), mode="full") / tx_energy
    full_peak_index = int(np.argmax(np.abs(full_corr))) if full_corr.size else 0

    # The frame phase is useful for tracking a continuous repeated probe stream.
    frame_phase = int(peak_index % frame_len)
    first_detected_frame = int(frame_starts[0])

    return {
        "detected": bool(detected),
        "iq": rx,
        "probe": tx,
        "taps": taps,
        "cir": taps,
        "pdp_db": _pdp_db(taps),
        "tap_delays_s": tap_delays_s,
        "metadata": {
            "sequence": normalize_sequence_name(sequence),
            "sequence_name": canonical_sequence_metadata_name(sequence),
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
            "detected_frame_starts": [int(start) for start in frame_starts],
            "window_start": int(window_start),
            "window_stop": int(window_stop),
            "full_correlation_peak_index": full_peak_index,
            "estimation_window_repetitions": int(estimation_window_repetitions),
            "num_repetitions": repetitions,
            "num_repetitions_used": int(len(frame_starts)),
            "guard_len": guard_samples,
            "frame_len": frame_len,
            "sample_rate_hz": sample_rate,
            "estimation_mode": mode,
            "tap_delays_s": tap_delays_s.tolist(),
        },
    }
