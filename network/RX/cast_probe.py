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


def _peak_near(norm_corr, center, radius, threshold):
    max_start = int(norm_corr.size - 1)
    if max_start < 0:
        return None
    center = int(center)
    radius = max(0, int(radius))
    left = max(0, center - radius)
    right = min(max_start + 1, center + radius + 1)
    if left >= right:
        return None
    local = norm_corr[left:right]
    local_index = int(np.argmax(local))
    candidate = int(left + local_index)
    if float(norm_corr[candidate]) >= float(threshold):
        return candidate
    return None


def _detected_frame_starts(
    norm_corr,
    peak_index,
    frame_len,
    repetition_threshold,
    requested_repetitions,
    frame_search_radius=None,
):
    max_start = int(norm_corr.size - 1)
    requested = max(1, int(requested_repetitions))
    if frame_len <= 0 or max_start < 0:
        return []
    search_radius = (
        max(2, int(round(0.015 * frame_len)))
        if frame_search_radius is None
        else max(0, int(frame_search_radius))
    )
    search_radius = min(search_radius, max(0, (frame_len - 1) // 3))

    first_start = int(peak_index)
    while first_start - frame_len >= 0:
        previous_start = _peak_near(
            norm_corr,
            center=first_start - frame_len,
            radius=search_radius,
            threshold=repetition_threshold,
        )
        if previous_start is None:
            break
        first_start = previous_start

    starts = []
    candidate = first_start
    threshold = float(repetition_threshold)
    while candidate <= max_start:
        next_start = _peak_near(
            norm_corr,
            center=candidate,
            radius=search_radius,
            threshold=threshold,
        )
        if next_start is not None and (not starts or next_start > starts[-1]):
            starts.append(int(next_start))
            if len(starts) >= requested:
                break
            candidate = next_start + frame_len
        else:
            candidate += frame_len

    if not starts and 0 <= int(peak_index) <= max_start:
        starts.append(int(peak_index))
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
    rx_sample_rate_hz=None,
    estimation_mode="matched_filter",
    min_repetitions_detected=1,
    repetition_detection_threshold=None,
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
    frame_search_radius = min(
        max(0, (frame_len - 1) // 3),
        max(2, int(round(0.015 * frame_len))),
    )
    sample_rate = float(sample_rate_hz)
    rx_sample_rate = sample_rate if rx_sample_rate_hz is None else float(rx_sample_rate_hz)
    sample_rate_ratio = rx_sample_rate / sample_rate if sample_rate > 0 else 1.0
    decimation = 1
    if sample_rate_ratio > 1.0 and abs(sample_rate_ratio - round(sample_rate_ratio)) < 1e-6:
        decimation = int(round(sample_rate_ratio))
    repetition_threshold = (
        max(float(detection_threshold), 0.25)
        if repetition_detection_threshold is None
        else float(repetition_detection_threshold)
    )
    min_repetitions = max(1, int(min_repetitions_detected))
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
                "frame_search_radius": frame_search_radius,
                "raw_frame_len": int(frame_len * decimation),
                "num_repetitions": repetitions,
                "sample_rate_hz": sample_rate,
                "rx_sample_rate_hz": rx_sample_rate,
                "estimation_decimation": decimation,
                "estimation_mode": mode,
                "min_repetitions_detected": min_repetitions,
                "repetition_detection_threshold": repetition_threshold,
            },
        }

    candidates = []
    for phase in range(decimation):
        estimation_rx = rx[phase::decimation] if decimation > 1 else rx
        if estimation_rx.size < tx.size:
            continue
        valid_corr, norm_corr, tx_energy = _normalized_valid_correlation(estimation_rx, tx)
        peak_index = int(np.argmax(norm_corr))
        normalized_peak = float(norm_corr[peak_index])
        frame_starts = _detected_frame_starts(
            norm_corr=norm_corr,
            peak_index=peak_index,
            frame_len=frame_len,
            repetition_threshold=repetition_threshold,
            requested_repetitions=repetitions,
            frame_search_radius=frame_search_radius,
        )
        periodic_values = [float(norm_corr[start]) for start in frame_starts if 0 <= start < norm_corr.size]
        periodic_mean = float(np.mean(periodic_values)) if periodic_values else 0.0
        score = periodic_mean + 0.05 * len(frame_starts) + normalized_peak
        candidates.append({
            "phase": phase,
            "rx": estimation_rx,
            "valid_corr": valid_corr,
            "norm_corr": norm_corr,
            "tx_energy": tx_energy,
            "peak_index": peak_index,
            "normalized_peak": normalized_peak,
            "frame_starts": frame_starts,
            "periodic_mean": periodic_mean,
            "score": score,
        })

    if not candidates:
        return {
            "detected": False,
            "iq": rx,
            "probe": tx,
            "taps": empty_taps,
            "cir": empty_taps,
            "pdp_db": _pdp_db(empty_taps),
            "tap_delays_s": tap_delays_s,
            "metadata": {
                "reason": "decimated_rx_shorter_than_probe",
                "sequence": normalize_sequence_name(sequence),
                "sequence_name": canonical_sequence_metadata_name(sequence),
                "sequence_length": period,
                "rx_samples": int(rx.size),
                "num_taps": tap_count,
                "guard_len": guard_samples,
                "frame_len": frame_len,
                "frame_search_radius": frame_search_radius,
                "raw_frame_len": int(frame_len * decimation),
                "num_repetitions": repetitions,
                "sample_rate_hz": sample_rate,
                "rx_sample_rate_hz": rx_sample_rate,
                "estimation_decimation": decimation,
                "estimation_mode": mode,
                "min_repetitions_detected": min_repetitions,
                "repetition_detection_threshold": repetition_threshold,
            },
        }

    best = max(candidates, key=lambda candidate: candidate["score"])
    estimation_rx = best["rx"]
    valid_corr = best["valid_corr"]
    norm_corr = best["norm_corr"]
    tx_energy = best["tx_energy"]
    peak_index = int(best["peak_index"])
    normalized_peak = float(best["normalized_peak"])
    matched_peak = valid_corr[peak_index] / tx_energy
    frame_starts = list(best["frame_starts"]) or [peak_index]
    detected = normalized_peak >= float(detection_threshold) and len(frame_starts) >= min_repetitions

    response_len = tx.size + tap_count - 1
    per_probe_taps = [
        _matched_filter_taps(estimation_rx[start:start + response_len], tx, tap_count, tx_energy)
        for start in frame_starts
    ]
    taps = np.mean(np.stack(per_probe_taps, axis=0), axis=0).astype(np.complex64)

    window_start = int(frame_starts[0])
    window_stop = int(min(estimation_rx.size, frame_starts[-1] + response_len))
    rx_window = estimation_rx[window_start:window_stop]
    full_corr = np.convolve(rx_window, np.conj(tx[::-1]), mode="full") / tx_energy
    full_peak_index = int(np.argmax(np.abs(full_corr))) if full_corr.size else 0

    # The frame phase is useful for tracking a continuous repeated probe stream.
    frame_phase = int(peak_index % frame_len)
    first_detected_frame = int(frame_starts[0])
    phase = int(best["phase"])
    raw_peak_index = int(peak_index * decimation + phase)
    raw_frame_len = int(frame_len * decimation)
    raw_frame_starts = [int(start * decimation + phase) for start in frame_starts]
    raw_window_start = int(window_start * decimation + phase)
    raw_window_stop = int(min(rx.size, window_stop * decimation + phase))

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
            "peak_index": raw_peak_index,
            "frame_phase": int(raw_peak_index % max(raw_frame_len, 1)),
            "first_detected_frame": raw_frame_starts[0],
            "detected_frame_starts": raw_frame_starts,
            "window_start": raw_window_start,
            "window_stop": raw_window_stop,
            "estimation_peak_index": peak_index,
            "estimation_frame_phase": frame_phase,
            "estimation_first_detected_frame": first_detected_frame,
            "estimation_detected_frame_starts": [int(start) for start in frame_starts],
            "estimation_window_start": int(window_start),
            "estimation_window_stop": int(window_stop),
            "full_correlation_peak_index": full_peak_index,
            "estimation_window_repetitions": int(estimation_window_repetitions),
            "num_repetitions": repetitions,
            "num_repetitions_used": int(len(frame_starts)),
            "min_repetitions_detected": min_repetitions,
            "periodic_peak_mean": float(best["periodic_mean"]),
            "repetition_detection_threshold": repetition_threshold,
            "guard_len": guard_samples,
            "frame_len": frame_len,
            "frame_search_radius": frame_search_radius,
            "raw_frame_len": raw_frame_len,
            "sample_rate_hz": sample_rate,
            "rx_sample_rate_hz": rx_sample_rate,
            "estimation_decimation": decimation,
            "decimation_phase": phase,
            "estimation_mode": mode,
            "tap_delays_s": tap_delays_s.tolist(),
        },
    }
