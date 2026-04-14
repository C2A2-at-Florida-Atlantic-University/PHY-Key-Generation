from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

POWDER_ROOT = Path(__file__).resolve().parents[1]
PIDML_ROOT = POWDER_ROOT.parent / "PIDML-PHY-SEC"
sys.path.insert(0, str(POWDER_ROOT))
sys.path.insert(0, str(PIDML_ROOT))

from network.RX.cast_probe import estimate_cast_probe_channel, load_cast_sequence
from src.channel_estimation.cast_estimation import CastChannelEstimator


def test_powderkeygen_guarded_cast_matches_pidml_matched_filter(tmp_path: Path) -> None:
    sequence = load_cast_sequence("glfsr_bpsk").real.astype(np.float32)
    cast_repo = tmp_path / "cast-main"
    sequence_dir = cast_repo / "radio_api" / "code_sequences"
    sequence_dir.mkdir(parents=True)
    np.savetxt(sequence_dir / "glfsr_bpsk.csv", sequence, delimiter=",")

    taps = np.zeros(16, dtype=np.complex64)
    taps[:5] = np.array(
        [1.0 + 0.0j, 0.22 + 0.08j, -0.12 + 0.04j, 0.05 - 0.03j, 0.02 + 0.01j],
        dtype=np.complex64,
    )

    pidml_estimator = CastChannelEstimator(
        sample_rate_hz=1e6,
        num_taps=16,
        cast_repo_path=str(cast_repo),
        sequence_name="glfsr_bpsk",
        num_repetitions=8,
        guard_len=32,
        cfr_bins=64,
        estimation_mode="matched_filter",
    )
    pidml_payload = pidml_estimator.simulate_and_estimate(taps, snr_db=float("inf"))

    powder_payload = estimate_cast_probe_channel(
        np.repeat(pidml_payload["rx_waveform"], 2),
        sequence="glfsr_bpsk",
        num_taps=16,
        detection_threshold=0.01,
        estimation_window_repetitions=8,
        num_repetitions=8,
        guard_len=32,
        sample_rate_hz=1e6,
        rx_sample_rate_hz=2e6,
        estimation_mode="matched_filter",
    )

    assert powder_payload["detected"]
    assert powder_payload["metadata"]["estimation_decimation"] == 2
    assert powder_payload["metadata"]["num_repetitions_used"] == 8
    np.testing.assert_allclose(
        powder_payload["taps"],
        pidml_payload["estimate"].taps,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        powder_payload["tap_delays_s"],
        pidml_payload["estimate"].tap_delays_s,
        rtol=0,
        atol=0,
    )


def test_continuous_guarded_cast_n_plus_two_capture_recovers_requested_repetitions() -> None:
    sequence = load_cast_sequence("glfsr_bpsk")
    guard_len = 32
    requested_repetitions = 8
    rx_to_tx_rate = 2
    raw_frame_len = (sequence.size + guard_len) * rx_to_tx_rate
    capture_samples = (requested_repetitions + 2) * raw_frame_len

    taps = np.zeros(16, dtype=np.complex64)
    taps[:4] = np.array([1.0 + 0.0j, 0.2 + 0.08j, -0.08 + 0.04j, 0.03 - 0.02j], dtype=np.complex64)
    tx_frame = np.concatenate([sequence, np.zeros(guard_len, dtype=np.complex64)])
    tx_stream = np.tile(tx_frame, requested_repetitions + 8)
    rx_stream = np.convolve(tx_stream, taps, mode="full")[:tx_stream.size]
    rx_stream = np.repeat(rx_stream, rx_to_tx_rate)
    rx_capture = rx_stream[123:123 + capture_samples]

    payload = estimate_cast_probe_channel(
        rx_capture,
        sequence="glfsr_bpsk",
        num_taps=16,
        detection_threshold=0.25,
        estimation_window_repetitions=requested_repetitions,
        num_repetitions=requested_repetitions,
        guard_len=guard_len,
        sample_rate_hz=1e6,
        rx_sample_rate_hz=2e6,
        estimation_mode="matched_filter",
        min_repetitions_detected=3,
        repetition_detection_threshold=0.25,
    )

    assert payload["detected"]
    assert payload["metadata"]["num_repetitions_used"] == requested_repetitions
    np.testing.assert_allclose(payload["taps"][:4], taps[:4], rtol=0, atol=2e-2)
