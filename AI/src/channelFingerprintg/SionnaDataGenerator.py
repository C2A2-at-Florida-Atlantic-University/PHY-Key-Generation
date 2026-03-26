"""Generate synthetic channel-reciprocity training data via Sionna ray tracing.

Usage from the training pipeline:

    from SionnaDataGenerator import generate_sionna_training_data

    data, labels, rx, tx = generate_sionna_training_data(
        alice="EBC", bob="Guest House", eve="Moran",
        num_probes=100,
        signal_type="Sinusoid",   # or "deltaPulse"
        num_samples=8192,
    )

The returned arrays match the format expected by DatasetHandler.load_data():
  - data:   np.ndarray of complex64, shape (4*num_probes, num_samples)
            ordered as [AB, AE, BA, BE] quadruplets
  - labels: np.ndarray of int, shape (4*num_probes,)
  - rx:     np.ndarray of int, shape (4*num_probes,)
  - tx:     np.ndarray of int, shape (4*num_probes,)
"""

import sys
import os
import numpy as np

SIONNA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../SionnaPHYSec"))
if SIONNA_ROOT not in sys.path:
    sys.path.insert(0, SIONNA_ROOT)

NODE_NAME_TO_ID = {"EBC": 1, "Guest House": 2, "Moran": 3, "USTAR": 4}
NODE_ID_TO_NAME = {v: k for k, v in NODE_NAME_TO_ID.items()}
BASE_STATION_POSITIONS = {
    "EBC": [59.391, -174.7, 13.632],
    "Guest House": [230.06, -116.5, 2.9765],
    "Moran": [8.449, 151.12, 4.7896],
    "USTAR": [-147.4, -15.07, 2.2198],
}


def _resolve_node(node):
    """Accept node name (str) or integer ID and return (name, int_id)."""
    if isinstance(node, int):
        name = NODE_ID_TO_NAME.get(node)
        if name is None:
            raise ValueError(f"Unknown node ID {node}. Known: {NODE_ID_TO_NAME}")
        return name, node
    name = str(node)
    nid = NODE_NAME_TO_ID.get(name)
    if nid is None:
        raise ValueError(f"Unknown node name '{name}'. Known: {list(NODE_NAME_TO_ID)}")
    return name, nid


def _ensure_mitsuba_variant():
    """Set the Mitsuba backend variant once.

    Uses llvm_ad_rgb (CPU) by default because cuda_ad_rgb requires OptiX
    (libnvoptix.so.1) which is often absent in container environments.
    Set SIONNA_MITSUBA_VARIANT to override (e.g. "cuda_ad_rgb").
    """
    import mitsuba as mi
    if mi.variant() is not None:
        return
    preferred = os.environ.get("SIONNA_MITSUBA_VARIANT",
                               "llvm_ad_mono_polarized")
    candidates = [preferred] if preferred != "llvm_ad_mono_polarized" else []
    candidates.append("llvm_ad_mono_polarized")
    candidates.append("cuda_ad_mono_polarized")
    for variant in candidates:
        if variant in mi.variants():
            try:
                mi.set_variant(variant)
                print(f"[SionnaDataGenerator] Mitsuba variant set to '{variant}'")
                return
            except Exception:
                continue
    raise RuntimeError(
        f"No usable Mitsuba variant found. Available: {mi.variants()}"
    )


def _compute_taps(alice_name, bob_name):
    """Load the POWDER scene and compute channel taps from alice -> bob.

    Temporarily changes CWD to SIONNA_ROOT because POWDER_env_loader uses
    relative paths to locate scene files.
    """
    _ensure_mitsuba_variant()
    from POWDER_env_loader import LoadPOWDERScene, ComputeChannelImpulseResponse

    prev_cwd = os.getcwd()
    try:
        os.chdir(SIONNA_ROOT)
        scene, bs_positions = LoadPOWDERScene()
        _, _, _, _, _, taps, _, _, _ = ComputeChannelImpulseResponse(
            scene, bs_positions, alice_name, bob_name
        )
    finally:
        os.chdir(prev_cwd)

    taps_list = [complex(t) for t in taps[0, 0, 0, 0, 0, :]]
    return taps_list


def _make_probe_signal(num_samples, signal_type="Sinusoid", num_bins=1024):
    """Create the probe waveform (before channel convolution).

    signal_type:
      - "Sinusoid": 1 kHz complex sinusoid
      - "deltaPulse": OFDM-style IFFT delta pulse from gr-delta_pulse
    """
    if signal_type == "deltaPulse":
        from delta_pulse.deltaPulse import sdr_delta_pulse
        single_pulse = sdr_delta_pulse(
            N=num_bins, amplitude=0.8, window=True, center=True,
        )
        reps = int(np.ceil(num_samples / len(single_pulse)))
        sig = np.tile(single_pulse, reps)[:num_samples]
    else:
        samp_rate = 1_000_000
        freq = 1000
        t = np.arange(num_samples, dtype=np.float64) / samp_rate
        sig = np.exp(1j * 2 * np.pi * freq * t).astype(np.complex64)
    return sig.astype(np.complex64)


def _generate_iq_gnuradio(taps, num_samples, signal_type="Sinusoid",
                          num_bins=1024):
    """Run a GNU Radio flowgraph to push a signal through the channel taps.

    For deltaPulse, uses the delta_pulse_source block from gr-delta_pulse.
    For Sinusoid, uses a standard complex sinusoid source.
    """
    from gnuradio import analog, blocks, channels, gr
    import time as _time

    samp_rate = 1_000_000

    tb = gr.top_block("SionnaIQGen", catch_exceptions=True)

    if signal_type == "deltaPulse":
        from delta_pulse import delta_pulse_source
        src = delta_pulse_source(
            num_bins=num_bins, amplitude=0.8, window=True,
            center=True, repeat=True, num_pulses=-1,
        )
    else:
        src = analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, 1000, 1, 0, 0)

    chan = channels.channel_model(
        noise_voltage=0.01,
        frequency_offset=0.0,
        epsilon=1.0,
        taps=taps,
        noise_seed=0,
        block_tags=False,
    )
    vec_len = 512
    s2v = blocks.stream_to_vector(gr.sizeof_gr_complex, vec_len)
    sink = blocks.vector_sink_c(vec_len, vec_len)

    tb.connect(src, chan)
    tb.connect(chan, s2v)
    tb.connect(s2v, sink)

    tb.start()
    _time.sleep(0.15)
    iq = np.array(sink.data(), dtype=np.complex64)
    tb.stop()
    tb.wait()

    return iq[-num_samples:]


def _generate_iq_numpy(taps, num_samples, signal_type="Sinusoid",
                       noise_voltage=0.01, seed=None, num_bins=1024):
    """Pure-NumPy IQ generation (no GNU Radio dependency).

    Convolves the probe signal with the channel taps and adds AWGN.
    For deltaPulse, the probe is generated by sdr_delta_pulse from
    gr-delta_pulse (OFDM-style IFFT delta pulse).
    """
    rng = np.random.RandomState(seed)
    taps_arr = np.asarray(taps, dtype=np.complex64)

    sig = _make_probe_signal(num_samples, signal_type, num_bins=num_bins)

    rx = np.convolve(sig, taps_arr, mode="full")[:num_samples]

    if noise_voltage > 0:
        noise = noise_voltage * (
            rng.randn(num_samples) + 1j * rng.randn(num_samples)
        ).astype(np.complex64) / np.sqrt(2)
        rx = rx + noise

    return rx


def generate_probe_quadruplet(
    alice_name,
    bob_name,
    eve_name,
    num_samples=8192,
    signal_type="Sinusoid",
    backend="numpy",
    noise_voltage=0.01,
    seed=None,
    num_bins=1024,
):
    """Generate one [AB, AE, BA, BE] quadruplet of IQ probes.

    Returns four complex-valued 1D arrays of length num_samples.
    """
    taps_ab = _compute_taps(alice_name, bob_name)
    taps_ba = _compute_taps(bob_name, alice_name)
    taps_ae = _compute_taps(alice_name, eve_name)
    taps_be = _compute_taps(bob_name, eve_name)

    gen = _generate_iq_numpy if backend == "numpy" else _generate_iq_gnuradio

    kwargs = {"num_samples": num_samples, "signal_type": signal_type,
              "num_bins": num_bins}
    if backend == "numpy":
        kwargs["noise_voltage"] = noise_voltage

    if seed is not None and backend == "numpy":
        iq_ab = gen(taps_ab, seed=seed, **kwargs)
        iq_ae = gen(taps_ae, seed=seed + 1, **kwargs)
        iq_ba = gen(taps_ba, seed=seed + 2, **kwargs)
        iq_be = gen(taps_be, seed=seed + 3, **kwargs)
    else:
        iq_ab = gen(taps_ab, **kwargs)
        iq_ae = gen(taps_ae, **kwargs)
        iq_ba = gen(taps_ba, **kwargs)
        iq_be = gen(taps_be, **kwargs)

    return iq_ab, iq_ae, iq_ba, iq_be


def generate_sionna_training_data(
    alice,
    bob,
    eve,
    num_probes=100,
    signal_type="Sinusoid",
    num_samples=8192,
    backend="numpy",
    noise_voltage=0.01,
    seed=42,
    num_bins=1024,
):
    """Generate a full synthetic training dataset for one scenario.

    Parameters
    ----------
    alice, bob, eve : str or int
        Node names ("EBC", "Guest House", ...) or integer IDs (1-4).
    num_probes : int
        Number of probe exchanges to simulate. Total samples = 4 * num_probes.
    signal_type : str
        "Sinusoid" or "deltaPulse".
    num_samples : int
        IQ samples per probe (8192 for Sinusoid, 16384 for deltaPulse).
    backend : str
        "numpy" (pure convolution, fast, no GNU Radio) or "gnuradio".
    noise_voltage : float
        AWGN standard deviation (numpy backend only).
    seed : int or None
        Base seed for reproducibility.
    num_bins : int
        FFT/IFFT size for delta pulse generation (default 1024).

    Returns
    -------
    data : np.ndarray, shape (4*num_probes, num_samples), complex64
    labels : np.ndarray, shape (4*num_probes,), int
    rx : np.ndarray, shape (4*num_probes,), int
    tx : np.ndarray, shape (4*num_probes,), int
    """
    alice_name, alice_id = _resolve_node(alice)
    bob_name, bob_id = _resolve_node(bob)
    eve_name, eve_id = _resolve_node(eve)

    print(f"[SionnaDataGenerator] Generating {num_probes} probes: "
          f"Alice={alice_name}({alice_id}), Bob={bob_name}({bob_id}), Eve={eve_name}({eve_id})")
    print(f"  signal_type={signal_type}, num_samples={num_samples}, backend={backend}")

    taps_ab = _compute_taps(alice_name, bob_name)
    taps_ba = _compute_taps(bob_name, alice_name)
    taps_ae = _compute_taps(alice_name, eve_name)
    taps_be = _compute_taps(bob_name, eve_name)

    gen = _generate_iq_numpy if backend == "numpy" else _generate_iq_gnuradio

    all_data = []
    all_labels = []
    all_rx = []
    all_tx = []

    for i in range(num_probes):
        probe_seed = (seed + i * 4) if seed is not None else None

        kwargs = {"num_samples": num_samples, "signal_type": signal_type,
                  "num_bins": num_bins}
        if backend == "numpy":
            kwargs["noise_voltage"] = noise_voltage

        if probe_seed is not None and backend == "numpy":
            iq_ab = gen(taps_ab, seed=probe_seed, **kwargs)
            iq_ae = gen(taps_ae, seed=probe_seed + 1, **kwargs)
            iq_ba = gen(taps_ba, seed=probe_seed + 2, **kwargs)
            iq_be = gen(taps_be, seed=probe_seed + 3, **kwargs)
        else:
            iq_ab = gen(taps_ab, **kwargs)
            iq_ae = gen(taps_ae, **kwargs)
            iq_ba = gen(taps_ba, **kwargs)
            iq_be = gen(taps_be, **kwargs)

        # Quadruplet order: [AB, AE, BA, BE]
        all_data.extend([iq_ab, iq_ae, iq_ba, iq_be])

        # channel label: 1 for all (same scenario)
        all_labels.extend([1, 2, 1, 3])

        # TX/RX for each link
        all_tx.extend([alice_id, alice_id, bob_id, bob_id])
        all_rx.extend([bob_id, eve_id, alice_id, eve_id])

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  Generated probe {i + 1}/{num_probes}")

    data = np.array(all_data, dtype=np.complex64)
    labels = np.array(all_labels, dtype=int)
    rx_arr = np.array(all_rx, dtype=int)
    tx_arr = np.array(all_tx, dtype=int)

    print(f"[SionnaDataGenerator] Done. data.shape={data.shape}")
    return data, labels, rx_arr, tx_arr


def generate_multi_scenario_data(
    scenarios,
    num_probes=100,
    signal_type="Sinusoid",
    num_samples=8192,
    backend="numpy",
    noise_voltage=0.01,
    seed=42,
    num_bins=1024,
):
    """Generate training data for multiple [Alice, Bob, Eve] scenarios.

    Parameters
    ----------
    scenarios : list of [alice, bob, eve] triples
        Each element is a list/tuple of 3 node names or IDs.
        Example: [["EBC", "Guest House", "Moran"], [1, 3, 4]]

    Returns
    -------
    Same as generate_sionna_training_data but concatenated across scenarios.
    """
    all_data, all_labels, all_rx, all_tx = [], [], [], []
    for idx, (alice, bob, eve) in enumerate(scenarios):
        sc_seed = (seed + idx * num_probes * 4) if seed is not None else None
        d, l, r, t = generate_sionna_training_data(
            alice, bob, eve,
            num_probes=num_probes,
            signal_type=signal_type,
            num_samples=num_samples,
            backend=backend,
            noise_voltage=noise_voltage,
            seed=sc_seed,
            num_bins=num_bins,
        )
        all_data.append(d)
        all_labels.append(l)
        all_rx.append(r)
        all_tx.append(t)

    return (
        np.concatenate(all_data, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_rx, axis=0),
        np.concatenate(all_tx, axis=0),
    )

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    node_names = ["EBC", "Guest House", "Moran"]
    signal_type = "Sinusoid" # "Sinusoid" or "deltaPulse"
    data, labels, rx, tx = generate_sionna_training_data(
        alice=node_names[0], bob=node_names[1], eve=node_names[2],
        num_probes=1,
        signal_type=signal_type,   
        backend="gnuradio", # "numpy" or "gnuradio"
        num_samples=8192,
        num_bins=1024,
    )
    print(data.shape)
    print(labels.shape)
    print(rx.shape)
    print(tx.shape)
    
    # Plot the data for the quadruplet
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    # Only plot the first 4
    for i, data in enumerate(data):
        axes[i, 0].plot(data.real, label='Real')
        axes[i, 0].plot(data.imag, label='Imaginary')
        axes[i, 0].set_title(f'{i} Real and Imaginary')
        axes[i, 0].legend()
        # Second axis is the spectrogram
        FFT_size = 2048 if signal_type == "deltaPulse" else 512
        axes[i, 1].specgram(data, NFFT=FFT_size)
        axes[i, 1].set_title(f'{i} Spectrogram')
        axes[i, 1].set_xlabel('Time')
        axes[i, 1].set_ylabel('Frequency')

    plt.savefig(f'sionna_data_plot_{node_names[0]}_{node_names[1]}_{node_names[2]}.png')