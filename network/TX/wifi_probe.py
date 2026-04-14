#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import sysconfig
import pmt
from gnuradio import blocks, gr, uhd
import foo
import ieee802_11


def _get_grc_state_directory():
    old_path = os.path.expanduser("~/.grc_gnuradio")
    try:
        from gnuradio.gr import paths  # type: ignore

        state_path = paths.persistent()
    except (ImportError, NameError, AttributeError):
        state_path = os.path.join(
            os.getenv("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
            "gnuradio",
        )

    os.makedirs(state_path, exist_ok=True)
    if os.path.isdir(old_path):
        return old_path
    return state_path


def _get_python_site_packages():
    try:
        return sysconfig.get_path("platlib") or ""
    except Exception:
        return ""


def _candidate_wifi_phy_hier_grc_files():
    tx_dir = os.path.dirname(__file__)
    workspace_root = os.path.abspath(os.path.join(tx_dir, "../../../"))
    phy_root = os.path.abspath(os.path.join(tx_dir, "../../"))
    prefixes = [os.environ.get("VIRTUAL_ENV", ""), sys.prefix, sys.base_prefix]

    candidates = []
    for prefix in prefixes:
        if prefix:
            candidates.append(
                os.path.join(prefix, "share", "gnuradio", "examples", "ieee802_11", "wifi_phy_hier.grc")
            )
    candidates.extend(
        [
            os.path.join(workspace_root, "gr-ieee802-11", "examples", "wifi_phy_hier.grc"),
            os.path.join(phy_root, "external", "gr-ieee802-11", "examples", "wifi_phy_hier.grc"),
            os.path.join(workspace_root, "external", "gr-ieee802-11", "examples", "wifi_phy_hier.grc"),
        ]
    )
    return [path for path in dict.fromkeys(os.path.abspath(path) for path in candidates if path)]


def _grc_block_paths():
    block_paths = []
    for prefix in [os.environ.get("VIRTUAL_ENV", ""), sys.prefix, sys.base_prefix]:
        if not prefix:
            continue
        candidate = os.path.join(prefix, "share", "gnuradio", "grc", "blocks")
        if os.path.isdir(candidate):
            block_paths.append(candidate)
    return [path for path in dict.fromkeys(block_paths)]


def _prepend_env_path(env, key, values):
    entries = [value for value in values if value]
    existing = env.get(key, "")
    if existing:
        entries.extend(part for part in existing.split(os.pathsep) if part)
    if entries:
        env[key] = os.pathsep.join(dict.fromkeys(entries))


def _ensure_wifi_phy_hier_generated():
    grcc_path = shutil.which("grcc")
    if not grcc_path:
        return

    grc_file = next((path for path in _candidate_wifi_phy_hier_grc_files() if os.path.isfile(path)), "")
    if not grc_file:
        return

    out_dir = os.environ.get("WIFI_GRC_HIER_PATH", "") or os.environ.get("GRC_HIER_PATH", "")
    if not out_dir:
        out_dir = _get_python_site_packages() or _get_grc_state_directory()
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    _prepend_env_path(env, "PYTHONPATH", [_get_python_site_packages()])
    _prepend_env_path(env, "GRC_BLOCKS_PATH", _grc_block_paths())
    subprocess.run(
        [sys.executable, grcc_path, "-o", out_dir, grc_file],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def _import_wifi_phy_hier():
    try:
        from wifi_phy_hier import wifi_phy_hier  # type: ignore
        return wifi_phy_hier
    except ImportError:
        pass

    candidate_paths = [
        os.environ.get("WIFI_GRC_HIER_PATH", ""),
        os.environ.get("GRC_HIER_PATH", ""),
        _get_python_site_packages(),
        _get_grc_state_directory(),
    ]
    for path in candidate_paths:
        if path and path not in sys.path and os.path.isdir(path):
            sys.path.append(path)

    try:
        from wifi_phy_hier import wifi_phy_hier  # type: ignore
        return wifi_phy_hier
    except ImportError:
        _ensure_wifi_phy_hier_generated()
        for path in candidate_paths:
            if path and path not in sys.path and os.path.isdir(path):
                sys.path.append(path)
        from wifi_phy_hier import wifi_phy_hier  # type: ignore
        return wifi_phy_hier


class WiFiProbeTx(gr.top_block):
    def __init__(
        self,
        samp_rate=10e6,
        gain=30,
        freq=2412e6,
        buffer_size=0x800,
        bandwidth=20e6,
        SDR_ADDR="",
        encoding=0,
        interval_ms=300,
        tx_amplitude=0.6,
        payload="probe_request",
    ):
        gr.top_block.__init__(self, "WiFiProbeTx")

        wifi_phy_hier = _import_wifi_phy_hier()

        self.samp_rate = samp_rate
        self.gain = gain
        self.freq = freq
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.SDR_ADDR = SDR_ADDR
        self.encoding = int(encoding)
        self.interval_ms = int(interval_ms)
        self.tx_amplitude = float(tx_amplitude)
        self.payload = payload
        self.max_buf = 1024 * 1024  # Request 1MB, but GNU Radio will cap to 8192
        self.wifi_phy_hier_0 = wifi_phy_hier(
            bandwidth=self.samp_rate,
            chan_est=ieee802_11.LS,
            encoding=ieee802_11.Encoding(self.encoding),
            frequency=self.freq,
            sensitivity=0.56,
        )

        self.usrp_sink = uhd.usrp_sink(
            ",".join((self.SDR_ADDR, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            ),
            "packet_len",
        )
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.usrp_sink.set_center_freq(self.freq, 0)
        self.usrp_sink.set_gain(self.gain, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_max_output_buffer(self.max_buf)

        self.ieee802_11_mac_0 = ieee802_11.mac(
            [0x23, 0x23, 0x23, 0x23, 0x23, 0x23],
            [0x42, 0x42, 0x42, 0x42, 0x42, 0x42],
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        )
        self.blocks_vector_source_x_0 = blocks.vector_source_c((0,), False, 1, [])
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(self.tx_amplitude)
        self.foo_packet_pad2_0 = foo.packet_pad2(False, False, 0.01, 100, 1000)
        self.foo_packet_pad2_0.set_min_output_buffer(max(96000, int(self.buffer_size)))
        self.blocks_message_strobe_0 = blocks.message_strobe(
            pmt.intern(str(self.payload)), self.interval_ms
        )

        self.msg_connect(
            (self.blocks_message_strobe_0, "strobe"), (self.ieee802_11_mac_0, "app in")
        )
        self.msg_connect(
            (self.ieee802_11_mac_0, "phy out"), (self.wifi_phy_hier_0, "mac_in")
        )
        self.connect((self.blocks_vector_source_x_0, 0), (self.wifi_phy_hier_0, 0))
        self.connect((self.wifi_phy_hier_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.foo_packet_pad2_0, 0))
        self.connect((self.foo_packet_pad2_0, 0), (self.usrp_sink, 0))

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.wifi_phy_hier_0.set_bandwidth(self.samp_rate)

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_sink.set_gain(self.gain, 0)

    def set_freq(self, freq):
        self.freq = freq
        self.usrp_sink.set_center_freq(self.freq, 0)
        self.wifi_phy_hier_0.set_frequency(self.freq)

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size
        self.foo_packet_pad2_0.set_min_output_buffer(max(96000, int(self.buffer_size)))

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.set_samp_rate(self.bandwidth)

    def set_interval_ms(self, interval_ms):
        self.interval_ms = int(interval_ms)
        self.blocks_message_strobe_0.set_period(self.interval_ms)

    def set_payload(self, payload):
        self.payload = payload
        self.blocks_message_strobe_0.set_msg(pmt.intern(str(self.payload)))

    def set_encoding(self, encoding):
        self.encoding = int(encoding)
        self.wifi_phy_hier_0.set_encoding(ieee802_11.Encoding(self.encoding))

    def set_tx_amplitude(self, tx_amplitude):
        self.tx_amplitude = float(tx_amplitude)
        self.blocks_multiply_const_vxx_0.set_k(self.tx_amplitude)
