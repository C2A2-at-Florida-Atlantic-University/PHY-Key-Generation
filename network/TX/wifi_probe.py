#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pmt
from gnuradio import blocks, gr, uhd
import foo
import ieee802_11


def _import_wifi_phy_hier():
    try:
        from wifi_phy_hier import wifi_phy_hier  # type: ignore
        return wifi_phy_hier
    except ImportError:
        pass

    candidate_paths = [
        os.environ.get("WIFI_GRC_HIER_PATH", ""),
        os.environ.get("GRC_HIER_PATH", ""),
        os.path.expanduser("~/.grc_gnuradio"),
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../gr-ieee802-11/examples")
        ),
    ]
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
