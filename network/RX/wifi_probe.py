#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import blocks, fft, gr, network, pdu, uhd
from gnuradio.fft import window
import ieee802_11


class WiFiProbeRx(gr.top_block):
    def __init__(
        self,
        samp_rate=10e6,
        gain=50,
        freq=2412e6,
        buffer_size=0x800,
        bandwidth=20e6,
        SDR_ADDR="",
        UDP_port=40868,
        symbols_udp_port=40869,
        pilots_udp_port=40870,
        csi_udp_port=40871,
        chan_est_udp_port=40872,
        sync_length=320,
        window_size=48,
        chan_est=0,
    ):
        gr.top_block.__init__(self, "WiFiProbeRx")

        self.samp_rate = samp_rate
        self.gain = gain
        self.freq = freq
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.SDR_ADDR = SDR_ADDR
        self.UDP_port = UDP_port
        self.symbols_udp_port = symbols_udp_port
        self.pilots_udp_port = pilots_udp_port
        self.csi_udp_port = csi_udp_port
        self.chan_est_udp_port = chan_est_udp_port
        self.sync_length = sync_length
        self.window_size = window_size
        self.chan_est = chan_est
        self.max_buf = 1024 * 1024  # Request 1MB, but GNU Radio will cap to 8192
        self.usrp_source = uhd.usrp_source(
            ",".join((self.SDR_ADDR, "")),
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            ),
        )
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.usrp_source.set_center_freq(self.freq, 0)
        self.usrp_source.set_gain(self.gain, 0)
        self.usrp_source.set_antenna("RX2", 0)
        self.usrp_source.set_max_output_buffer(self.max_buf)

        # Stream raw IQ through UDP so existing API retrieve_IQ()/rx_recordIQ continue working.
        self.udp_sink = network.udp_sink(
            gr.sizeof_gr_complex, 1, "127.0.0.1", self.UDP_port, 0, self.buffer_size, False
        )
        self.blocks_complex_to_mag_squared_0 = blocks.complex_to_mag_squared(1)
        self.blocks_complex_to_mag_0 = blocks.complex_to_mag(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_gr_complex * 1, 16)
        self.blocks_delay_0 = blocks.delay(gr.sizeof_gr_complex * 1, self.sync_length)
        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()
        self.blocks_divide_xx_0 = blocks.divide_ff(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(
            self.window_size + 16, 1, 4000, 1
        )
        self.blocks_moving_average_xx_1 = blocks.moving_average_cc(
            self.window_size, 1, 4000, 1
        )
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_gr_complex * 1, 64)
        self.fft_vxx_0 = fft.fft_vcc(64, True, window.rectangular(64), True, 1)
        self.ieee802_11_sync_short_0 = ieee802_11.sync_short(0.56, 2, False, False)
        self.ieee802_11_sync_long_0 = ieee802_11.sync_long(self.sync_length, False, False)
        self.ieee802_11_frame_equalizer_0 = ieee802_11.frame_equalizer(
            ieee802_11.Equalizer(self.chan_est), self.freq, self.samp_rate, False, False
        )
        self.ieee802_11_decode_mac_0 = ieee802_11.decode_mac(True, False)
        self.ieee802_11_parse_mac_0 = ieee802_11.parse_mac(False, True)
        self.pdu_pdu_to_tagged_stream_symbols = pdu.pdu_to_tagged_stream(
            gr.types.complex_t, "packet_len"
        )
        self.pdu_pdu_to_tagged_stream_pilots = pdu.pdu_to_tagged_stream(
            gr.types.complex_t, "packet_len"
        )
        self.pdu_pdu_to_tagged_stream_csi = pdu.pdu_to_tagged_stream(
            gr.types.complex_t, "packet_len"
        )
        self.pdu_pdu_to_tagged_stream_chan_est = pdu.pdu_to_tagged_stream(
            gr.types.complex_t, "packet_len"
        )
        self.udp_sink_symbols = network.udp_sink(
            gr.sizeof_gr_complex, 1, "127.0.0.1", self.symbols_udp_port, 0, self.buffer_size, False
        )
        self.udp_sink_pilots = network.udp_sink(
            gr.sizeof_gr_complex, 1, "127.0.0.1", self.pilots_udp_port, 0, self.buffer_size, False
        )
        self.udp_sink_csi = network.udp_sink(
            gr.sizeof_gr_complex, 1, "127.0.0.1", self.csi_udp_port, 0, self.buffer_size, False
        )
        self.udp_sink_chan_est = network.udp_sink(
            gr.sizeof_gr_complex, 1, "127.0.0.1", self.chan_est_udp_port, 0, self.buffer_size, False
        )

        self.msg_connect(
            (self.ieee802_11_decode_mac_0, "out"), (self.ieee802_11_parse_mac_0, "in")
        )
        self.msg_connect(
            (self.ieee802_11_frame_equalizer_0, "symbols"),
            (self.pdu_pdu_to_tagged_stream_symbols, "pdus"),
        )
        self.msg_connect(
            (self.ieee802_11_frame_equalizer_0, "pilots"),
            (self.pdu_pdu_to_tagged_stream_pilots, "pdus"),
        )
        self.msg_connect(
            (self.ieee802_11_frame_equalizer_0, "csi"),
            (self.pdu_pdu_to_tagged_stream_csi, "pdus"),
        )
        self.msg_connect(
            (self.ieee802_11_frame_equalizer_0, "chan_est_samples"),
            (self.pdu_pdu_to_tagged_stream_chan_est, "pdus"),
        )
        self.connect((self.usrp_source, 0), (self.udp_sink, 0))
        self.connect((self.usrp_source, 0), (self.blocks_complex_to_mag_squared_0, 0))
        self.connect((self.usrp_source, 0), (self.blocks_delay_0_0, 0))
        self.connect((self.usrp_source, 0), (self.blocks_multiply_xx_0, 0))
        self.connect((self.blocks_complex_to_mag_squared_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_moving_average_xx_0, 0), (self.blocks_divide_xx_0, 1))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_conjugate_cc_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.ieee802_11_sync_short_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_moving_average_xx_1, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.blocks_complex_to_mag_0, 0))
        self.connect((self.blocks_moving_average_xx_1, 0), (self.ieee802_11_sync_short_0, 1))
        self.connect((self.blocks_complex_to_mag_0, 0), (self.blocks_divide_xx_0, 0))
        self.connect((self.blocks_divide_xx_0, 0), (self.ieee802_11_sync_short_0, 2))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.blocks_delay_0, 0))
        self.connect((self.ieee802_11_sync_short_0, 0), (self.ieee802_11_sync_long_0, 0))
        self.connect((self.blocks_delay_0, 0), (self.ieee802_11_sync_long_0, 1))
        self.connect((self.ieee802_11_sync_long_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.fft_vxx_0, 0))
        self.connect((self.fft_vxx_0, 0), (self.ieee802_11_frame_equalizer_0, 0))
        self.connect((self.ieee802_11_frame_equalizer_0, 0), (self.ieee802_11_decode_mac_0, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_symbols, 0), (self.udp_sink_symbols, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_pilots, 0), (self.udp_sink_pilots, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_csi, 0), (self.udp_sink_csi, 0))
        self.connect((self.pdu_pdu_to_tagged_stream_chan_est, 0), (self.udp_sink_chan_est, 0))

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.ieee802_11_frame_equalizer_0.set_bandwidth(self.samp_rate)

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_source.set_gain(self.gain, 0)

    def set_freq(self, freq):
        self.freq = freq
        self.usrp_source.set_center_freq(self.freq, 0)
        self.ieee802_11_frame_equalizer_0.set_frequency(self.freq)

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.set_samp_rate(self.bandwidth)

    def set_chan_est(self, chan_est):
        self.chan_est = int(chan_est)
        self.ieee802_11_frame_equalizer_0.set_algorithm(
            ieee802_11.Equalizer(self.chan_est)
        )
        
    def start(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0x10, 0x10, 0)
        super().start()

    def stop(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "DDR",  0xFFFFFFFF, 0x0, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT",  0xFFFFFFFF, 0x0, 0)
        return super().stop()
