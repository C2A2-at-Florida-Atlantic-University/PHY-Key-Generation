#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: File Source TX
# Author: Jose Sanchez
# GNU Radio version: 3.8.5.0

from gnuradio import blocks
import pmt
from gnuradio import gr
import sys
import signal
import iio

class FileSource(gr.top_block):

    def __init__(self,samp_rate=1000000,gain=0,freq=2400000000,
        buffer_size=0x800,bandwidth=20000000,SDR_ID="ip:192.168.2.1",
        filename="/home/siwn/siwn-node/network/Matlab/BPSK.dat"):
        gr.top_block.__init__(self, "File Source TX")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.gain = gain
        self.freq = freq
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.SDR_ID = SDR_ID
        self.filename = filename

        ##################################################
        # Blocks
        ##################################################

        self.iio_pluto_sink_0 = iio.pluto_sink(self.SDR_ID, self.freq, self.samp_rate, self.bandwidth, self.buffer_size, True, self.gain, '', True)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, self.filename, True, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.iio_pluto_sink_0, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_filename(self):
        return self.gain

    def set_filename(self, filename):
        self.filename = filename
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, self.filename, True, 0, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_SDR_ID(self):
        return self.SDR_ID

    def set_SDR_ID(self, SDR_ID):
        self.SDR_ID = SDR_ID

def main(top_block_cls=FileSource, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
