#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Tx
# Author: root
# GNU Radio version: v3.8.5.0-6-g57bd109d

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from gnuradio import analog
from gnuradio import gr
import sys
import iio
import time

class Sinusoid(gr.top_block):

    def __init__(self,samp_rate=1000000,gain=0,freq=2400000000,
        buffer_size=8192,bandwidth=20000000,SDR_ID="ip:192.168.2.1"):
        gr.top_block.__init__(self, "Sinusoid")
        ##################################################
        # Variables
        ##################################################
        self.samp_rate=samp_rate 
        self.gain=gain 
        self.freq=freq 
        self.buffer_size=buffer_size 
        self.bandwidth=bandwidth 
        self.SDR_ID=SDR_ID 

        ##################################################
        # Blocks
        ##################################################
        self.iio_pluto_sink_0 = iio.pluto_sink(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, gain, '', True)
        self.analog_sig_source_x_0=analog.sig_source_c(samp_rate, analog.GR_SIN_WAVE, freq/128, 1, 0, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.iio_pluto_sink_0, 0))

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain=gain
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq=freq
        self.analog_sig_source_x_0.set_frequency(self.freq)
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size=buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth=bandwidth
        self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_SDR_ID(self):
        return self.SDR_ID

    def set_SDR_ID(self, SDR_ID):
        self.SDR_ID=SDR_ID

def main(top_block_cls=Sinusoid, options=None):
    tb = top_block_cls()
    tb.start()
    time.sleep(20)
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()
