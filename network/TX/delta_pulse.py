#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Delta Pulse TX
# Author: root
# GNU Radio version: v3.8.5.0-6-g57bd109d

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except Exception:
            print("Warning: failed to XInitThreads()")

from delta_pulse import delta_pulse_source
from gnuradio import gr, uhd
import sys
import time

class DeltaPulse(gr.top_block):

    def __init__(self,
                samp_rate=1000000,
                gain=31,
                freq=3.55e9,
                buffer_size=8192,
                bandwidth=20000000,
                SDR_ADDR="",
                num_bins=512,
                amplitude=1,
                center=False,
                repeat=True,
                window=True,
                num_pulses=-1):
        gr.top_block.__init__(self, "DeltaPulse")
        ##################################################
        # Variables
        ##################################################
        self.samp_rate=samp_rate 
        self.gain=gain 
        self.freq=freq 
        self.buffer_size=buffer_size 
        self.bandwidth=bandwidth 
        self.SDR_ADDR=SDR_ADDR 
        self.num_bins=num_bins
        self.amplitude=amplitude
        self.center=center
        self.repeat=repeat
        self.window=window
        self.num_pulses=num_pulses

        ##################################################
        # Blocks
        ##################################################
        self.max_buf = 1024*1024  
        self.usrp_sink = uhd.usrp_sink(
            # device address string: blank => first USRP found
            ",".join((self.SDR_ADDR, "")),
            # stream args: one channel of complex floats
            uhd.stream_args(
                cpu_format="fc32",
                args="",
                channels=[0],
            ),
            ""  # XML or args string (unused here)
        )
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.usrp_sink.set_center_freq(self.freq, 0)
        self.usrp_sink.set_gain(self.gain, 0)
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_clock_rate(30.72e6, uhd.ALL_MBOARDS)
        self.usrp_sink.set_max_output_buffer(self.max_buf)
        self.usrp_sink.set_time_unknown_pps(uhd.time_spec())
        
        # Set DC offset to zero to minimize DC offset and LO leakage (especially for x310 radios)
        try:
            self.usrp_sink.set_dc_offset(0.0, 0)
        except Exception:
            # Some USRP models may not support set_dc_offset, ignore if not available
            pass
        
        # Try to enable automatic DC offset correction if available
        try:
            self.usrp_sink.set_auto_dc_offset(True, 0)
        except Exception:
            # Auto DC offset may not be available on all USRP models, ignore if not available
            pass
        
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
        self.delta_pulse_source_0 = delta_pulse_source(
            self.num_bins,
            self.amplitude,
            self.center,
            self.repeat,
            self.window,
            self.num_pulses
        )
        ##################################################
        # Connections
        ##################################################
        self.connect((self.delta_pulse_source_0, 0), (self.usrp_sink, 0))

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate=samp_rate
        self.usrp_sink.set_samp_rate(self.samp_rate)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain=gain
        self.usrp_sink.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq=freq
        self.usrp_sink.set_center_freq(self.freq, 0)
        # Re-apply DC offset correction when frequency changes (DC offset can be frequency-dependent)
        try:
            self.usrp_sink.set_dc_offset(0.0, 0)
        except Exception:
            pass
        try:
            self.usrp_sink.set_auto_dc_offset(True, 0)
        except Exception:
            pass

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size=buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth=bandwidth

    def get_num_bins(self):
        return self.num_bins

    def set_num_bins(self, num_bins):
        self.num_bins=num_bins
        # Note: delta_pulse_source doesn't support runtime parameter changes
        # Would need to recreate the block if this needs to be changed

    def get_amplitude(self):
        return self.amplitude

    def set_amplitude(self, amplitude):
        self.amplitude=amplitude
        # Note: delta_pulse_source doesn't support runtime parameter changes
        # Would need to recreate the block if this needs to be changed
        
    def start(self):
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)
        super().start()
        
    def stop(self):
        self.usrp_sink.set_gpio_attr("FP0", "DDR", 0xFFFFFFFF, 0x0, 0)
        self.usrp_sink.set_gpio_attr("FP0", "OUT", 0xFFFFFFFF, 0x0, 0)
        return super().stop()

def main(top_block_cls=DeltaPulse, options=None):
    tb = top_block_cls()
    tb.start()
    time.sleep(20)
    tb.stop()
    tb.wait()

if __name__ == '__main__':
    main()

