#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QT RX scope for debugging CaST probe visibility on POWDER dense B210 nodes."""

import argparse
import signal
import sys

from PyQt5 import Qt, QtCore
from gnuradio import blocks, gr, qtgui, uhd
from gnuradio.fft import window

try:
    import sip
except ImportError:  # pragma: no cover - depends on the PyQt packaging.
    from PyQt5 import sip


def _qt_widget(block):
    if hasattr(block, "qwidget"):
        return sip.wrapinstance(block.qwidget(), Qt.QWidget)
    return sip.wrapinstance(block.pyqwidget(), Qt.QWidget)


class CastProbeRxQtDebug(gr.top_block, Qt.QWidget):
    def __init__(
        self,
        freq=3.5e9,
        samp_rate=2e6,
        gain=90,
        bandwidth=20e6,
        clock_rate=32e6,
        antenna="RX2",
        device_args="",
        fft_size=2048,
        time_samples=4096,
    ):
        gr.top_block.__init__(self, "CaST Probe RX QT Debug")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("CaST Probe RX QT Debug")
        qtgui.util.check_set_qss()

        self.freq = float(freq)
        self.samp_rate = float(samp_rate)
        self.gain = float(gain)
        self.bandwidth = float(bandwidth)
        self.clock_rate = float(clock_rate)
        self.antenna = antenna
        self.device_args = device_args
        self.fft_size = int(fft_size)
        self.time_samples = int(time_samples)

        self.top_layout = Qt.QVBoxLayout(self)
        self.controls_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.controls_layout)

        self._freq_range = qtgui.Range(3.3e9, 3.8e9, 1e6, self.freq, 200)
        self._freq_win = qtgui.RangeWidget(
            self._freq_range, self.set_freq, "Frequency (Hz)", "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.controls_layout.addWidget(self._freq_win, 0, 0)

        self._samp_rate_range = qtgui.Range(0.5e6, 4.0e6, 0.25e6, self.samp_rate, 200)
        self._samp_rate_win = qtgui.RangeWidget(
            self._samp_rate_range,
            self.set_samp_rate,
            "Sample rate (S/s)",
            "counter_slider",
            float,
            QtCore.Qt.Horizontal,
        )
        self.controls_layout.addWidget(self._samp_rate_win, 1, 0)

        self._gain_range = qtgui.Range(0, 90, 1, self.gain, 200)
        self._gain_win = qtgui.RangeWidget(
            self._gain_range, self.set_gain, "RX gain", "counter_slider", float, QtCore.Qt.Horizontal
        )
        self.controls_layout.addWidget(self._gain_win, 2, 0)

        self.usrp_source = uhd.usrp_source(
            ",".join((self.device_args, "")),
            uhd.stream_args(cpu_format="fc32", args="", channels=[0]),
        )
        self.usrp_source.set_clock_rate(self.clock_rate, uhd.ALL_MBOARDS)
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.usrp_source.set_center_freq(self.freq, 0)
        self.usrp_source.set_antenna(self.antenna, 0)
        self.usrp_source.set_bandwidth(self.bandwidth, 0)
        self.usrp_source.set_gain(self.gain, 0)
        self.usrp_source.set_max_output_buffer(1024 * 1024)
        self.usrp_source.set_time_unknown_pps(uhd.time_spec(0))
        self._enable_frontend()

        self.power = blocks.complex_to_mag_squared(1)
        self.power_probe = blocks.probe_signal_f()

        self.time_sink = qtgui.time_sink_c(
            self.time_samples,
            self.samp_rate,
            "RX IQ time",
            1,
            None,
        )
        self.time_sink.set_update_time(0.10)
        self.time_sink.enable_autoscale(True)
        self.time_sink.enable_grid(True)
        self.time_sink.enable_axis_labels(True)
        self.time_sink.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")

        self.freq_sink = qtgui.freq_sink_c(
            self.fft_size,
            window.WIN_BLACKMAN_hARRIS,
            self.freq,
            self.samp_rate,
            "RX spectrum",
            1,
            None,
        )
        self.freq_sink.set_update_time(0.10)
        self.freq_sink.set_y_axis(-120, 10)
        self.freq_sink.enable_autoscale(False)
        self.freq_sink.enable_grid(True)
        self.freq_sink.enable_axis_labels(True)

        self.top_layout.addWidget(_qt_widget(self.time_sink))
        self.top_layout.addWidget(_qt_widget(self.freq_sink))

        self.status = Qt.QLabel(self)
        self.top_layout.addWidget(self.status)
        self.status_timer = Qt.QTimer(self)
        self.status_timer.timeout.connect(self._refresh_status)
        self.status_timer.start(500)

        self.connect((self.usrp_source, 0), (self.time_sink, 0))
        self.connect((self.usrp_source, 0), (self.freq_sink, 0))
        self.connect((self.usrp_source, 0), (self.power, 0))
        self.connect((self.power, 0), (self.power_probe, 0))

    def _enable_frontend(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x0)
        self.usrp_source.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT", 0x10, 0x10, 0)

    def _disable_frontend(self):
        self.usrp_source.set_gpio_attr("FP0", "CTRL", 0x0)
        self.usrp_source.set_gpio_attr("FP0", "DDR", 0x10, 0x10, 0)
        self.usrp_source.set_gpio_attr("FP0", "OUT", 0x0, 0x10, 0)

    def _refresh_status(self):
        try:
            power = float(self.power_probe.level())
        except Exception:
            power = float("nan")
        self.status.setText(
            f"freq={self.freq:.0f} Hz, samp_rate={self.samp_rate:.0f} S/s, "
            f"gain={self.gain:.1f} dB, instantaneous |IQ|^2={power:.4g}"
        )

    def start(self):
        self._enable_frontend()
        return super().start()

    def stop(self):
        self._disable_frontend()
        return super().stop()

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

    def set_freq(self, freq):
        self.freq = float(freq)
        self.usrp_source.set_center_freq(self.freq, 0)
        self.freq_sink.set_frequency_range(self.freq, self.samp_rate)

    def set_samp_rate(self, samp_rate):
        self.samp_rate = float(samp_rate)
        self.usrp_source.set_samp_rate(self.samp_rate)
        self.time_sink.set_samp_rate(self.samp_rate)
        self.freq_sink.set_frequency_range(self.freq, self.samp_rate)

    def set_gain(self, gain):
        self.gain = float(gain)
        self.usrp_source.set_gain(self.gain, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="QT RX scope for CaST probe debugging.")
    parser.add_argument("--freq", type=float, default=3.5e9, help="RF center frequency in Hz.")
    parser.add_argument("--samp-rate", type=float, default=2e6, help="RX sample rate in samples/s.")
    parser.add_argument("--gain", type=float, default=90, help="RX gain.")
    parser.add_argument("--bandwidth", type=float, default=20e6, help="Analog bandwidth in Hz.")
    parser.add_argument("--clock-rate", type=float, default=32e6, help="B210 master clock rate in Hz.")
    parser.add_argument("--antenna", default="RX2", help="RX antenna port.")
    parser.add_argument("--device-args", default="", help="UHD device args.")
    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size for spectrum plot.")
    parser.add_argument("--time-samples", type=int, default=4096, help="Samples in time plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    qapp = Qt.QApplication(sys.argv)
    tb = CastProbeRxQtDebug(
        freq=args.freq,
        samp_rate=args.samp_rate,
        gain=args.gain,
        bandwidth=args.bandwidth,
        clock_rate=args.clock_rate,
        antenna=args.antenna,
        device_args=args.device_args,
        fft_size=args.fft_size,
        time_samples=args.time_samples,
    )
    tb.start()
    tb.show()

    def stop_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        qapp.quit()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    qapp.exec_()


if __name__ == "__main__":
    main()
