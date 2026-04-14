#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: PN Sequence TX
# Author: Jose Sanchez
# GNU Radio version: 3.8.5.0
# Ref: https://github.com/wineslab/cast/blob/main/radio_api/gnuradio_gui/tx.py

from pathlib import Path

from gnuradio import blocks, gr, uhd
import sys
import signal

CAST_SEQUENCE_DIR = Path(__file__).resolve().parent / "cast_sequences"
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


def _load_bpsk_csv(csv_path):
    values = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for token in line.strip().split(","):
                if token:
                    values.append(float(token))
    return tuple(values)


def load_cast_sequence(sequence="glfsr"):
    canonical_name = normalize_sequence_name(sequence)
    csv_stem = CAST_SEQUENCE_CSV_STEMS.get(canonical_name)
    if csv_stem is None:
        raise ValueError(f"Unsupported CaST probe sequence: {sequence!r}")
    return _load_bpsk_csv(CAST_SEQUENCE_DIR / f"{csv_stem}.csv")


class pnSequence(gr.top_block):

    def __init__(self,
                samp_rate=1000000,
                gain=0,
                freq=2400000000,
                buffer_size=32768,
                bandwidth=20000000,
                SDR_ADDR="",
                sequence="glfsr",
                guard_len=0):
        gr.top_block.__init__(self, "PN Sequence TX")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.gain = gain
        self.freq = freq
        self.buffer_size = buffer_size
        self.bandwidth = bandwidth
        self.SDR_ADDR = SDR_ADDR
        self.guard_len = max(0, int(guard_len))
        self.sequence = "glfsr"
        ### See https://ece.northeastern.edu/wineslab/papers/villa2022wintech.pdf for sequences ###
        self.sequences = {"ls2":(1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1),
                        "ls1_all":(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,-1,1,1,1,1,-1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1),
                        "ls1":(1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,1),
                        "glfsr_0s":(1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
                        "glfsr150":(1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1),
                        "glfsr":(1,-1,1,1,-1,-1,-1,1,1,1,1,-1,1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1),
                        "ga128":(1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1)}
        
        self._load_cast_sequences()
        self.sequence = self._resolve_sequence_name(sequence)

        ##################################################
        # Blocks
        ##################################################

        # self.iio_pluto_sink_0=iio.pluto_sink(SDR_ID, freq, samp_rate, bandwidth, buffer_size, True, gain, '', True)
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
        self.usrp_sink.set_clock_rate(32e6, uhd.ALL_MBOARDS)
        self.usrp_sink.set_samp_rate(self.samp_rate)
        self.usrp_sink.set_center_freq(self.freq, 0)
        self.usrp_sink.set_gain(self.gain, 0)
        # choose TX port on B200-series / X300-series
        self.usrp_sink.set_antenna("TX/RX", 0)
        self.usrp_sink.set_max_output_buffer(self.max_buf)
        self.blocks_vector_source_x_1 = blocks.vector_source_f(self._sequence_with_guard(), True, 1, [])
        self.blocks_null_source_1 = blocks.null_source(gr.sizeof_float*1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_float_to_complex_0, 0), (self.usrp_sink, 0))
        self.connect((self.blocks_null_source_1, 0), (self.blocks_float_to_complex_0, 1))
        self.connect((self.blocks_vector_source_x_1, 0), (self.blocks_float_to_complex_0, 0))

    def _load_cast_sequences(self):
        for canonical_name, csv_stem in CAST_SEQUENCE_CSV_STEMS.items():
            csv_path = CAST_SEQUENCE_DIR / f"{csv_stem}.csv"
            if not csv_path.exists():
                continue
            samples = _load_bpsk_csv(csv_path)
            self.sequences[canonical_name] = samples
            self.sequences[csv_stem] = samples

        for alias, canonical_name in CAST_SEQUENCE_ALIASES.items():
            if canonical_name in self.sequences:
                self.sequences[alias] = self.sequences[canonical_name]

    def _resolve_sequence_name(self, sequence):
        sequence_name = normalize_sequence_name(sequence)
        if sequence_name not in self.sequences:
            available = ", ".join(sorted(self.sequences))
            raise ValueError(f"Unknown PN/CaST sequence {sequence!r}. Available sequences: {available}")
        return sequence_name

    def _sequence_with_guard(self):
        samples = tuple(self.sequences[self.sequence])
        if self.guard_len <= 0:
            return samples
        return samples + (0.0,) * self.guard_len

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_sink.set_samp_rate(self.samp_rate)

    def get_available_sequences(self):
        return [sequence for sequence in self.sequences]
    
    def get_sequences(self):
        return self.sequences
    
    def get_ls2(self):
        return self.sequences["ls2"]

    def get_ls1_all(self):
        return self.sequences["ls1_all"]

    def get_ls1(self):
        return self.sequences["ls1"]

    def get_glfsr_0s(self):
        return self.sequences["glfsr_0s"]

    def get_glfsr150(self):
        return self.sequences["glfsr150"]

    def get_glfsr(self):
        return self.sequences["glfsr"]
    
    def get_ga128(self):
        return self.sequences["ga128"]

    def get_gold(self):
        return self.sequences["gold"]

    def get_gain(self):
        return self.gain
    
    def set_sequence(self,sequence):
        self.sequence = self._resolve_sequence_name(sequence)
        self.blocks_vector_source_x_1.set_data(self._sequence_with_guard(), [])

    def get_guard_len(self):
        return self.guard_len

    def set_guard_len(self, guard_len):
        self.guard_len = max(0, int(guard_len))
        self.blocks_vector_source_x_1.set_data(self._sequence_with_guard(), [])

    def set_gain(self, gain):
        self.gain = gain
        self.usrp_sink.set_gain(self.gain, 0)
    
    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.usrp_sink.set_center_freq(self.freq, 0)

    def get_buffer_size(self):
        return self.buffer_size

    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def get_bandwidth(self):
        return self.bandwidth

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth
        # self.iio_pluto_sink_0.set_params(self.freq, self.samp_rate, self.bandwidth, self.gain, '', True)

    def get_SDR_ID(self):
        return self.SDR_ID

    def set_SDR_ID(self, SDR_ID):
        self.SDR_ID = SDR_ID

def main(top_block_cls=pnSequence, options=None):
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
