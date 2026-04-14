# CaST Probe Sequences

These BPSK code-sequence CSVs are copied from `cast-main/radio_api/code_sequences` so PHY-Key-Generation can transmit the same CaST probe waveforms through the existing GNU Radio `pnSequence` transmitter.

CaST defaults to `glfsr_bpsk` (`CODE_SEQUENCE = 1`), which is exposed in PHY-Key-Generation as `cast`, `cast_glfsr`, `glfsr`, and `glfsr_bpsk`.
