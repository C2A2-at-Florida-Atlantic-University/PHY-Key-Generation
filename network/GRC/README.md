GNU Radio Companion files and generated/debug Python flowgraphs for TX/RX data and sinusoid.

CaST dense-node RF debug:

```bash
# TX node: continuously send glfsr_bpsk + 32 zero guard samples and plot the TX frame.
python3 cast_probe_tx_debug.py --freq 3.5e9 --samp-rate 1e6 --gain 89

# RX node: open QT time/spectrum plots with tunable frequency, sample rate, and gain.
python3 cast_probe_rx_qt_debug.py --freq 3.5e9 --samp-rate 2e6 --gain 90
```

Both debug flowgraphs explicitly enable the RF frontend GPIO used by the dense nodes when started and release it on stop.
