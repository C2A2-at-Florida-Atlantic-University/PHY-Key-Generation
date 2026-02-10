from datetime import datetime
import time
import uuid
import json
if __name__ != '__main__':
    from TX.mpsk import MPSK
    from TX.sinusoid import Sinusoid
    from TX.delta_pulse import DeltaPulse
    from TX.pkt_xmt_gr38 import packetTransmit
    from TX.pnSequence import pnSequence
    from TX.FileSource import FileSource
    try:
        from TX.wifi_probe import WiFiProbeTx
    except Exception:
        WiFiProbeTx = None
else:
    from mpsk import MPSK
    from sinusoid import Sinusoid
    from delta_pulse import DeltaPulse
    from pkt_xmt_gr38 import packetTransmit
    from pnSequence import pnSequence
    from FileSource import FileSource
    try:
        from wifi_probe import WiFiProbeTx
    except Exception:
        WiFiProbeTx = None

class Transmitter():
    def __init__(self,
                gain,
                samp_rate,
                freq=3.550e9,
                bandwidth=20000000,
                buffer_size=0x800,
                SDR_ADDR=""):
        self.gain = gain
        self.samp_rate = samp_rate
        self.freq = freq
        self.bandwidth = bandwidth
        self.buffer_size = buffer_size
        self.SDR_ADDR = SDR_ADDR
        self.tx = None
        self.type = "None"
        # Don't initialize any transmitter by default - wait for explicit type selection
        
    def setFreq(self,freq):
        self.freq = freq
        self.tx.set_freq(self.freq)
    
    def setGain(self,gain):
        self.gain = gain
        self.tx.set_gain(self.gain)
    
    def setSamplingRate(self,samp_rate):
        self.samp_rate = samp_rate
        self.tx.set_samp_rate(self.samp_rate)
    
    def setBandwidth(self,bandwidth):
        self.bandwidth = bandwidth
        self.tx.set_bandwidth(self.bandwidth)
    
    def set_buffer_size(self,buffer_size):
        self.buffer_size = buffer_size
        self.tx.set_buffer_size(self.buffer_size)

    def str_to_length_and_decimals(self,text):
        if isinstance(text, bytes):
            values=[c for c in text]
        else:
            values=[ord(c) for c in text]
        length=len(values)
        return length, values

    def _cleanup_tx(self):
        """Helper method to properly stop and cleanup existing transmitter"""
        if self.tx is not None:
            try:
                # Try to stop the transmitter if it's running
                self.tx.stop()
            except Exception:
                pass
            try:
                # Wait for it to finish stopping
                self.tx.wait()
            except Exception:
                pass
            try:
                # Delete the transmitter object
                del self.tx
            except Exception:
                pass
        self.tx = None

    def set_tx_data(self,data):
        json_string=json.dumps(data)
        length, values=self.str_to_length_and_decimals(json_string)
        sps=2
        self.type = "data"
        self._cleanup_tx()
        self.tx=packetTransmit(
            input=values,
            input_len=length,
            samp_rate=self.samp_rate,
            sps=sps,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR)
        
    def set_tx_M_PSK(self,M):
        self.type = "mpsk"
        self._cleanup_tx()
        self.tx = MPSK(
            samp_rate=self.samp_rate,
            sps=4,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            M=M
        )

    def set_tx_sinusoid(self):
        self.type = "sinusoid"
        self._cleanup_tx()
        self.tx=Sinusoid(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR)
    
    def set_tx_delta_pulse(self, num_bins=512, amplitude=1, center=False, repeat=True, window=True, num_pulses=-1):
        self.type = "deltaPulse"
        self._cleanup_tx()
        self.tx=DeltaPulse(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            num_bins=num_bins,
            amplitude=amplitude,
            center=center,
            repeat=repeat,
            window=window,
            num_pulses=num_pulses)
        
    def set_tx_pnSequence(self,sequence="glfsr"):
        self.type = "pnSequence"
        self._cleanup_tx()
        self.tx=pnSequence(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            sequence=sequence
        )
    
    def set_tx_fileSource(self,filename="'/home/siwn/siwn-node/network/Matlab/BPSK.dat"):
        self.type = "fileSource"
        self._cleanup_tx()
        self.tx=FileSource(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            filename=filename
        )

    def set_tx_wifi_probe(
        self,
        payload="probe_request",
        interval_ms=300,
        encoding=0,
        tx_amplitude=0.6,
    ):
        if WiFiProbeTx is None:
            raise RuntimeError(
                "WiFi probe TX dependencies are missing. Install/import gr-ieee802-11 modules (foo, ieee802_11, wifi_phy_hier)."
            )
        self.type = "wifiProbe"
        self._cleanup_tx()
        self.tx = WiFiProbeTx(
            samp_rate=self.samp_rate,
            gain=self.gain,
            freq=self.freq,
            buffer_size=self.buffer_size,
            bandwidth=self.bandwidth,
            SDR_ADDR=self.SDR_ADDR,
            encoding=encoding,
            interval_ms=interval_ms,
            tx_amplitude=tx_amplitude,
            payload=payload,
        )
    
    def start(self):
        if self.tx is None:
            raise RuntimeError("No transmitter type has been set. Call set_tx_* method first.")
        self.tx.start()

    def stop(self):
        if self.tx is not None:
            try:
                self.tx.stop()
                self.tx.wait()
            except Exception:
                pass
            try:
                del self.tx
            except Exception:
                pass
        self.tx = None

def create_data_packet(data):
    UUID=str(uuid.uuid4())
    now=datetime.now()
    timestamp=now.strftime("%Y%m%d%H%M%S")
    data_size=len(data)
    data_type=type(data).__name__
    content={"UUID":UUID,"ts":timestamp,"size":data_size,"type":data_type,"data":data}
    return content

def TestTransmitter():
    samp_rate=1e6
    gain=80
    freq=3.55e9
    transmitter = Transmitter(
        gain=gain,
        samp_rate=samp_rate,
        freq=freq,
        bandwidth=20000000,
        buffer_size=8192,
        SDR_ADDR=""
    )
    txType = input("Transmitter Type [Data (1), Sinusoid (2)]:")
    if txType == "1":
        print("Writing input")
        data=input("Input: ")
        content=create_data_packet(data)
        print("Setting data transmitter")
        transmitter.set_tx_data(content)
    elif txType == "2":
        print("Setting sinusoid transmitter")
        transmitter.set_tx_sinusoid()
    else:
        return
    print("Start TX")
    transmitter.start()
    t=15
    for i in range(t):
        print(i+1)
        time.sleep(1)
    print("Stop TX")
    transmitter.stop()

def TestTxSinusoid():
    print("Testing Tx Sinusoid")
    print("Setting up parameters")
    samp_rate=1e6
    gain=80
    freq=3.55e9
    buffer_size=8192
    bandwidth=20000000
    SDR_ADDR=""
    print("Setting transmitter")
    tx = Sinusoid(samp_rate=samp_rate,gain=gain,freq=freq,
        buffer_size=buffer_size,bandwidth=bandwidth,SDR_ADDR=SDR_ADDR)
    print("Transmitter Start")
    tx.start()
    t = 15
    print("Transmitting for " + str(t)+" seconds")
    for i in range(t):
        print(i+1)
        time.sleep(1)
    print("Transmitter Stop")
    tx.stop()
    print("Transmitter Wait")
    tx.wait()

if __name__ == '__main__':
    #TestTxSinusoid()
    TestTransmitter()
    print("Done")