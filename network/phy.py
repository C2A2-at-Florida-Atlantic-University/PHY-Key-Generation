import time
import json
from RX.receiver import Receiver
from TX.transmitter import Transmitter, create_data_packet
from phyAPI import phyAPI

class PHY:
    def __init__(self,txSamplingRate=1e6,rxSamplingRate=1e6*2,txGain=80,rxGain=70,freq1=3.55e9,freq2=3.56e9, bandwidth=20000000, buffer_size=0x800):
        conf = getConf(file="../config.json")
        self.NodeIP = conf['node']['ip']
        self.NetPort = conf['NET']['port']
        self.availableFreq = {1:freq1,2:freq2}
        self.freq = {"rx":self.availableFreq[1],"tx":self.availableFreq[1]}
        self.mode = {"rx":"none","tx":"none"}
        self.samplingRate = {"rx":rxSamplingRate,"tx":txSamplingRate}
        self.gain = {"rx":rxGain,"tx":txGain}
        self.buffer_size=buffer_size
        self.bandwidth={"rx":int(0.8 * rxSamplingRate),"tx":int(0.8 * txSamplingRate)} if bandwidth is None else {"rx":bandwidth,"tx":bandwidth}
        self.transmitter = Transmitter(
            gain=self.gain["tx"],
            samp_rate=self.samplingRate["tx"],
            freq=self.freq["tx"], 
            bandwidth=self.bandwidth["tx"], 
            buffer_size=self.buffer_size,
            SDR_ADDR=""
        )
        self.receiver = Receiver(
            gain=self.gain["rx"],
            samp_rate=self.samplingRate["rx"],
            freq=self.freq["rx"],
            bandwidth=self.bandwidth["rx"],
            buffer_size=self.buffer_size,
            SDR_ADDR="",
            UDP_port=40868,
            UDP_IP="127.0.0.1"
        )
    
    def startAPI(self, apiPort=5002):
        self.api = phyAPI()
        self.api.injectNode(self)
        self.api.start(apiPort,self.NodeIP)
        
    def setFreq(self,freq,x):
        self.freq[x] = freq
        if x == "rx":
            self.receiver.setFreq(freq)
        elif x == "tx":
            self.transmitter.setFreq(freq)
    
    def changeFreq(self,freqN,x):
        if self.freq[x] != self.availableFreq[freqN]:
            self.setFreq(self.availableFreq[freqN],x)
    
    def setSamplingRate(self,samplingRate,x):
        self.samplingRate[x]=samplingRate
        if x == "rx":
            self.receiver.setSamplingRate(self.samplingRate[x])
        elif x == "tx":
            self.transmitter.setSamplingRate(self.samplingRate[x])
    
    def setRxSamplingRate(self,rxSamplingRate):
        self.rxSamplingRate = rxSamplingRate
    
    def setGain(self,gain,x):
        self.gain[x]=gain
        if x == "rx":
            self.receiver.setGain(self.gain[x])
        elif x == "tx":
            self.transmitter.setGain(self.gain[x])

    def setBandwidth(self,bandwidth,x):
        self.bandwidth[x]=bandwidth
        if x == "rx":
            self.receiver.setBandwidth(self.bandwidth[x])
        elif x == "tx":
            self.transmitter.setBandwidth(self.bandwidth[x])
    
    def setBufferSize(self,buffer_size,x):
        self.buffer_size=buffer_size
        if x == "rx":
            self.receiver.set_buffer_size(self.buffer_size)
        elif x == "tx":
            self.transmitter.set_buffer_size(self.buffer_size)
            
    ### TRANSMITTER FUNCTIONS ###
    def transmit_data(self,data,t):
        self.setTxData(data)
        self.transmitter.start()
        for i in range(t):
            time.sleep(0.1)
        self.transmitter.stop()
    
    def setTxData(self,data):
        content=create_data_packet(data)
        self.mode["tx"] = "data"
        self.transmitter.set_tx_data(content)
        
    def setTxSinusoid(self):
        self.mode["tx"] = "sinusoid"
        self.transmitter.set_tx_sinusoid() 
    
    def setTxDeltaPulse(self, num_bins=512, amplitude=1, center=False, repeat=True, window=True, num_pulses=-1):
        self.mode["tx"] = "deltaPulse"
        self.transmitter.set_tx_delta_pulse(num_bins=num_bins, amplitude=amplitude, center=center, repeat=repeat, window=window, num_pulses=num_pulses)
        
    def setTxPnSequence(self,sequence="glfsr"):
        self.mode["tx"] = "pnSequence"
        self.transmitter.set_tx_pnSequence(sequence)
        
    def setTxFileSource(self,filename="/home/siwn/siwn-node/network/Matlab/BPSK.dat"):
        self.mode["tx"] = "fileSource"
        self.transmitter.set_tx_fileSource(filename)
        
    def setTxMPSK(self,M):
        self.mode["tx"] = str(M)+"PSK"
        self.transmitter.set_tx_M_PSK(M)

    def setTxWiFiProbe(self, payload="probe_request", interval_ms=300, encoding=0, tx_amplitude=0.6):
        self.mode["tx"] = "wifiProbe"
        self.transmitter.set_tx_wifi_probe(
            payload=payload,
            interval_ms=interval_ms,
            encoding=encoding,
            tx_amplitude=tx_amplitude,
        )
        
    ### RECEIVER FUNCTIONS ###
    def set_receive_data(self):
        #if self.mode["rx"] != "data":
        self.mode["rx"] = "data"
        self.receiver.set_rx_data()

    def set_receive_IQ(self):
        self.receiver.set_rx_IQ()
        if self.mode["rx"] != "IQ":
            self.mode["rx"] = "IQ"
            return "Setting mode: "+self.mode["rx"]
        else:
            return "Omitting setting mode: "+self.mode["rx"]
        
    def set_receive_MPSK(self,M=2):
        if self.mode["rx"] != str(M)+"PSK":
            self.mode["rx"] = str(M)+"PSK"
            self.receiver.set_rx_MPSK(M)

    def set_receive_wifi_probe(self, chan_est=0):
        self.mode["rx"] = "wifiProbe"
        self.receiver.set_rx_wifi_probe(chan_est=chan_est)

    def _wifi_probe_data_ready(self, eq_data, required_counts):
        if eq_data is None:
            return False
        for key, min_count in required_counts.items():
            if key not in eq_data or len(eq_data[key]) < min_count:
                return False
        return True

    def _normalize_wifi_probe_required_counts(self, samples):
        if isinstance(samples, dict):
            return {
                "symbols": int(samples.get("symbols", samples.get("iq", 1))),
                "pilots": int(samples.get("pilots", 1)),
                "csi": int(samples.get("csi", 1)),
                "chan_est": int(samples.get("chan_est", samples.get("chan_est_samples", 1))),
            }
        scalar_samples = max(1, int(samples))
        return {
            "symbols": scalar_samples,
            "pilots": scalar_samples,
            "csi": scalar_samples,
            "chan_est": scalar_samples,
        }

    def record_wifi_probe_data(
        self,
        samples=1024,
        warmup_retries=3,
        warmup_sleep_s=0.1,
        warmup_timeout_s=0.5,
        read_timeout_s=1.5,
    ):
        if self.mode["rx"] != "wifiProbe":
            self.set_receive_wifi_probe()
        self.receiver.start()
        try:
            # Warmup: ensure each WiFi probe stream has arrived before final read.
            required_counts = self._normalize_wifi_probe_required_counts(samples)
            warmup_counts = {key: 1 for key in required_counts.keys()}
            for _ in range(max(0, int(warmup_retries))):
                warmup_data = self.receiver.retrieve_wifi_probe_data(
                    samples=warmup_counts,
                    max_wait_s=warmup_timeout_s
                )
                if self._wifi_probe_data_ready(warmup_data, warmup_counts):
                    break
                time.sleep(max(0.0, float(warmup_sleep_s)))

            eq = self.receiver.retrieve_wifi_probe_data(samples=samples)
            if not self._wifi_probe_data_ready(eq, required_counts):
                # One immediate retry after warmup if any stream is short.
                eq = self.receiver.retrieve_wifi_probe_data(
                    samples=samples,
                    max_wait_s=read_timeout_s
                )
            return eq
        finally:
            self.receiver.stop()

    def get_data(self):
        received = False
        print("Retrieving UDP data")
        while not(received):
            try:
                UUID,ts,size,type,contents=self.receiver.retrieve_data()
                received = True
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
                pass
        return received,UUID,ts,size,type,contents

    def get_raw_data(self):
        received = False
        print("Retrieving UDP data")
        while not(received):
            try:
                data=self.receiver.retrieve_raw_data(dataSize=4096)
                received = True
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
                pass
        return received,data
        
    def await_confirmation(self):
        print("setting to receive data at freq", self.freq["rx"])
        self.receiver.set_rx_data()
        print("Setting receiver to channel 2")
        self.changeFreq(2,"rx")
        print("start receive data")
        self.receiver.clear_UDP_socket()
        self.receiver.start()
        confirmed=False
        print("getting confirmation")
        while not(confirmed):
            try:
                UUID,ts,size,type,contents=self.receiver.retrieve_data()
                if contents == "confirmation":
                    confirmed=True
                    print("package received")
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
                pass
        print("setting to receive data")
        self.changeFreq(1,"rx")    
        print("stopping receiver")
        self.receiver.stop()
        self.receiver.clear_UDP_socket()
        return confirmed,UUID,ts,size,type,contents

    def send_confirmation(self):
        print("setting to transmit confirmation at freq", self.freq["tx"])
        self.set_TX_confirmation()
        print("Setting transmitter to channel 2")
        self.changeFreq(2,"tx")
        print("start transmitter")
        self.transmitter.start() #send confirmation
        time.sleep(0.4)
        receiving = True
        timeout = 3
        t = 0
        #Stop receiving when transmitter stops using a timeout
        print("Start receiver at channel 1 with timeout")
        while receiving:
            try:
                UUID,ts,size,type,contents=self.receiver.retrieve_data(dataSize=4096)
                t = 0
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)
                t = t+1
                if t == timeout:
                    receiving = False
                pass
        print("setting back to TX channel 1")
        self.changeFreq(1,"tx")
        print("stopping transmitter")
        self.transmitter.stop()

    def send_confirmation_sinusoid(self):
        print("setting to transmit confirmation at freq", self.freq["tx"])
        self.set_TX_confirmation()
        print("Setting transmitter to channel 2")
        self.changeFreq(2,"tx")
        print("start transmitter")
        self.transmitter.start() #send confirmation
        #Send confirmation while it is receiving sinusoid, use classifier or channel sounder
        #12 sec delay in the meanwhile
        timeout = 2
        time.sleep(timeout)
        print("setting back to TX channel 1")
        self.changeFreq(1,"tx")
        print("stop transmitting")
        self.transmitter.stop()
        

    def set_TX_confirmation(self):
        data="confirmation"
        self.setTxData(data)

    def send_data_wait_confirmation(self,data="Hello"):
        print("Setting to transmit data:",data)
        self.setTxData(data)
        print("Start transmitter at freq", self.freq["tx"])
        self.transmitter.start()
        print("await for confirmation")
        self.await_confirmation()
        print("stop transmitter")
        self.transmitter.stop()

    def receive_data_send_confirmation(self):
        print("setting to receive data")
        self.set_receive_data()
        print("Start receiver at freq", self.freq["rx"])
        self.receiver.clear_UDP_socket()
        self.receiver.start()
        print("getting received data")
        received,UUID,ts,size,type,contents = self.get_data()
        if received:
            print("data received, sending confirmation")
            self.send_confirmation()
        print("stopping receiver")
        self.receiver.stop()
        self.receiver.clear_UDP_socket()
        return received,UUID,ts,size,type,contents

    def receive_raw_data_send_confirmation(self):
        print("setting to receive raw data")
        self.set_receive_data()
        print("Start receiver at freq", self.freq["rx"])
        self.receiver.clear_UDP_socket()
        self.receiver.start()
        print("getting received data")
        received,data = self.get_raw_data()
        if received:
            print("data received, sending confirmation")
            self.send_confirmation()
        print("stopping receiver")
        self.receiver.stop()
        self.receiver.clear_UDP_socket()
        return received,data

    def transmit_sinusoid_await_confirmation(self):
        print("setting to transmit sinusoid")
        self.setTxSinusoid()
        print("start transmitter at freq", self.freq["tx"])
        self.transmitter.start()
        print("awaiting for confirmation")
        self.await_confirmation()
        print("stopping transmitter")
        self.transmitter.stop()

    def receive_sinusoid_send_confirmation(self):
        print("receiving IQ data")
        self.set_receive_IQ()
        time.sleep(0.5)
        print("starting receiver at freq", self.freq["rx"])
        self.receiver.clear_UDP_socket()
        self.receiver.start()
        print("retrieving IQ data")
        IQ_data = self.receiver.retrieve_IQ(dataSize=8192,samples=8192)
        #print(IQ_data)
        print("Stopping receiver")
        self.receiver.stop()
        self.receiver.clear_UDP_socket()
        print("sending confirmation")
        self.send_confirmation_sinusoid()
        return IQ_data

    def sendSinusoidAsync(self,t):
        self.setTxSinusoid()
        self.transmitter.start()
        time.sleep(t)
        self.transmitter.stop()
    
    def receiveIQAsync(self):
        self.set_receive_IQ()
        self.receiver.clear_UDP_socket()
        self.receiver.start()
        IQ_data = self.receiver.retrieve_IQ(dataSize=8192,samples=8192)
        self.receiver.stop()
        self.receiver.clear_UDP_socket()
        return IQ_data
        
def getConf(file="../config.json"):
    with open(file) as json_data_file:
        conf = json.load(json_data_file)
    return conf

def testPHY():
    try:
        print("Testing PHY")
        L_phy = PHY()
        L_phy.sendSinusoidAsync(10)
        IQ = L_phy.receiveIQAsync()
        print(IQ)
        # Delete the PHY object
        del L_phy
        print("PHY working properly")
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)
        pass
    
if __name__ == '__main__':
    # testPHY()
    L_phy = PHY()
    L_phy.startAPI()
    print("Done")