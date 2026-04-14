
import struct
import socket
import json
from collections import deque
import select
import threading

from flask import Flask, jsonify
import numpy as np  
if __name__ != '__main__':
    from RX.mpsk import MPSK
    from RX.pkt_rcv_gr38 import packetReceive
    from RX.sinusoid import Sinusoid
    try:
        from RX.wifi_probe import WiFiProbeRx
    except Exception:
        WiFiProbeRx = None
    try:
        from RX.cast_probe import estimate_cast_probe_channel
    except Exception:
        estimate_cast_probe_channel = None
else:
    from mpsk import MPSK
    from pkt_rcv_gr38 import packetReceive
    from sinusoid import Sinusoid
    try:
        from wifi_probe import WiFiProbeRx
    except Exception:
        WiFiProbeRx = None
    try:
        from cast_probe import estimate_cast_probe_channel
    except Exception:
        estimate_cast_probe_channel = None
import time

WIFI_PROBE_FRAME_SAMPLES = {
    "symbols": 720,
    "pilots": 4,
    "csi": 52,
    "chan_est": 128,
}
class Receiver():
    def __init__(self,
                gain,
                samp_rate,
                freq,
                bandwidth=20000000,
                buffer_size=0x800,
                SDR_ADDR="",
                UDP_port=40868,
                UDP_IP="127.0.0.1",
                socket_timeout=0.1):
        self.gain = gain
        self.samp_rate = samp_rate
        self.freq = freq
        self.bandwidth = bandwidth
        self.buffer_size =buffer_size
        self.SDR_ADDR = SDR_ADDR
        self.UDP_port=UDP_port
        self.UDP_IP=UDP_IP
        self.socket_timeout = socket_timeout
        self.sock = self.set_UDP_socket()
        self.wifi_probe_udp_ports = {
            "symbols": self.UDP_port + 1,
            "pilots": self.UDP_port + 2,
            "csi": self.UDP_port + 3,
            "chan_est": self.UDP_port + 4,
        }
        self.wifi_probe_socks = {}
        self.last_wifi_probe_timing = {}
        self.rx = None
        self._rx_lock = threading.Lock()
        print("Receiver Initialized")
        print("Gain: "+str(self.gain))
        print("Sampling Rate: "+str(self.samp_rate))
        print("Frequency: "+str(self.freq))
        print("Bandwidth: "+str(self.bandwidth))
        print("Buffer Size: "+str(self.buffer_size))
        print("SDR ID: "+str(self.SDR_ADDR))
        print("UDP Port: "+str(self.UDP_port))
        print("UDP IP: "+str(self.UDP_IP))

    def setFreq(self,freq):
        self.freq = freq
        with self._rx_lock:
            if self.rx is not None:
                self.rx.set_freq(self.freq)
        
    def setSamplingRate(self,samp_rate):
        self.samp_rate=samp_rate
        with self._rx_lock:
            if self.rx is not None:
                self.rx.set_samp_rate(self.samp_rate)
    
    def setGain(self,gain):
        self.gain=gain
        with self._rx_lock:
            if self.rx is not None:
                self.rx.set_gain(self.gain)
    
    def setBandwidth(self,bandwidth):
        self.bandwidth=bandwidth
        with self._rx_lock:
            if self.rx is not None:
                self.rx.set_bandwidth(self.bandwidth)
    
    def set_buffer_size(self,buffer_size):
        self.buffer_size=buffer_size
        with self._rx_lock:
            if self.rx is not None:
                self.rx.set_buffer_size(self.buffer_size)
        
    def set_rx_data(self):
        sps = 2 #symbols per sample
        with self._rx_lock:
            self._close_wifi_probe_sockets()
            self._cleanup_rx()
            self.rx=packetReceive(
                samp_rate=self.samp_rate,
                sps=sps,
                gain=self.gain,
                freq=self.freq,
                buffer_size=self.buffer_size,
                bandwidth=self.bandwidth,
                SDR_ADDR=self.SDR_ADDR,
                UDP_port=self.UDP_port
            )
        
    def set_rx_MPSK(self,M):
        with self._rx_lock:
            self._close_wifi_probe_sockets()
            self._cleanup_rx()
            self.rx=MPSK(
                samp_rate=self.samp_rate,
                sps=4,
                gain=self.gain,
                freq=self.freq,
                buffer_size=self.buffer_size,
                bandwidth=self.bandwidth,
                SDR_ADDR=self.SDR_ADDR,
                UDP_port=self.UDP_port,
                M=M
            )
        
    def set_rx_IQ(self):
        with self._rx_lock:
            self._close_wifi_probe_sockets()
            self._cleanup_rx()
            self.rx=Sinusoid(
                samp_rate=self.samp_rate,
                gain=self.gain,
                freq=self.freq,
                buffer_size=self.buffer_size,
                bandwidth=self.bandwidth,
                SDR_ADDR=self.SDR_ADDR,
                UDP_port=self.UDP_port
            )

    def set_rx_cast_probe(self):
        # CaST probing uses the raw IQ capture graph; frame detection happens after capture.
        self.set_rx_IQ()

    def set_rx_wifi_probe(self, chan_est=0):
        if WiFiProbeRx is None:
            raise RuntimeError(
                "WiFi probe RX dependencies are missing. Install/import gr-ieee802-11 modules (ieee802_11)."
            )
        with self._rx_lock:
            self._init_wifi_probe_sockets()
            self._cleanup_rx()
            self.rx = WiFiProbeRx(
                samp_rate=self.samp_rate,
                gain=self.gain,
                freq=self.freq,
                buffer_size=self.buffer_size,
                bandwidth=self.bandwidth,
                SDR_ADDR=self.SDR_ADDR,
                UDP_port=self.UDP_port,
                symbols_udp_port=self.wifi_probe_udp_ports["symbols"],
                pilots_udp_port=self.wifi_probe_udp_ports["pilots"],
                csi_udp_port=self.wifi_probe_udp_ports["csi"],
                chan_est_udp_port=self.wifi_probe_udp_ports["chan_est"],
                chan_est=chan_est,
            )

    def _cleanup_rx(self):
        if self.rx is not None:
            try:
                self.rx.stop()
            except Exception:
                pass
            try:
                self.rx.wait()
            except Exception:
                pass
            try:
                del self.rx
            except Exception:
                pass
        self.rx = None
        
    #Set UDP_port=40860 for retrieving IQ
    def set_UDP_socket(self):
        return self._create_udp_socket(self.UDP_port)

    def _create_udp_socket(self, port):
        sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.UDP_IP, port))
        sock.settimeout(self.socket_timeout)
        return sock

    def _init_wifi_probe_sockets(self):
        self._close_wifi_probe_sockets()
        self.wifi_probe_socks = {
            key: self._create_udp_socket(port)
            for key, port in self.wifi_probe_udp_ports.items()
        }

    def _close_wifi_probe_sockets(self):
        for sock in self.wifi_probe_socks.values():
            try:
                sock.close()
            except Exception:
                pass
        self.wifi_probe_socks = {}
    
    def data2IQ(self,data,bps=8):
        samples = []
        for i in range(0, len(data), bps):
            real, imag = struct.unpack('ff', data[i:i+bps])
            complex_num = complex(real, imag)
            samples.append(complex_num)
        return samples
    
    def retrieve_IQ(self,dataSize=2**16,samples=2**10,max_wait_s=2.0):
        target_samples = max(1, int(samples))
        latest_samples = deque(maxlen=target_samples)
        start_time = time.time()
        has_received_data = False

        # Drain available packets and keep only the most recent samples.
        while time.time() - start_time < max_wait_s:
            try:
                data = self.retrieve_raw_data(dataSize)
                valid_len = len(data) - (len(data) % 8)
                if valid_len == 0:
                    continue
                float_view = np.frombuffer(data[:valid_len], dtype=np.float32)
                complex_view = float_view[0::2] + 1j * float_view[1::2]
                latest_samples.extend(complex_view.tolist())
                has_received_data = True
            except socket.timeout:
                # Once stream goes idle after receiving data, return latest window.
                if has_received_data:
                    break
        return list(latest_samples)
    
    def retrieve_cast_probe_data(
        self,
        samples=32768,
        dataSize=2**16,
        max_wait_s=2.0,
        sequence="cast",
        num_taps=128,
        detection_threshold=0.05,
        estimation_window_repetitions=4,
        num_repetitions=None,
        guard_len=0,
        sample_rate_hz=1e6,
        rx_sample_rate_hz=None,
        estimation_mode="matched_filter",
        min_repetitions_detected=1,
        repetition_detection_threshold=None,
    ):
        if estimate_cast_probe_channel is None:
            raise RuntimeError("CaST probe estimator is unavailable. Check RX/cast_probe.py imports.")
        wait_s = 2.0 if max_wait_s is None else float(max_wait_s)
        iq_samples = self.retrieve_IQ(dataSize=dataSize, samples=samples, max_wait_s=wait_s)
        return estimate_cast_probe_channel(
            iq_samples,
            sequence=sequence,
            num_taps=num_taps,
            detection_threshold=detection_threshold,
            estimation_window_repetitions=estimation_window_repetitions,
            num_repetitions=num_repetitions,
            guard_len=guard_len,
            sample_rate_hz=sample_rate_hz,
            rx_sample_rate_hz=self.samp_rate if rx_sample_rate_hz is None else rx_sample_rate_hz,
            estimation_mode=estimation_mode,
            min_repetitions_detected=min_repetitions_detected,
            repetition_detection_threshold=repetition_detection_threshold,
        )

    def clear_UDP_socket(self):
        print("clearing socket")
        try:
            while True:
                _, _ = self.sock.recvfrom(4096)
        except:  # noqa: E722
            pass
        print("done clearing socket")

    def clear_wifi_probe_sockets(self):
        for name, sock in self.wifi_probe_socks.items():
            print("clearing socket", name)
            try:
                while True:
                    _, _ = sock.recvfrom(4096)
            except Exception:
                pass

    def _retrieve_complex_samples(self, sock, samples=1024, dataSize=2**16, max_wait_s=2.0):
        target_samples = max(1, int(samples))
        values = deque(maxlen=target_samples)
        start_time = time.time()
        has_received_data = False
        while time.time() - start_time < max_wait_s:
            try:
                data, _ = sock.recvfrom(dataSize)
                valid_len = len(data) - (len(data) % 8)
                if valid_len == 0:
                    continue
                float_view = np.frombuffer(data[:valid_len], dtype=np.float32)
                complex_view = float_view[0::2] + 1j * float_view[1::2]
                values.extend(complex_view.tolist())
                has_received_data = True
            except socket.timeout:
                if has_received_data:
                    break
        return list(values)

    def retrieve_wifi_probe_data(self, samples=1024, dataSize=2**16, max_wait_s=None):
        if not self.wifi_probe_socks:
            raise RuntimeError(
                "WiFi probe sockets are not initialized. Call set_rx_wifi_probe() first."
            )
        if isinstance(samples, dict):
            per_stream_samples = {
                "symbols": int(samples.get("symbols", samples.get("iq", WIFI_PROBE_FRAME_SAMPLES["symbols"]))),
                "pilots": int(samples.get("pilots", WIFI_PROBE_FRAME_SAMPLES["pilots"])),
                "csi": int(samples.get("csi", WIFI_PROBE_FRAME_SAMPLES["csi"])),
                "chan_est": int(samples.get("chan_est", samples.get("chan_est_samples", WIFI_PROBE_FRAME_SAMPLES["chan_est"]))),
            }
        else:
            scalar_samples = int(samples)
            per_stream_samples = {
                "symbols": scalar_samples,
                "pilots": scalar_samples,
                "csi": scalar_samples,
                "chan_est": scalar_samples,
            }
        stream_buffers = {
            key: deque(maxlen=max(1, int(per_stream_samples[key])))
            for key in self.wifi_probe_socks.keys()
        }
        sock_to_key = {sock: key for key, sock in self.wifi_probe_socks.items()}
        stream_ready_time = {
            key: (0.0 if int(per_stream_samples[key]) <= 0 else None)
            for key in self.wifi_probe_socks.keys()
        }
        t0 = time.time()
        deadline = None if max_wait_s is None else t0 + max(0.0, float(max_wait_s))

        while True:
            select_timeout = None
            if deadline is not None:
                remaining_s = deadline - time.time()
                if remaining_s <= 0:
                    break
                select_timeout = min(self.socket_timeout, remaining_s)

            ready_socks, _, _ = select.select(
                list(self.wifi_probe_socks.values()),
                [],
                [],
                select_timeout,
            )
            if not ready_socks and deadline is not None and time.time() >= deadline:
                break

            for sock in ready_socks:
                try:
                    data, _ = sock.recvfrom(dataSize)
                except socket.timeout:
                    continue

                valid_len = len(data) - (len(data) % 8)
                if valid_len <= 0:
                    continue
                float_view = np.frombuffer(data[:valid_len], dtype=np.float32)
                complex_view = float_view[0::2] + 1j * float_view[1::2]
                key = sock_to_key[sock]
                stream_buffers[key].extend(complex_view.tolist())
                target = int(per_stream_samples[key])
                if target <= 0:
                    continue
                if (
                    stream_ready_time[key] is None and
                    len(stream_buffers[key]) >= target
                ):
                    stream_ready_time[key] = time.time() - t0

            if all(
                (int(per_stream_samples[key]) <= 0) or
                (len(stream_buffers[key]) >= int(per_stream_samples[key]))
                for key in stream_buffers.keys()
            ):
                break

        total_elapsed = time.time() - t0
        self.last_wifi_probe_timing = {
            "required_samples": {key: int(per_stream_samples[key]) for key in per_stream_samples.keys()},
            "available_samples": {key: len(stream_buffers[key]) for key in stream_buffers.keys()},
            "stream_ready_s": stream_ready_time,
            "total_elapsed_s": total_elapsed,
        }

        print(
            "[WiFiProbe RX timings] "
            + ", ".join(
                f"{key}: {stream_ready_time[key]:.3f}s" if stream_ready_time[key] is not None else f"{key}: n/a"
                for key in ["symbols", "pilots", "csi", "chan_est"]
            )
            + f", total: {total_elapsed:.3f}s"
        )

        return {key: list(values) for key, values in stream_buffers.items()}

    def retrieve_data(self,dataSize=8192):
        data, _ =self.sock.recvfrom(dataSize) 
        print(data)
        data=data.decode('ascii')
        data=json.loads(data)
        UUID=data['UUID']
        ts=data['ts']
        size=data['size']
        type=data['type']
        contents=data['data']
        return UUID,ts,size,type,contents

    def retrieve_raw_data(self,dataSize=8192):
        data, _ =self.sock.recvfrom(dataSize) 
        return data

    def start(self):
        with self._rx_lock:
            if self.rx is None:
                raise RuntimeError("Receiver type has not been set. Call set_rx_* before start().")
            self.rx.start()

    def stop(self):
        with self._rx_lock:
            if self.rx is None:
                return
            self.rx.stop()
            self.rx.wait()
        self.clear_UDP_socket()
        if self.wifi_probe_socks:
            self.clear_wifi_probe_sockets()

def testReceiver():
    samp_rate=1e6
    gain=70
    freq=3.55e9

    print("Radio Setup Parameters")
    rx = Receiver(
        gain,
        samp_rate,
        freq,
        bandwidth=20000000,
        buffer_size=8192,
        SDR_ADDR="",
        UDP_port=40868,
        UDP_IP="127.0.0.1")
    
    rxType = input("Receiver Type [Data (1), IQ (2)]:")
    if rxType == "1":
        print("Setting data receiver")
        rx.set_rx_data()
    elif rxType == "2":
        print("Setting sinusoid receiver")
        rx.set_rx_IQ()
    else:
        print("no option "+rxType)
        return
    #receiver.set_rx_data()
    print("Starting RX")
    rx.clear_UDP_socket()
    rx.start()
    if rxType == "1":
        received = False
        while not(received):
            try:
                print("attempting to retrieving data")
                #UUID,ts,size,type,contents=rx.retrieve_data(dataSize=8192)
                contents=rx.retrieve_raw_data(dataSize=8192)
                received = True
            except Exception as err:
                print("Error:",err)
                pass
            time.sleep(2)
        print(contents)
    elif rxType == "2":
        received = False
        print("Retrieving UDP data")
        while not(received):
            try:
                data=rx.retrieve_IQ(samples=1024)
                received = True
            except Exception as error:
                print("An exception occurred:", error)
                pass
        # samples = rx.data2IQ(data)
        real_data = np.real(data)
        imag_data = np.imag(data)
        callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
        print("Callback:", callback)
        app = Flask(__name__)
        with app.app_context():
            response = jsonify(callback)
        print(response)
        # print(data)
        print(len(data))

    print("Stopping RX")
    rx.stop()
    rx.clear_UDP_socket()

if __name__ == '__main__':
    testReceiver()
    print("Done")
