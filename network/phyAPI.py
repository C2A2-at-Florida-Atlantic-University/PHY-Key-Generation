from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
import numpy as np
import time

phy = None

class phyAPI(FlaskView):
    def __init__(self):
        self.app = Flask(__name__)
    
    def start(self,port,ip):
        phyAPI.register(self.app, route_base='/')
        # GNU Radio/UHD graph teardown/recreation is not thread-safe in this app.
        # Keep Flask single-threaded so all RX/TX graph transitions happen on one thread.
        self.app.run(host=ip, port=port, threaded=False, use_reloader=False)
    
    def injectNode(self, injectedNode):
        global phy
        phy = injectedNode

    def _normalize_wifi_probe_samples(self, sample_counts, scalar_samples):
        if sample_counts is not None:
            return {
                "iq": int(sample_counts.get("iq", 96)),
                "pilots": int(sample_counts.get("pilots", 8)),
                "csi": int(sample_counts.get("csi", 104)),
                "chan_est_samples": int(sample_counts.get("chan_est_samples", 128)),
            }
        samples = int(scalar_samples)
        return samples

    def _receiver_required_counts(self, samples, strict_counts=False):
        if strict_counts:
            if isinstance(samples, dict):
                return {
                    "symbols": int(samples["iq"]),
                    "pilots": int(samples["pilots"]),
                    "csi": int(samples["csi"]),
                    "chan_est": int(samples["chan_est_samples"]),
                }
            count = max(1, int(samples))
            return {"symbols": count, "pilots": count, "csi": count, "chan_est": count}
        return {"symbols": 1, "pilots": 1, "csi": 1, "chan_est": 1}

    def _wifi_probe_available_counts(self, eq_data):
        if eq_data is None:
            return {"symbols": 0, "pilots": 0, "csi": 0, "chan_est": 0}
        return {
            "symbols": len(eq_data.get("symbols", [])),
            "pilots": len(eq_data.get("pilots", [])),
            "csi": len(eq_data.get("csi", [])),
            "chan_est": len(eq_data.get("chan_est", [])),
        }

    def _wifi_probe_has_required_counts(self, eq_data, required_counts):
        available_counts = self._wifi_probe_available_counts(eq_data)
        for key, min_count in required_counts.items():
            if available_counts.get(key, 0) < int(min_count):
                return False
        return True

    @route('/tx/data', methods=['POST'])
    def tx_data(self):
        #{"mode":"data","data":"Hello","time":1}
        data=request.get_json()
        message=data["message"]
        phy.send_data_wait_confirmation(data=message)
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/tx/sinusoid', methods=['POST'])
    def tx_sinusoid(self):
        #{"mode":"data","data":"Hello","time":1}
        phy.transmit_sinusoid_await_confirmation()
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/tx/set/sinusoid', methods=['POST'])
    def tx_set_sinusoid(self):
        phy.setTxSinusoid()
        callback = {"contents": "setTxSinusoid" }
        return jsonify(callback), 201

    @route('/tx/set/deltaPulse', methods=['POST'])
    def tx_set_deltaPulse(self):
        data=request.get_json()
        num_bins = data.get("num_bins", 512)
        amplitude = data.get("amplitude", 1)
        center = data.get("center", True)
        repeat = data.get("repeat", True)
        window = data.get("window", True)
        num_pulses = data.get("num_pulses", -1)
        phy.setTxDeltaPulse(num_bins=num_bins, amplitude=amplitude, center=center, repeat=repeat, window=window, num_pulses=num_pulses)
        callback = {"contents": "setTxDeltaPulse" }
        return jsonify(callback), 201

    @route('/tx/set/MPSK', methods=['POST'])
    def tx_set_MPSK(self):
        data=request.get_json()
        M=data["M"]
        phy.setTxMPSK(M)
        callback = {"contents": "setTxMPSK" }
        return jsonify(callback), 201
    
    @route('/tx/set/pnSequence', methods=['POST'])
    def tx_set_pnSequence(self):
        data=request.get_json()
        sequence=data["sequence"]
        phy.setTxPnSequence(sequence)
        callback = {"contents": "setTxPnSequence" }
        return jsonify(callback), 201
    
    @route('/tx/set/fileSource', methods=['POST'])
    def tx_set_fileSource(self):
        data=request.get_json()
        fileSource=data["fileSource"]
        phy.setTxFileSource(fileSource)
        callback = {"contents": "setTxFileSource" }
        return jsonify(callback), 201

    @route('/tx/set/wifiProbe', methods=['POST'])
    def tx_set_wifiProbe(self):
        data = request.get_json() or {}
        payload = data.get("payload", "probe_request")
        interval_ms = data.get("interval_ms", 300)
        encoding = data.get("encoding", 0)
        tx_amplitude = data.get("tx_amplitude", 0.6)
        phy.setTxWiFiProbe(
            payload=payload,
            interval_ms=interval_ms,
            encoding=encoding,
            tx_amplitude=tx_amplitude,
        )
        callback = {"contents": "setTxWiFiProbe"}
        return jsonify(callback), 201

    @route('/tx/start', methods=['POST'])
    def tx_start(self):
        phy.transmitter.start()
        callback = {"contents": "transmitting" }
        return jsonify(callback), 201
    
    @route('/tx/stop', methods=['POST'])
    def tx_stop(self):
        phy.transmitter.stop()
        callback = {"contents": "done" }
        return jsonify(callback), 201

    @route('/rx/data', methods=['GET'])
    def rx_data(self):
        # Replace with your logic to retrieve items
        received,UUID,ts,size,type,contents = phy.receive_data_send_confirmation()
        callback = {"received":received,"UUID":UUID,"ts":ts,"size":size,"type":type,"contents": contents }
        return jsonify(callback), 200

    @route('/rx/raw_data', methods=['GET'])
    def raw_data(self):
        # Replace with your logic to retrieve items
        received,data = phy.receive_raw_data_send_confirmation()
        #print(data)
        callback = {"received":received,"data":data}
        return jsonify(callback), 200

    @route('/rx/sinusoid', methods=['GET'])
    def rx_sinusoid(self):
        IQ_data = phy.receive_sinusoid_send_confirmation()
        real_data = np.real(IQ_data)
        imag_data = np.imag(IQ_data)
        callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
        return jsonify(callback), 200

    @route('/rx/set/IQ', methods=['GET'])
    def rx_setIQ(self):
        try:
            contents = phy.set_receive_IQ()
            callback = {"contents": contents}
            return jsonify(callback), 200
        except Exception as error:
            callback = {"error": str(error)}
            print("Error: ", error)
            return jsonify(callback), 500

    @route('/rx/set/MPSK', methods=['POST'])
    def rx_setMPSK(self):
        data=request.get_json()
        M=data["M"]
        phy.set_receive_MPSK(M)
        callback = {"contents": "done"}
        return jsonify(callback), 200

    @route('/rx/set/wifiProbe', methods=['POST'])
    def rx_set_wifiProbe(self):
        data = request.get_json() or {}
        chan_est = data.get("chan_est", 0)
        phy.set_receive_wifi_probe(chan_est=chan_est)
        callback = {"contents": "done"}
        return jsonify(callback), 200

    @route('/rx/recordIQ', methods=['POST'])
    def rx_recordIQ(self):
        try:
            data=request.get_json()
            phy.receiver.start()
            IQ_data = phy.receiver.retrieve_IQ(samples=data["samples"])
            phy.receiver.stop()
            real_data = np.real(IQ_data)
            imag_data = np.imag(IQ_data)
            callback = {"real": real_data.tolist(), "imag": imag_data.tolist()}
            return jsonify(callback), 200
        except Exception as error:
            callback = {"error": str(error)}
            print("Error: ", error)
            return jsonify(callback), 500

    @route('/rx/recordWiFiProbe', methods=['POST'])
    def rx_record_wifi_probe(self):
        try:
            data = request.get_json() or {}
            sample_counts = data.get("sample_counts", None)
            warmup_retries = int(data.get("warmup_retries", 3))
            warmup_sleep_s = float(data.get("warmup_sleep_s", 0.03))
            warmup_timeout_s = float(data.get("warmup_timeout_s", 0.3))
            read_timeout_s = float(data.get("read_timeout_s", 1.5))
            poll_interval_s = float(data.get("poll_interval_s", 0.02))
            strict_counts = bool(data.get("strict_counts", False))
            # Keep this at 0 by default to avoid restarting RX capture sessions.
            api_retries = int(data.get("api_retries", 0))
            api_retry_sleep_s = float(data.get("api_retry_sleep_s", 0.02))
            samples = self._normalize_wifi_probe_samples(sample_counts, data.get("samples", 1024))
            required_counts = self._receiver_required_counts(samples, strict_counts=strict_counts)

            eq_data = None
            for attempt_idx in range(max(0, api_retries) + 1):
                eq_data = phy.record_wifi_probe_data(
                    samples=samples,
                    warmup_retries=warmup_retries,
                    warmup_sleep_s=warmup_sleep_s,
                    warmup_timeout_s=warmup_timeout_s,
                    read_timeout_s=read_timeout_s,
                    poll_interval_s=poll_interval_s,
                )
                if self._wifi_probe_has_required_counts(eq_data, required_counts):
                    break
                if attempt_idx < api_retries:
                    time.sleep(max(0.0, api_retry_sleep_s))

            if not self._wifi_probe_has_required_counts(eq_data, required_counts):
                callback = {
                    "error": "Incomplete WiFi probe data; not sending empty streams",
                    "required_counts": required_counts,
                    "available_counts": self._wifi_probe_available_counts(eq_data),
                }
                return jsonify(callback), 503

            callback = {
                "iq": {
                    "real": np.real(eq_data["symbols"]).tolist(),
                    "imag": np.imag(eq_data["symbols"]).tolist(),
                },
                "pilots": {
                    "real": np.real(eq_data["pilots"]).tolist(),
                    "imag": np.imag(eq_data["pilots"]).tolist(),
                },
                "csi": {
                    "real": np.real(eq_data["csi"]).tolist(),
                    "imag": np.imag(eq_data["csi"]).tolist(),
                },
                "chan_est_samples": {
                    "real": np.real(eq_data["chan_est"]).tolist(),
                    "imag": np.imag(eq_data["chan_est"]).tolist(),
                },
            }
            if "__timings__" in eq_data:
                callback["timings"] = eq_data["__timings__"]
            return jsonify(callback), 200
        except Exception as error:
            callback = {"error": str(error)}
            print("Error: ", error)
            return jsonify(callback), 500
        
    
    @route('/set/PHY', methods=['POST'])
    def setTx(self):
        try:
            data=request.get_json()
            if "freq" in data:
                phy.setFreq(data["freq"],data["x"])
            if "SamplingRate" in data:
                phy.setSamplingRate(data["SamplingRate"],data["x"])
            if "gain" in data:
                phy.setGain(data["gain"],data["x"])
            if "bandwidth" in data:
                phy.setBandwidth(data["bandwidth"],data["x"])
            if "buffer_size" in data:
                phy.setBufferSize(data["buffer_size"],data["x"])
            callback = {"contents": "done"}
            return jsonify(callback), 200
        except Exception as error:
            callback = {"error": str(error)}
            print("Error: ", error)
            return jsonify(callback), 500