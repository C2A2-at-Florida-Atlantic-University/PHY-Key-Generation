from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
import numpy as np

phy = None

class phyAPI(FlaskView):
    def __init__(self):
        self.app = Flask(__name__)
    
    def start(self,port,ip):
        phyAPI.register(self.app, route_base='/')
        self.app.run(host=ip, port=port)
    
    def injectNode(self, injectedNode):
        global phy
        phy = injectedNode

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
            if sample_counts is not None:
                samples = {
                    "iq": int(sample_counts.get("iq", 96)),
                    "pilots": int(sample_counts.get("pilots", 8)),
                    "csi": int(sample_counts.get("csi", 104)),
                    "chan_est_samples": int(sample_counts.get("chan_est_samples", 128)),
                }
            else:
                samples = int(data.get("samples", 1024))
            eq_data = phy.record_wifi_probe_data(samples=samples)
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