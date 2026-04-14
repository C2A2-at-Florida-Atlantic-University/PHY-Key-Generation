import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException

_tx_config_cache = {}
_rx_config_cache = {}
_node_active_role = {}
_last_tx_setup = {}
_last_rx_setup = {}
_recovery_in_progress = set()
REQUEST_RETRY_DELAY_S = 1
REQUEST_TIMEOUT_S = 10
PIDML_DENSE_NODE_NAMES = {
    1: "EBC",
    2: "Guest House",
    3: "Moran",
    4: "USTAR",
}
CAST_INSTANCE_ROLES = {
    1: "BA",
    2: "BE",
    3: "AB",
    4: "AE",
}
PIDML_CAST_PROFILE = {
    "name": "pidml_cast",
    "type": "castProbe",
    "node_ids": [1, 2, 3, 4],
    "examples": 10,
    "freq": 2485000000,
    "tx_sampling_rate_hz": 1000000,
        "rx_sampling_rate_hz": 2000000,
        "castProbe": {
            "sequence": "glfsr_bpsk",
            "samples": 32768,
            "num_taps": 16,
            "num_repetitions": 8,
            "estimation_window_repetitions": 8,
            "guard_len": 32,
            "sample_rate_hz": 1000000,
            "rx_sample_rate_hz": 2000000,
            "estimation_mode": "matched_filter",
            "max_wait_s": 2.0,
            "record_timeout_s": 15.0,
            "detection_threshold": 0.25,
            "repetition_detection_threshold": 0.25,
            "min_repetitions_detected": 3,
            "max_capture_retries": 5,
            "estimator_name": "CaST Matched Filter",
            "frame_shape": "sequence_plus_guard_zeros",
        "generate_plots": True,
        "plot_iq_samples": 4096,
        "plot_pause_s": 0.1,
    },
}

def APILink(IP,port,path):
    return "http://"+IP+":"+port+path    

def _wait_for_node_health(nodeID, port):
    health_url = APILink(NodeIPs[nodeID], port, "/health")
    while True:
        try:
            response = requests.get(health_url, timeout=3)
            if response.status_code == 200:
                print(f"[INFO] Node {nodeID} API is back online.")
                return
        except RequestException:
            pass
        print(f"[INFO] Waiting for node {nodeID} API health check...")
        time.sleep(REQUEST_RETRY_DELAY_S)

def _reconfigure_node_after_reconnect(nodeID, role_hint=None):
    if nodeID in _recovery_in_progress:
        return

    role = role_hint or _node_active_role.get(nodeID)
    if role not in ("tx", "rx"):
        return

    _recovery_in_progress.add(nodeID)
    try:
        # Force reapply because node process restart invalidates previous config.
        _tx_config_cache.pop(nodeID, None)
        _rx_config_cache.pop(nodeID, None)

        if role == "tx" and nodeID in _last_tx_setup:
            setup = _last_tx_setup[nodeID]
            print(f"[INFO] Reconfiguring node {nodeID} as TX after reconnect...")
            setTXNode(setup["params"], setup["type"], nodeID, setup["metadata"])
        elif role == "rx" and nodeID in _last_rx_setup:
            setup = _last_rx_setup[nodeID]
            print(f"[INFO] Reconfiguring node {nodeID} as RX after reconnect...")
            setRXNode(setup["params"], nodeID, type=setup["type"], metadata=setup["metadata"])
    finally:
        _recovery_in_progress.discard(nodeID)

def _request_with_recovery(method, nodeID, port, path, data=None, retry_on_5xx=True, timeout_s=REQUEST_TIMEOUT_S):
    headers = {'Content-Type': 'application/json'}
    url = APILink(NodeIPs[nodeID],port,path)
    payload = json.dumps(data) if data is not None else None

    while True:
        try:
            if method == "POST":
                response = requests.post(url, data=payload, headers=headers, timeout=timeout_s)
            elif method == "GET":
                response = requests.get(url, data=payload, headers=headers, timeout=timeout_s)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Retry on transient server-side failures.
            if response.status_code >= 500 and retry_on_5xx:
                print(
                    f"[WARN] Node {nodeID} returned {response.status_code} for {method} {path}. "
                    f"Waiting {REQUEST_RETRY_DELAY_S}s before retry..."
                )
                time.sleep(REQUEST_RETRY_DELAY_S)
                continue

            return response
        except RequestException as error:
            print(
                f"[WARN] Node {nodeID} unavailable for {method} {path}: {error}. "
                f"Waiting {REQUEST_RETRY_DELAY_S}s for node recovery..."
            )
            _wait_for_node_health(nodeID, port)
            if path.startswith("/tx"):
                _reconfigure_node_after_reconnect(nodeID, role_hint="tx")
            elif path.startswith("/rx"):
                _reconfigure_node_after_reconnect(nodeID, role_hint="rx")
            else:
                _reconfigure_node_after_reconnect(nodeID)

def recordIQ(nodeID,port,samples):
    path = "/rx/recordIQ"
    data = {
        "samples": samples
    }
    response_json = None
    try:
        response = _request_with_recovery("POST", nodeID, port, path, data=data, timeout_s=None)
        response_json = response.json()
        imag = response_json["imag"]
        real = response_json["real"]
        return real,imag
    except Exception as error:
        print("Error: ", error)
        print("Response:",response_json)
        return None, None

def recordWiFiProbe(nodeID,port,samples,warmup=None):
    path = "/rx/recordWiFiProbe"
    data = {}
    if isinstance(samples, dict):
        data["sample_counts"] = samples
    else:
        data["samples"] = samples
    if warmup:
        data["api_retries"] = int(warmup.get("api_retries", 0))
        data["strict_counts"] = bool(warmup.get("strict_counts", False))
        if "max_wait_s" in warmup:
            data["max_wait_s"] = float(warmup["max_wait_s"])
        record_timeout_s = warmup.get("record_timeout_s", REQUEST_TIMEOUT_S)
    else:
        record_timeout_s = REQUEST_TIMEOUT_S
    response_json = None
    try:
        response = _request_with_recovery("POST", nodeID, port, path, data=data, timeout_s=record_timeout_s)
        response_json = response.json()
        required_sections = ["iq", "pilots", "csi", "chan_est_samples"]
        for section in required_sections:
            if section not in response_json:
                raise KeyError(f"Missing '{section}' in WiFi probe response")
            if "real" not in response_json[section] or "imag" not in response_json[section]:
                raise KeyError(f"Missing 'real'/'imag' in WiFi probe section '{section}'")
        return response_json
    except Exception as error:
        print("Error: ", error)
        print("Response:",response_json)
        return None

def recordCastProbe(nodeID,port,samples,warmup=None):
    path = "/rx/recordCastProbe"
    warmup = warmup or {}
    data = {
        "samples": int(warmup.get("samples", samples)),
        "sequence": warmup.get("sequence", "cast"),
        "num_taps": int(warmup.get("num_taps", 128)),
        "detection_threshold": float(warmup.get("detection_threshold", 0.05)),
        "estimation_window_repetitions": int(warmup.get("estimation_window_repetitions", warmup.get("num_repetitions", 4))),
        "num_repetitions": int(warmup.get("num_repetitions", warmup.get("estimation_window_repetitions", 4))),
        "guard_len": int(warmup.get("guard_len", 0)),
        "sample_rate_hz": float(warmup.get("sample_rate_hz", warmup.get("tx_sampling_rate_hz", 1e6))),
        "rx_sample_rate_hz": float(warmup.get("rx_sample_rate_hz", warmup.get("sample_rate_hz", 1e6))),
        "estimation_mode": warmup.get("estimation_mode", "matched_filter"),
        "min_repetitions_detected": int(warmup.get("min_repetitions_detected", 1)),
    }
    if "repetition_detection_threshold" in warmup:
        data["repetition_detection_threshold"] = float(warmup["repetition_detection_threshold"])
    if "max_wait_s" in warmup:
        data["max_wait_s"] = float(warmup["max_wait_s"])
    record_timeout_s = warmup.get("record_timeout_s", REQUEST_TIMEOUT_S)
    response_json = None
    try:
        response = _request_with_recovery("POST", nodeID, port, path, data=data, timeout_s=record_timeout_s)
        response_json = response.json()
        for section in ["iq", "probe", "cir", "taps"]:
            if section not in response_json:
                raise KeyError(f"Missing '{section}' in CaST probe response")
            if "real" not in response_json[section] or "imag" not in response_json[section]:
                raise KeyError(f"Missing 'real'/'imag' in CaST probe section '{section}'")
        return response_json
    except Exception as error:
        print("Error: ", error)
        print("Response:",response_json)
        return None

def setRxIQ(nodeID,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    response = _request_with_recovery("GET", nodeID, port, path, data=data)
    return response

def setRxWiFiProbe(nodeID,port,chan_est=0):
    path = "/rx/set/wifiProbe"
    data = {
        "chan_est": chan_est
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def setRxCastProbe(nodeID,port):
    path = "/rx/set/castProbe"
    data = {
        "contents": "castProbe"
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def setPHY(nodeID,port,params):
    path = "/set/PHY"
    data = {
        "x": params["x"],
        "freq": params["freq"],
        "SamplingRate": params["SamplingRate"],
        "gain": params["gain"][nodeID][params["x"]]
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response
    
def set_tx_sinusoid(nodeID,port):
    path = "/tx/set/sinusoid"
    data = {
        "message": "sinusoid"
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def set_tx_deltaPulse(nodeID,port,metadata):
    path = "/tx/set/deltaPulse"
    data = {
        "num_bins": metadata["num_bins"],
        "amplitude": metadata["amplitude"],
        "center": metadata["center"],
        "repeat": metadata["repeat"],
        "window": metadata["window"],
        "num_pulses": metadata["num_pulses"]
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def set_tx_MPSK(nodeID,port,M):
    path = "/tx/set/MPSK"
    data = {
        "M": M
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def set_tx_pnSequence(nodeID,port,sequence):
    path = "/tx/set/pnSequence"
    data = {
        "sequence": sequence
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def _get_sequence_value(metadata, key, default):
    metadata = metadata or {}
    value = metadata.get(key, default)
    if isinstance(value, dict):
        return value.get("sequence", default)
    return value if value is not None else default

def set_tx_castProbe(nodeID,port,metadata=None):
    path = "/tx/set/castProbe"
    metadata = metadata or {}
    data = {
        "sequence": _get_sequence_value({"castProbe": metadata}, "castProbe", "cast"),
        "guard_len": int(metadata.get("guard_len", 0)),
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def set_tx_wifiProbe(nodeID,port,metadata):
    path = "/tx/set/wifiProbe"
    data = {
        "payload": metadata.get("payload", "probe_request"),
        "interval_ms": metadata.get("interval_ms", 300),
        "encoding": metadata.get("encoding", 0),
        "tx_amplitude": metadata.get("tx_amplitude", 0.6)
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def start_tx(nodeID,port,retry_on_5xx=True):
    path = "/tx/start"
    data = {
        "message": "TX Start"
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data, retry_on_5xx=retry_on_5xx)
    return response
    
def stop_tx(nodeID,port):
    path = "/tx/stop"
    data = {
        "message": "sinusoid"
    }
    response = _request_with_recovery("POST", nodeID, port, path, data=data)
    return response

def plotTimeDomain(i_samples,q_samples,samples=-1,id=0):
    plt.plot(i_samples[0:samples], color='red')
    plt.plot(q_samples[0:samples], color='blue')
    plt.xlabel('Time')
    plt.ylabel('IQ')
    plt.title('Time Domain Plot Node: '+str(id))
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    # plt.show()
    # Show for 0.5 seconds
    plt.pause(0.1)
    plt.clf()  # Clear the figure for the next plot

def plotTimeDomainSideBySide(I1, Q1, I2, Q2, samples=-1, id1=0, id2=0, ax1=None, ax2=None, fig=None):
    
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        # Check if axes exist, if not recreate them
        if len(fig.axes) == 0:
            ax1, ax2 = fig.subplots(1, 2)
        else:
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
    
    # Clear the axes before plotting new data
    ax1.clear()
    ax2.clear()
        
    # Plot first subplot
    ax1.plot(I1[0:samples], color='red')
    ax1.plot(Q1[0:samples], color='blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('IQ')
    ax1.set_title('Time Domain Plot Node: '+str(id1))
    ax1.grid(True)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.axvline(0, color='black', linewidth=0.5)
    
    # Plot second subplot
    ax2.plot(I2[0:samples], color='red')
    ax2.plot(Q2[0:samples], color='blue')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('IQ')
    ax2.set_title('Time Domain Plot Node: '+str(id2))
    ax2.grid(True)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.pause(0.1)

def setTXNode(params,type,nodeID,metadata = {"pnSequence":"glfsr"}):
    metadata = metadata or {}
    print("type:",type)
    _node_active_role[nodeID] = "tx"
    _last_tx_setup[nodeID] = {"params": params, "type": type, "metadata": metadata}
    tx_gain = params["tx"]["gain"][nodeID]["tx"]
    tx_signature = (
        type,
        params["tx"]["freq"],
        params["tx"]["SamplingRate"],
        tx_gain,
        json.dumps(metadata.get(type, {}), sort_keys=True),
    )

    # Stop any existing transmitter first to avoid interference
    try:
        stop_tx(nodeID,port["radio"])
        time.sleep(0.02)  # Keep transition delay short to reduce collection latency
    except Exception as e:
        print(f"Warning: Could not stop existing transmitter: {e}")
    
    # Node switched to TX mode, so RX cache for this node is now stale.
    _rx_config_cache.pop(nodeID, None)

    def _configure_tx():
        if type == "sinusoid":
            set_tx_sinusoid(nodeID,port["radio"])
        elif type == "pnSequence":
            set_tx_pnSequence(nodeID,port["radio"],_get_sequence_value(metadata, type, "glfsr"))
        elif type == "castProbe":
            set_tx_castProbe(nodeID,port["radio"],metadata.get(type, {}))
        elif type == "deltaPulse":
            set_tx_deltaPulse(nodeID,port["radio"],metadata[type])
        elif type == "wifiProbe":
            set_tx_wifiProbe(nodeID,port["radio"],metadata.get(type, {}))
        else:
            raise ValueError(f"Unsupported TX type: {type}")
        setPHY(nodeID,port["radio"],params["tx"])
        _tx_config_cache[nodeID] = tx_signature

    if type == "castProbe" or _tx_config_cache.get(nodeID) != tx_signature:
        _configure_tx()

    while True:
        response = start_tx(nodeID,port["radio"],retry_on_5xx=False)
        if response.status_code < 500:
            break
        print(
            f"[WARN] Node {nodeID} returned {response.status_code} for /tx/start. "
            "Reconfiguring transmitter before retry..."
        )
        _tx_config_cache.pop(nodeID, None)
        _configure_tx()
        time.sleep(REQUEST_RETRY_DELAY_S)
    
def setRXNode(params,nodeID,type="IQ",metadata=None):
    metadata = metadata or {}
    _node_active_role[nodeID] = "rx"
    _last_rx_setup[nodeID] = {"params": params, "type": type, "metadata": metadata}
    rx_gain = params["rx"]["gain"][nodeID]["rx"]
    rx_signature = (
        type,
        params["rx"]["freq"],
        params["rx"]["SamplingRate"],
        rx_gain,
        json.dumps(metadata.get(type, {}), sort_keys=True),
    )
    if type != "castProbe" and _rx_config_cache.get(nodeID) == rx_signature:
        return
    if type == "wifiProbe":
        chan_est = metadata.get(type, {}).get("chan_est", 0)
        response = setRxWiFiProbe(nodeID,port["radio"],chan_est=chan_est)
        print("Response SetRXNode WiFiProbe: ", response)
    elif type == "castProbe":
        response = setRxCastProbe(nodeID,port["radio"])
        print("Response SetRXNode CastProbe: ", response)
    else:
        response = setRxIQ(nodeID,port["radio"])
        print("Response SetRXNode IQ: ", response)
    response = setPHY(nodeID,port["radio"],params["rx"])
    print("Response SetRXNode PHY: ", response)
    _rx_config_cache[nodeID] = rx_signature
    
def RecordNodeData(nodeID,samples,type="IQ",metadata=None):
    if type == "wifiProbe":
        metadata = metadata or {}
        warmup_cfg = metadata.get("wifiProbe", {})
        return recordWiFiProbe(nodeID,port["radio"],samples,warmup=warmup_cfg)
    if type == "castProbe":
        metadata = metadata or {}
        cast_cfg = metadata.get("castProbe", {})
        return recordCastProbe(nodeID,port["radio"],samples,warmup=cast_cfg)
    return recordIQ(nodeID,port["radio"],samples)

def setRXNodesParallel(params, nodeIDs, type="IQ", metadata=None):
    with ThreadPoolExecutor(max_workers=len(nodeIDs)) as executor:
        futures = [
            executor.submit(setRXNode, params, nodeID, type, metadata)
            for nodeID in nodeIDs
        ]
        for future in futures:
            future.result()

def _recordNodeDataWithTimestamp(nodeID, samples, type="IQ", metadata=None):
    try:
        data = RecordNodeData(nodeID, samples=samples, type=type, metadata=metadata)
    except Exception as error:
        print(f"[WARN] recordNodeData failed for node {nodeID}: {error}")
        data = None
    return nodeID, data, int(time.time())

def recordNodesParallel(nodeIDs, samples, type="IQ", metadata=None):
    results = {}
    with ThreadPoolExecutor(max_workers=len(nodeIDs)) as executor:
        futures = [
            executor.submit(_recordNodeDataWithTimestamp, nodeID, samples, type, metadata)
            for nodeID in nodeIDs
        ]
        for future in futures:
            nodeID, data, ts = future.result()
            results[nodeID] = {"data": data, "timestamp": ts}
    return results
    
def stopTXNode(nodeID):
    stop_tx(nodeID,port["radio"])
    _tx_config_cache.pop(nodeID, None)

def setupNodesPingPong(RX1,RX2,TX,params,type,metadata = {"pnSequence":"glfsr"}):
    setRXNode(params,TX,type=type,metadata=metadata)    
    setRXNode(params,RX1,type=type,metadata=metadata)
    setRXNode(params,RX2,type=type,metadata=metadata)
    setTXNode(params,type,TX,metadata)
    
def collect_data_ping_pong_3Nodes(params, nodes, packages, type, channel_Labels = [1,2,3],metadata = {"pnSequence":"glfsr"}):
    i = 1
    i_samples = []
    q_samples = []
    channel = []
    instance = []
    ids = []
    tx = []
    rx = []
    timestamp = []
    id = 0
    timeSleep = 0.1
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    numberOfSamples = 32768
    print("Setting RX Node: ", Eve)
    
    generatePlots = True
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    while i<packages*2+1:
        print(i)
        if i%2 == 0:
            print("Setting TX Node: ", Bob)
            setTXNode(params,type,Bob,metadata)
            print("Setting RX Nodes: ", Alice, Eve)
            setRXNodesParallel(params, [Alice, Eve], type=type, metadata=metadata)
            time.sleep(timeSleep)
            
            valid_capture = False
            real1, imaginary1 = None, None
            real2, imaginary2 = None, None
            timestamp1, timestamp2 = None, None
            retry_idx = 0
            while True:
                print("Recording RX Nodes: ", Alice, Eve)
                rx_results = recordNodesParallel([Alice, Eve], samples=numberOfSamples, type=type, metadata=metadata)
                real1, imaginary1 = _extract_iq_record(rx_results[Alice]["data"])
                timestamp1 = rx_results[Alice]["timestamp"]
                real2, imaginary2 = _extract_iq_record(rx_results[Eve]["data"])
                timestamp2 = rx_results[Eve]["timestamp"]

                valid_capture = _iq_samples_valid(real1, imaginary1, min_samples=1) and _iq_samples_valid(real2, imaginary2, min_samples=1)
                if valid_capture:
                    break
                retry_idx += 1
                print(f"Retrying read (no RX rearm): empty/invalid IQ samples detected (attempt {retry_idx})")
                time.sleep(timeSleep)
            
            stopTXNode(Bob)
            
            if valid_capture:
                i_samples.append(real1[-numberOfSamples:])
                q_samples.append(imaginary1[-numberOfSamples:])
                channel.append(channel_Labels[0])
                instance.append(1)
                ids.append(id)
                tx.append(Bob)
                rx.append(Alice)
                timestamp.append(timestamp1)
                
                i_samples.append(real2[-numberOfSamples:])
                q_samples.append(imaginary2[-numberOfSamples:])
                channel.append(channel_Labels[1])  
                instance.append(2)   
                ids.append(id)
                tx.append(Bob)
                rx.append(Eve)
                timestamp.append(timestamp2)
                
                if(generatePlots):
                    plotTimeDomainSideBySide(
                        real1[-numberOfSamples:], 
                        imaginary1[-numberOfSamples:], 
                        real2[-numberOfSamples:], 
                        imaginary2[-numberOfSamples:], 
                        samples=numberOfSamples, id1=Alice, id2=Eve, ax1=ax1, ax2=ax2, fig=fig)
                    time.sleep(timeSleep)
        else:
            id = id + 1
            print(("Setting TX Node: ", Alice))
            setTXNode(params,type,Alice,metadata)
            print("Setting RX Nodes: ", Bob, Eve)
            setRXNodesParallel(params, [Bob, Eve], type=type, metadata=metadata)
            time.sleep(timeSleep)
            
            valid_capture = False
            real1, imaginary1 = None, None
            real2, imaginary2 = None, None
            timestamp1, timestamp2 = None, None
            retry_idx = 0
            while True:
                print("Recording RX Nodes: ", Bob, Eve)
                rx_results = recordNodesParallel([Bob, Eve], samples=numberOfSamples, type=type, metadata=metadata)
                real1, imaginary1 = _extract_iq_record(rx_results[Bob]["data"])
                timestamp1 = rx_results[Bob]["timestamp"]
                real2, imaginary2 = _extract_iq_record(rx_results[Eve]["data"])
                timestamp2 = rx_results[Eve]["timestamp"]

                valid_capture = _iq_samples_valid(real1, imaginary1, min_samples=1) and _iq_samples_valid(real2, imaginary2, min_samples=1)
                if valid_capture:
                    break
                retry_idx += 1
                print(f"Retrying read (no RX rearm): empty/invalid IQ samples detected (attempt {retry_idx})")
                time.sleep(timeSleep)

            stopTXNode(Alice)

            if valid_capture:
                i_samples.append(real1[-numberOfSamples:])
                q_samples.append(imaginary1[-numberOfSamples:])
                channel.append(channel_Labels[0])
                instance.append(3)
                ids.append(id)
                tx.append(Alice)
                rx.append(Bob)
                timestamp.append(timestamp1)
                
                i_samples.append(real2[-numberOfSamples:])
                q_samples.append(imaginary2[-numberOfSamples:])
                channel.append(channel_Labels[2]) # Changed from "labels" to "channel"
                instance.append(4)
                ids.append(id)
                tx.append(Alice)
                rx.append(Eve)
                timestamp.append(timestamp2)
                if(generatePlots):    
                    
                    plotTimeDomainSideBySide(
                        real1[-numberOfSamples:], 
                        imaginary1[-numberOfSamples:], 
                        real2[-numberOfSamples:], 
                        imaginary2[-numberOfSamples:], 
                        samples=numberOfSamples, id1=Bob, id2=Eve, ax1=ax1, ax2=ax2, fig=fig)
                time.sleep(timeSleep)
        
        i = i + 1
    return i_samples, q_samples, channel, instance, ids, tx, rx, timestamp

def _slice_samples(values, samples):
    if samples <= 0:
        return []
    if samples == -1:
        return values
    return values[-samples:]

def _node_name(node_id, metadata=None):
    cast_metadata = (metadata or {}).get("castProbe", {})
    node_names = cast_metadata.get("node_names", PIDML_DENSE_NODE_NAMES)
    return node_names.get(node_id, node_names.get(str(node_id), str(node_id)))

def _iq_samples_valid(real_values, imag_values, min_samples=1):
    if real_values is None or imag_values is None:
        return False
    if len(real_values) < min_samples or len(imag_values) < min_samples:
        return False
    if len(real_values) != len(imag_values):
        return False
    return True

def _extract_iq_record(record):
    if isinstance(record, dict) and "iq" in record:
        return record["iq"].get("real"), record["iq"].get("imag")
    return record

def _wifi_probe_is_valid(probe_data, sample_counts=None):
    if probe_data is None:
        return False
    required_sections = ["iq", "pilots", "csi", "chan_est_samples"]
    min_counts = sample_counts or {}
    for section in required_sections:
        if section not in probe_data:
            return False
        if "real" not in probe_data[section] or "imag" not in probe_data[section]:
            return False
        min_section_samples = int(min_counts.get(section, 1))
        if not _iq_samples_valid(
            probe_data[section]["real"],
            probe_data[section]["imag"],
            min_samples=min_section_samples
        ):
            return False
    return True

def _cast_probe_is_valid(probe_data, min_iq_samples=1, min_taps=1):
    if probe_data is None or not bool(probe_data.get("detected", False)):
        return False
    for section, min_samples in [("iq", min_iq_samples), ("taps", min_taps), ("cir", min_taps)]:
        if section not in probe_data:
            return False
        if "real" not in probe_data[section] or "imag" not in probe_data[section]:
            return False
        if not _iq_samples_valid(probe_data[section]["real"], probe_data[section]["imag"], min_samples=min_samples):
            return False
    return True

def _cast_probe_debug_summary(node_id, probe_data):
    if probe_data is None:
        return f"node {node_id}: no response"
    metadata = probe_data.get("metadata", {})
    iq = _section_complex(probe_data, "iq")
    iq_power = float(np.mean(np.abs(iq) ** 2)) if iq.size else 0.0
    return (
        f"node {node_id}: detected={bool(probe_data.get('detected', False))}, "
        f"peak={float(metadata.get('normalized_peak', 0.0)):.3f}, "
        f"reps={int(metadata.get('num_repetitions_used', 0))}/"
        f"{int(metadata.get('min_repetitions_detected', 1))}, "
        f"decim={int(metadata.get('estimation_decimation', 1))}, "
        f"phase={int(metadata.get('decimation_phase', 0))}, "
        f"power={iq_power:.4g}"
    )

def _append_cast_probe_sample(feature_store, probe_data, numberOfSamples, num_taps):
    metadata = probe_data.get("metadata", {})
    iq_real, iq_imag = _extract_iq_record(probe_data)
    tap_delays_s = probe_data.get("tap_delays_s", metadata.get("tap_delays_s", []))
    if tap_delays_s is None or len(tap_delays_s) == 0:
        sample_rate_hz = float(metadata.get("sample_rate_hz", 1e6))
        tap_delays_s = (np.arange(num_taps, dtype=np.float32) / sample_rate_hz).tolist()
    feature_store["iq_I"].append(_slice_samples(iq_real, numberOfSamples))
    feature_store["iq_Q"].append(_slice_samples(iq_imag, numberOfSamples))
    feature_store["taps_I"].append(_slice_samples(probe_data["taps"]["real"], num_taps))
    feature_store["taps_Q"].append(_slice_samples(probe_data["taps"]["imag"], num_taps))
    feature_store["cir_I"].append(_slice_samples(probe_data["cir"]["real"], num_taps))
    feature_store["cir_Q"].append(_slice_samples(probe_data["cir"]["imag"], num_taps))
    feature_store["tap_delays_s"].append(_slice_samples(tap_delays_s, num_taps))
    feature_store["pdp_db"].append(_slice_samples(probe_data.get("pdp_db", []), num_taps))
    feature_store["detected"].append(bool(probe_data.get("detected", False)))
    feature_store["normalized_peak"].append(float(metadata.get("normalized_peak", 0.0)))
    feature_store["pathloss_db"].append(float(metadata.get("pathloss_db", 0.0)))
    feature_store["frame_phase"].append(int(metadata.get("frame_phase", -1)))
    feature_store["peak_index"].append(int(metadata.get("peak_index", -1)))

def _get_wifi_probe_sample_counts(metadata):
    # Default to one complete WiFi-probe frame so CSI, pilots, and symbols stay aligned.
    sample_counts = {
        "iq": 720,
        "pilots": 4,
        "csi": 52,
        "chan_est_samples": 128,
    }
    wifi_metadata = metadata.get("wifiProbe", {})
    if "sample_counts" in wifi_metadata:
        for section in sample_counts.keys():
            if section in wifi_metadata["sample_counts"]:
                sample_counts[section] = int(wifi_metadata["sample_counts"][section])
    return sample_counts

def _append_wifi_probe_sample(feature_store, probe_data, sample_counts):
    for section in ["iq", "pilots", "csi", "chan_est_samples"]:
        section_samples = sample_counts[section]
        feature_store[section]["I"].append(_slice_samples(probe_data[section]["real"], section_samples))
        feature_store[section]["Q"].append(_slice_samples(probe_data[section]["imag"], section_samples))

def _section_complex(probe_data, section):
    if probe_data is None or section not in probe_data:
        return np.zeros(0, dtype=np.complex64)
    payload = probe_data[section]
    real_values = np.asarray(payload.get("real", []), dtype=np.float32)
    imag_values = np.asarray(payload.get("imag", []), dtype=np.float32)
    if real_values.size == 0 and imag_values.size == 0:
        return np.zeros(0, dtype=np.complex64)
    keep = min(real_values.size, imag_values.size)
    return (real_values[:keep] + 1j * imag_values[:keep]).astype(np.complex64)

def _plot_cast_probe_pair(
        probe_data_1,
        probe_data_2,
        tx_node,
        rx_node_1,
        rx_node_2,
        role_1,
        role_2,
        num_taps,
        iq_samples,
        metadata=None,
        fig=None,
        pause_s=0.1
    ):
    if fig is None:
        fig = plt.figure(figsize=(14, 11))

    fig.clf()
    axes = fig.subplots(4, 2)
    cast_metadata = (metadata or {}).get("castProbe", {})
    tx_label = _node_name(tx_node, metadata)

    for col_idx, (probe_data, rx_node, role) in enumerate([
        (probe_data_1, rx_node_1, role_1),
        (probe_data_2, rx_node_2, role_2),
    ]):
        rx_label = _node_name(rx_node, metadata)
        capture_metadata = probe_data.get("metadata", {}) if probe_data else {}

        probe = _section_complex(probe_data, "probe")
        guard_len = int(capture_metadata.get("guard_len", cast_metadata.get("guard_len", 0)))
        if probe.size and guard_len > 0:
            probe = np.concatenate([probe, np.zeros(guard_len, dtype=np.complex64)])
        probe_ax = axes[0, col_idx]
        probe_ax.plot(probe.real, color="red", linewidth=0.8, label="I")
        probe_ax.plot(probe.imag, color="blue", linewidth=0.8, label="Q")
        if guard_len > 0 and probe.size >= guard_len:
            probe_ax.axvspan(probe.size - guard_len, probe.size, color="gray", alpha=0.15, label="guard")
        probe_ax.set_title(f"{role}: CaST probe frame")
        probe_ax.set_xlabel("Sample")
        probe_ax.set_ylabel("Amplitude")
        probe_ax.grid(True)
        probe_ax.legend(loc="upper right")

        iq = _section_complex(probe_data, "iq")
        peak_index = int(capture_metadata.get("peak_index", -1))
        frame_starts = capture_metadata.get("detected_frame_starts", [])
        if isinstance(frame_starts, list) and frame_starts:
            anchor_index = int(frame_starts[0])
        else:
            anchor_index = peak_index if peak_index >= 0 else 0
        raw_frame_len = int(capture_metadata.get(
            "raw_frame_len",
            capture_metadata.get("frame_len", probe.size if probe.size else 1)
        ))
        window_len = int(iq_samples) if int(iq_samples) > 0 else int(iq.size)
        plot_start = max(0, anchor_index - raw_frame_len)
        plot_stop = min(iq.size, plot_start + window_len)
        if plot_stop - plot_start < window_len:
            plot_start = max(0, plot_stop - window_len)
        iq_window = iq[plot_start:plot_stop] if iq.size else iq
        iq_offset = plot_start

        iq_ax = axes[1, col_idx]
        iq_ax.plot(iq_window.real, color="red", linewidth=0.8, label="I")
        iq_ax.plot(iq_window.imag, color="blue", linewidth=0.8, label="Q")
        if iq_offset <= peak_index < iq_offset + iq_window.size:
            iq_ax.axvline(peak_index - iq_offset, color="black", linestyle=":", linewidth=1.0, label="peak")
        if isinstance(frame_starts, list):
            for frame_start in frame_starts:
                frame_start = int(frame_start)
                if iq_offset <= frame_start < iq_offset + iq_window.size:
                    iq_ax.axvline(frame_start - iq_offset, color="green", linestyle="--", linewidth=0.8)
        iq_ax.set_title(f"{role}: {tx_label} -> {rx_label} IQ")
        iq_ax.set_xlabel("Sample")
        iq_ax.set_ylabel("Amplitude")
        iq_ax.grid(True)
        iq_ax.legend(loc="upper right")

        taps = _section_complex(probe_data, "taps")[:int(num_taps)]
        tap_axis = np.arange(taps.size, dtype=np.int32)
        tap_ax = axes[2, col_idx]
        tap_ax.plot(tap_axis, taps.real, color="red", marker="o", linewidth=1.0, label="Real")
        tap_ax.plot(tap_axis, taps.imag, color="blue", marker="o", linewidth=1.0, label="Imag")
        tap_ax.set_title(
            f"{role}: taps, peak={float(capture_metadata.get('normalized_peak', 0.0)):.3f}"
        )
        tap_ax.set_xlabel("Tap")
        tap_ax.set_ylabel("Coefficient")
        tap_ax.grid(True)
        tap_ax.legend(loc="upper right")

        pdp = np.asarray(probe_data.get("pdp_db", []), dtype=np.float32)[:int(num_taps)] if probe_data else np.zeros(0, dtype=np.float32)
        pdp_ax = axes[3, col_idx]
        pdp_ax.plot(np.arange(pdp.size, dtype=np.int32), pdp, color="green", marker="o", linewidth=1.0)
        pdp_ax.set_title(
            f"{role}: PDP, reps={capture_metadata.get('num_repetitions_used', cast_metadata.get('num_repetitions', 'n/a'))}"
        )
        pdp_ax.set_xlabel("Tap")
        pdp_ax.set_ylabel("dB")
        pdp_ax.grid(True)

    fig.suptitle(f"CaST probe capture from {_node_name(tx_node, metadata)}", fontsize=12)
    plt.tight_layout()
    plt.pause(float(pause_s))

def _plot_wifi_probe_sections_side_by_side(probe_data_1, probe_data_2, sample_counts, id1, id2, ax1=None, ax2=None, fig=None):
    sections = ["iq", "csi", "chan_est_samples"]
    if fig is None:
        fig = plt.figure(figsize=(12, 10))

    # Reuse the same figure and redraw a 3x2 grid every time.
    fig.clf()
    axes = fig.subplots(len(sections), 2)
    if len(sections) == 1:
        axes = np.array([axes])

    for row_idx, section in enumerate(sections):
        if int(sample_counts.get(section, 0)) <= 0:
            continue

        node1_i = _slice_samples(probe_data_1[section]["real"], sample_counts[section])
        node1_q = _slice_samples(probe_data_1[section]["imag"], sample_counts[section])
        node2_i = _slice_samples(probe_data_2[section]["real"], sample_counts[section])
        node2_q = _slice_samples(probe_data_2[section]["imag"], sample_counts[section])

        left_ax = axes[row_idx, 0]
        right_ax = axes[row_idx, 1]

        left_ax.plot(node1_i, color='red')
        left_ax.plot(node1_q, color='blue')
        left_ax.set_xlabel('Time')
        left_ax.set_ylabel('IQ')
        left_ax.set_title(f'Node {id1} - {section}')
        left_ax.grid(True)
        left_ax.axhline(0, color='black', linewidth=0.5)
        left_ax.axvline(0, color='black', linewidth=0.5)

        right_ax.plot(node2_i, color='red')
        right_ax.plot(node2_q, color='blue')
        right_ax.set_xlabel('Time')
        right_ax.set_ylabel('IQ')
        right_ax.set_title(f'Node {id2} - {section}')
        right_ax.grid(True)
        right_ax.axhline(0, color='black', linewidth=0.5)
        right_ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.pause(0.1)

def collect_data_ping_pong_3Nodes_wifi_probe(params, nodes, packages, metadata, channel_Labels=[1,2,3]):
    i = 1
    channel = []
    instance = []
    ids = []
    tx = []
    rx = []
    timestamp = []
    id = 0
    timeSleep = 0.05
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    sample_counts = _get_wifi_probe_sample_counts(metadata)
    wifi_metadata = metadata.get("wifiProbe", {})
    max_capture_retries = int(wifi_metadata.get("max_capture_retries", 5))

    feature_store = {
        "iq": {"I": [], "Q": []},
        "pilots": {"I": [], "Q": []},
        "csi": {"I": [], "Q": []},
        "chan_est_samples": {"I": [], "Q": []},
    }

    generatePlots = bool(wifi_metadata.get("generate_plots", False))
    fig = None
    ax1 = None
    ax2 = None
    if generatePlots:
        fig = plt.figure(figsize=(12, 10))

    while i < packages * 2 + 1:
        print(i)
        if i % 2 == 0:
            print("Setting TX Node: ", Bob)
            setTXNode(params, "wifiProbe", Bob, metadata)
            print("Setting RX Nodes: ", Alice, Eve)
            setRXNodesParallel(params, [Alice, Eve], type="wifiProbe", metadata=metadata)
            time.sleep(timeSleep)

            valid_capture = False
            probe_data_1 = None
            probe_data_2 = None
            timestamp1, timestamp2 = None, None
            retry_idx = 0
            while retry_idx <= max_capture_retries:
                print("Recording RX Nodes: ", Alice, Eve)
                rx_results = recordNodesParallel([Alice, Eve], samples=sample_counts, type="wifiProbe", metadata=metadata)
                probe_data_1 = rx_results[Alice]["data"]
                timestamp1 = rx_results[Alice]["timestamp"]
                probe_data_2 = rx_results[Eve]["data"]
                timestamp2 = rx_results[Eve]["timestamp"]

                valid_capture = _wifi_probe_is_valid(probe_data_1, sample_counts=sample_counts) and _wifi_probe_is_valid(probe_data_2, sample_counts=sample_counts)
                if valid_capture:
                    break
                retry_idx += 1
                if retry_idx <= max_capture_retries:
                    print(f"Retrying read (no RX rearm): empty/invalid WiFi probe samples detected (attempt {retry_idx})")
                    time.sleep(timeSleep)

            if not valid_capture:
                print(f"[WARN] Skipping WiFi probe capture for TX node {Bob} after {max_capture_retries + 1} attempts")
            stopTXNode(Bob)

            if valid_capture:
                _append_wifi_probe_sample(feature_store, probe_data_1, sample_counts)
                channel.append(channel_Labels[0])
                instance.append(1)
                ids.append(id)
                tx.append(Bob)
                rx.append(Alice)
                timestamp.append(timestamp1)

                _append_wifi_probe_sample(feature_store, probe_data_2, sample_counts)
                channel.append(channel_Labels[1])
                instance.append(2)
                ids.append(id)
                tx.append(Bob)
                rx.append(Eve)
                timestamp.append(timestamp2)

                if generatePlots:
                    _plot_wifi_probe_sections_side_by_side(
                        probe_data_1,
                        probe_data_2,
                        sample_counts,
                        id1=Alice,
                        id2=Eve,
                        ax1=ax1,
                        ax2=ax2,
                        fig=fig
                    )
                time.sleep(timeSleep)
        else:
            id = id + 1
            print(("Setting TX Node: ", Alice))
            setTXNode(params, "wifiProbe", Alice, metadata)
            print("Setting RX Nodes: ", Bob, Eve)
            setRXNodesParallel(params, [Bob, Eve], type="wifiProbe", metadata=metadata)
            time.sleep(timeSleep)

            valid_capture = False
            probe_data_1 = None
            probe_data_2 = None
            timestamp1, timestamp2 = None, None
            retry_idx = 0
            while retry_idx <= max_capture_retries:
                print("Recording RX Nodes: ", Bob, Eve)
                rx_results = recordNodesParallel([Bob, Eve], samples=sample_counts, type="wifiProbe", metadata=metadata)
                probe_data_1 = rx_results[Bob]["data"]
                timestamp1 = rx_results[Bob]["timestamp"]
                probe_data_2 = rx_results[Eve]["data"]
                timestamp2 = rx_results[Eve]["timestamp"]

                valid_capture = _wifi_probe_is_valid(probe_data_1, sample_counts=sample_counts) and _wifi_probe_is_valid(probe_data_2, sample_counts=sample_counts)
                if valid_capture:
                    break
                retry_idx += 1
                if retry_idx <= max_capture_retries:
                    print(f"Retrying read (no RX rearm): empty/invalid WiFi probe samples detected (attempt {retry_idx})")
                    time.sleep(timeSleep)
                
            if not valid_capture:
                print(f"[WARN] Skipping WiFi probe capture for TX node {Alice} after {max_capture_retries + 1} attempts")
            stopTXNode(Alice)

            if valid_capture:
                _append_wifi_probe_sample(feature_store, probe_data_1, sample_counts)
                channel.append(channel_Labels[0])
                instance.append(3)
                ids.append(id)
                tx.append(Alice)
                rx.append(Bob)
                timestamp.append(timestamp1)

                _append_wifi_probe_sample(feature_store, probe_data_2, sample_counts)
                channel.append(channel_Labels[2])
                instance.append(4)
                ids.append(id)
                tx.append(Alice)
                rx.append(Eve)
                timestamp.append(timestamp2)

                if generatePlots:
                    _plot_wifi_probe_sections_side_by_side(
                        probe_data_1,
                        probe_data_2,
                        sample_counts,
                        id1=Bob,
                        id2=Eve,
                        ax1=ax1,
                        ax2=ax2,
                        fig=fig
                    )
                time.sleep(timeSleep)
        
        i = i + 1

    if fig is not None:
        plt.close(fig)

    return (
        feature_store["iq"]["I"],
        feature_store["iq"]["Q"],
        feature_store["pilots"]["I"],
        feature_store["pilots"]["Q"],
        feature_store["csi"]["I"],
        feature_store["csi"]["Q"],
        feature_store["chan_est_samples"]["I"],
        feature_store["chan_est_samples"]["Q"],
        channel,
        instance,
        ids,
        tx,
        rx,
        timestamp,
    )

def collect_data_ping_pong_3Nodes_cast_probe(params, nodes, packages, metadata, channel_Labels=[1,2,3]):
    i = 1
    channel = []
    instance = []
    ids = []
    tx = []
    rx = []
    timestamp = []
    sample_id = 0
    timeSleep = 0.1
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    cast_metadata = metadata.get("castProbe", {})
    numberOfSamples = int(cast_metadata.get("samples", 32768))
    num_taps = int(cast_metadata.get("num_taps", 128))
    max_capture_retries = int(cast_metadata.get("max_capture_retries", 5))
    generatePlots = bool(cast_metadata.get("generate_plots", False))
    plot_iq_samples = int(cast_metadata.get("plot_iq_samples", min(numberOfSamples, 4096)))
    plot_pause_s = float(cast_metadata.get("plot_pause_s", 0.1))
    fig = plt.figure(figsize=(14, 11)) if generatePlots else None

    feature_store = {
        "iq_I": [],
        "iq_Q": [],
        "taps_I": [],
        "taps_Q": [],
        "cir_I": [],
        "cir_Q": [],
        "tap_delays_s": [],
        "pdp_db": [],
        "detected": [],
        "normalized_peak": [],
        "pathloss_db": [],
        "frame_phase": [],
        "peak_index": [],
        "role": [],
        "tx_name": [],
        "rx_name": [],
    }

    while i < packages * 2 + 1:
        if i % 2 == 0:
            tx_node = Bob
            rx_nodes = [Alice, Eve]
            labels = [channel_Labels[0], channel_Labels[1]]
            instances = [1, 2]
        else:
            sample_id += 1
            tx_node = Alice
            rx_nodes = [Bob, Eve]
            labels = [channel_Labels[0], channel_Labels[2]]
            instances = [3, 4]

        print("Setting TX Node: ", tx_node)
        setTXNode(params, "castProbe", tx_node, metadata)
        print("Setting RX Nodes: ", *rx_nodes)
        setRXNodesParallel(params, rx_nodes, type="castProbe", metadata=metadata)
        time.sleep(timeSleep)

        rx_results = None
        valid_capture = False
        retry_idx = 0
        while retry_idx <= max_capture_retries:
            print("Recording RX Nodes: ", *rx_nodes)
            rx_results = recordNodesParallel(rx_nodes, samples=numberOfSamples, type="castProbe", metadata=metadata)
            valid_capture = all(
                _cast_probe_is_valid(rx_results[node]["data"], min_iq_samples=1, min_taps=num_taps)
                for node in rx_nodes
            )
            if valid_capture:
                break
            print(
                "[CaSTProbe RX] "
                + "; ".join(
                    _cast_probe_debug_summary(node, rx_results[node]["data"])
                    for node in rx_nodes
                )
            )
            retry_idx += 1
            if retry_idx <= max_capture_retries:
                print(f"Retrying read: invalid/undetected CaST probe capture (attempt {retry_idx})")
                time.sleep(timeSleep)

        if not valid_capture:
            print(f"[WARN] Skipping CaST probe capture for TX node {tx_node} after {max_capture_retries + 1} attempts")
        stopTXNode(tx_node)

        if valid_capture:
            if generatePlots:
                _plot_cast_probe_pair(
                    rx_results[rx_nodes[0]]["data"],
                    rx_results[rx_nodes[1]]["data"],
                    tx_node=tx_node,
                    rx_node_1=rx_nodes[0],
                    rx_node_2=rx_nodes[1],
                    role_1=CAST_INSTANCE_ROLES.get(instances[0], ""),
                    role_2=CAST_INSTANCE_ROLES.get(instances[1], ""),
                    num_taps=num_taps,
                    iq_samples=plot_iq_samples,
                    metadata=metadata,
                    fig=fig,
                    pause_s=plot_pause_s,
                )
            for rx_node, label, inst in zip(rx_nodes, labels, instances):
                probe_data = rx_results[rx_node]["data"]
                _append_cast_probe_sample(feature_store, probe_data, numberOfSamples, num_taps)
                channel.append(label)
                instance.append(inst)
                ids.append(sample_id)
                tx.append(tx_node)
                rx.append(rx_node)
                timestamp.append(rx_results[rx_node]["timestamp"])
                feature_store["role"].append(CAST_INSTANCE_ROLES.get(inst, ""))
                feature_store["tx_name"].append(_node_name(tx_node, metadata))
                feature_store["rx_name"].append(_node_name(rx_node, metadata))

        i += 1

    if fig is not None:
        plt.close(fig)

    return (
        feature_store["iq_I"],
        feature_store["iq_Q"],
        feature_store["taps_I"],
        feature_store["taps_Q"],
        feature_store["cir_I"],
        feature_store["cir_Q"],
        feature_store["tap_delays_s"],
        feature_store["pdp_db"],
        feature_store["detected"],
        feature_store["normalized_peak"],
        feature_store["pathloss_db"],
        feature_store["frame_phase"],
        feature_store["peak_index"],
        feature_store["role"],
        feature_store["tx_name"],
        feature_store["rx_name"],
        channel,
        instance,
        ids,
        tx,
        rx,
        timestamp,
    )

def create_dataset(filename, i_samples, q_samples, channel, instance, ids, tx, rx, timestamp):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("I", data=i_samples)
        data_file.create_dataset("Q", data=q_samples)
        data_file.create_dataset("ids", data=[ids])
        data_file.create_dataset("instance", data=[instance])
        data_file.create_dataset("channel", data=[channel])
        data_file.create_dataset("tx", data=[tx])
        data_file.create_dataset("rx", data=[rx])
        data_file.create_dataset("timestamp", data=[timestamp])
    # Save dataset to file
    print("Dataset saved to", filename)

def create_cast_probe_dataset(
        filename,
        iq_I,
        iq_Q,
        taps_I,
        taps_Q,
        cir_I,
        cir_Q,
        tap_delays_s,
        pdp_db,
        detected,
        normalized_peak,
        pathloss_db,
        frame_phase,
        peak_index,
        role,
        tx_name,
        rx_name,
        channel,
        instance,
        ids,
        tx,
        rx,
        timestamp,
        attrs=None
    ):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("iq_I", data=iq_I)
        data_file.create_dataset("iq_Q", data=iq_Q)
        data_file.create_dataset("taps_I", data=taps_I)
        data_file.create_dataset("taps_Q", data=taps_Q)
        data_file.create_dataset("cir_I", data=cir_I)
        data_file.create_dataset("cir_Q", data=cir_Q)
        data_file.create_dataset("tap_delays_s", data=tap_delays_s)
        data_file.create_dataset("pdp_db", data=pdp_db)
        data_file.create_dataset("detected", data=detected)
        data_file.create_dataset("normalized_peak", data=normalized_peak)
        data_file.create_dataset("pathloss_db", data=pathloss_db)
        data_file.create_dataset("frame_phase", data=frame_phase)
        data_file.create_dataset("peak_index", data=peak_index)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        data_file.create_dataset("role", data=np.asarray(role, dtype=object), dtype=string_dtype)
        data_file.create_dataset("tx_name", data=np.asarray(tx_name, dtype=object), dtype=string_dtype)
        data_file.create_dataset("rx_name", data=np.asarray(rx_name, dtype=object), dtype=string_dtype)
        data_file.create_dataset("ids", data=[ids])
        data_file.create_dataset("instance", data=[instance])
        data_file.create_dataset("channel", data=[channel])
        data_file.create_dataset("tx", data=[tx])
        data_file.create_dataset("rx", data=[rx])
        data_file.create_dataset("timestamp", data=[timestamp])
        for key, value in (attrs or {}).items():
            if isinstance(value, (dict, list, tuple)):
                data_file.attrs[key] = json.dumps(value, sort_keys=True)
            else:
                data_file.attrs[key] = value
    print("CaST probe dataset saved to", filename)

def create_wifi_probe_dataset(
        filename,
        iq_I,
        iq_Q,
        pilots_I,
        pilots_Q,
        csi_I,
        csi_Q,
        chan_est_samples_I,
        chan_est_samples_Q,
        channel,
        instance,
        ids,
        tx,
        rx,
        timestamp
    ):
    with h5py.File(filename, "w") as data_file:
        data_file.create_dataset("iq_I", data=iq_I)
        data_file.create_dataset("iq_Q", data=iq_Q)
        data_file.create_dataset("pilots_I", data=pilots_I)
        data_file.create_dataset("pilots_Q", data=pilots_Q)
        data_file.create_dataset("csi_I", data=csi_I)
        data_file.create_dataset("csi_Q", data=csi_Q)
        data_file.create_dataset("chan_est_samples_I", data=chan_est_samples_I)
        data_file.create_dataset("chan_est_samples_Q", data=chan_est_samples_Q)
        data_file.create_dataset("ids", data=[ids])
        data_file.create_dataset("instance", data=[instance])
        data_file.create_dataset("channel", data=[channel])
        data_file.create_dataset("tx", data=[tx])
        data_file.create_dataset("rx", data=[rx])
        data_file.create_dataset("timestamp", data=[timestamp])
    print("WiFi probe dataset saved to", filename)

def generateNodeConfigs(nodeIDs):
    # Generate all possible triplet (Alice, Bob, Eve) combinations for the given node IDs
    # Constraints:
    # - Alice and Bob cannot be the same node
    # - Eve cannot be the same node as Alice or Bob
    # - [Alice, Bob, Eve] is considered the same as [Bob, Alice, Eve] (only include one)
    nodeConfigs = []
    for Alice in nodeIDs:
        for Bob in nodeIDs:
            # Ensure Alice != Bob and enforce ordering to avoid duplicates
            # (only consider cases where Bob > Alice to avoid [Bob, Alice, Eve] duplicates)
            if Bob <= Alice:
                continue
            for Eve in nodeIDs:
                # Eve must be different from both Alice and Bob
                if Eve != Alice and Eve != Bob:
                    nodeConfigs.append([Alice, Bob, Eve])
    return nodeConfigs

def loadOTALabConfig(
        gainConfigs={
            "x310":{"tx":80,"rx":80},
            "b210":{"tx":89,"rx":80}
            },
        nodeIDs = None
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"pc743.emulab.net",       # x310 radio node 1
        2:"pc750.emulab.net",       # x310 radio node 2
        3:"pc745.emulab.net",       # x310 radio node 3
        4:"pc730.emulab.net",        # x310 radio node 4
        5:"ota-nuc1.emulab.net",    # b210 nuc node 1
        6:"ota-nuc2.emulab.net",    # b210 nuc node 2
        7:"ota-nuc3.emulab.net",    # b210 nuc node 3
        8:"ota-nuc4.emulab.net",    # b210 nuc node 4
    }
    NodeGains = {
        1:gainConfigs["x310"],
        2:gainConfigs["x310"],
        3:gainConfigs["x310"],
        4:gainConfigs["x310"],
        5:gainConfigs["b210"],
        6:gainConfigs["b210"],
        7:gainConfigs["b210"],
        8:gainConfigs["b210"],
    }
    
    nodeIDs = nodeIDs if nodeIDs is not None else NodeIPs.keys()
    NodeConfigs = generateNodeConfigs(nodeIDs)
    
    # From the x310 nodes, remove them as transmitters (first two nodes in the list)
    x310_nodes = [1,2,3,4]
    NodeConfigs = [config for config in NodeConfigs if config[0] not in x310_nodes and config[1] not in x310_nodes]
    return NodeIPs, NodeGains, NodeConfigs

def loadOTADenseConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":89,"rx":90}
            },
        nodeIDs = None
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"cnode-ebc.emulab.net",       # EBC dense node with b210
        2:"cnode-guesthouse.emulab.net",# Guesthouse dense node with b210
        3:"cnode-moran.emulab.net",     # Moran dense node with b210
        4:"cnode-ustar.emulab.net",     # Ustar dense node with b210
        5:"cnode-mario.emulab.net",     # Mario dense node with b210
        6:"localhost",                  # Local computer with b210
        7:"162.168.10.101",             # Jetson nano 1 with b210
        8:"162.168.10.102"              # Jetson nano 2 with b210
    }
    NodeGains = {
        1:gainConfigs["b210"],
        2:gainConfigs["b210"],
        3:gainConfigs["b210"],
        4:gainConfigs["b210"],
        5:gainConfigs["b210"],
        6:gainConfigs["b210"],
        7:gainConfigs["b210"],
        8:gainConfigs["b210"],
    }
    if nodeIDs is None:
        nodeIDs = NodeIPs.keys()
    NodeConfigs = generateNodeConfigs(nodeIDs)
    return NodeIPs, NodeGains, NodeConfigs

def loadOTARooftopConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":80}
            },
        nodeIDs = None
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"cnode-ebc.emulab.net",       # EBC dense node with b210
        2:"cnode-guesthouse.emulab.net",# Guesthouse dense node with b210
        3:"cnode-moran.emulab.net",     # Moran dense node with b210
        4:"cnode-ustar.emulab.net",     # Ustar dense node with b210
        5:"localhost",                  # Local computer with b210
        6:"162.168.10.101",             # Jetson nano 1 with b210
        7:"162.168.10.102"              # Jetson nano 2 with b210
    }
    NodeGains = {
        1:gainConfigs["x310"],
        2:gainConfigs["x310"],
        3:gainConfigs["x310"],
        4:gainConfigs["x310"],
        5:gainConfigs["x310"],
        6:gainConfigs["x310"],
        7:gainConfigs["x310"]
    }
    if nodeIDs is None:
        nodeIDs = NodeIPs.keys()
    NodeConfigs = generateNodeConfigs(nodeIDs)
    return NodeIPs, NodeGains, NodeConfigs

if __name__ == "__main__":
    
    # nodeIDs = [2,3,4,5,6,7,8] #[1,2,3,4,5,6,7,8]
    collection_profile = "pidml_cast"
    nodeIDs = PIDML_CAST_PROFILE["node_ids"] if collection_profile == "pidml_cast" else [1,2,3,4] #[1,2,3,4,5,6,7,8]
    # NodeIPs, NodeGains, nodeConfigs = loadOTALabConfig(nodeIDs = nodeIDs)
    NodeIPs, NodeGains, nodeConfigs = loadOTADenseConfig(nodeIDs = nodeIDs)
    # Removing certain nodes that data has been collected
    # configsToRemove = [[5, 6, 2], [5, 6, 3], [5, 6, 4], 
    #                     [5, 6, 7], [5, 6, 8], [5, 7, 2], 
    #                     [5, 7, 3], [5, 7, 4], [5, 7, 6], 
    #                     [5, 7, 8], [5, 8, 2], [5, 8, 3], 
    #                     [5, 8, 4], [5, 8, 6], [5, 8, 7], 
    #                     [6, 7, 2]]
    configsToRemove = []
    nodeConfigs = [config for config in nodeConfigs if config not in configsToRemove]
    # nodeConfigs = [[8,7,5]] # Testing with only one config
    print("Node configs: ", nodeConfigs)
    print(len(nodeConfigs))
    # exit()
    port = {'orch':'5001','radio':'5002','ai':'5003'}

    examples = PIDML_CAST_PROFILE["examples"] if collection_profile == "pidml_cast" else 100
    freq = PIDML_CAST_PROFILE["freq"] if collection_profile == "pidml_cast" else 3.550e9
    samp_rate = PIDML_CAST_PROFILE["tx_sampling_rate_hz"] if collection_profile == "pidml_cast" else 1e6

    paramsTx = {
        "x":"tx",
        "freq":freq,
        "SamplingRate":samp_rate,
        "gain":NodeGains
    }
    paramsRx = {
        "x":"rx",
        "freq":paramsTx["freq"],
        "SamplingRate":PIDML_CAST_PROFILE["rx_sampling_rate_hz"] if collection_profile == "pidml_cast" else int(paramsTx["SamplingRate"]*2),
        "gain":NodeGains
    }
    params = {"tx":paramsTx,"rx":paramsRx}

    type = PIDML_CAST_PROFILE["type"] if collection_profile == "pidml_cast" else "wifiProbe" #pnSequence, castProbe, MPSK, sinusoid, deltaPulse, wifiProbe
        
    metadata = {
        "pnSequence": "glfsr",
        "castProbe": {
            "sequence": "cast",
            "samples": 32768,
            "num_taps": 128,
            "max_wait_s": 2.0,
            "record_timeout_s": 15.0,
            "detection_threshold": 0.05,
            "estimation_window_repetitions": 4
        },
        "deltaPulse": {
            "num_bins": 1024,
            "amplitude": 1,
            "center": True,
            "repeat": True,
            "window": False,  # Set to False to avoid windowing artifacts that may create continuous signal
            "num_pulses": -1
        },
        "wifiProbe": {
            "payload": "probe_request",
            "interval_ms": 300,
            "encoding": 0,
            "tx_amplitude": 0.6,
            "chan_est": 0,
            "api_retries": 1,
            "strict_counts": True,
            "record_timeout_s": 20.0,
            "max_wait_s": 8.0,
            "max_capture_retries": 5,
            "generate_plots": False,
            "sample_counts": {
                "iq": 720,
                "pilots": 4,
                "csi": 52,
                "chan_est_samples": 128
            }
        }
    }
    if collection_profile == "pidml_cast":
        metadata["castProbe"].update(PIDML_CAST_PROFILE["castProbe"])
        metadata["castProbe"]["node_names"] = PIDML_DENSE_NODE_NAMES
        metadata["castProbe"]["profile"] = collection_profile
    
    for nodes in nodeConfigs:
        print(f"Collecting data for node config: Alice={nodes[0]}, Bob={nodes[1]}, Eve={nodes[2]}")
        ts = int(time.time())
        dataset_name = "Dataset_OTADense_Channels_"+type+"_"+str(examples)+"_"+"".join(str(node) for node in nodes)+"_"+str(ts)+".hdf5"

        if type == "wifiProbe":
            iq_I, iq_Q, pilots_I, pilots_Q, csi_I, csi_Q, chan_est_samples_I, chan_est_samples_Q, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes_wifi_probe(
                                                params,
                                                nodes,
                                                examples,
                                                metadata=metadata
                                            )

            create_wifi_probe_dataset(
                dataset_name,
                np.array(iq_I),
                np.array(iq_Q),
                np.array(pilots_I),
                np.array(pilots_Q),
                np.array(csi_I),
                np.array(csi_Q),
                np.array(chan_est_samples_I),
                np.array(chan_est_samples_Q),
                np.array(channel),
                np.array(instance),
                np.array(ids),
                np.array(tx),
                np.array(rx),
                np.array(timestamp)
            )
        elif type == "castProbe":
            iq_I, iq_Q, taps_I, taps_Q, cir_I, cir_Q, tap_delays_s, pdp_db, detected, normalized_peak, pathloss_db, frame_phase, peak_index, role, tx_name, rx_name, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes_cast_probe(
                                                params,
                                                nodes,
                                                examples,
                                                metadata=metadata
                                            )

            create_cast_probe_dataset(
                dataset_name,
                np.array(iq_I),
                np.array(iq_Q),
                np.array(taps_I),
                np.array(taps_Q),
                np.array(cir_I),
                np.array(cir_Q),
                np.array(tap_delays_s),
                np.array(pdp_db),
                np.array(detected),
                np.array(normalized_peak),
                np.array(pathloss_db),
                np.array(frame_phase),
                np.array(peak_index),
                np.array(role),
                np.array(tx_name),
                np.array(rx_name),
                np.array(channel),
                np.array(instance),
                np.array(ids),
                np.array(tx),
                np.array(rx),
                np.array(timestamp),
                attrs={
                    "profile": collection_profile,
                    "probe_type": type,
                    "freq_hz": int(paramsTx["freq"]),
                    "tx_sampling_rate_hz": int(paramsTx["SamplingRate"]),
                    "rx_sampling_rate_hz": int(paramsRx["SamplingRate"]),
                    "castProbe": metadata["castProbe"],
                    "node_names": PIDML_DENSE_NODE_NAMES,
                    "role_order": ["AB", "AE", "BA", "BE"],
                }
            )
        else:
            i_samples, q_samples, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes(
                                            params, 
                                            nodes, 
                                            examples, 
                                            type,
                                            metadata = metadata
                                        )
            create_dataset(
                dataset_name,
                np.array(i_samples),
                np.array(q_samples),
                np.array(channel), 
                np.array(instance), 
                np.array(ids),
                np.array(tx),
                np.array(rx),
                np.array(timestamp)
            )

    # ###############

    # nodes = [2,5,4]

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[0],nodes[1],nodes[2]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[2],nodes[0],nodes[1]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)

    # IQ_Samples, labels, instance, ids = collect_data_ping_pong_3Nodes(params, [nodes[1],nodes[2],nodes[2]], packages, type)

    # IQ_Samples_arr = np.concatenate((IQ_Samples_arr,np.array(IQ_Samples)),axis=0)
    # labels_arr = np.concatenate((labels_arr,np.array(labels)),axis=0)
    # instance_arr = np.concatenate((instance_arr,np.array(instance)),axis=0)
    # ids_arr = np.concatenate((ids_arr,np.array(ids)),axis=0)
