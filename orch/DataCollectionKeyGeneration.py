import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
from concurrent.futures import ThreadPoolExecutor

_tx_config_cache = {}
_rx_config_cache = {}

def APILink(IP,port,path):
    return "http://"+IP+":"+port+path    

def recordIQ(nodeID,port,samples):
    path = "/rx/recordIQ"
    data = {
        "samples": samples
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    # response_rx = requests.get(APILink(NodeIPs[nodeID],port,path))
    # print("Response:",response)
    response_json = response.json()
    try:
        imag = response_json["imag"]
        real = response_json["real"]
        return real,imag
    except Exception as error:
        print("Error: ", error)
        print("Response:",response_json)
        return None, None

def recordWiFiProbe(nodeID,port,samples):
    path = "/rx/recordWiFiProbe"
    data = {}
    if isinstance(samples, dict):
        data["sample_counts"] = samples
    else:
        data["samples"] = samples
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    response_json = response.json()
    try:
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

def setRxIQ(nodeID,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.get(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def setRxWiFiProbe(nodeID,port,chan_est=0):
    path = "/rx/set/wifiProbe"
    data = {
        "chan_est": chan_est
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def setPHY(nodeID,port,params):
    path = "/set/PHY"
    data = {
        "x": params["x"],
        "freq": params["freq"],
        "SamplingRate": params["SamplingRate"],
        "gain": params["gain"][nodeID][params["x"]]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response
    
def set_tx_sinusoid(nodeID,port):
    path = "/tx/set/sinusoid"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
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
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_MPSK(nodeID,port,M):
    path = "/tx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_pnSequence(nodeID,port,sequence):
    path = "/tx/set/pnSequence"
    data = {
        "sequence": sequence
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_wifiProbe(nodeID,port,metadata):
    path = "/tx/set/wifiProbe"
    data = {
        "payload": metadata.get("payload", "probe_request"),
        "interval_ms": metadata.get("interval_ms", 300),
        "encoding": metadata.get("encoding", 0),
        "tx_amplitude": metadata.get("tx_amplitude", 0.6)
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def start_tx(nodeID,port):
    path = "/tx/start"
    data = {
        "message": "TX Start"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response
    
def stop_tx(nodeID,port):
    path = "/tx/stop"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
    return response

def plotTimeDomain(I,Q,samples=-1,id=0):
    plt.plot(I[0:samples], color='red')
    plt.plot(Q[0:samples], color='blue')
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
    print("type:",type)
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

    if _tx_config_cache.get(nodeID) != tx_signature:
        if type == "sinusoid":
            response = set_tx_sinusoid(nodeID,port["radio"])
        elif type == "pnSequence":
            response = set_tx_pnSequence(nodeID,port["radio"],metadata[type])
        elif type == "deltaPulse":
            response = set_tx_deltaPulse(nodeID,port["radio"],metadata[type])
        elif type == "wifiProbe":
            response = set_tx_wifiProbe(nodeID,port["radio"],metadata.get(type, {}))
        else:
            raise ValueError(f"Unsupported TX type: {type}")
        response = setPHY(nodeID,port["radio"],params["tx"])
        _tx_config_cache[nodeID] = tx_signature

    response = start_tx(nodeID,port["radio"])
    
def setRXNode(params,nodeID,type="IQ",metadata=None):
    metadata = metadata or {}
    rx_gain = params["rx"]["gain"][nodeID]["rx"]
    rx_signature = (
        type,
        params["rx"]["freq"],
        params["rx"]["SamplingRate"],
        rx_gain,
        json.dumps(metadata.get(type, {}), sort_keys=True),
    )
    if _rx_config_cache.get(nodeID) == rx_signature:
        return
    if type == "wifiProbe":
        chan_est = metadata.get(type, {}).get("chan_est", 0)
        response = setRxWiFiProbe(nodeID,port["radio"],chan_est=chan_est)
        print("Response SetRXNode WiFiProbe: ", response)
    else:
        response = setRxIQ(nodeID,port["radio"])
        print("Response SetRXNode IQ: ", response)
    response = setPHY(nodeID,port["radio"],params["rx"])
    print("Response SetRXNode PHY: ", response)
    _rx_config_cache[nodeID] = rx_signature
    
def RecordNodeData(nodeID,samples,type="IQ"):
    if type == "wifiProbe":
        return recordWiFiProbe(nodeID,port["radio"],samples)
    return recordIQ(nodeID,port["radio"],samples)

def setRXNodesParallel(params, nodeIDs, type="IQ", metadata=None):
    with ThreadPoolExecutor(max_workers=len(nodeIDs)) as executor:
        futures = [
            executor.submit(setRXNode, params, nodeID, type, metadata)
            for nodeID in nodeIDs
        ]
        for future in futures:
            future.result()

def _recordNodeDataWithTimestamp(nodeID, samples, type="IQ"):
    return nodeID, RecordNodeData(nodeID, samples=samples, type=type), int(time.time())

def recordNodesParallel(nodeIDs, samples, type="IQ"):
    results = {}
    with ThreadPoolExecutor(max_workers=len(nodeIDs)) as executor:
        futures = [
            executor.submit(_recordNodeDataWithTimestamp, nodeID, samples, type)
            for nodeID in nodeIDs
        ]
        for future in futures:
            nodeID, data, ts = future.result()
            results[nodeID] = {"data": data, "timestamp": ts}
    return results
    
def stopTXNode(nodeID):
    response = stop_tx(nodeID,port["radio"])

def setupNodesPingPong(RX1,RX2,TX,params,type,metadata = {"pnSequence":"glfsr"}):
    setRXNode(params,TX,type=type,metadata=metadata)    
    setRXNode(params,RX1,type=type,metadata=metadata)
    setRXNode(params,RX2,type=type,metadata=metadata)
    setTXNode(params,type,TX,metadata)
    
def collect_data_ping_pong_3Nodes(params, nodes, packages, type, channel_Labels = [1,2,3],metadata = {"pnSequence":"glfsr"}):
    i = 1
    # IQ_Samples = np.array([])
    I = []
    Q = []
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
            setRXNodesParallel(params, [Alice, Eve])
            time.sleep(timeSleep)
            
            print("Recording RX Nodes: ", Alice, Eve)
            rx_results = recordNodesParallel([Alice, Eve], samples=numberOfSamples)
            real1, imaginary1 = rx_results[Alice]["data"]
            timestamp1 = rx_results[Alice]["timestamp"]
            real2, imaginary2 = rx_results[Eve]["data"]
            timestamp2 = rx_results[Eve]["timestamp"]
            
            stopTXNode(Bob)
            
            if real2 is None or real1 is None:
                id = id - 1
                i = i - 1
                # continue
                setRXNode(params,Eve)
            
            if real1 is not None and real2 is not None:
                I.append(real1[-numberOfSamples:])
                Q.append(imaginary1[-numberOfSamples:])
                channel.append(channel_Labels[0])
                instance.append(1)
                ids.append(id)
                tx.append(Bob)
                rx.append(Alice)
                timestamp.append(timestamp1)
                
                I.append(real2[-numberOfSamples:])
                Q.append(imaginary2[-numberOfSamples:])
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
            setRXNodesParallel(params, [Bob, Eve])
            time.sleep(timeSleep)
            
            print("Recording RX Nodes: ", Bob, Eve)
            rx_results = recordNodesParallel([Bob, Eve], samples=numberOfSamples)
            real1, imaginary1 = rx_results[Bob]["data"]
            timestamp1 = rx_results[Bob]["timestamp"]
            real2, imaginary2 = rx_results[Eve]["data"]
            timestamp2 = rx_results[Eve]["timestamp"]
            if real2 is None or real1 is None:
                id = id - 1
                i = i - 1
                setRXNode(params,Eve)
            
            stopTXNode(Alice)
            
            if real1 is not None and real2 is not None:
                I.append(real1[-numberOfSamples:])
                Q.append(imaginary1[-numberOfSamples:])
                channel.append(channel_Labels[0])
                instance.append(3)
                ids.append(id)
                tx.append(Alice)
                rx.append(Bob)
                timestamp.append(timestamp1)
                
                I.append(real2[-numberOfSamples:])
                Q.append(imaginary2[-numberOfSamples:])
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
    return I, Q, channel, instance, ids, tx, rx, timestamp

def _slice_samples(values, samples):
    if samples == -1:
        return values
    return values[-samples:]

def _wifi_probe_is_valid(probe_data):
    if probe_data is None:
        return False
    required_sections = ["iq", "pilots", "csi", "chan_est_samples"]
    for section in required_sections:
        if section not in probe_data:
            return False
        if "real" not in probe_data[section] or "imag" not in probe_data[section]:
            return False
    return True

def _get_wifi_probe_sample_counts(metadata):
    # Default to twice the nominal WiFi probe lengths.
    sample_counts = {
        "iq": 96,
        "pilots": 8,
        "csi": 104,
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

def _plot_wifi_probe_sections_side_by_side(probe_data_1, probe_data_2, sample_counts, id1, id2, ax1=None, ax2=None, fig=None):
    for section in ["iq", "pilots", "csi", "chan_est_samples"]:
        plotTimeDomainSideBySide(
            _slice_samples(probe_data_1[section]["real"], sample_counts[section]),
            _slice_samples(probe_data_1[section]["imag"], sample_counts[section]),
            _slice_samples(probe_data_2[section]["real"], sample_counts[section]),
            _slice_samples(probe_data_2[section]["imag"], sample_counts[section]),
            samples=sample_counts[section],
            id1=f"{id1} {section}",
            id2=f"{id2} {section}",
            ax1=ax1,
            ax2=ax2,
            fig=fig
        )

def collect_data_ping_pong_3Nodes_wifi_probe(params, nodes, packages, metadata, channel_Labels=[1,2,3]):
    i = 1
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
    sample_counts = _get_wifi_probe_sample_counts(metadata)

    feature_store = {
        "iq": {"I": [], "Q": []},
        "pilots": {"I": [], "Q": []},
        "csi": {"I": [], "Q": []},
        "chan_est_samples": {"I": [], "Q": []},
    }

    generatePlots = True
    fig = None
    ax1 = None
    ax2 = None
    if generatePlots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    while i < packages * 2 + 1:
        print(i)
        if i % 2 == 0:
            print("Setting TX Node: ", Bob)
            setTXNode(params, "wifiProbe", Bob, metadata)

            print("Setting RX Nodes: ", Alice, Eve)
            setRXNodesParallel(params, [Alice, Eve], type="wifiProbe", metadata=metadata)
            time.sleep(timeSleep)

            print("Recording RX Nodes: ", Alice, Eve)
            rx_results = recordNodesParallel([Alice, Eve], samples=sample_counts, type="wifiProbe")
            probe_data_1 = rx_results[Alice]["data"]
            timestamp1 = rx_results[Alice]["timestamp"]
            probe_data_2 = rx_results[Eve]["data"]
            timestamp2 = rx_results[Eve]["timestamp"]

            stopTXNode(Bob)

            if (not _wifi_probe_is_valid(probe_data_1)) or (not _wifi_probe_is_valid(probe_data_2)):
                id = id - 1
                i = i - 1
                setRXNode(params, Eve, type="wifiProbe", metadata=metadata)

            if _wifi_probe_is_valid(probe_data_1) and _wifi_probe_is_valid(probe_data_2):
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

            print("Recording RX Nodes: ", Bob, Eve)
            rx_results = recordNodesParallel([Bob, Eve], samples=sample_counts, type="wifiProbe")
            probe_data_1 = rx_results[Bob]["data"]
            timestamp1 = rx_results[Bob]["timestamp"]
            probe_data_2 = rx_results[Eve]["data"]
            timestamp2 = rx_results[Eve]["timestamp"]

            if (not _wifi_probe_is_valid(probe_data_1)) or (not _wifi_probe_is_valid(probe_data_2)):
                id = id - 1
                i = i - 1
                setRXNode(params, Eve, type="wifiProbe", metadata=metadata)

            stopTXNode(Alice)

            if _wifi_probe_is_valid(probe_data_1) and _wifi_probe_is_valid(probe_data_2):
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

def create_dataset(filename, I, Q, channel, instance, ids, tx, rx, timestamp):
    with h5py.File(filename, "w") as data_file:
        dset = data_file.create_dataset("I", data=I)
        dset = data_file.create_dataset("Q", data=Q)
        dset = data_file.create_dataset("ids", data=[ids])
        dset = data_file.create_dataset("instance", data=[instance])
        dset = data_file.create_dataset("channel", data=[channel])
        dset = data_file.create_dataset("tx", data=[tx])
        dset = data_file.create_dataset("rx", data=[rx])
        dset = data_file.create_dataset("timestamp", data=[timestamp])
    # Save dataset to file
    print("Dataset saved to", filename)

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
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":70}
            },
        nodeIDs = None
    ):
    # OTA Lab node IPs
    NodeIPs = {
        1:"pc743.emulab.net",       # x310 radio node 1
        2:"pc780.emulab.net",       # x310 radio node 2
        3:"pc774.emulab.net",       # x310 radio node 3
        4:"pc768.emulab.net",        # x310 radio node 4
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
            "b210":{"tx":80,"rx":70}
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
        1:gainConfigs["b210"],
        2:gainConfigs["b210"],
        3:gainConfigs["b210"],
        4:gainConfigs["b210"],
        5:gainConfigs["b210"],
        6:gainConfigs["b210"],
        7:gainConfigs["b210"]
    }
    if nodeIDs is None:
        nodeIDs = NodeIPs.keys()
    NodeConfigs = generateNodeConfigs(nodeIDs)
    return NodeIPs, NodeGains, NodeConfigs

def loadOTARooftopConfig(
        gainConfigs={
            "x310":{"tx":31,"rx":31},
            "b210":{"tx":80,"rx":70}
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
    
    nodeIDs = [2,3,4,5,6,7,8] #[1,2,3,4,5,6,7,8]
    NodeIPs, NodeGains, nodeConfigs = loadOTALabConfig(nodeIDs = nodeIDs)
    # NodeIPs, NodeGains, nodeConfigs = loadOTADenseConfig()
    # Removing certain nodes that data has been collected
    # configsToRemove = [[5, 6, 2], [5, 6, 3], [5, 6, 4], [5, 6, 7], [5, 6, 8], [5, 7, 2], [5, 7, 3], [5, 7, 4]]
    # nodeConfigs = [config for config in nodeConfigs if config not in configsToRemove]
    print("Node configs: ", nodeConfigs)
    print(len(nodeConfigs))
    # exit()
    port = {'orch':'5001','radio':'5002','ai':'5003'}

    examples= 100
    freq = 3.450e9
    samp_rate = 1000e3

    paramsTx = {
        "x":"tx",
        "freq":freq,
        "SamplingRate":samp_rate,
        "gain":NodeGains
    }
    paramsRx = {
        "x":"rx",
        "freq":paramsTx["freq"],
        "SamplingRate":int(paramsTx["SamplingRate"]*2),
        "gain":NodeGains
    }
    params = {"tx":paramsTx,"rx":paramsRx}

    type = "wifiProbe" #pnSequence, MPSK, sinusoid, deltaPulse, wifiProbe
        
    metadata = {
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
            "sample_counts": {
                "iq": 96,
                "pilots": 8,
                "csi": 104,
                "chan_est_samples": 128
            }
        }
    }
    
    # Lets test a single node setting it up, then starting the transmitter and then stopping it after 10 seconds
    # print("Testing single node setup")
    # nodeID = 3
    # port = {'orch':'5001','radio':'5002','ai':'5003'}
    # setTXNode(params,type,nodeID,metadata)
    # time.sleep(10)
    # stopTXNode(nodeID)
    # print("Transmitter stopped")
    # exit()
    
    for nodes in nodeConfigs:
        print(f"Collecting data for node config: Alice={nodes[0]}, Bob={nodes[1]}, Eve={nodes[2]}")
        ts = int(time.time())
        dataset_name = "Dataset_OTALab_Channels_"+type+"_"+str(examples)+"_"+"".join(str(node) for node in nodes)+"_"+str(ts)+".hdf5"

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
        else:
            I, Q, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes(
                                                params, 
                                                nodes, 
                                                examples, 
                                                type,
                                                metadata = metadata
                                            )
            create_dataset(
                dataset_name,
                np.array(I),
                np.array(Q),
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