import requests
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py

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
    print("Response:",response)
    response_json = response.json()
    imag = response_json["imag"]
    real = response_json["real"]
    return real,imag

def setRxIQ(nodeID,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.get(APILink(NodeIPs[nodeID],port,path), data=json.dumps(data), headers=headers)
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
    plt.pause(0.5)
    plt.clf()  # Clear the figure for the next plot

def setTXNode(params,type,nodeID,metadata = {"pnSequence":"glfsr"}):
    print("type:",type)
    if type == "sinusoid":
        response = set_tx_sinusoid(nodeID,port["radio"])
    elif type == "pnSequence":
        response = set_tx_pnSequence(nodeID,port["radio"],metadata[type])
    response = setPHY(nodeID,port["radio"],params["tx"])
    response = start_tx(nodeID,port["radio"])
    
def setRXNode(params,nodeID):
    response = setRxIQ(nodeID,port["radio"])
    response = setPHY(nodeID,port["radio"],params["rx"])
    
def RecordNodeData(nodeID,samples):
    return recordIQ(nodeID,port["radio"],samples)
    
def stopTXNode(nodeID):
    response = stop_tx(nodeID,port["radio"])

def setupNodesPingPong(RX1,RX2,TX,params,type,metadata = {"pnSequence":"glfsr"}):
    setRXNode(params,TX)    
    setRXNode(params,RX1)
    setRXNode(params,RX2)
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
    timeSleep = 0.15
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    numberOfSamples = 8192
    setRXNode(params,Eve)
    generatePlots = True
    while i<packages*2+1:
        print(i)
        if i%2 == 0:
            setTXNode(params,type,Bob,metadata)
            setRXNode(params,Alice)
            time.sleep(timeSleep)
            
            real1, imaginary1 = RecordNodeData(Alice, samples=numberOfSamples)
            # IQ_N1 = np.concatenate((imaginary1[0:numberOfSamples], real1[0:numberOfSamples]), axis=None)
            # IQ_Samples = np.concatenate((IQ_Samples,[IQ_N1]), axis=0)
            I.append(real1[0:numberOfSamples])
            Q.append(imaginary1[0:numberOfSamples])
            channel.append(channel_Labels[0])
            instance.append(1)
            ids.append(id)
            tx.append(Bob)
            rx.append(Alice)
            timestamp.append(int(time.time()))
            
            real2, imaginary2 = RecordNodeData(Eve, samples=numberOfSamples)
            # IQ_N3_1 = np.concatenate((imaginary2[0:numberOfSamples], real2[0:numberOfSamples]), axis=None)
            # IQ_Samples = np.concatenate((IQ_Samples,[IQ_N3_1]), axis=0)
            I.append(real2[0:numberOfSamples])
            Q.append(imaginary2[0:numberOfSamples])
            channel.append(channel_Labels[1])  
            instance.append(2)   
            ids.append(id)
            tx.append(Bob)
            rx.append(Eve)
            timestamp.append(int(time.time()))
            
            if(generatePlots):
                plotTimeDomain(real1, imaginary1, samples=numberOfSamples, id=Alice)
                plotTimeDomain(real2, imaginary2, samples=numberOfSamples, id=Eve)
            
            # plot_spectrogram("Spectrogram Bob TX & Alice RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_waveform("Waveform Bob TX & Alice RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_spectrogram("Spectrogram Bob TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            # plot_waveform("Waveform Bob TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            stopTXNode(Bob)
            time.sleep(timeSleep)

        else:
            id = id + 1
            setTXNode(params,type,Alice,metadata)
            setRXNode(params,Bob)
            time.sleep(timeSleep)
            
            real1, imaginary1 = RecordNodeData(Bob, samples=numberOfSamples)
            # IQ_N2 = np.concatenate((imaginary1[0:numberOfSamples], real1[0:numberOfSamples]), axis=None)
            # if(i == 1):
            #     IQ_Samples = np.array([IQ_N2])
            # else:
            #     IQ_Samples = np.concatenate((IQ_Samples,[IQ_N2]), axis=0)
            I.append(real1[0:numberOfSamples])
            Q.append(imaginary1[0:numberOfSamples])
            channel.append(channel_Labels[0])
            instance.append(3)
            ids.append(id)
            tx.append(Alice)
            rx.append(Bob)
            timestamp.append(int(time.time()))

            real2, imaginary2 = RecordNodeData(Eve, samples=numberOfSamples)
            # IQ_N3_2 = np.concatenate((imaginary2[0:numberOfSamples], real2[0:numberOfSamples]), axis=None)
            # IQ_Samples = np.concatenate((IQ_Samples,[IQ_N3_2]), axis=0)
            I.append(real2[0:numberOfSamples])
            Q.append(imaginary2[0:numberOfSamples])
            channel.append(channel_Labels[2]) # Changed from "labels" to "channel"
            instance.append(4)
            ids.append(id)
            tx.append(Alice)
            rx.append(Eve)
            timestamp.append(int(time.time()))

            stopTXNode(Alice)
            
            if(generatePlots):
                plotTimeDomain(real1, imaginary1, samples=numberOfSamples, id=Bob)
                plotTimeDomain(real2, imaginary2, samples=numberOfSamples, id=Eve)
            time.sleep(timeSleep)
        
        i = i + 1
    return I, Q, channel, instance, ids, tx, rx, timestamp

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

def loadOTALabConfig():
    # OTA Lab node IPs
    NodeIPs = {
        1:"ota-nuc1.emulab.net",
        2:"ota-nuc2.emulab.net",
        3:"ota-nuc3.emulab.net",
        4:"ota-nuc4.emulab.net",
        5:"pc761.emulab.net",
        6:"pc775.emulab.net",
        7:"pc761.emulab.net",
        8:"pc761.emulab.net"
    }
    NodeGains = {
        1:{"tx":80,"rx":70},
        2:{"tx":80,"rx":70},
        3:{"tx":80,"rx":70},
        4:{"tx":80,"rx":70},
        5:{"tx":31,"rx":31},
        6:{"tx":31,"rx":31},
        7:{"tx":31,"rx":31},
        8:{"tx":31,"rx":31}
    }
    return NodeIPs, NodeGains

def loadOTADenseConfig():
    # OTA Lab node IPs
    NodeIPs = {
        1:"cnode-moran.emulab.net",
        2:"cnode-ebc.emulab.net",
        3:"localhost",
        4:"ota-nuc4.emulab.net",
        5:"pc761.emulab.net",
        6:"pc775.emulab.net",
        7:"pc761.emulab.net",
        8:"pc761.emulab.net"
    }
    NodeGains = {
        1:{"tx":89,"rx":76},
        2:{"tx":89,"rx":76},
        3:{"tx":89,"rx":76},
        4:{"tx":31,"rx":31},
        5:{"tx":31,"rx":31},
        6:{"tx":31,"rx":31},
        7:{"tx":31,"rx":31},
        8:{"tx":31,"rx":31}
    }
    return NodeIPs, NodeGains

if __name__ == "__main__":

    # NodeIPs, NodeGains = loadOTALabConfig()
    NodeIPs, NodeGains = loadOTADenseConfig()

    port = {'orch':'5001','radio':'5002','ai':'5003'}

    examples= 5
    freq = 3.558e9
    samp_rate = 600e3

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

    type = "sinusoid" #pnSequence, MPSK, sinusoid

    nodes = [1,2,3]

    I, Q, channel, instance, ids, tx, rx, timestamp = collect_data_ping_pong_3Nodes(
                                        params, 
                                        nodes, 
                                        examples, 
                                        type
                                    )

    I_arr = np.array(I)
    Q_arr = np.array(Q)
    channel_arr = np.array(channel)
    instance_arr = np.array(instance)
    ids_arr = np.array(ids)
    tx_arr = np.array(tx)
    rx_arr = np.array(rx)
    timestamp_arr = np.array(timestamp)

    print("I shape:",I_arr.shape)
    print("Q shape:",Q_arr.shape)
    print("channel shape:",channel_arr.shape)
    print("instance shape:",instance_arr.shape)
    print("ids shape:",ids_arr.shape)   
    ts = int(time.time())
    create_dataset(
        "Dataset_Channels_"+type+"_"+str(examples)+"_"+"".join(str(node) for node in nodes)+"_"+str(ts)+".hdf5",
        I_arr,
        Q_arr,
        channel_arr, 
        instance_arr, 
        ids_arr,
        tx_arr,
        rx_arr,
        timestamp_arr
    )




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