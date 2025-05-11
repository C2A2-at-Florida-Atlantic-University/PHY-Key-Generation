import requests
import json
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
import math
import h5py
from threading import Thread, Event
import seaborn as sns


import datetime as dt
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str
import os 

#Dataset Generator for creating datasets
#Creates IQ datasets within the SigMF format
#IQ files as well as signal metadata
class sigMFDataset():
    def __init__(self):
        self.date_time = dt.datetime.utcnow().isoformat()+'Z'
        self.metadataIsSet = False
        
    def setData(self,data,label,samplesPerExample):
        self.data = data
        self.label = label
        self.SPE = samplesPerExample
        
    def createDataset(self):
        if self.metadataIsSet:
            self.createFolder()
            self.createIQFile()
            self.createMetadata()
        else:
            print("Set metadata first with setMetadata()")
    
    def createFolder(self):
        parent_dir = os.getcwd()
        directory = self.fileName+"_"+self.author+"_"+self.date_time
        self.path = os.path.join(parent_dir,directory)
        os.mkdir(self.path) 
        print("Directory '% s' created" % directory) 
    
    def createIQFile(self):
        self.data.tofile(self.fileName+'.sigmf-data')
    
    def setMetadata(self):
        # create the metadata
        self.fileName = input("File Name:")
        self.samp_rate = input("Sampling Rate:")
        self.freq = input("Sampling Frequency:")
        self.author = input("Author Email:")
        self.description = input("Description:")
        self.metadataIsSet = True
        
    def setMetadata(self,fileName,samp_rate,freq,author,description):
        # create the metadata
        self.fileName = fileName
        self.samp_rate = samp_rate
        self.freq = freq
        self.author = author
        self.description = description
        self.metadataIsSet = True
    
    def createMetadata(self):
        self.metadata = SigMFFile(
            data_file=self.fileName+'.sigmf-data', # extension is optional
            global_info = {
                SigMFFile.DATATYPE_KEY: get_data_type_str(self.data),  # in this case, 'cf32_le'
                SigMFFile.SAMPLE_RATE_KEY: self.samp_rate,
                SigMFFile.AUTHOR_KEY: self.author,
                SigMFFile.DESCRIPTION_KEY: self.description,
                SigMFFile.FREQUENCY_KEY: self.freq,
                SigMFFile.DATETIME_KEY: self.date_time,
            }
        )
        self.metadata.tofile(self.fileName+'.sigmf-meta')
    
def APILink(IP,port,path):
    return "http://"+IP+":"+port+path    

def recordIQ(IP,port):
    path = "/rx/recordIQ"
    response_rx = requests.get(APILink(IP,port,path))
    response_json = response_rx.json()
    imag = response_json["imag"]
    real = response_json["real"]
    return real,imag

def setRxIQ(IP,port):
    path = "/rx/set/IQ"
    data = {
        "contents": "IQ"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.get(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response
    
def setRxMPSK(IP,port,M):
    path = "/rx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response

def setPHY(IP,port,params):
    path = "/set/PHY"
    data = {
        "x": params["x"],
        "freq": params["freq"],
        "SamplingRate": params["SamplingRate"],
        "gain": params["gain"]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response
    
def set_tx_sinusoid(IP,port):
    path = "/tx/set/sinusoid"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_MPSK(IP,port,M):
    path = "/tx/set/MPSK"
    data = {
        "M": M
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_pnSequence(IP,port,sequence):
    path = "/tx/set/pnSequence"
    data = {
        "sequence": sequence
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response

def set_tx_fileSource(IP,port,fileSource):
    path = "/tx/set/fileSource"
    data = {
        "fileSource": fileSource
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response

def start_tx(IP,port):
    path = "/tx/start"
    data = {
        "message": "TX Start"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
    return response
    
def stop_tx(IP,port):
    path = "/tx/stop"
    data = {
        "message": "sinusoid"
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(APILink(IP,port,path), data=json.dumps(data), headers=headers)
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
    plt.show()

def plotConstellationDiagram(I,Q,samples=-1,id=0):
    plt.scatter(I[0:samples], Q[0:samples], color='red')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.title('Constellation Diagram Plot Node: '+str(id))
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()

def plotSpectrogram(I,Q,fs,samples=-1,id=0):
    x = np.array([complex(Q[i],I[i]) for i in range(len(I))])
    f, t, Sxx = signal.spectrogram(x, fs)
    plt.title('Spectrogram Plot Node: '+str(id))
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    
def RoundRecordData(params,type,tx,RX,samples=1024,packets=1, metadata = {}):
    print("type:",type)
    if type == "sinusoid":
        response = set_tx_sinusoid(NodeIP[tx],port["radio"])
    elif type == "MPSK":
        response = set_tx_MPSK(NodeIP[tx],port["radio"],metadata[type])
    elif type == "pnSequence":
        response = set_tx_pnSequence(NodeIP[tx],port["radio"],metadata[type])
    elif type == "fileSource":
        response = set_tx_fileSource(NodeIP[tx],port["radio"],metadata[type])
    print(response)
    response = setPHY(NodeIP[tx],port["radio"],params["tx"])
    print(response)
    response = start_tx(NodeIP[tx],port["radio"])
    print(response)
    time.sleep(2)
    data = {}
    for rx in RX:
        data[rx] = np.array([])
        response = setRxIQ(NodeIP[rx],port["radio"])
        response = setPHY(NodeIP[rx],port["radio"],params["rx"])

    for i in range(packets):
        print("packet "+str(i)+" of "+str(packets))
        for rx in RX:  
            # print("starting RX",rx)
            I,Q = recordIQ(NodeIP[rx],port["radio"])
            complexIQ = np.array(I)+np.array(Q)*1j
            data[rx] = np.append(data[rx],complexIQ)
        
    time.sleep(2)
    # print("stop TX",tx)
    response = stop_tx(NodeIP[tx],port["radio"])
    # print(response)
    return data
    
def RecordDataAmbient(params,RX,samples=1024):
    for rx in RX:
        print("start RX ",rx)
        response = setRxIQ(NodeIP[rx],port["radio"])
        print(response)
        I,Q = recordIQ(NodeIP[rx],port["radio"])
        print("stop RX",rx)
        # plotTimeDomain(I,Q,samples=1024)
        # plotConstellationDiagram(I,Q,samples=1024)
        # plotSpectrogram(I_Sig,Q_Sig,params["rx"]["SamplingRate"],samples=1024)
    return data
    
def formatData(Data,modulation):
    DS_node = np.array([])
    DS_labels = np.array([])
    DS_data = np.array([])
    for node in Data:
        for N in range(examples):
            DS_node = np.append(DS_node,node)
            DS_labels = np.append(DS_labels,modulation)
            complex_data = np.array(Data[node][N*samplesPerExample:(N+1)*samplesPerExample])
            imag_data = complex_data.imag
            real_data = complex_data.real
            real_imag_data = np.array([[real_data,  imag_data]])
            if len(DS_data) == 0: 
                DS_data = real_imag_data
            else:
                DS_data = np.append(DS_data,real_imag_data,axis=0)
            print(DS_data.shape)

    return DS_data, DS_node, DS_labels

def recordDataSignalClassification(TXs,RXs,params,Ms = [2,4]):
    for tx in TXs:
        env = {"rx":RXs,"tx":tx}
        Ms = [2,4]
        for M in Ms:
            Data = RoundRecordData(params,type="MPSK",tx=env["tx"],RX=env["rx"],
                                   samples=samplesPerExample,packets=int(packetsNeeded),
                                   metadata={"MPSK":M})
            if M == 2:
                Modulation = "BPSK"
            elif M == 4:
                Modulation = "QPSK"    
            DS_data, DS_node, DS_labels = formatData(Data,Modulation)
            filename = Modulation+"_"+str(env["tx"])+"_TX_8_2_RX_01032024"
            file = h5py.File(filename+".h5", "w")
            # Create datasets within the HDF5 file
            file.create_dataset("data", data=DS_data)
            file.create_dataset("node", data=DS_node)
            file.close()
            
def recordDataSignalClassification(TXs,RXs,params,modulations = ["BPSK"]):
    for tx in TXs:
        env = {"rx":RXs,"tx":tx}
        for modulation in modulations:
            print("Recording data for "+modulation+" TX=" +str(tx))
            fileSource = "Matlab/"+modulation+".dat"
            Data = RoundRecordData(params,type="fileSource",tx=env["tx"],RX=env["rx"],
                                   samples=samplesPerExample,packets=int(packetsNeeded),
                                   metadata={"fileSource":fileSource})   
            DS_data, DS_node, DS_labels = formatData(Data,modulation)
            filename = modulation+"_"+str(env["tx"])+"_TX_8_2_RX_01032024"
            file = h5py.File(filename+".h5", "w")
            # Create datasets within the HDF5 file
            file.create_dataset("data", data=DS_data)
            file.create_dataset("node", data=DS_node)
            file.close()
            
def recordDataChannelFingerprinting(TXs,RXs,params,sequence = "glfsr"):
    for tx in TXs:
        print("TX:",tx)
        env = {"rx":RXs,"tx":tx}
        Data = RoundRecordData(params,type="pnSequence",tx=env["tx"],RX=env["rx"],samples=samplesPerExample,packets=int(packetsNeeded),M=1,sequence=sequence)
        DS_data, DS_node, DS_labels = formatData(Data,sequence)
        filename = "pnSequence"+"_"+str(env["tx"])+"_TX_8_2_RX_01032024"
        file = h5py.File(filename+".h5", "w")
        # Create datasets within the HDF5 file
        file.create_dataset("data", data=DS_data)
        file.create_dataset("node", data=DS_node)
        file.close()

def setTXNode(params,type,tx,metadata = {"pnSequence":"glfsr"}):
    print("type:",type)
    if type == "sinusoid":
        response = set_tx_sinusoid(NodeIP[tx],port["radio"])
    elif type == "pnSequence":
        response = set_tx_pnSequence(NodeIP[tx],port["radio"],metadata[type])
    response = setPHY(NodeIP[tx],port["radio"],params["tx"])
    response = start_tx(NodeIP[tx],port["radio"])
    
def setRXNode(params,rx):
    response = setRxIQ(NodeIP[rx],port["radio"])
    response = setPHY(NodeIP[rx],port["radio"],params["rx"])
    
def RecordNodeData(rx):
    return recordIQ(NodeIP[rx],port["radio"])
    
def stopTXNode(tx):
    response = stop_tx(NodeIP[tx],port["radio"])

def setupNodesPingPong(RX1,RX2,TX,params,type,metadata = {"pnSequence":"glfsr"}):
    setRXNode(params,TX)    
    setRXNode(params,RX1)
    setRXNode(params,RX2)
    setTXNode(params,type,TX,metadata)
    
def plot_waveform(plot_title,real,imaginary):
    plt.title(plot_title)
    plt.plot(real[:], )
    plt.plot(imaginary[:])
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

#Plot Spectrogram
def plot_spectrogram(plot_title,real,imaginary):
    # Calculate power spectral density (frequency domain version of signal)
    samples = np.array(real) + 1j*np.array(imaginary)
    samples = samples[:]
    win_len=256/2
    overlap=win_len/2
    f, t, spec = signal.stft(samples, 
                    window='boxcar', 
                    nperseg= win_len, 
                    noverlap= overlap, 
                    nfft= win_len,
                    return_onesided=False, 
                    padded = False, 
                    boundary = None)
    # FFT shift to adjust the central frequency.
    spec = np.fft.fftshift(spec, axes=0)
    # Take the logarithm of the magnitude.      
    spec_amp = np.log10(np.abs(spec)**2)
    num_row = spec_amp.shape[0]
    spec_amp = spec_amp[round(num_row*0.3):round(num_row*0.7)]
    plt.title(plot_title)
    sns.heatmap(spec_amp,xticklabels=[], yticklabels=[], cmap='Blues', cbar=False)
    plt.gca().invert_yaxis()
    plt.show()
    
def collect_data_ping_pong_3Nodes(params, nodes, packages, type, channel_Labels = [1,2,3],metadata = {"pnSequence":"glfsr"}):
    i = 1
    IQ_Samples = np.array([])
    labels = []
    instance = []
    ids = []
    id = 0
    timeSleep = 0.15
    Alice = nodes[0]
    Bob = nodes[1]
    Eve = nodes[2]
    numberOfSamples = 8192
    setRXNode(params,Eve)
    while i<packages*2+1:
        print(i)
        if i%2 == 0:
            setTXNode(params,type,Bob,metadata)
            setRXNode(params,Alice)
            time.sleep(timeSleep)
            
            real1, imaginary1 = RecordNodeData(rx=Alice)
            IQ_N1 = np.concatenate((imaginary1[0:numberOfSamples], real1[0:numberOfSamples]), axis=None)
            IQ_Samples = np.concatenate((IQ_Samples,[IQ_N1]), axis=0)
            labels.append(channel_Labels[0])
            instance.append(1)
            ids.append(id)
            
            real2, imaginary2 = RecordNodeData(rx=Eve)
            IQ_N3_1 = np.concatenate((imaginary2[0:numberOfSamples], real2[0:numberOfSamples]), axis=None)
            IQ_Samples = np.concatenate((IQ_Samples,[IQ_N3_1]), axis=0)
            labels.append(channel_Labels[1])  
            instance.append(2)   
            ids.append(id)
            
            # plot_spectrogram("Spectrogram Bob TX & Alice RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_waveform("Waveform Bob TX & Alice RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_spectrogram("Spectrogram Bob TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            # plot_waveform("Waveform Bob TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            stopTXNode(tx=Bob)
            time.sleep(timeSleep)

        else:
            id = id + 1
            setTXNode(params,type,Alice,metadata)
            setRXNode(params,Bob)
            time.sleep(timeSleep)
            
            real1, imaginary1 = RecordNodeData(rx=Bob)
            IQ_N2 = np.concatenate((imaginary1[0:numberOfSamples], real1[0:numberOfSamples]), axis=None)
            if(i == 1):
                IQ_Samples = np.array([IQ_N2])
            else:
                IQ_Samples = np.concatenate((IQ_Samples,[IQ_N2]), axis=0)
            labels.append(channel_Labels[0])
            instance.append(3)
            ids.append(id)

            real2, imaginary2 = RecordNodeData(rx=Eve)
            IQ_N3_2 = np.concatenate((imaginary2[0:numberOfSamples], real2[0:numberOfSamples]), axis=None)
            IQ_Samples = np.concatenate((IQ_Samples,[IQ_N3_2]), axis=0)
            labels.append(channel_Labels[2])
            instance.append(4)
            ids.append(id)
            # plot_spectrogram("Spectrogram Alice TX & Bob RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_waveform("Waveform Alice TX & Bob RX",real1[0:numberOfSamples],imaginary1[0:numberOfSamples])
            # plot_spectrogram("Spectrogram Alice TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            # plot_waveform("Waveform Alice TX & Eve RX",real2[0:numberOfSamples],imaginary2[0:numberOfSamples])
            stopTXNode(tx=Alice)
            time.sleep(timeSleep)
        
        i = i + 1
    return IQ_Samples, labels, instance, ids

def create_dataset(filename, IQ_Samples, labels, instance, ids):
    with h5py.File(filename, "w") as data_file:
        dset = data_file.create_dataset("ids", data=[ids])
        dset = data_file.create_dataset("instance", data=[instance])
        dset = data_file.create_dataset("data", data=IQ_Samples)
        dset = data_file.create_dataset("label", data=[labels])
        
NodeIP = {1:"127.0.0.1",2:"100.100.54.44",3:"100.71.206.44",4:"100.81.87.38",5:"100.91.5.85",
      6:"127.0.0.1",7:"100.105.49.48",8:"100.75.87.114",9:"127.0.0.1",10:"127.0.0.1"}

port = {'orch':'5001','radio':'5002','ai':'5003'}

examples= 3000
samplesPerExample = 1024
samplesPerPacket = 8192
freq = 2.4e9
samp_rate = 6e5
gainRX = 60
gainTX = 0

paramsTx = {"x":"tx","freq":freq,"SamplingRate":samp_rate,"gain":gainTX}
paramsRx = {"x":"rx","freq":paramsTx["freq"],"SamplingRate":int(paramsTx["SamplingRate"]*2),"gain":gainRX}
params = {"tx":paramsTx,"rx":paramsRx}

packages = 100
type = "pnSequence"
