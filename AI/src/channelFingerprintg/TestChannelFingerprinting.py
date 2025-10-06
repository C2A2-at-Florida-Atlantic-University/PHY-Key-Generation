import shutil
import tensorflow as tf
import pandas as pd
tf.keras.backend.clear_session()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Allow memory growth for GPU
        for gpu in gpus:
            print(gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
from datetime import datetime
from statistics import mean
import numpy as np
import unireedsolomon as rs
from tqdm import tqdm
import matplotlib.pyplot as plt
from nistrng import *
import hashlib
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from deep_learning_models import QuadrupletNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram
import os
# Import scipy
from scipy.stats import entropy
import h5py
import sys

def feature_quantization(features):
    mean_features = mean(features)
    threshold = mean_features #0
    features_quatized = []
    for i in features:
        if i >= threshold:
            features_quatized.append(1)
        else:
            features_quatized.append(0)
    return features_quatized  

def arr2str(arr):
    str_arr = ''
    for i in arr:
        str_arr += str(i)
    return str_arr

def str2arr(string):
    arr = []
    integer = int(string, 16)
    binary_string = format(integer, '0>42b')
    for i in binary_string:
        arr.append(int(i))
    return arr

def reconcile(A,B,n=255,k=128):
    A = hex(int(arr2str(A), 2))
    A = str(A[2:]) #A hex key
    B = hex(int(arr2str(B), 2))
    B = str(B[2:]) #B hex key
    coder = rs.RSCoder(n,k)
    Aencode = coder.encode(A) #Encode A
    Aparity = Aencode[k:] #A parity bits
    BAparity = B+Aparity #B key + A parity bits
    try:
        Bdecode = coder.decode(BAparity) #Decode B key + A parity bits
        Breconciled = Bdecode[0] #Reconcilieted key
    except:
        Breconciled = B
    return [A == Breconciled,Breconciled]

def reconciliation_rate(data,n,k):
    j = 0
    reconciliation_data1 = []
    reconciliation_data2 = []
    reconciliation_data3 = []
    reconciled_data = []
    pbar = tqdm(total = data.shape[0]/4+1)
    
    while j <= data.shape[0]-3:
        reconciliation_data1.append(reconcile(data[j],data[j+2],n,k))
        reconciliation_data2.append(reconcile(data[j],data[j+1],n,k))
        reconciliation_data3.append(reconcile(data[j+2],data[j+3],n,k))
        j = j + 4
        pbar.update(1)
        
    return reconciliation_data1, reconciliation_data2, reconciliation_data3

def privacyAmplification(data):
    # encode the string
    encoded_str = data.encode()
    # create sha3-256 hash objects
    obj_sha3_256 = hashlib.new("sha3_512", encoded_str)
    return(obj_sha3_256.hexdigest())

#https://github.com/InsaneMonster/NistRng/blob/master/benchmarks/numpy_rng_test.py
def NIST_RNG_test(data):
    #Eligible test from NIST-SP800-22r1a:
    #-monobit
    #-runs
    #-dft
    #-non_overlapping_template_matching
    #-approximate_entropy
    #-cumulative sums
    #-random_excursion
    #-random_excursion_variant
    eligible_battery: dict = check_eligibility_all_battery(np.array(data[0]), SP800_22R1A_BATTERY)
    num_packets = len(data)
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
    results_passed = {"Monobit" : [],"Frequency Within Block" : [],"Runs" : [],"Longest Run Ones In A Block" : [],
                      "Discrete Fourier Transform" : [],"Non Overlapping Template Matching" : [],"Serial" : [],
                      "Approximate Entropy" : [],"Cumulative Sums" : [],"Random Excursion" : [],
                      "Random Excursion Variant" : []}
    results_score = {"Monobit" : [],"Frequency Within Block" : [],"Runs" : [],"Longest Run Ones In A Block" : [],
                     "Discrete Fourier Transform" : [],"Non Overlapping Template Matching" : [],"Serial" : [],
                     "Approximate Entropy" : [],"Cumulative Sums" : [],"Random Excursion" : [],
                     "Random Excursion Variant" : []}
    data_results = []
    
    for i in range(0,num_packets):
        eligible_battery: dict = check_eligibility_all_battery(np.array(data[i]), SP800_22R1A_BATTERY)
        results = run_all_battery(np.array(data[i]), eligible_battery, False)
        data_results.append(results)
        for result, elapsed_time in results:
            score = np.round(result.score, 3)
            name = result.name
            results_score[name].append(score)
            if result.passed:
                passed = 1
            else:
                passed = 0
            results_passed[name].append(passed)
    
    for i in results_score:
        passing_score = sum(results_passed[i])/num_packets
        if round(passing_score) == 1:
            print("- PASSED ("+str(passing_score)+") - score: " + str(np.round(sum(results_score[i])/num_packets, 3)) + " - " + i)
        else:
            print("- FAILED ("+str(passing_score)+") - score: " + str(np.round(sum(results_score[i])/num_packets, 3)) + " - " + i)
            
    return results_score, results_passed
        
def KDR(A,B):
    kdr = np.bitwise_xor(A,B)
    kdr = np.sum(kdr)
    kdr = kdr/len(A)
    return kdr
    
def KDR_data(data):
    j = 0
    KDR_AB = []
    KDR_AC = []
    KDR_BC = []
    pbar = tqdm(total = data.shape[0]/4+1)
    while j <= data.shape[0]-3:
        KDR_AB.append(KDR(data[j],data[j+2]))
        KDR_AC.append(KDR(data[j],data[j+1]))
        KDR_BC.append(KDR(data[j+2],data[j+3]))
        j = j + 4
        pbar.update(1)
    return KDR_AB, KDR_AC, KDR_BC

def groupAverage(arr, n):
    result = []
    i=0
    while i <len(arr):
        sum_n = 0
        j = 0
        while j < n:
            sum_n = sum_n + arr[i]
            j = j + 1
        result.append(sum_n/n)
        i = i + n
    return result

def get_latest_results_file(results_directory, results_file):
    # The results file begins with the results_file name
    # Get the latest results file
    results_files = [f for f in os.listdir(results_directory) if f.endswith('.h5') and f.startswith(results_file)]
    results_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_directory, x)))
    return results_files[-1]

def extract_features_data(data, feature_extractor):
    ChannelSpectrogramObj = ChannelSpectrogram()
    data_spectrogram = ChannelSpectrogramObj.channel_spectrogram(np.array(data),256)
    features = feature_extractor.predict(data_spectrogram)
    return features

def quantize_data(features):
    quantized_data = []
    for i in features:
        features_quatized = feature_quantization(i)
        quantized_data.append(features_quatized)
    
    quantized_data = np.array(quantized_data)
    return quantized_data

def bdr_data(quantized_data):
    KDR_AB, KDR_AC, KDR_BC = KDR_data(quantized_data)
    KDR_AB_average = np.sum(KDR_AB)/(len(KDR_AB))
    KDR_AC_average = np.sum(KDR_AC)/(len(KDR_AC))
    KDR_BC_average = np.sum(KDR_BC)/(len(KDR_BC))
    return KDR_AB_average, KDR_AC_average, KDR_BC_average, KDR_AB, KDR_AC, KDR_BC
        
def test_model(feature_extractor_name, node_configurations, home="/home/Research/POWDER/", generate_results=True):
    dataset_name = node_configurations['dataset_name']
    repo_name = node_configurations['repo_name']
    node_Ids = node_configurations['node_Ids']
    config_name = node_configurations['config_name']

    extractor_name = feature_extractor_name.split("/")[-1]
    extractor_name = extractor_name.split(".")[0]
    test_configs_name = repo_name+"_"+dataset_name+"_"+str(node_Ids)+"_"+extractor_name
    results_directory = home+"Results/"
    results_file = "Results_"+test_configs_name

    # Load the dataset first and feed that to the model
    for idx, node_ids in enumerate(node_Ids):
        node_config_name = config_name+"-"+"".join(str(node) for node in node_ids)
        print("Config name: ", node_config_name)
        if idx == 0:
            dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
        else:
            dataset.add_dataset(dataset_name, node_config_name, repo_name)
    dataset.get_dataframe_Info()
    data, labels = dataset.load_data()
    
    if generate_results:
        # with tf.device('/CPU:0'):
        #     print("loading model")
        feature_extractor = load_model(feature_extractor_name)
        features = extract_features_data(data, feature_extractor)
        t_start = time.time()
        quantized_data = quantize_data(features)
        t_end = time.time()
        print("Time for quantization: ", t_end-t_start)
        quantized_data = quantized_data[:]
        avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, KDR_AB, KDR_AC, KDR_BC = bdr_data(quantized_data)
        print("Average KDR Alice-Bob: ", avg_KDR_AB)
        print("Average KDR Bob-Eve: ", avg_KDR_AC)
        print("Average KDR Alice-Eve: ", avg_KDR_BC)
        
        #Reconciliation
        k = int(512/4)
        k_n = 2
        n1 = int(k+(k/k_n)-1)
        s1 = int(n1-k)
        t_start = time.time()
        reconciliation11,reconciliation21,reconciliation31=reconciliation_rate(quantized_data,n1,k)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        k_n_2 = k_n**2
        n2 = int(k+(k/(k_n_2))-1)
        s2 = int(n2-k)
        t_start = time.time()
        reconciliation12,reconciliation22,reconciliation32=reconciliation_rate(quantized_data,n2,k)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        k_n_3 = k_n**3
        n3 = int(k+(k/(k_n_3))-1)
        s3 = int(n3-k)
        t_start = time.time()
        reconciliation13,reconciliation23,reconciliation33=reconciliation_rate(quantized_data,n3,k)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        #Privacy Amplification
        priv_amp_data = []
        i = 0
        rec_data1 = np.array(reconciliation11)
        t_start = time.time()
        while i < rec_data1.shape[0]:
            priv_amp = privacyAmplification(rec_data1[i][1])
            priv_amp_data.append(priv_amp)
            i = i+1
        t_end = time.time()
        print("Time for privacy amplification: ", t_end-t_start)
                
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with h5py.File(results_directory+results_file+"_"+time_stamp+".h5", "w") as f:
            f.create_dataset("features", data=features)
            f.create_dataset("quantized_data", data=quantized_data)
            f.create_dataset("KDR_AB", data=KDR_AB)
            f.create_dataset("KDR_AC", data=KDR_AC)
            f.create_dataset("KDR_BC", data=KDR_BC)
            f.create_dataset("reconciliation11", data=[reconciliation11[i][0] for i in range(len(reconciliation11))])
            f.create_dataset("reconciliation12", data=[reconciliation12[i][0] for i in range(len(reconciliation12))])
            f.create_dataset("reconciliation13", data=[reconciliation13[i][0] for i in range(len(reconciliation13))])
            f.create_dataset("reconciliation21", data=[reconciliation21[i][0] for i in range(len(reconciliation21))])
            f.create_dataset("reconciliation22", data=[reconciliation22[i][0] for i in range(len(reconciliation22))])
            f.create_dataset("reconciliation23", data=[reconciliation23[i][0] for i in range(len(reconciliation23))])
            f.create_dataset("reconciliation31", data=[reconciliation31[i][0] for i in range(len(reconciliation31))])
            f.create_dataset("reconciliation32", data=[reconciliation32[i][0] for i in range(len(reconciliation32))])
            f.create_dataset("reconciliation33", data=[reconciliation33[i][0] for i in range(len(reconciliation33))])
            f.create_dataset("priv_amp_data", data=priv_amp_data)
            # f.create_dataset("priv_amp_bin_data", data=priv_amp_bin_data)
            f.close()
            
    results_file = get_latest_results_file(results_directory, results_file)
    with h5py.File(results_directory+results_file, "r") as f:
        features = f["features"][:]
        quantized_data = f["quantized_data"][:]
        KDR_AB = f["KDR_AB"][:]
        KDR_AC = f["KDR_AC"][:]
        KDR_BC = f["KDR_BC"][:]
        reconciliation11 = f["reconciliation11"][:]
        reconciliation12 = f["reconciliation12"][:]
        reconciliation13 = f["reconciliation13"][:]
        reconciliation21 = f["reconciliation21"][:]
        reconciliation22 = f["reconciliation22"][:]
        reconciliation23 = f["reconciliation23"][:]
        reconciliation31 = f["reconciliation31"][:]
        reconciliation32 = f["reconciliation32"][:]
        reconciliation33 = f["reconciliation33"][:]
        priv_amp_data = f["priv_amp_data"][:]
        f.close()
    
    scenario = "Scenario 2"
    
    # Separate real and imaginary parts
    t = np.arange(len(data[0]))
    real_parts = np.real(data[0])
    imaginary_parts = np.imag(data[0])
        
    # Plot results
    plt.plot(t, real_parts, 'r', t, imaginary_parts, 'b')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Complex Data Plot')
    plt.legend(['Real Part', 'Imaginary Part'])
    plt.show()
    # Save the plot
    plt.savefig(results_directory+results_file+"_complex_data.png")
    
    # Make an analysis of the samples
    
    # Plot the entropy of the data
    entropy_data = []
    for i in features:
        entropy_data.append(entropy(i))
    plt.plot(entropy_data)
    plt.xlabel('Time')
    plt.ylabel('Entropy')
    plt.title('Entropy of the Data')
    plt.show()
    # Save the plot
    plt.savefig(results_directory+results_file+"_entropy_data.png")
    
    batch_size = 2
    KDR_AB_average_batch = groupAverage(KDR_AB, batch_size)
    KDR_AC_average_batch = groupAverage(KDR_AC, batch_size)
    KDR_BC_average_batch = groupAverage(KDR_BC, batch_size)

    plt.figure(figsize=[15,3])

    plt.subplot(1, 3,1)
    plt.plot(KDR_AB)
    # naming the x axis
    plt.xlabel('N Key Generated')
    # naming the y axis
    plt.ylabel('KDR')
    plt.ylim(-0.05, 1)
    plt.title("KDR Alice Bob")

    plt.subplot(1, 3,2)
    plt.plot(KDR_AC)
    # naming the x axis
    plt.xlabel('N Key Generated')
    # naming the y axis
    plt.ylabel('KDR')
    plt.ylim(-0.05, 1)
    plt.title("KDR Alice Eve")

    plt.subplot(1, 3,3)
    plt.plot(KDR_BC)
    # naming the x axis
    plt.xlabel('N Key Generated')
    # naming the y axis
    plt.ylabel('KDR')
    plt.ylim(-0.05, 1)
    plt.title("KDR Bob Eve")
    
    plt.show()
    # Save the plot
    plt.savefig(results_directory+results_file+"_KDR_data.png")

    # KDR Bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    AB = KDR_AB_average_batch
    AE = KDR_AC_average_batch
    BE = KDR_BC_average_batch
    
    # Set position of bar on X axis
    br1 = np.arange(len(AB))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, AB, color ='b', width = barWidth,
            edgecolor ='grey', label ='Alice-Bob')
    plt.bar(br2, AE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Alice-Eve')
    plt.bar(br3, BE, color ='m', width = barWidth,
            edgecolor ='grey', label ='Bob-Eve')
    
    # Adding Xticks
    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
    plt.ylabel('BDR', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(AB))],
        np.arange(1, len(AB)+1)*batch_size)

    plt.title("Bit Dissagreement Ratio "+scenario)
    plt.legend()
    plt.show()
    
    # save the plot
    plt.savefig(results_directory+results_file+"_KDR_data_bar.png")
    
    
    # # Reconciliation Rate Plots
    
    # k = int(512/4)
    # n1 = int(k+(k/1)-1)
    # s1 = int(n1-k)
    
    # n2 = int(k+(k/2)-1)
    # s2 = int(n2-k)
    
    # n3 = int(k+(k/3)-1)
    # s3 = int(n3-k)
    
    rec_rate_data11 = []
    rec_rate_data21 = []
    rec_rate_data31 = []

    rec_rate_data12 = []
    rec_rate_data22 = []
    rec_rate_data32 = []

    rec_rate_data13 = []
    rec_rate_data23 = []
    rec_rate_data33 = []

    i = 0
    while i < len(reconciliation11):
        rec_rate_data11.append(reconciliation11[i])
        rec_rate_data21.append(reconciliation21[i])
        rec_rate_data31.append(reconciliation31[i])
        
        rec_rate_data12.append(reconciliation12[i])
        rec_rate_data22.append(reconciliation22[i])
        rec_rate_data32.append(reconciliation32[i])
        
        rec_rate_data13.append(reconciliation13[i])
        rec_rate_data23.append(reconciliation23[i])
        rec_rate_data33.append(reconciliation33[i])
        i = i+1

    rec_rate_AB_average1 = np.sum(rec_rate_data11)/(len(rec_rate_data11))
    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average1)
    rec_rate_AC_average1 = np.sum(rec_rate_data21)/(len(rec_rate_data21))
    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average1)
    rec_rate_BC_average1 = np.sum(rec_rate_data31)/(len(rec_rate_data31))
    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average1)

    rec_rate_AB_average2 = np.sum(rec_rate_data12)/(len(rec_rate_data12))
    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average2)
    rec_rate_AC_average2 = np.sum(rec_rate_data22)/(len(rec_rate_data22))
    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average2)
    rec_rate_BC_average2 = np.sum(rec_rate_data32)/(len(rec_rate_data32))
    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average2)

    rec_rate_AB_average3 = np.sum(rec_rate_data13)/(len(rec_rate_data13))
    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average3)
    rec_rate_AC_average3 = np.sum(rec_rate_data23)/(len(rec_rate_data23))
    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average3)
    rec_rate_BC_average3 = np.sum(rec_rate_data33)/(len(rec_rate_data33))
    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average3)

    plt.figure(figsize=[15,3])
    plt.subplot(1, 3,1)
    plt.plot(rec_rate_data11)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Bob for RS("+str(n1)+","+str(k)+")")

    plt.subplot(1, 3,2)
    plt.plot(rec_rate_data21)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Eve for RS("+str(n1)+","+str(k)+")")

    plt.subplot(1, 3,3)
    plt.plot(rec_rate_data31)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Bob Eve for RS("+str(n1)+","+str(k)+")")

    plt.figure(figsize=[15,3])
    plt.subplot(1, 3,1)
    plt.plot(rec_rate_data12)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Bob for RS("+str(n2)+","+str(k)+")")

    plt.subplot(1, 3,2)
    plt.plot(rec_rate_data22)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Eve for RS("+str(n2)+","+str(k)+")")

    plt.subplot(1, 3,3)
    plt.plot(rec_rate_data32)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Bob Eve for RS("+str(n2)+","+str(k)+")")

    plt.figure(figsize=[15,3])
    plt.subplot(1, 3,1)
    plt.plot(rec_rate_data13)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Bob for RS("+str(n3)+","+str(k)+")")

    plt.subplot(1, 3,2)
    plt.plot(rec_rate_data23)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Alice Eve for RS("+str(n3)+","+str(k)+")")

    plt.subplot(1, 3,3)
    plt.plot(rec_rate_data33)
    # naming the x axis
    plt.xlabel('N Batch')
    # naming the y axis
    plt.ylabel('Reconciliation Rate')
    plt.ylim(-0.05, 1.05)
    plt.title("Reconciliation Rate Bob Eve for RS("+str(n3)+","+str(k)+")")

    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    AB = [rec_rate_AB_average1, rec_rate_AB_average2, rec_rate_AB_average3]
    AE = [rec_rate_AC_average1, rec_rate_AC_average2, rec_rate_AC_average3]
    BE = [rec_rate_BC_average1, rec_rate_BC_average2, rec_rate_BC_average3]

    # Set position of bar on X axis
    br1 = np.arange(len(AB))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, AB, color ='b', width = barWidth,
            edgecolor ='grey', label ='Alice-Bob')
    plt.bar(br2, AE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Alice-Eve')
    plt.bar(br3, BE, color ='m', width = barWidth,
            edgecolor ='grey', label ='Bob-Eve')
    
    # Adding Xticks
    plt.xlabel('RS(N,K)', fontweight ='bold', fontsize = 15)
    plt.ylabel('Reconciliation Success Rate', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(AB))],
            ["RS("+str(n1)+","+str(k)+")", "RS("+str(n2)+","+str(k)+")", "RS("+str(n3)+","+str(k)+")"])

    plt.title("Average Reconciliation Success Rate with variations on RS(N,K) - "+scenario, fontsize = 15)
    plt.legend()
    plt.show()
    # Save the plot
    plt.savefig(results_directory+results_file+"_reconciliation_rate_data_bar.png")

    # Reconciliation Bar 1
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    AB = groupAverage(rec_rate_data11,batch_size)
    AE = groupAverage(rec_rate_data21,batch_size)
    BE = groupAverage(rec_rate_data31,batch_size)
    
    # Set position of bar on X axis
    br1 = np.arange(len(AB))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, AB, color ='b', width = barWidth,
            edgecolor ='grey', label ='Alice-Bob')
    plt.bar(br2, AE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Alice-Eve')
    plt.bar(br3, BE, color ='m', width = barWidth,
            edgecolor ='grey', label ='Bob-Eve')
    
    # Adding Xticks
    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(AB))],
        np.arange(1, len(AB)+1)*batch_size)
    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

    plt.title("Reconciliation Success Rate "+scenario+" - RS("+str(n1)+","+str(k)+")")
    plt.legend()
    plt.show()
    
    plt.show()
    # Save the plot
    plt.savefig(results_directory+results_file+"_reconciliation_rate_data_1_bar.png")

    # Reconciliation Bar 2
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    AB = groupAverage(rec_rate_data12,batch_size)
    AE = groupAverage(rec_rate_data22,batch_size)
    BE = groupAverage(rec_rate_data32,batch_size)
    
    # Set position of bar on X axis
    br1 = np.arange(len(AB))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, AB, color ='b', width = barWidth,
            edgecolor ='grey', label ='Alice-Bob')
    plt.bar(br2, AE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Alice-Eve')
    plt.bar(br3, BE, color ='m', width = barWidth,
            edgecolor ='grey', label ='Bob-Eve')
    
    # Adding Xticks
    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(AB))],
        np.arange(1, len(AB)+1)*batch_size)
    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

    plt.title("Reconciliation Success Rate "+scenario+" - RS("+str(n2)+","+str(k)+")")
    plt.legend()
    plt.show()

    # Save the plot
    plt.savefig(results_directory+results_file+"_reconciliation_rate_data_2_bar.png")


    # Reconciliation Bar 3
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    AB = groupAverage(rec_rate_data13,1)
    AE = groupAverage(rec_rate_data23,1)
    BE = groupAverage(rec_rate_data33,1)
    
    # Set position of bar on X axis
    br1 = np.arange(len(AB))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    
    # Make the plot
    plt.bar(br1, AB, color ='b', width = barWidth,
            edgecolor ='grey', label ='Alice-Bob')
    plt.bar(br2, AE, color ='r', width = barWidth,
            edgecolor ='grey', label ='Alice-Eve')
    plt.bar(br3, BE, color ='m', width = barWidth,
            edgecolor ='grey', label ='Bob-Eve')
    
    # Adding Xticks
    plt.xlabel('Probe Number', fontweight ='bold', fontsize = 15)
    plt.ylabel('Reconciliation Success', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(AB))],
        np.arange(1, len(AB)+1)*batch_size)
    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

    plt.title("Reconciliation Success Rate "+scenario+" - RS("+str(n3)+","+str(k)+")")
    plt.legend()
    plt.show()
    
    # Save the plot
    plt.savefig(results_directory+results_file+"_reconciliation_rate_data_3_bar.png")

    #NIST Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
    priv_amp_bin_data = []
    for P_A_data in priv_amp_data:
        priv_amp_bin = str2arr(P_A_data)
        priv_amp_bin_data.append(priv_amp_bin)
    # print(priv_amp_bin_data[0])

    results_score, results_passed = NIST_RNG_test(priv_amp_bin_data)

def calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC):
    KDR_eve_avg = (avg_KDR_AC+avg_KDR_BC)/2
    eve_KDR_ratio = KDR_eve_avg/avg_KDR_AB
    return eve_KDR_ratio
    
def find_best_model(configuration, homeDir, node_configurations):
    
    batch_size = 128
    fft_len = 256
    patience = 50
    maxEpochs = 1000
    lr = 0.1
    val_size = 0.15
    factor = 0.5

    # Alpha:  1 Beta:  1 Gamma:  0.5
    alphas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # # alphas = [1]
    betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # betas = [1]
    gammas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # gammas = [0.5]
    optimizers = ["SGD"]
    
    # results = {"alpha":[],"beta":[],"gamma":[],"optimizer":[],"KDR_AB":[],"KDR_AC":[],"KDR_BC":[], "Models":[], "KDR_ratio":[]}
    # change results as a pandas dataframe
    results = pd.DataFrame(columns=["alpha","beta","gamma","optimizer","KDR_AB","KDR_AC","KDR_BC","KDR_ratio","Models"])
    # test only model and obtain BDR
    dataset_name = node_configurations['dataset_name']
    repo_name = node_configurations['repo_name']
    node_Ids = node_configurations['node_Ids']
    config_name = node_configurations['config_name']
    
    # Load the dataset first and feed that to the model
    for idx, node_ids in enumerate(node_Ids):
        node_config_name = config_name+"-"+"".join(str(node) for node in node_ids)
        print("Config name: ", node_config_name)
        if idx == 0:
            dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
        else:
            dataset.add_dataset(dataset_name, node_config_name, repo_name)
    dataset.get_dataframe_Info()
    data, _ = dataset.load_data()
    
    for a in alphas:
        for b in betas:
            for g in gammas:
                for o in optimizers:
                    # for file in os.listdir(modelsDir):
                    print("Alpha: ", a, "Beta: ", b, "Gamma: ", g, "Optimizer: ", o)
                    train_configurations = {
                        "QuadrupletNet": {
                            "alpha": a,
                            "beta": b,
                            "gamma": g,
                            "fft_len": fft_len,
                            "batch_size": batch_size,
                            "validation_size": val_size,
                            "LearningRate": lr,
                            "epochs": maxEpochs,
                            "patience": patience,
                            "factor": factor,
                            "optimizer": o
                        },
                        "TripletNet": {
                            "alpha": a,
                            "beta": b,
                            "fft_len": fft_len,
                            "batch_size": batch_size,
                            "validation_size": val_size,
                            "LearningRate": lr,
                            "epochs": maxEpochs,
                            "patience": patience,
                            "factor": factor,
                            "optimizer": o
                        }
                    }
                    model_type = "QuadrupletNet"
                    
                    filename_start = 'FeatureExtractor_'+str(train_configurations[model_type]["fft_len"]) \
                            +'_alpha'+str(train_configurations[model_type]["alpha"]) \
                            +'_beta'+str(train_configurations[model_type]["beta"]) \
                            +('_gamma'+str(train_configurations[model_type]["gamma"]) if "gamma" in train_configurations[model_type] else "") \
                            +'_'+train_configurations[model_type]['optimizer'] \
                            +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                            +'_'+configuration["config_name"]
                    
                    ModelsDir = homeDir+"Models/"
                    for file in os.listdir(ModelsDir):
                        if file.startswith(filename_start):
                            complete_filename = ModelsDir+file
                            feature_extractor = load_model(complete_filename)
                            features = extract_features_data(data, feature_extractor)
                            quantized_data = quantize_data(features)
                            quantized_data = quantized_data[:]
                            avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, _, _, _ = bdr_data(quantized_data)
                            print("Average KDR Alice-Bob:", avg_KDR_AB)
                            print("Average KDR Bob-Eve:", avg_KDR_AC)
                            print("Average KDR Alice-Eve:", avg_KDR_BC)
                            # if (avg_KDR_AC > 0.1 and avg_KDR_BC > 0.1):
                            KDR_ratio = calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC)
                            print("Average KDR ratio: ", KDR_ratio)
                            results = pd.concat([results, pd.DataFrame([{"alpha": a, "beta": b, "gamma": g, "optimizer": o, "KDR_AB": avg_KDR_AB, "KDR_AC": avg_KDR_AC, "KDR_BC": avg_KDR_BC, "KDR_ratio": KDR_ratio, "Models": file}])], ignore_index=True)
                            break        

    # Save the results dataframe in a file
    ResultsDir = homeDir+"Results/"
    results.to_csv(ResultsDir+"results_BDR_models.csv", index=False)
        
    KDR_ratios = results["KDR_ratio"]
    best_KDR_ratio_index = np.argmax(KDR_ratios)
    print(best_KDR_ratio_index)
    print("Best KDR ratio: ", KDR_ratios[best_KDR_ratio_index])
    print("Best average KDR_AB: ", results["KDR_AB"][best_KDR_ratio_index])
    print("Best average KDR_AC: ", results["KDR_AC"][best_KDR_ratio_index])
    print("Best average KDR_BC: ", results["KDR_BC"][best_KDR_ratio_index])
    print("Best KDR ratio index: ", best_KDR_ratio_index)
    print("Best model: ", results["Models"][best_KDR_ratio_index])
    
    return results["Models"][best_KDR_ratio_index]
    
if __name__ == "__main__":
    homeDir = "/home/Research/POWDER/"
    ModelsDir = homeDir+"Models/"
    ResultsDir = homeDir+"Results/"
    test_node_configurations = {
            'OTA-Lab': {
                'dataset_name': 'Key-Generation',
                'config_name': 'Sinusoid-Powder-OTA-Lab-Nodes',
                'repo_name': 'CAAI-FAU',
                'node_Ids': [
                    [1,2,3],
                    [1,4,5],
                    [1,4,8],
                    [2,4,3],
                    [4,2,5],
                    [4,2,8],
                    [4,8,5],
                    [5,7,8],
                    [5,8,7],
                    [8,4,1],
                    [8,5,1],
                    [8,5,4]
                ]
            },
            'OTA-Dense': {
                'dataset_name': 'Key-Generation',
                'config_name': 'Sinusoid-Powder-OTA-Dense-Nodes',
                'repo_name': 'CAAI-FAU',
                'node_Ids': [
                    # [1,2,3], 
                    [1,2,5], # Second scenario
                    # [1,3,2],
                    # [4,3,5] # First scenario
                ]
            }
        }
    # Get name of file from command line
    # If there is no command line argument, use the default model name
    if len(sys.argv) == 1:
        # model_name = "FeatureExtractor_512_alpha0.5_beta0.5_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_1758227515"
        # Use the model with the best validation loss
        test_type = "BDR" # "BDR", "loss"
        if test_type == "loss":
            saved_validation_loss = []
            saved_filename = []
            for file in os.listdir(ResultsDir):
                # If the file begins with History_ and ends with .h5, then add the validation loss open the file and get the validation loss
                if file.startswith("History_") and file.endswith(".h5"):
                    with h5py.File(ResultsDir+file, "r") as f:
                        saved_validation_loss.append(f["validation_loss"][-1])
                        saved_filename.append(file)
            # Sort the list by validation loss
            # Get the index of the best validation loss
            best_validation_loss_index = saved_validation_loss.index(min(saved_validation_loss))
            model_name = saved_filename[best_validation_loss_index]
            # Trim History_ from the filename
            model_name = model_name.split("History_")[1]
            # Trim .h5 from the filename
            print("Model name: ", model_name)
            print("Best validation loss: ", min(saved_validation_loss))
            # Get the filename with the best validation loss
            # model_name = saved_filename[-1]
        else:
            model_name = find_best_model(test_node_configurations['OTA-Lab'], homeDir, test_node_configurations['OTA-Dense'])
            # Copy the file as a best model under models directory
            shutil.copy(ModelsDir+model_name, ModelsDir+"Best_Model.h5")
    else:
        # model_name = sys.argv[1]
        #0.2,0.4,0.3,SGD
        alpha = sys.argv[1]
        beta = sys.argv[2]
        gamma = sys.argv[3]
        optimizer = sys.argv[4]
        model_name = "FeatureExtractor_256_alpha"+alpha+"_beta"+beta+"_gamma"+gamma+"_"+optimizer+"_lr0.1"
        # Find the file that starts with the model_name
        found = False
        print("Model name: ", model_name)
        for file in os.listdir(ModelsDir):
            if file.startswith(model_name):
                model_name = file
                found = True
                break
        if not found:
            print("Model not found")
            
            exit()
    
    # model_name = "FeatureExtractor_512_alpha0.5_beta0.5_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_1758225804"
    feature_extractor_name = ModelsDir+model_name
    test_model(feature_extractor_name, test_node_configurations['OTA-Dense'], home=homeDir, generate_results=True)