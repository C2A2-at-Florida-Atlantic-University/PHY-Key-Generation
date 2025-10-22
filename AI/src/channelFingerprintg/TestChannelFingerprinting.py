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
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from nistrng import *
import hashlib
import time
from tensorflow.keras.models import load_model
import os
from scipy.stats import entropy
import h5py
import sys
from DatasetHandler import DatasetHandler, ChannelSpectrogram, ChannelIQ, ChannelPolar
from reconciliation import ReedSolomonReconciliation, BitToSymbolTransformation
from Quantization import Quantization

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

def extract_features_data(data, feature_extractor, fft_len, data_type="Spectrogram"):
    data = np.array(data)
    if data_type == "IQ":
        ChannelIQObj = ChannelIQ()
        data = ChannelIQObj.channel_iq(data)
    elif data_type == "Polar":
        ChannelPolarObj = ChannelPolar()
        data = ChannelPolarObj.channel_polar(data)
    elif data_type == "Spectrogram":
        ChannelSpectrogramObj = ChannelSpectrogram()
        data = ChannelSpectrogramObj.channel_spectrogram(
            data,
            fft_len
        )
    features = feature_extractor.predict(data)
    return features

def quantize_data(features, quantization_method={"type": "floating_point", "precision": 1}):
    quantization = Quantization()
    quantized_data = []
    for i in features:
        # features_quatized = feature_quantization(i)
        if quantization_method["type"] == "floating_point":
            features_quatized = quantization.floating_point_quantization(i, quantization_method["precision"])
        elif quantization_method["type"] == "mean":
            features_quatized = quantization.mean_quantization(i)
        else:
            if quantization_method["type"] != "threshold":
                print("Invalid quantization method: ", quantization_method["type"])
                print("Using threshold quantization with threshold 0.5")
                quantization_method["threshold"] = 0.5
            features_quatized = quantization.threshold_quantization(i, quantization_method["threshold"])
            
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
    data, _, _, _ = dataset.load_data()
    quantization_methods = [{"type": "floating_point", "precision": 1}, {"type": "mean"}, {"type": "threshold", "threshold": 0.5}]
    quantization_method = quantization_methods[0]
    if generate_results:
        # with tf.device('/CPU:0'):
        #     print("loading model")
        feature_extractor_path = home+"Models/"+feature_extractor_name
        feature_extractor = load_model(feature_extractor_path, custom_objects={"K": tf.keras.backend})
        # Infer fft_len from the model input shape if available
        try:
            inferred_fft_len = int(feature_extractor.input_shape[1]) if feature_extractor.input_shape is not None else 256
        except Exception:
            inferred_fft_len = 256
        features = extract_features_data(data, feature_extractor, inferred_fft_len)
        t_start = time.time()
        quantized_data = quantize_data(features, quantization_method)
        t_end = time.time()
        print("Time for quantization: ", t_end-t_start)
        quantized_data = quantized_data[:]
        avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, KDR_AB, KDR_AC, KDR_BC = bdr_data(quantized_data)
        print("Average KDR Alice-Bob: ", avg_KDR_AB)
        print("Average KDR Bob-Eve: ", avg_KDR_AC)
        print("Average KDR Alice-Eve: ", avg_KDR_BC)
        
        #Reconciliation
        L = data[0].shape[0]
        bits_per_symbol = 8
        K = int(L/bits_per_symbol)
        S = 1
        N = int(K + S)
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, N)
        k = int(L/bits_per_symbol)
        s1 = (2**1)-1
        n1 = int(k+s1)
        
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n1)
        t_start = time.time()
        reconciliation11,reconciliation21,reconciliation31=reconciliation.reconcile_rate(quantized_data)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        s2 = (2**2)-1
        n2 = int(k+s2)
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n2)
        t_start = time.time()
        reconciliation12,reconciliation22,reconciliation32=reconciliation.reconcile_rate(quantized_data)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        s3 = (2**3)-1
        n3 = int(k+s3)
        t_start = time.time()
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n3)
        reconciliation13,reconciliation23,reconciliation33=reconciliation.reconcile_rate(quantized_data)
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
    print("Results file: ", results_file)
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
    
    plotting = True
    if plotting:
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
    
    return 

def calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC):
    KDR_eve_avg = (avg_KDR_AC+avg_KDR_BC)/2
    eve_KDR_ratio = KDR_eve_avg/avg_KDR_AB
    return eve_KDR_ratio

def find_best_model(configuration, homeDir, node_configurations):
    
    fft_len = 256
    # lr = 0.1
    lr = 0.0001

    # Alpha:  1 Beta:  1 Gamma:  0.5
    alphas = [0.5]# [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # # alphas = [1]
    # betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    betas = [0.5]
    # gammas = []
    gammas = [0.1]
    optimizers = ["Adam"]
    
    
    network_types = ["RNN"] # "ResNet", "FeedForward", "RNN"
    FFT_lengths = [256] # [256, 512, 1024]
    output_lengths = [128, 256, 512] # [128, 256, 512]
    
    # results = {"alpha":[],"beta":[],"gamma":[],"optimizer":[],"KDR_AB":[],"KDR_AC":[],"KDR_BC":[], "Models":[], "KDR_ratio":[]}
    # change results as a pandas dataframe
    results = pd.DataFrame(columns=["alpha","beta","gamma","optimizer","L","data_type","quantization_method","KDR_AB","KDR_AC","KDR_BC","KDR_ratio","Models"])
    # test only model and obtain BDR
    results_reconciliation = pd.DataFrame(columns=["alpha","beta","optimizer","L","data_type","bps","K","S","N","quantization_method","rec_rate_AB","rec_rate_AC","rec_rate_BC","Models"])
    print("Node configurations: ", node_configurations)
    # Check keys of node_configurations
    print("Keys of node_configurations: ", node_configurations.keys())
    node_configs_names = ""
    idx = 0
    for dataset_type in node_configurations.keys():
        dataset_name = node_configurations[dataset_type]["dataset_name"]
        repo_name = node_configurations[dataset_type]["repo_name"]
        node_Ids = node_configurations[dataset_type]["node_Ids"]
        config_name = node_configurations[dataset_type]["config_name"]
        # Load the dataset first and feed that to the model
        # If node Ids is not empty, then load the dataset
        if len(node_Ids) > 0:
            node_configs_names += config_name if node_configs_names == "" else "_"+config_name
            for node_ids in node_Ids:
                node_names =  "-"+"".join(str(node) for node in node_ids)
                node_config_name = config_name+node_names
                node_configs_names += node_names
                print("Config name: ", node_config_name)
                if idx == 0:
                    dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
                    
                else:
                    dataset.add_dataset(dataset_name, node_config_name, repo_name)
                idx = idx + 1
    dataset.get_dataframe_Info()
    print("Node configs names: ", node_configs_names)
    REPRO_SEED = 42
    data, _, _, _ = dataset.load_data(shuffle=False, seed=REPRO_SEED)
    
    quantization_methods = [{"type": "floating_point", "precision": 1}, {"type": "mean"}, {"type": "threshold", "threshold": 0.5}]
    quantization_methods = quantization_methods[1:]
    data_types = ["IQ", "Polar", "Spectrogram"]
    # data_type = data_types[2]
    for data_type in data_types:
        for network_type in network_types:
            for fft_len in FFT_lengths:
                for output_length in output_lengths:
                    for a in alphas:
                        for b in betas:
                            for g in gammas:
                                for o in optimizers:
                                    for quantization_method in quantization_methods:
                                        # for file in os.listdir(modelsDir):
                                        print("Alpha: ", a, "Beta: ", b, "Gamma: ", g, "Optimizer: ", o)
                                        model_configurations = {
                                            "QuadrupletNet": {
                                                "alpha": a,
                                                "beta": b,
                                                "gamma": g,
                                                "fft_len": fft_len,
                                                "data_type": data_type,
                                                "output_length": output_length,
                                                "LearningRate": lr,
                                                "optimizer": o
                                            },
                                            "TripletNet": {
                                                "alpha": a,
                                                "beta": b,
                                                "fft_len": fft_len,
                                                "data_type": data_type,
                                                "LearningRate": lr,
                                                "optimizer": o
                                            }
                                        }
                                        model_type = "QuadrupletNet"
                                        
                                        filename_start = ((str(model_configurations[model_type]["data_type"])+'_') if model_configurations[model_type]["data_type"] != "Spectrogram" else '') \
                                                + 'FeatureExtractor_'+network_type+'_in'+str(model_configurations[model_type]["fft_len"]) \
                                                +'_out'+str(model_configurations[model_type]["output_length"]) \
                                                +'_alpha'+str(model_configurations[model_type]["alpha"]) \
                                                +'_beta'+str(model_configurations[model_type]["beta"]) \
                                                +('_gamma'+str(model_configurations[model_type]["gamma"]) if "gamma" in model_configurations[model_type] else "") \
                                                +'_'+model_configurations[model_type]['optimizer'] \
                                                +'_lr'+str(model_configurations[model_type]["LearningRate"]) \
                                                +'_'+configuration["config_name"]
                                        
                                        ModelsDir = homeDir+"Models/"
                                        fileFound = False
                                        for file in os.listdir(ModelsDir):
                                            if file.startswith(filename_start):
                                                print("Model found for filename: ", filename_start)
                                                fileFound = True
                                                complete_filename = ModelsDir+file
                                                feature_extractor = load_model(complete_filename, custom_objects={"K": tf.keras.backend})
                                                features = extract_features_data(data, feature_extractor, model_configurations[model_type]["fft_len"], model_configurations[model_type]["data_type"])
                                                quantized_data = quantize_data(features, quantization_method)
                                                quantized_data = quantized_data[:]
                                                avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, _, _, _ = bdr_data(quantized_data)
                                                print("Average KDR Alice-Bob:", avg_KDR_AB)
                                                print("Average KDR Bob-Eve:", avg_KDR_AC)
                                                print("Average KDR Alice-Eve:", avg_KDR_BC)
                                                # if (avg_KDR_AC > 0.1 and avg_KDR_BC > 0.1):
                                                KDR_ratio = calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC)
                                                print("Average KDR ratio: ", KDR_ratio)
                                                # reconcile data
                                                L = quantized_data[0].shape[0]
                                                
                                                bits_per_symbol = [2, 4, 8]
                                                b2s_transform = BitToSymbolTransformation()
                                                
                                                for bps in bits_per_symbol:
                                                    print("Bits per symbol: ", bps)
                                                    K = int(L/bps)
                                                    Ss = [(2**i)-1 for i in range(1, int(np.log2(K)+1))]
                                                    # Remove any values over 255
                                                    Ss = [S for S in Ss if S < 255]
                                                    for S in Ss:
                                                        N = int(K + S)
                                                        print("K: ", K, "S: ", S)
                                                        print("N: ", N)
                                                        reconciliation = ReedSolomonReconciliation(L, bps, K, N)
                                                        
                                                        quantizedSymbols = b2s_transform.group_b2s(quantized_data, bps)
                                                        
                                                        rec_AB, rec_AC, rec_BC = reconciliation.reconcile_rate(quantizedSymbols)
                                                        rec_rate_dataAB = []
                                                        rec_rate_dataAC = []
                                                        rec_rate_dataBC = []
                                                        i = 0
                                                        while i < len(rec_AB):
                                                            rec_rate_dataAB.append(rec_AB[i][0])
                                                            rec_rate_dataAC.append(rec_AC[i][0])
                                                            rec_rate_dataBC.append(rec_BC[i][0])
                                                            i = i+1
                                                        rec_rate_AB_average1 = np.sum(rec_rate_dataAB)/(len(rec_rate_dataAB))
                                                        print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average1)
                                                        rec_rate_AC_average1 = np.sum(rec_rate_dataAC)/(len(rec_rate_dataAC))
                                                        print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average1)
                                                        rec_rate_BC_average1 = np.sum(rec_rate_dataBC)/(len(rec_rate_dataBC))
                                                        print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average1)
                                                        results_reconciliation = pd.concat([results_reconciliation, pd.DataFrame([{"alpha": a, "beta": b, "optimizer": o, "L": output_length, "data_type": data_type, "bps": bps, "K": K, "S": S, "N": N, "rec_rate_AB": rec_rate_AB_average1, "rec_rate_AC": rec_rate_AC_average1, "rec_rate_BC": rec_rate_BC_average1, "Models": file}])], ignore_index=True)
                                                
                                                results = pd.concat([results, pd.DataFrame([{"alpha": a, "beta": b, "gamma": g, "optimizer": o, "L": output_length, "data_type": data_type, "quantization_method": quantization_method["type"], "KDR_AB": avg_KDR_AB, "KDR_AC": avg_KDR_AC, "KDR_BC": avg_KDR_BC, "KDR_ratio": KDR_ratio, "Models": file}])], ignore_index=True)
                                                break      
                                        if not fileFound:
                                            print("No model found for filename: ", filename_start)
                                            fileFound = False
                                        
    # Save the results dataframe in a file
    ResultsDir = homeDir+"Results/"
    results_file = "results_BDR_models_"+node_configs_names+"_RNN.csv"
    results.to_csv(ResultsDir+results_file, index=False)
    print("Results file: ", ResultsDir+results_file)
    reconciliation_file = "results_reconciliation_models_"+node_configs_names+"_RNN.csv"
    results_reconciliation.to_csv(ResultsDir+reconciliation_file, index=False)
    print("Reconciliation file: ", ResultsDir+reconciliation_file)
    # exit()
    KDR_ratios = results["KDR_ratio"]
    best_KDR_ratio_index = np.argmax(KDR_ratios)
    print(best_KDR_ratio_index)
    print("Best KDR ratio: ", KDR_ratios[best_KDR_ratio_index])
    print("Best average KDR_AB: ", results["KDR_AB"][best_KDR_ratio_index])
    print("Best average KDR_AC: ", results["KDR_AC"][best_KDR_ratio_index])
    print("Best average KDR_BC: ", results["KDR_BC"][best_KDR_ratio_index])
    print("Best KDR ratio index: ", best_KDR_ratio_index)
    print("Best model: ", results["Models"][best_KDR_ratio_index])
    
    # plot_bdr_results_CSV(ResultsDir, results_file)
    
    return results["Models"][best_KDR_ratio_index]

def plot_bdr_results_CSV(csv_file, title, save_path):
    """Create grouped bar charts of BDR vs key size L for each input type.

    For each input type (IQ, Polar, Spectrogram), generates a plot where:
      - X-axis: key size L
      - Y-axis: BDR (KDR_AB, KDR_AC, KDR_BC)
      - Within each L group, bars are grouped by metric color (AB/AC/BC) and
        subdivided by quantization method using distinct hatches.

    Saves three figures using save_path as a base, e.g.,
    base+'_IQ.png', base+'_Polar.png', base+'_Spectrogram.png'.
    """
    import os
    import numpy as np
    from matplotlib.patches import Patch

    results = pd.read_csv(csv_file)

    # Normalize data_type naming to a canonical set
    def normalize_dtype(x):
        if isinstance(x, str):
            xl = x.strip().lower()
            if xl == "iq":
                return "IQ"
            if xl == "polar":
                return "Polar"
            if xl in ("spectrogram", "spectogram"):
                return "Spectrogram"
        return x

    if "data_type" not in results.columns:
        raise ValueError("CSV missing required column 'data_type'.")
    results["data_type_norm"] = results["data_type"].apply(normalize_dtype)

    # Ensure L exists; if missing, attempt extracting from Models (pattern '_out{L}')
    if "L" not in results.columns or results["L"].dropna().empty:
        if "Models" in results.columns:
            extracted = results["Models"].str.extract(r"_out(\d+)")[0]
            results["L"] = pd.to_numeric(extracted, errors="coerce")
        else:
            raise ValueError("CSV missing 'L' and 'Models' to derive L.")
    else:
        results["L"] = pd.to_numeric(results["L"], errors="coerce")
    results = results.dropna(subset=["L"]).copy()
    results["L"] = results["L"].astype(int)

    # Required columns
    required_cols = {"quantization_method", "KDR_AB", "KDR_AC", "KDR_BC"}
    missing = required_cols.difference(results.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    metric_names = ["KDR_AB", "KDR_AC", "KDR_BC"]
    metric_labels = {"KDR_AB": "Alice-Bob", "KDR_AC": "Alice-Eve", "KDR_BC": "Bob-Eve"}
    metric_colors = {"KDR_AB": "#d62728", "KDR_AC": "#1f77b4", "KDR_BC": "#2ca02c"}
    hatch_styles = ["", "//", "x", "-", ".", "o", "*", "+", "O"]

    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".png"

    desired_types = ["IQ", "Polar", "Spectrogram"]
    present_types = [t for t in desired_types if t in results["data_type_norm"].unique().tolist()]
    print("Present types: ", present_types)
    for dtype in present_types:
        sub = results[results["data_type_norm"] == dtype].copy()
        if sub.empty:
            continue

        # Axis keys
        L_values = sorted(sub["L"].unique().tolist())
        quant_methods = list(dict.fromkeys(sub["quantization_method"].astype(str).tolist()))
        num_L = len(L_values)
        num_metrics = len(metric_names)
        num_quants = max(1, len(quant_methods))

        # Aggregate mean per L and quantization
        grouped = sub.groupby(["L", "quantization_method"], as_index=False)[metric_names].mean()
        hatch_map = {qm: hatch_styles[i % len(hatch_styles)] for i, qm in enumerate(quant_methods)}

        # Geometry
        x_base = np.arange(num_L)
        group_width = 0.8
        metric_slot_width = group_width / num_metrics
        bar_width = metric_slot_width / num_quants

        fig, ax = plt.subplots(figsize=(12, 6))
        for m_idx, metric in enumerate(metric_names):
            for q_idx, qname in enumerate(quant_methods):
                x_offsets = -group_width / 2 + m_idx * metric_slot_width + q_idx * bar_width + bar_width / 2
                xs = x_base + x_offsets
                heights = []
                for Lval in L_values:
                    row = grouped[(grouped["L"] == Lval) & (grouped["quantization_method"].astype(str) == qname)]
                    if not row.empty:
                        heights.append(float(row.iloc[0][metric]))
                    else:
                        heights.append(0.0)
                ax.bar(
                    xs,
                    heights,
                    width=bar_width,
                    color=metric_colors[metric],
                    hatch=hatch_map[qname],
                    edgecolor="black",
                    label=None,
                )

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(Lv) for Lv in L_values])
        ax.set_xlabel("Key Size L")
        ax.set_ylabel("BDR")
        plot_title = title if title else "BDR vs Key Size L"
        ax.set_title(f"{dtype} - {plot_title}")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        color_handles = [
            Patch(facecolor=metric_colors[m], edgecolor="black", label=metric_labels[m])
            for m in metric_names
        ]
        hatch_handles = [
            Patch(facecolor="#cccccc", edgecolor="black", hatch=hatch_map[q], label=str(q))
            for q in quant_methods
        ]
        legend1 = ax.legend(handles=color_handles, title="Node Pair", loc="upper left")
        ax.add_artist(legend1)
        ax.legend(handles=hatch_handles, title="Quantization", loc="upper right")

        plt.tight_layout()
        out_path = f"{base}_{dtype}{ext}"
        print("Saving BDR plot to: ", out_path)
        plt.savefig(out_path)
        plt.close(fig)

def plot_rec_rate_by_S(csv_path, title=None, save_path=None):
    """Plot rec_rate_AB vs S for each unique (L,bps,K) line from a results CSV.

    - X axis: S
    - Y axis: rec_rate_AB
    - One line per (L,bps,K)
    """
    df = pd.read_csv(csv_path)
    required_cols = {"S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Filter to spectrogram-only rows if data_type column exists
    if "data_type" in df.columns:
        mask = df["data_type"].astype(str).str.strip().str.lower().isin(["spectrogram", "spectogram"])
        df = df[mask].copy()
        if df.empty:
            raise ValueError("No rows with data_type equal to 'spectrogram'/'spectogram' found in the CSV.")

    # Ensure numeric types
    df["S"] = pd.to_numeric(df["S"], errors="coerce")
    df["rec_rate_AB"] = pd.to_numeric(df["rec_rate_AB"], errors="coerce")
    df["rec_rate_AC"] = pd.to_numeric(df["rec_rate_AC"], errors="coerce")
    df["rec_rate_BC"] = pd.to_numeric(df["rec_rate_BC"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["bps"] = pd.to_numeric(df["bps"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"]).copy()

    # Aggregate in case there are multiple rows per S within a (L,bps,K)
    agg = (
        df.groupby(["L", "bps", "K", "S"], as_index=False)[["rec_rate_AB", "rec_rate_AC", "rec_rate_BC"]].mean()
        .sort_values(["L", "bps", "K", "S"])  # sort for consistent lines
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    # Color per (L,bps,K), styles per pair
    from matplotlib import cm
    from matplotlib.lines import Line2D
    groups = agg[["L", "bps", "K"]].drop_duplicates().reset_index(drop=True)
    cmap = cm.get_cmap("tab20", len(groups))
    group_to_color = {tuple(groups.loc[i, ["L", "bps", "K"]].astype(int)): cmap(i) for i in range(len(groups))}

    pair_to_col = {
        "Alice-Bob": "rec_rate_AB",
        "Alice-Eve": "rec_rate_AC",
        "Bob-Eve": "rec_rate_BC",
    }
    pair_to_style = {"Alice-Bob": "-", "Alice-Eve": "--", "Bob-Eve": ":"}

    for (L_val, bps_val, K_val), sub in agg.groupby(["L", "bps", "K"], sort=False):
        color = group_to_color[(int(L_val), int(bps_val), int(K_val))]
        # Plot three lines with same color and different styles
        for pair, col_name in pair_to_col.items():
            label = f"L={int(L_val)}, bps={int(bps_val)}, K={int(K_val)}" if pair == "Alice-Bob" else None
            ax.plot(
                sub["S"].values,
                sub[col_name].values,
                marker="o" if pair == "Alice-Bob" else None,
                linestyle=pair_to_style[pair],
                color=color,
                label=label,
            )

    ax.set_xlabel("Parity Symbol Length S")
    ax.set_ylabel("Reconciliation Rate")
    # if title is None:
    title = "Reconciliation Rate for Parity Symbol Length S"
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    # First legend for groups (labels set only on AB lines)
    group_legend = ax.legend(loc="best", title="Groups (L,bps,K)")
    ax.add_artist(group_legend)
    # Second legend for pair linestyles
    style_handles = [
        Line2D([0], [0], color="black", linestyle=pair_to_style[pair], label=pair)
        for pair in ["Alice-Bob", "Alice-Eve", "Bob-Eve"]
    ]
    ax.legend(handles=style_handles, title="Node Pair", loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_rec_rate_by_S_for_L(csv_path, title=None, save_path=None, L=512):
    """Plot rec_rate_AB vs S for a single key size L from a results CSV.

    - X axis: S
    - Y axis: reconciliation rate
    - One line per (bps,K), with three linestyles for AB/AC/BC
    """
    df = pd.read_csv(csv_path)
    required_cols = {"S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    # Filter to spectrogram-only rows if data_type column exists
    if "data_type" in df.columns:
        mask = df["data_type"].astype(str).str.strip().str.lower().isin(["spectrogram", "spectogram"])
        df = df[mask].copy()
        if df.empty:
            raise ValueError("No rows with data_type equal to 'spectrogram'/'spectogram' found in the CSV.")

    # Ensure numeric types
    df["S"] = pd.to_numeric(df["S"], errors="coerce")
    df["rec_rate_AB"] = pd.to_numeric(df["rec_rate_AB"], errors="coerce")
    df["rec_rate_AC"] = pd.to_numeric(df["rec_rate_AC"], errors="coerce")
    df["rec_rate_BC"] = pd.to_numeric(df["rec_rate_BC"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")
    df["bps"] = pd.to_numeric(df["bps"], errors="coerce")
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df = df.dropna(subset=["S", "rec_rate_AB", "rec_rate_AC", "rec_rate_BC", "L", "bps", "K"]).copy()

    # Filter by the requested L
    target_L = int(L)
    df = df[df["L"] == target_L].copy()
    if df.empty:
        raise ValueError(f"No rows found for L={target_L} after filtering.")

    # Aggregate in case there are multiple rows per S within a (bps,K)
    agg = (
        df.groupby(["bps", "K", "S"], as_index=False)[["rec_rate_AB", "rec_rate_AC", "rec_rate_BC"]].mean()
        .sort_values(["bps", "K", "S"])  # sort for consistent lines
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    # Color per (bps,K), styles per pair
    from matplotlib import cm
    from matplotlib.lines import Line2D
    groups = agg[["bps", "K"]].drop_duplicates().reset_index(drop=True)
    cmap = cm.get_cmap("tab20", len(groups))
    group_to_color = {tuple(groups.loc[i, ["bps", "K"]].astype(int)): cmap(i) for i in range(len(groups))}

    pair_to_col = {
        "Alice-Bob": "rec_rate_AB",
        "Alice-Eve": "rec_rate_AC",
        "Bob-Eve": "rec_rate_BC",
    }
    pair_to_style = {"Alice-Bob": "-", "Alice-Eve": "--", "Bob-Eve": ":"}

    for (bps_val, K_val), sub in agg.groupby(["bps", "K"], sort=False):
        color = group_to_color[(int(bps_val), int(K_val))]
        # Plot three lines with same color and different styles
        for pair, col_name in pair_to_col.items():
            label = f"bps={int(bps_val)}, K={int(K_val)}" if pair == "Alice-Bob" else None
            ax.plot(
                sub["S"].values,
                sub[col_name].values,
                marker="o" if pair == "Alice-Bob" else None,
                linestyle=pair_to_style[pair],
                color=color,
                label=label,
            )

    ax.set_xlabel("Parity Symbol Length S")
    ax.set_ylabel("Reconciliation Rate")
    plot_title = title if title is not None else f"Reconciliation Rate for Parity Symbol Length S (L={target_L})"
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", alpha=0.4)
    # First legend for groups (labels set only on AB lines)
    group_legend = ax.legend(loc="best", title="Groups (bps,K)")
    ax.add_artist(group_legend)
    # Second legend for pair linestyles
    style_handles = [
        Line2D([0], [0], color="black", linestyle=pair_to_style[pair], label=pair)
        for pair in ["Alice-Bob", "Alice-Eve", "Bob-Eve"]
    ]
    ax.legend(handles=style_handles, title="Node Pair", loc="upper right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
if __name__ == "__main__":
    '''
    Results file:  /home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv
    Reconciliation file:  /home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv
    '''
    ResultsDir = "/home/Research/POWDER/Results/"
    results_file = "results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
    reconciliation_file = "results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
    plot_rec_rate_by_S(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate (A-B) vs S", save_path="rec_rate_by_S_OTA-All-123_RNN.png")
    plot_rec_rate_by_S_for_L(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate vs Parity Symbol Length S (L=512)", save_path="rec_rate_by_S_OTA-All-123_RNN_L512.png", L=512)
    plot_bdr_results_CSV(csv_file=ResultsDir+results_file, title="BDR vs key size L", save_path="BDR_vs_L_OTA-All-123_RNN.png")
    exit()
    
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
                    # [1,4,5],
                    # [1,4,8], # Fourth scenario
                    # [2,4,3],
                    # [4,2,5],
                    # [4,2,8],
                    # [4,8,5],
                    # [5,7,8],
                    # [5,8,7],
                    # [8,4,1],
                    # [8,5,1],
                    # [8,5,4]
                ]
            },
            'OTA-Dense': {
                'dataset_name': 'Key-Generation',
                'config_name': 'Sinusoid-Powder-OTA-Dense-Nodes',
                'repo_name': 'CAAI-FAU',
                'node_Ids': [
                    [1,2,3], # Third scenario
                    # [1,2,5], # Second scenario
                    # [1,3,2],
                    # [4,3,5] # First scenario
                ]
            }
        }
    test_node_configurations["OTA-All"] = {"OTA-Lab": test_node_configurations["OTA-Lab"], "OTA-Dense": test_node_configurations["OTA-Dense"]}
    print("Test node configurations: ", test_node_configurations["OTA-All"])
    # Get name of file from command line
    # If there is no command line argument, use the default model name
    data_collection_type_train = "OTA-Lab" # "OTA-Lab", "OTA-Dense"
    data_collection_type_test = "OTA-All" # "OTA-Lab", "OTA-Dense"
    # exit()
    if len(sys.argv) == 1:
        # model_name = "FeatureExtractor_512_alpha0.5_beta0.5_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_1758227515"
        # Use the model with the best validation loss
        test_type = "Rec" # "BDR", "loss", "plot_BDR"
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
        elif test_type == "plot_BDR":
            results_file = "results_BDR_models_OTA-Dense.csv" # "results_BDR_models_OTA-Lab.csv" or "results_BDR_models_OTA-Dense.csv"
            alpha = 0.5
            plot_bdr_results_CSV(ResultsDir, results_file, alpha)
        else:
            model_name = find_best_model(test_node_configurations[data_collection_type_train], homeDir, test_node_configurations[data_collection_type_test])
            
            # results_reconciliation_file = ResultsDir+"results_reconciliation_models_OTA-All-123_RNN.csv"
            # plot_rec_rate_by_S(csv_path=results_reconciliation_file, title="Reconciliation Rate (A-B) vs S", save_path="rec_rate_by_S_OTA-All-123_RNN.png")
            
            # results_file = ResultsDir+"results_BDR_models_OTA-All-123_RNN.csv"
            # plot_bdr_results_CSV(results_file, "BDR vs L", ResultsDir+"BDR_vs_L_OTA-All-123_RNN.png")
    
            # Copy the file as a best model under models directory
            # shutil.copy(ModelsDir+model_name, ModelsDir+"Best_Model.h5")
    else:
        # model_name = sys.argv[1]
        #0.2,0.4,0.3,SGD
        arg_idx = 1
        alpha = sys.argv[arg_idx]
        arg_idx += 1
        beta = sys.argv[arg_idx]
        arg_idx += 1
        # Check if next argument is a floating point number
        if sys.argv[arg_idx].replace('.', '', 1).isdigit():
            gamma = sys.argv[arg_idx]
            arg_idx += 1
        else:
            gamma = None
        optimizer = sys.argv[arg_idx]
        arg_idx += 1
        network_type = sys.argv[arg_idx]
        arg_idx += 1
        input_length = sys.argv[arg_idx]
        arg_idx += 1
        output_length = sys.argv[arg_idx]
        arg_idx += 1
        print("Input length: ", input_length)
        print("Output length: ", output_length)
        print("Alpha: ", alpha)
        print("Beta: ", beta)
        print("Gamma: ", gamma)
        print("Optimizer: ", optimizer)
        print("Network type: ", network_type)
        model_name = "FeatureExtractor_"+network_type \
                +"_in"+input_length+"_out"+output_length \
                +"_alpha"+alpha+"_beta"+beta+("_gamma"+gamma+"_" if gamma is not None else "_") \
                +optimizer+"_lr0.0001"
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
    # feature_extractor_name = ModelsDir+model_name
    # test_model(model_name, test_node_configurations[data_collection_type_test], home=homeDir, generate_results=True)