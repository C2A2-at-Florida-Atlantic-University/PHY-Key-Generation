import tensorflow as tf

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
    
def BDR_data(data):
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

if __name__ == "__main__":
    node_configurations = {
        'OTA-lab': {
            'dataset_name': 'Key-Generation',
            'config_name': 'Sinusoid-Powder-OTA-Lab-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': [
                # [1,2,3],
                # [1,4,5],
                # [1,4,8],
                # [2,4,3],
                # [4,2,5],
                # [4,2,8],
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
                # [1,2,5],
                # [1,3,2],
                [4,3,5]
            ]
        }
    }
    home = "/Users/josea/Workspaces/Powder/"
    models_directory = home+"Models/"
    feature_extractor_name = "QExtractor2_256_alpha0.5_beta0_batch64_val0.1_RMS0.1_DSsin2.4dev1278-800"
    
    configuration = node_configurations['OTA-Dense']
    dataset_name = configuration['dataset_name']
    repo_name = configuration['repo_name']
    node_Ids = configuration['node_Ids']
        
    node_config_name = repo_name+"_"+dataset_name+"_"+str(node_Ids)+"_"+feature_extractor_name
    results_directory = home+"Results/"
    results_file = "Results_"+node_config_name

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
    
    generate_results = False
    if generate_results:
        with tf.device('/CPU:0'):
            print("loading model")
            feature_extractor = load_model(models_directory+feature_extractor_name+".h5")
        #feature_extractor.summary()
        #plot_model(feature_extractor, to_file='featureExtractor.png', show_shapes=True, show_layer_names=True)
        
        ChannelSpectrogramObj = ChannelSpectrogram()
        # Load the classification dataset. (IQ samples and labels)
        # print(data[0])
        # Convert IQ samples to channel independent spectrograms. (classification data)
        t_start = time.time()
        print("converting IQ to spectrograms")
        data_spectrogram = ChannelSpectrogramObj.channel_spectrogram(np.array(data),256)
        t_end = time.time()
        print("Time for signal processing: ", t_end-t_start)
        # print(data_spectrogram.shape)
        t_start = time.time()
        # Extract RFFs from channel independent spectrograms.
        print("extracting features")
        features = feature_extractor.predict(data_spectrogram)
        t_end = time.time()
        print("Time for fingerprinting: ", t_end-t_start)
        del data_spectrogram
        #feature quantization
        quantized_data = []
        t_start = time.time()
        for i in features:
            features_quatized = feature_quantization(i)
            quantized_data.append(features_quatized)
        
        quantized_data = np.array(quantized_data)
        t_end = time.time()
        print("Time for quantization: ", t_end-t_start)
        
        #Bit Dissagreement Ratio
        
        quantized_data = quantized_data[:]
        
        KDR_AB, KDR_AC, KDR_BC = KDR_data(quantized_data)
        KDR_AB_average = np.sum(KDR_AB)/(len(KDR_AB))
        print("Average KDR Alice-Bob:", KDR_AB_average)
        KDR_AC_average = np.sum(KDR_AC)/(len(KDR_AC))
        print("Average KDR Alice-Eve:", KDR_AC_average)
        KDR_BC_average = np.sum(KDR_BC)/(len(KDR_BC))
        print("Average KDR Bob-Eve:", KDR_BC_average)
        
        #Reconciliation
        k = int(512/4)
        n1 = int(k+(k/1)-1)
        s1 = int(n1-k)
        t_start = time.time()
        reconciliation11,reconciliation21,reconciliation31=reconciliation_rate(quantized_data,n1,k)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        n2 = int(k+(k/2)-1)
        s2 = int(n2-k)
        t_start = time.time()
        reconciliation12,reconciliation22,reconciliation32=reconciliation_rate(quantized_data,n2,k)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        n3 = int(k+(k/3)-1)
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
    
    batch_size = 2
    KDR_AB_average_batch = groupAverage(KDR_AB, batch_size)
    KDR_AC_average_batch = groupAverage(KDR_AC, batch_size)
    KDR_BC_average_batch = groupAverage(KDR_BC, batch_size)

    plt.figure(figsize=[15,3])
    
    plot_titles = ["KDR Alice-Bob", "KDR Alice-Eve", "KDR Bob-Eve"]
    for i in range(len(KDR)):
        plt.subplot(1, 3,i+1)
        plt.plot(KDR[i])
        # naming the x axis
        plt.xlabel('N Key Generated')
        # naming the y axis
        plt.ylabel('KDR')
        plt.ylim(-0.05, 1)
        plt.title(plot_titles[i])

    # KDR Bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    
    # set height of bar
    AB = KDR_average_batch[0]
    AE = KDR_average_batch[1]
    BE = KDR_average_batch[2]
    
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

    plt.title("Bit Dissagreement Ratio Scenario")
    plt.legend()
    plt.show()
    
    # Reconciliation Rate Plots
    
    k = int(512/4)
    n1 = int(k+(k/1)-1)
    s1 = int(n1-k)
    
    n2 = int(k+(k/2)-1)
    s2 = int(n2-k)
    
    n3 = int(k+(k/3)-1)
    s3 = int(n3-k)
    
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

    rec_rate_AB_average2 = np.sum(rec_rate_data12)/(len(rec_rate_data12))
    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average2)
    rec_rate_AC_average2 = np.sum(rec_rate_data22)/(len(rec_rate_data22))
    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average2)
    rec_rate_BC_average2 = np.sum(rec_rate_data32)/(len(rec_rate_data32))
    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average2)
    
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

    rec_rate_AB_average3 = np.sum(rec_rate_data13)/(len(rec_rate_data13))
    print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average3)
    rec_rate_AC_average3 = np.sum(rec_rate_data23)/(len(rec_rate_data23))
    print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average3)
    rec_rate_BC_average3 = np.sum(rec_rate_data33)/(len(rec_rate_data33))
    print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average3)
    
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

    plt.title("Average Reconciliation Success Rate with variations on RS(N,K)", fontsize = 15)
    plt.legend()
    # plt.show()
    plt.savefig('Reconciliation.png')

    # Reconciliation Bar 1
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))

    batch_size = 2
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

    plt.title("Reconciliation Success Rate - RS("+str(n1)+","+str(k)+")")
    plt.legend()
    # plt.show()
    plt.savefig('Reconciliation_Success_Rate_RS_'+str(n1)+'_'+str(k)+'.png')

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

    plt.title("Reconciliation Success Rate Scenario 1 - RS("+str(n2)+","+str(k)+")")
    plt.legend()
    # plt.show()
    plt.savefig('Reconciliation_Success_Rate_RS_'+str(n2)+'_'+str(k)+'.png')
    
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
        np.arange(1, len(AB)+1)*32)
    plt.yticks([0,1],["Unsuccessful", "Successful"]) 

    plt.title("Reconciliation Success Rate Scenario 1 - RS("+str(n3)+","+str(k)+")")
    plt.legend()
    plt.show()
    
    
    #NIST Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
    priv_amp_bin_data = []
    for P_A_data in priv_amp_data:
        priv_amp_bin = str2arr(P_A_data)
        priv_amp_bin_data.append(priv_amp_bin)
    # print(priv_amp_bin_data[0])

    results_score, results_passed = NIST_RNG_test(priv_amp_bin_data)