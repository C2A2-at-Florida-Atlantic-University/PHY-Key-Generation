import tensorflow as tf
import deep_learning_models  # ensure Lambda deserialization finds this module
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
import time
from tensorflow.keras.models import load_model
import os
from scipy.stats import entropy
import h5py
import sys
from DatasetHandler import DatasetHandler, ChannelSpectrogram, ChannelIQ, ChannelPolar
from reconciliation import ReedSolomonReconciliation, BitToSymbolTransformation
from Quantization import Quantization
from privacy_amplification import PrivacyAmplification
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


# Define the font size for the plot
title_font_size = 25
label_font_size = 25
legend_font_size = 16

# title_font_size = 18
# label_font_size = 18
# legend_font_size = 12

tick_font_size = 13

img_type = ".png" # png, eps, pdf

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

def KDR_data_rayTracing(data, dataRayTracing):
    KDR_AA = []
    KDR_BB = []
    i,j = 0,0
    print("Data shape: ", data.shape[0], "DataRayTracing shape: ", dataRayTracing.shape[0])
    while i <= data.shape[0]-3:
        KDR_AA.append(KDR(data[i], dataRayTracing[j]))
        KDR_BB.append(KDR(data[i+2], dataRayTracing[j+1]))
        j += 2
        i += 4
    return KDR_AA, KDR_BB

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
            # features_quatized = quantization.L2_threshold_quantization(i)
            
        quantized_data.append(features_quatized)
    
    quantized_data = np.array(quantized_data)
    return quantized_data

def avg_KDR_data(quantized_data):
    KDR_AB, KDR_AC, KDR_BC = KDR_data(quantized_data)
    KDR_AB_average = np.sum(KDR_AB)/(len(KDR_AB))
    KDR_AC_average = np.sum(KDR_AC)/(len(KDR_AC))
    KDR_BC_average = np.sum(KDR_BC)/(len(KDR_BC))
    return KDR_AB_average, KDR_AC_average, KDR_BC_average, KDR_AB, KDR_AC, KDR_BC

def avg_KDR_data_RayTracing(quantized_data, quantized_dataRayTracing):
    KDR_AA, KDR_BB = KDR_data_rayTracing(quantized_data, quantized_dataRayTracing)
    KDR_AA_average = np.sum(KDR_AA)/(len(KDR_AA))
    KDR_BB_average = np.sum(KDR_BB)/(len(KDR_BB))
    return KDR_AA_average, KDR_BB_average, KDR_AA, KDR_BB
        
def test_model(feature_extractor_name, node_configurations, home="/home/Research/POWDER/", generate_results=True):
    node_configs_names = ""
    idx = 0
    # for dataset_type in node_configurations.keys():
    #     dataset_name = node_configurations[dataset_type]["dataset_name"]
    #     repo_name = node_configurations[dataset_type]["repo_name"]
    #     node_Ids = node_configurations[dataset_type]["node_Ids"]
    #     config_name = node_configurations[dataset_type]["config_name"]
    #     # Load the dataset first and feed that to the model
    #     # If node Ids is not empty, then load the dataset
    #     if len(node_Ids) > 0:
    #         node_configs_names += config_name if node_configs_names == "" else "_"+config_name
    #         for node_ids in node_Ids:
    #             node_names =  "-"+"".join(str(node) for node in node_ids)
    #             node_config_name = config_name+node_names
    #             node_configs_names += node_names
    #             print("Config name: ", node_config_name)
    #             if idx == 0:
    #                 dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
    #             else:
    #                 dataset.add_dataset(dataset_name, node_config_name, repo_name)
    #             idx = idx + 1
    
    dataset_type = "OTA-Dense"  # "OTA-Lab", "OTA-Dense", "Sionna-Ray-Tracing"
    dataset_name = node_configurations[dataset_type]["dataset_name"]
    repo_name = node_configurations[dataset_type]['repo_name']
    node_Ids = node_configurations[dataset_type]['node_Ids']
    config_name = node_configurations[dataset_type]['config_name']

    extractor_name = feature_extractor_name.split("/")[-1]
    extractor_name = extractor_name.split(".")[0]
    test_configs_name = repo_name+"_"+dataset_name+"_"+str(node_Ids)+"_"+extractor_name
    results_directory = home+"Results/"
    results_file = "Results_"+test_configs_name

    # Load the dataset first and feed that to the model
    for idx, node_ids in enumerate(node_Ids):
        node_config_name = config_name+"-"+"".join(str(node) for node in node_ids)
        print("Config name: ", node_config_name)
        # Check if config contains "Sionna-Ray-Tracing"
        if idx == 0:
            dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
        else:
            dataset.add_dataset(dataset_name, node_config_name, repo_name)
    dataset.get_dataframe_Info()
    data, _, _, _ = dataset.load_data()
    
    min_num_samples = 8192
    # Resize the data in examples to the least number of samples
    data = np.array([example[-min_num_samples:] for example in data])
    
    # Load the Sionna-Ray-Tracing dataset
    dataRayTracing = False
    if dataRayTracing:
        print("Loading Sionna-Ray-Tracing dataset")
        dataset_type_rayTracing = "Sionna-Ray-Tracing"
        dataset_name_rayTracing = node_configurations[dataset_type_rayTracing]["dataset_name"]
        repo_name_rayTracing = node_configurations[dataset_type_rayTracing]["repo_name"]
        node_Ids_rayTracing = node_configurations[dataset_type_rayTracing]["node_Ids"]
        config_name_rayTracing = node_configurations[dataset_type_rayTracing]["config_name"]
        errs = 0
        for idx, node_ids in enumerate(node_Ids):
            # get the first two node ids
            alice_node = node_ids[0]
            bob_node = node_ids[1]
            rayTracingConfigName = config_name_rayTracing+"-"+str(alice_node)+str(bob_node)
            print("Config name: ", rayTracingConfigName)
            if idx-errs == 0:
                try:
                    datasetRayTracing = DatasetHandler(dataset_name_rayTracing, rayTracingConfigName, repo_name_rayTracing)
                except Exception:
                    print("Error loading Sionna-Ray-Tracing dataset: ", rayTracingConfigName)
                    errs += 1
                    continue
            else:
                try:
                    datasetRayTracing.add_dataset(dataset_name_rayTracing, rayTracingConfigName, repo_name_rayTracing)
                except Exception:
                    print("Error adding Sionna-Ray-Tracing dataset: ", rayTracingConfigName)
                    errs += 1
                    continue
        datasetRayTracing.get_dataframe_Info()
        dataRayTracing, _, _, _ = datasetRayTracing.load_data(shuffle=False)
        print("Sionna-Ray-Tracing dataset loaded successfully")
        # exit()
        
    quantization_methods = [{"type": "floating_point", "precision": 1}, {"type": "mean"}, {"type": "threshold", "threshold": 0.5}]
    quantization_method = quantization_methods[0]
    if generate_results:
        # with tf.device('/CPU:0'):
        #     print("loading model")
        feature_extractor_path = home+"Models/"+feature_extractor_name
        feature_extractor = load_model(
            feature_extractor_path,
            custom_objects={"K": tf.keras.backend, "tf": tf},
            compile=False
        )
        # Infer fft_len from the model input shape if available
        try:
            inferred_fft_len = int(feature_extractor.input_shape[1]) if feature_extractor.input_shape is not None else 256
        except Exception:
            inferred_fft_len = 256
        features = extract_features_data(data, feature_extractor, inferred_fft_len)
        if dataRayTracing:
            featuresRayTracing = extract_features_data(dataRayTracing, feature_extractor, inferred_fft_len)
        
        # =====================================================================
        # DIAGNOSTIC: Analyze raw embeddings before any test-time quantization
        # =====================================================================
        print("\n" + "="*70)
        print("EMBEDDING DIAGNOSTICS (before test-time quantization)")
        print("="*70)
        
        # Check if model already outputs binary (from STE quantization layer)
        unique_vals = np.unique(features[:10].flatten())
        print(f"Unique values in first 10 embeddings: {unique_vals[:20]}{'...' if len(unique_vals)>20 else ''}")
        print(f"Number of unique values: {len(unique_vals)}")
        
        # Check value distribution
        print(f"\nValue statistics:")
        print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}")
        print(f"  Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        # Check if embeddings are binary {0, 1}
        is_binary = np.allclose(unique_vals, np.array([0., 1.])) or len(unique_vals) <= 2
        print(f"  Embeddings appear binary: {is_binary}")
        
        # Check fraction of 1s vs 0s (should be ~50/50 for good entropy)
        ones_fraction = features.mean()
        print(f"  Fraction of 1s: {ones_fraction:.4f} (ideal: ~0.5)")
        
        # Check embedding diversity - are all embeddings the same?
        print(f"\nEmbedding diversity check:")
        sample_indices = [0, 1, 2, 3]  # Alice1, Eve1, Alice2, Eve2 from different quadruplets
        for i, idx in enumerate(sample_indices[:min(4, len(features))]):
            emb = features[idx]
            print(f"  Sample {idx}: first 20 bits = {emb[:20].astype(int)}")
        
        # Compute pairwise Hamming distance between first few samples
        print(f"\nPairwise KDR (raw embeddings, no test-time quantization):")
        if len(features) >= 4:
            # Assuming data layout: [Alice_AB, Alice_AE, Bob_AB, Bob_BE, ...]
            # or [A1, E1, B1, E2, A2, E3, B2, E4, ...]
            # Check KDR between consecutive pairs
            raw_kdr_01 = np.mean(np.abs(features[0] - features[1]))  # Diff XOR for binary
            raw_kdr_02 = np.mean(np.abs(features[0] - features[2]))
            raw_kdr_03 = np.mean(np.abs(features[0] - features[3]))
            print(f"  KDR(sample 0, sample 1): {raw_kdr_01:.4f}")
            print(f"  KDR(sample 0, sample 2): {raw_kdr_02:.4f}")
            print(f"  KDR(sample 0, sample 3): {raw_kdr_03:.4f}")
        
        # Check if model output is already binary - skip redundant quantization
        model_has_quantization = "QuantizationLayer" in feature_extractor_path or "Quantization" in feature_extractor_path
        print(f"\nModel appears to have built-in quantization: {model_has_quantization}")
        print("="*70 + "\n")
        
        quantization_method["precision"] = 1 # 1, 2, 4
        t_start = time.time()
        
        # If model already outputs binary, use threshold quantization to preserve it
        if is_binary or model_has_quantization:
            print("NOTE: Model outputs binary values, using threshold quantization (threshold=0.5)")
            quantized_data = quantize_data(features, {"type": "threshold", "threshold": 0.5})
        else:
            quantized_data = quantize_data(features, quantization_method)
        
        if dataRayTracing:
            if is_binary or model_has_quantization:
                quantized_dataRayTracing = quantize_data(featuresRayTracing, {"type": "threshold", "threshold": 0.5})
            else:
                quantized_dataRayTracing = quantize_data(featuresRayTracing, quantization_method)
        t_end = time.time()
        print("Time for quantization: ", t_end-t_start)
        quantized_data = quantized_data[:]
        avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, KDR_AB, KDR_AC, KDR_BC = avg_KDR_data(quantized_data)
        print("Average KDR Alice-Bob: ", avg_KDR_AB)
        print("Average KDR Bob-Eve: ", avg_KDR_AC)
        print("Average KDR Alice-Eve: ", avg_KDR_BC)
        
        if dataRayTracing:
            avg_KDR_AA_RayTracing, avg_KDR_BB_RayTracing, KDR_AA, KDR_BB =  avg_KDR_data_RayTracing(quantized_data,quantized_dataRayTracing)
            print("Average KDR Alice-Eve (Ray Tracing): ", avg_KDR_AA_RayTracing)
            print("Average KDR Bob-Eve (Ray Tracing): ", avg_KDR_BB_RayTracing)
        exit()
        #Reconciliation
        L = quantized_data[0].shape[0]
        bits_per_symbol = 8
        K = int(L/bits_per_symbol)
        S = 31
        N = int(K + S)
        # N = 255
        # reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, N)
        k = int(L/bits_per_symbol)
        # s1 = S
        S = N-K
        n1 = int(k+S)
        
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n1)
        t_start = time.time()
        reconciliation11,reconciliation21,reconciliation31=reconciliation.reconcile_rate(quantized_data)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        print("Reconciliation results: ", reconciliation11[0])
        # exit()
        # Get only keys that were successfully reconciled
        success_keys = []
        print("reconciliation11[0]: ", reconciliation11[0])
        for i in range(len(reconciliation11)):
            if reconciliation11[i][0] == True:
                success_keys.append(reconciliation11[i][1])
        print("L=", L, "K=", K, "S=", S, "N=", n1)
        print("Number of success keys: ", len(success_keys))
        print("Percentage of success keys: ", len(success_keys)/len(reconciliation11)*100, "%")
        exit()
        # Ampplification and NIST Test Suite shortcut
        #Privacy Amplification
        print("Privacy Amplification and NIST Test Suite shortcut")
        # exit()
        privacy_amplification = PrivacyAmplification()
        priv_amp_data = []
        i = 0
        rec_data1 = np.array(reconciliation11)
        t_start = time.time()
        while i < rec_data1.shape[0]:
            priv_amp = privacy_amplification.privacyAmplification(rec_data1[i][1])
            priv_amp_data.append(priv_amp)
            i = i+1
        t_end = time.time()
        print("Time for privacy amplification: ", t_end-t_start)
        
        #NIST Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
        priv_amp_bin_data = []
        for P_A_data in priv_amp_data:
            priv_amp_bin = privacy_amplification.str2arr(P_A_data)
            priv_amp_bin_data.append(priv_amp_bin)

        # results_score, results_passed = privacy_amplification.NIST_RNG_test(priv_amp_bin_data)
        # Test a single key for NIST Test Suite
        key = priv_amp_bin_data[-2]
        results_score, results_passed = privacy_amplification.NIST_RNG_test([key])
        def checkresultsPassed(results_passed, results_score, measure="score"):
            tests_considered = ["Monobit", 
                                "Frequency Within Block", 
                                "Runs", 
                                "Longest Run Ones In A Block", 
                                "Discrete Fourier Transform", 
                                "Non Overlapping Template Matching", 
                                "Serial", 
                                "Approximate Entropy", 
                                "Cumulative Sums", 
                                # Random Excursion,
                                # "Random Excursion Variant"
                                ]
            for test in tests_considered:
                print(test, results_passed[test], results_score[test])
                if measure == "score":
                    if results_score[test][0] < 0.01:
                        return False
                elif measure == "passed":
                    if results_passed[test][0] == False:
                        return False
            return True
        results_passed = checkresultsPassed(results_passed, results_score, measure="score")
        Key_passed = results_passed
        print("Key passed: ", Key_passed)
        print("Results score: ", results_score)
        print("Results passed: ", results_passed)
        # exit()
        # Find how many keys passed the NIST Test Suite
        num_keys_passed = 0
        for key in priv_amp_bin_data:
            results_score, results_passed = privacy_amplification.NIST_RNG_test([key])
            if checkresultsPassed(results_passed, results_score, measure="score"):
                num_keys_passed += 1
        # print("Number of keys passed the NIST Test Suite: ", num_keys_passed)
        print("Percentage of keys passed the NIST Test Suite: ", num_keys_passed/len(priv_amp_bin_data)*100)
        
        exit()
        
        s2 = 15
        n2 = int(k+s2)
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n2)
        t_start = time.time()
        reconciliation12,reconciliation22,reconciliation32=reconciliation.reconcile_rate(quantized_data)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        s3 = 7
        n3 = int(k+s3)
        t_start = time.time()
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, K, n3)
        reconciliation13,reconciliation23,reconciliation33=reconciliation.reconcile_rate(quantized_data)
        t_end = time.time()
        print("Time for reconciliation (Alice,Bob,Eve): ", t_end-t_start)
        
        #Privacy Amplification
        privacy_amplification = PrivacyAmplification()
        priv_amp_data = []
        i = 0
        rec_data1 = np.array(reconciliation11)
        t_start = time.time()
        while i < rec_data1.shape[0]:
            priv_amp = privacy_amplification.privacyAmplification(rec_data1[i][1])
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
        plt.savefig(results_directory+results_file+"_complex_data"+img_type)
        
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
        plt.savefig(results_directory+results_file+"_entropy_data"+img_type)
        
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
        plt.savefig(results_directory+results_file+"_KDR_data"+img_type)

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
        plt.savefig(results_directory+results_file+"_KDR_data_bar"+img_type)
        
        
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
        plt.savefig(results_directory+results_file+"_reconciliation_rate_data_bar"+img_type)

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
        plt.savefig(results_directory+results_file+"_reconciliation_rate_data_1_bar"+img_type)

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
        plt.savefig(results_directory+results_file+"_reconciliation_rate_data_2_bar"+img_type)

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
        plt.savefig(results_directory+results_file+"_reconciliation_rate_data_3_bar"+img_type)

        #NIST Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
        priv_amp_bin_data = []
        for P_A_data in priv_amp_data:
            priv_amp_bin = privacy_amplification.str2arr(P_A_data)
            priv_amp_bin_data.append(priv_amp_bin)
        # print(priv_amp_bin_data[0])

        results_score, results_passed = privacy_amplification.NIST_RNG_test(priv_amp_bin_data)
    
    return 

def calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC):
    KDR_eve_avg = (avg_KDR_AC+avg_KDR_BC)/2
    eve_KDR_ratio = KDR_eve_avg/avg_KDR_AB
    return eve_KDR_ratio

def find_best_model(configuration, homeDir, node_configurations):

    # Alpha:  1 Beta:  1 Gamma:  0.5
    alphas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    betas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    gammas = [0.1]
    optimizers = ["RMSprop"] # "Adam", "SGD", "RMSprop"
    
    network_types = ["RNN"] # "ResNet", "FeedForward", "RNN"
    FFT_lengths = [256] # [256, 512, 1024]
    output_lengths = [128] # [128, 256, 512]
    
    # change results as a pandas dataframe
    # test only model and obtain BDR
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
                # Skip if config name contains "Sionna-Ray-Tracing"
                if "Sionna-Ray-Tracing" in node_config_name:
                    continue
                if idx == 0:
                    dataset = DatasetHandler(dataset_name, node_config_name, repo_name)
                    
                else:
                    dataset.add_dataset(dataset_name, node_config_name, repo_name)
                idx = idx + 1
        else:
            print("No node IDs found for dataset: ", dataset_type)
    dataset.get_dataframe_Info()
    print("Node configs names: ", node_configs_names)
    # print("Dataset names: ", dataset_names)
    # exit()
    REPRO_SEED = 42
    data, _, _, _ = dataset.load_data(shuffle=False, seed=REPRO_SEED)
    precision_levels = [1,2,4]
    quantization_methods = [{"type": "floating_point", "precision": precision_levels[0]}, {"type": "mean"}, {"type": "threshold", "threshold": 0.5}]
    quantization_methods = quantization_methods[0:1]
    # data_types = ["IQ", "Polar", "Spectrogram"]
    data_types = ["Spectrogram"]
    # data_type = data_types[2]
    EveRayTracing = False
    if EveRayTracing:
        results = pd.DataFrame(columns=["alpha","beta","gamma","optimizer","L","data_type","quantization_method","KDR_AB","KDR_AC","KDR_BC","KDR_AA","KDR_BB","KDR_ratio","Models"])
        results_reconciliation = pd.DataFrame(columns=["alpha","beta","optimizer","L","data_type","bps","K","S","N","quantization_method","rec_rate_AB","rec_rate_AC","rec_rate_BC","rec_rate_AA","rec_rate_BB","Models"])
    else:
        results = pd.DataFrame(columns=["alpha","beta","optimizer","L","data_type","quantization_method","KDR_AB","KDR_AC","KDR_BC","KDR_ratio","Models"])
        results_reconciliation = pd.DataFrame(columns=["alpha","beta","optimizer","L","data_type","bps","K","S","N","quantization_method","rec_rate_AB","rec_rate_AC","rec_rate_BC","Models"])


    if EveRayTracing:
        print("Loading Sionna-Ray-Tracing dataset")
        dataset_type_rayTracing = "Sionna-Ray-Tracing"
        dataset_name_rayTracing = node_configurations[dataset_type_rayTracing]["dataset_name"]
        repo_name_rayTracing = node_configurations[dataset_type_rayTracing]["repo_name"]
        node_Ids_rayTracing = node_configurations[dataset_type_rayTracing]["node_Ids"]
        config_name_rayTracing = node_configurations[dataset_type_rayTracing]["config_name"]
        errs = 0
        for idx, node_ids in enumerate(node_Ids):
            # get the first two node ids
            alice_node = node_ids[0]
            bob_node = node_ids[1]
            rayTracingConfigName = config_name_rayTracing+"-"+str(alice_node)+str(bob_node)
            print("Config name: ", rayTracingConfigName)
            if idx-errs == 0:
                try:
                    datasetRayTracing = DatasetHandler(dataset_name_rayTracing, rayTracingConfigName, repo_name_rayTracing)
                except Exception:
                    print("Error loading Sionna-Ray-Tracing dataset: ", rayTracingConfigName)
                    errs += 1
                    continue
            else:
                try:
                    datasetRayTracing.add_dataset(dataset_name_rayTracing, rayTracingConfigName, repo_name_rayTracing)
                except Exception:
                    print("Error adding Sionna-Ray-Tracing dataset: ", rayTracingConfigName)
                    errs += 1
                    continue
        # datasetRayTracing.get_dataframe_Info()
        dataRayTracing, _, _, _ = datasetRayTracing.load_data(shuffle=False)
        print("Sionna-Ray-Tracing dataset loaded successfully")
        # exit()
    for data_type in data_types:
        for network_type in network_types:
            for fft_len in FFT_lengths:
                for output_length in output_lengths:
                    for a in alphas:
                        for b in betas:
                            for g in gammas:
                                for o in optimizers:
                                    lr = 0.1 if o == "SGD" else 0.0001
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
                                        
                                        filename_start = model_configurations[model_type]["data_type"] \
                                                + '_FeatureExtractor_'+network_type+'_in'+str(model_configurations[model_type]["fft_len"]) \
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
                                                if quantization_method["type"] == "floating_point":
                                                    for precision in precision_levels:
                                                        print("Precision level: ", precision)
                                                        quantization_method["precision"] = precision
                                                        print("Model found for filename: ", filename_start)
                                                        fileFound = True
                                                        complete_filename = ModelsDir+file
                                                        feature_extractor = load_model(
                                                            complete_filename,
                                                            custom_objects={"K": tf.keras.backend, "tf": tf},
                                                            compile=False
                                                        )
                                                        features = extract_features_data(data, feature_extractor, model_configurations[model_type]["fft_len"], model_configurations[model_type]["data_type"])
                                                        quantized_data = quantize_data(features, quantization_method)
                                                        quantized_data = quantized_data[:]
                                                        avg_KDR_AB, avg_KDR_AC, avg_KDR_BC, _, _, _ = avg_KDR_data(quantized_data)
                                                        print("Average KDR Alice-Bob:", avg_KDR_AB)
                                                        print("Average KDR Bob-Eve:", avg_KDR_AC)
                                                        print("Average KDR Alice-Eve:", avg_KDR_BC)
                                                        # if (avg_KDR_AC > 0.1 and avg_KDR_BC > 0.1):
                                                        KDR_ratio = calculate_KDR_ratio(avg_KDR_AB, avg_KDR_AC, avg_KDR_BC)
                                                        print("Average KDR ratio: ", KDR_ratio)
                                                        if EveRayTracing:
                                                            featuresRayTracing = extract_features_data(dataRayTracing, feature_extractor, model_configurations[model_type]["fft_len"], model_configurations[model_type]["data_type"])
                                                            quantized_dataRayTracing = quantize_data(featuresRayTracing, quantization_method)
                                                            quantized_dataRayTracing = quantized_dataRayTracing[:]
                                                            avg_KDR_AA, avg_KDR_BB, _, _ = avg_KDR_data_RayTracing(quantized_data, quantized_dataRayTracing)
                                                            print("Average KDR Alice-Eve (Ray Tracing): ", avg_KDR_AA)
                                                            print("Average KDR Bob-Eve (Ray Tracing): ", avg_KDR_BB)
                                                        # reconcile data
                                                        L = quantized_data[0].shape[0]
                                                        bits_per_symbol = [8]
                                                        b2s_transform = BitToSymbolTransformation()
                                                        print("Bits per symbol: ", bits_per_symbol)
                                                        K2N_Selection = {16:31, 32:63, 64:127}
                                                        for bps in bits_per_symbol:
                                                            print("Bits per symbol: ", bps)
                                                            K = int(L/bps)
                                                            # Ss = [(2**i)-1 for i in range(1, int(np.log2(K)+1))]
                                                            # Remove any values over 255
                                                            # Ss = [S for S in Ss if S < 255]
                                                            # for S in Ss:
                                                            # N = int(K + S)
                                                            # Select N for the given K from the K2N_Selection
                                                            N = K2N_Selection[K]
                                                            S = N-K
                                                            print("K: ", K, "S: ", S)
                                                            print("N: ", N)
                                                            reconciliation = ReedSolomonReconciliation(L, bps, K, N)
                                                            
                                                            quantizedSymbols = b2s_transform.group_b2s(quantized_data, bps)
                                                            
                                                            if EveRayTracing:
                                                                rec_AB, rec_AC, rec_BC, rec_AA, rec_BB = reconciliation.reconcile_rate(quantizedSymbols, quantized_dataRayTracing)
                                                            else:
                                                                rec_AB, rec_AC, rec_BC, _, _ = reconciliation.reconcile_rate(quantizedSymbols)
                                                            rec_rate_dataAB = []
                                                            rec_rate_dataAC = []
                                                            rec_rate_dataBC = []
                                                            if EveRayTracing:
                                                                rec_rate_dataAA = []
                                                                rec_rate_dataBB = []
                                                            i = 0
                                                            while i < len(rec_AB):
                                                                rec_rate_dataAB.append(rec_AB[i][0])
                                                                rec_rate_dataAC.append(rec_AC[i][0])
                                                                rec_rate_dataBC.append(rec_BC[i][0])
                                                                if EveRayTracing:
                                                                    rec_rate_dataAA.append(rec_AA[i][0])
                                                                    rec_rate_dataBB.append(rec_BB[i][0])
                                                                i = i+1
                                                            rec_rate_AB_average1 = np.sum(rec_rate_dataAB)/(len(rec_rate_dataAB))
                                                            print("Average reconciliation rate Alice-Bob:", rec_rate_AB_average1)
                                                            rec_rate_AC_average1 = np.sum(rec_rate_dataAC)/(len(rec_rate_dataAC))
                                                            print("Average reconciliation rate Alice-Eve:", rec_rate_AC_average1)
                                                            rec_rate_BC_average1 = np.sum(rec_rate_dataBC)/(len(rec_rate_dataBC))
                                                            print("Average reconciliation rate Bob-Eve:", rec_rate_BC_average1)
                                                            if EveRayTracing:
                                                                rec_rate_AA_average1 = np.sum(rec_rate_dataAA)/(len(rec_rate_dataAA))
                                                                print("Average reconciliation rate Alice-Alice:", rec_rate_AA_average1)
                                                                rec_rate_BB_average1 = np.sum(rec_rate_dataBB)/(len(rec_rate_dataBB))
                                                                print("Average reconciliation rate Bob-Bob:", rec_rate_BB_average1)
                                                                results_dataframe_reconciliation = pd.DataFrame([{"alpha": a, "beta": b, "optimizer": o, 
                                                                                                                  "L": L, "data_type": data_type, "bps": bps,
                                                                                                                  "K": K, "S": S, "N": N, "quantization_method": quantization_method["type"], 
                                                                                                                  "rec_rate_AB": rec_rate_AB_average1, "rec_rate_AC": rec_rate_AC_average1, 
                                                                                                                  "rec_rate_BC": rec_rate_BC_average1, "rec_rate_AA": rec_rate_AA_average1, 
                                                                                                                  "rec_rate_BB": rec_rate_BB_average1, "Models": file}])
                                                            else:
                                                                results_dataframe_reconciliation = pd.DataFrame([{"alpha": a, "beta": b, "optimizer": o, 
                                                                                                                  "L": L, "data_type": data_type, "bps": bps, 
                                                                                                                  "K": K, "S": S, "N": N, "quantization_method": quantization_method["type"], 
                                                                                                                  "rec_rate_AB": rec_rate_AB_average1, "rec_rate_AC": rec_rate_AC_average1, 
                                                                                                                  "rec_rate_BC": rec_rate_BC_average1, "Models": file}])
                                                            results_reconciliation = pd.concat([results_reconciliation, results_dataframe_reconciliation], ignore_index=True)
                                                            
                                                        if EveRayTracing:
                                                            results_dataframe = pd.DataFrame([{"alpha": a, "beta": b, "gamma": g, 
                                                                                                "optimizer": o, "L": L, "data_type": data_type, 
                                                                                                "quantization_method": quantization_method["type"], 
                                                                                                "KDR_AB": avg_KDR_AB, "KDR_AC": avg_KDR_AC, "KDR_BC": avg_KDR_BC, 
                                                                                                "KDR_AA": avg_KDR_AA, "KDR_BB": avg_KDR_BB,
                                                                                                "KDR_ratio": KDR_ratio, "Models": file}])
                                                        else:
                                                            results_dataframe = pd.DataFrame([{"alpha": a, "beta": b, "gamma": g, 
                                                                                                "optimizer": o, "L": L, "data_type": data_type, 
                                                                                                "quantization_method": quantization_method["type"], 
                                                                                                "KDR_AB": avg_KDR_AB, "KDR_AC": avg_KDR_AC, "KDR_BC": avg_KDR_BC, 
                                                                                                "KDR_ratio": KDR_ratio, "Models": file}])
                                                        
                                                        results = pd.concat([results, results_dataframe], ignore_index=True)
                                                    break      
                                        if not fileFound:
                                            print("No model found for filename: ", filename_start)
                                            fileFound = False
                                        
    # Save the results dataframe in a file
    ResultsDir = homeDir+"Results/"
    model_type = "".join([network_type+"_" for network_type in network_types])[:-1]
    print("Model type: ", model_type)
    if EveRayTracing:
        results_file = "results_BDR_models_"+node_configs_names+"_"+model_type+"_RayTracing.csv"
    else:
        results_file = "results_BDR_models_"+node_configs_names+"_"+model_type+".csv"
    results.to_csv(ResultsDir+results_file, index=False)
    print("Results file: ", ResultsDir+results_file)
    if EveRayTracing:
        reconciliation_file = "results_reconciliation_models_"+node_configs_names+"_"+model_type+"_RayTracing.csv"
    else:
        reconciliation_file = "results_reconciliation_models_"+node_configs_names+"_"+model_type+".csv"
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
    
    plot_bdr_results_CSV(csv_file=ResultsDir+results_file, title="BDR vs key size L", save_path="BDR_vs_L_"+node_configs_names+"_"+model_type+img_type)
    
    plot_rec_rate_by_S(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate (A-B) vs S", save_path="rec_rate_by_S_"+node_configs_names+"_"+model_type+img_type)
    ## Figure out how to make the L selection based on the best KDR ratio
    L = 128
    quantization_method = "floating_point" # "threshold", "mean", "floating_point"
    Ls = [128, 256, 512]
    for L in Ls:
        plot_rec_rate_by_S_for_L(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate vs Parity Symbol Length S (D="+str(L)+")", save_path="rec_rate_by_S_"+node_configs_names+"_"+model_type+"_L"+str(L)+img_type, L=L, quantization_method=quantization_method)
    
    
    return results["Models"][best_KDR_ratio_index]

def plot_bdr_results_CSV(csv_file, title, save_path, EveRayTracing=False):
    """Create grouped bar charts of BDR vs key size L for each input type.

    For each input type (IQ, Polar, Spectrogram), generates a plot where:
      - X-axis: key size L
      - Y-axis: BDR (KDR_AB, KDR_AC, KDR_BC)
      - Within each L group, bars are grouped by metric color (AB/AC/BC) and
        subdivided by quantization method using distinct hatches.

    Saves three figures using save_path as a base, e.g.,
    base+'_IQ'+img_type, base+'_Polar'+img_type, base+'_Spectrogram'+img_type.
    """
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
    if EveRayTracing:
        metric_names.append("KDR_AA")
        metric_names.append("KDR_BB")
        metric_labels["KDR_AA"] = "Alice-Eve (Ray Tracing)"
        metric_labels["KDR_BB"] = "Bob-Eve (Ray Tracing)"
        metric_colors["KDR_AA"] = "#ff7f0e" # orange
        metric_colors["KDR_BB"] = "#800080" # purple
    hatch_styles = ["", "//", "x", "-", ".", "o", "*", "+", "O"]

    base, ext = os.path.splitext(save_path)
    if not ext:
        ext = img_type

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
        # Make the plot tighter
        # Geometry
        x_base = np.arange(num_L)
        group_width = 0.8
        metric_slot_width = group_width / (num_metrics-2) if EveRayTracing else num_metrics
        bar_width = metric_slot_width / num_quants

        fig, ax = plt.subplots(figsize=(10, 6))
        for m_idx, metric in enumerate(metric_names):
            for q_idx, qname in enumerate(quant_methods):
                x_offsets = -group_width / 2 + ((m_idx - 2) if EveRayTracing and metric in ["KDR_AA", "KDR_BB"] else m_idx) * metric_slot_width + q_idx * bar_width + bar_width / 2
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
                    alpha=1,
                )
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
        ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(Lv) for Lv in L_values], fontsize=tick_font_size)
        # y axis tick size
        ax.yaxis.set_tick_params(labelsize=tick_font_size)
        # Set the size of the x axis ticks
        # ax.xaxis.set_tick_params(labelsize=tick_font_size*2)
        ax.set_xlabel("Binary Feature Vector Length (B)", fontsize=label_font_size)
        ax.set_ylabel(r'$\overline{\mathrm{BDR}}$', fontsize=label_font_size)
        # plot_title = title if title else "Key Dissagreement Ratio vs Key Size L"
        # ax.set_title(f"{dtype} - {plot_title}", fontsize=title_font_size)
        # ax.set_title(plot_title, fontsize=title_font_size)
        
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        color_handles = [
            Patch(facecolor=metric_colors[m], edgecolor="black", label=metric_labels[m])
            for m in metric_names
        ]
        hatch_handles = [
            Patch(facecolor="#cccccc", edgecolor="black", hatch=hatch_map[q], label=str(q))
            for q in quant_methods
        ]
        legend1 = ax.legend(handles=color_handles, loc="upper left", fontsize=legend_font_size)
        ax.add_artist(legend1)
        # ax.legend(handles=hatch_handles, title="Quantization", loc="upper right")

        plt.tight_layout()
        out_path = f"{base}_{dtype}{ext}"
        print("Saving BDR plot to: ", out_path)
        plt.savefig(out_path)
        
        plt.close(fig)
        
def plot_bdr_results_CSV_for_scenario(csv_files, title, save_path, EveRayTracing=False):
    
    from matplotlib.patches import Patch
    
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
    # plot the BDR results per scenario in three different subplots
    
    # Create single figure with three subplots (share x-axis ticks)
    fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # Plot the BDR results for each scenario in each subplot
    Titles = ["Indoor Scenario", "Outdoor Scenario 1", "Outdoor Scenario 2"]
    for i, csv_file in enumerate(csv_files):
        results = pd.read_csv(csv_file)    

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
        if EveRayTracing and i > 0:
            metric_names.append("KDR_AA")
            metric_names.append("KDR_BB")
            metric_labels["KDR_AA"] = "Alice-Eve (Ray Tracing)"
            metric_labels["KDR_BB"] = "Bob-Eve (Ray Tracing)"
            metric_colors["KDR_AA"] = "#ff7f0e" # orange
            metric_colors["KDR_BB"] = "#800080" # purple

        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = img_type

        desired_types = ["IQ", "Polar", "Spectrogram"]
        present_types = [t for t in desired_types if t in results["data_type_norm"].unique().tolist()]
        print("Present types: ", present_types)
        if not present_types:
            continue
        dtype = "Spectrogram" if "Spectrogram" in present_types else present_types[0]
        
        for dtype in [dtype]:
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
            # Make the plot tighter
            # Geometry
            x_base = np.arange(num_L)
            group_width = 0.8
            if EveRayTracing and i > 0:
                metric_slot_width = group_width / max(1, (num_metrics - 2))
            else:
                metric_slot_width = group_width / max(1, num_metrics)
            bar_width = metric_slot_width / num_quants

            # Define the axis for the subplot
            ax = axs[i]
            # Lets set the title for the subplot to the right side of the subplot 
            ax.set_title(Titles[i], fontsize=title_font_size, loc="right")
            for m_idx, metric in enumerate(metric_names):
                for q_idx, qname in enumerate(quant_methods):
                    x_offsets = (
                        -group_width / 2
                        + ((m_idx - 2) if (EveRayTracing and i > 0 and metric in ["KDR_AA", "KDR_BB"]) else m_idx) * metric_slot_width
                        + q_idx * bar_width
                        + bar_width / 2
                    )
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
                        alpha=1
                    )
            ax.yaxis.set_major_locator(MultipleLocator(0.02))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
            ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
            ax.set_xticks(x_base)
            ax.set_xticklabels([str(Lv) for Lv in L_values], fontsize=tick_font_size)
            # y axis tick size
            ax.yaxis.set_tick_params(labelsize=tick_font_size)
            # Only show x-axis label on bottom subplot
            if i == len(csv_files) - 1:
                ax.set_xlabel("Binary Feature Vector Length (B)", fontsize=label_font_size)
                # Set the size of the x axis ticks
                ax.xaxis.set_tick_params(labelsize=label_font_size*0.9)
            else:
                # Hide x-axis tick labels on top subplots (they share x-axis with bottom)
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(r'$\overline{\mathrm{BDR}}$', fontsize=label_font_size)
            # plot_title = title if title else "Key Dissagreement Ratio vs Key Size L"
            # ax.set_title(f"{dtype} - {plot_title}", fontsize=title_font_size)
            # ax.set_title(plot_title, fontsize=title_font_size)
            # Set the y axis to always end at 0.5
            ax.set_ylim(0, 0.5)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            color_handles = [
                Patch(facecolor=metric_colors[m], edgecolor="black", label=metric_labels[m])
                for m in metric_names
            ]
            hatch_handles = [
                Patch(facecolor="#cccccc", edgecolor="black", hatch=hatch_map[q], label=str(q))
                for q in quant_methods
            ]
            legend1 = ax.legend(handles=color_handles, loc="upper left", fontsize=legend_font_size)
            ax.add_artist(legend1)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)
    print("Saved figure to: ", save_path)

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

def plot_rec_rate_by_S_for_L(csv_path, title=None, save_path=None, L=512, quantization_method="threshold"):
    """Plot rec_rate vs RS code rate for a single key size L from a results CSV.

    - X axis: RS code rate S/(K+S)
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

    # Filter for specific quantization method
    df = df[df["quantization_method"] == quantization_method].copy()
    # Aggregate in case there are multiple rows per S within a (bps,K)
    agg = (
        df.groupby(["bps", "K", "S"], as_index=False)[["rec_rate_AB", "rec_rate_AC", "rec_rate_BC"]].mean()
        .sort_values(["bps", "K", "S"])  # sort for consistent lines
    )

    fig, ax = plt.subplots(figsize=(12, 7))
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
            # label = f"bps={int(bps_val)}, K={int(K_val)}" if pair == "Alice-Bob" else None
            label = f"Z={int(bps_val)}" if pair == "Alice-Bob" else None
            # Increase line width and marker size
            linewidth = 2
            marker_size = 8
            ax.plot(
                (K_val) / (K_val + sub["S"].values),
                sub[col_name].values,
                marker="o" if pair == "Alice-Bob" else None,
                linestyle=pair_to_style[pair],
                color=color,
                label=label,
                linewidth=linewidth,
                markersize=marker_size,
            )
            # Annotate (N,K) above each marker for Alice-Bob line
            if pair == "Alice-Bob":
                x_vals = (K_val) / (K_val + sub["S"].values)
                y_vals = sub[col_name].values
                n_vals = (K_val + sub["S"].values).astype(int)
                for xv, yv, nv in zip(x_vals, y_vals, n_vals):
                    ax.annotate(
                        f"RS({int(nv)},{int(K_val)})",
                        xy=(xv, yv),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=legend_font_size,
                        color=color,
                        zorder=5,
                    )

    ax.set_xlabel("RS Code Rate K/N")
    # Increase the font size of the x axis label
    ax.xaxis.label.set_fontsize(label_font_size)
    ax.set_ylabel("Average Reconciliation Rate")
    # Increase the font size of the y axis label
    ax.yaxis.label.set_fontsize(label_font_size)
    plot_title = title if title is not None else f"Average Reconciliation Rate vs RS Code Rate (B={target_L})"
    ax.set_title(plot_title)
    # Increase the font size of the title
    ax.title.set_fontsize(title_font_size)
    # Configure ticks and grid for precise value reading
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # Make the x axis to tilt 45 degrees
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(which="major", axis="both", linestyle="--", alpha=0.5)
    ax.grid(which="minor", axis="both", linestyle=":", alpha=0.3)
    
    # Size of text x and y axis ticks
    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.tick_params(axis='y', labelsize=tick_font_size)
    
    # First legend for groups (labels set only on AB lines)
    group_legend = ax.legend(loc="upper left", title="Bits per Symbol", fontsize=legend_font_size, bbox_to_anchor=(0, 0.78))
    ax.add_artist(group_legend)
    # Second legend for pair linestyles
    style_handles = [
        Line2D([0], [0], color="black", linestyle=pair_to_style[pair], label=pair)
        for pair in ["Alice-Bob", "Alice-Eve", "Bob-Eve"]
    ]
    # Move the leggend down so that it does not overlap with previous legend
    ax.legend(handles=style_handles, title="Node Pair", loc="upper left", fontsize=legend_font_size)
    # ax.legend(handles=style_handles, title="Node Pair", loc="upper left", fontsize=12)
    # Invert x-axis so higher code rate appears on the left
    ax.invert_xaxis()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
if __name__ == "__main__":
    '''
    Results file:  /home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv
    Reconciliation file:  /home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv
    '''
    # ResultsDir = "/home/Research/POWDER/Results/"
    # # results_file = "results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
    # # reconciliation_file = "results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
    # # /home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-132_RNN.csv
    # model_type = "RNN" # "ResNet", "RNN"
    # #results_file = "results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123-132_RNN.csv"
    # #reconciliation_file = "results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_Sinusoid-Powder-OTA-Dense-Nodes-123-132_RNN.csv"
    # results_file = "results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-123_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-12_RNN_RayTracing.csv"
    # reconciliation_file = "results_reconciliation_models_Sinusoid-Powder-OTA-Dense-Nodes-123_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-12_RNN_RayTracing.csv"
    
    # # plot_bdr_results_CSV(csv_file=ResultsDir+results_file, title="BDR vs key size L", save_path="BDR_vs_L_OTA-Lab-123_132_"+model_type+img_type)
    # # plot_bdr_results_CSV(csv_file=ResultsDir+results_file, title="BDR vs key size L", save_path="BDR_vs_L_OTA-Dense-123_RayTracing_"+model_type+img_type, EveRayTracing=True)
    # # plot_rec_rate_by_S(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate (A-B) vs S", save_path="rec_rate_by_S_OTA-Lab-123_132_"+model_type+img_type)
    
    # scenario1BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_RNN.csv"
    # scenario2BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-123_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-12_RNN_RayTracing.csv"
    # scenario3BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-132_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-13_RNN_RayTracing.csv"
    # scenario1Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_RNN.csv"
    # scenario2Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Dense-Nodes-123_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-12_RNN_RayTracing.csv"
    # scenario3Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Dense-Nodes-132_Sinusoid-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes-13_RNN_RayTracing.csv"
    # ScenariosBDR = [scenario1BDR, scenario2BDR, scenario3BDR]    
    # plot_bdr_results_CSV_for_scenario(ScenariosBDR, title= "", save_path="BDR_vs_L_separateScenarios_RayTracing_"+model_type+img_type, EveRayTracing=True)    
    # exit()
    # L = 128
    # quantization_method = "floating_point" # "threshold", "mean", "floating_point"
    # Ls = [128, 256, 512]
    # for L in Ls:
    #     # plot_rec_rate_by_S_for_L(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate vs RS Code Rate (B="+str(L)+")", save_path="rec_rate_by_S_OTA-Lab-123_132_"+model_type+"_L"+str(L)+img_type, L=L, quantization_method=quantization_method)
    #     plot_rec_rate_by_S_for_L(csv_path=ResultsDir+reconciliation_file, title="Reconciliation Rate vs RS Code Rate (B="+str(L)+")", save_path="rec_rate_by_S_OTA-Lab-123_RayTracing_"+model_type+"_L"+str(L)+img_type, L=L, quantization_method=quantization_method)
    # exit() 
    # Lets plot the history of the training
    # file = "/home/Research/POWDER/Results/History_Spectrogram_FeatureExtractor_RNN_in256_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_Sinusoid-Powder-OTA-Lab-Nodes"
    # with h5py.File(file, "r") as f:
    #     # Check the keys in the file
    #     print("Keys in the file: ", f.keys())
    #     train_loss = f["train_loss"][:]
    #     validation_loss = f["validation_loss"][:]
    # # Create new figure
    # plt.figure()
    # plt.plot(train_loss)
    # plt.plot(validation_loss)
    # plt.title("Training and Validation Loss")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.legend(["Train", "Validation"], loc="upper left")
    # plt.show()
    # plt.savefig("train_and_validation_loss.eps")
    # exit()
    
    homeDir = "/home/Research/POWDER/"
    ModelsDir = homeDir+"Models/"
    ResultsDir = homeDir+"Results/"
    
    # Try this out with the new Ray Tracing scenario
    
    # exit()
    
    signal_type = "Sinusoid" # Sinusoid, PN-Sequence, deltaPulse
    
    node_Ids = {"Sinusoid":
                    {"OTA-Lab": [#[1,2,3],
                            # [1,4,5],[1,4,8],[2,4,3],
                            # [4,2,5],[4,2,8],[4,8,5],
                            # [5,7,8],[5,8,7],[8,4,1],
                            # [8,5,1],[8,5,4]
                            ],
                    "OTA-Dense": [[1,2,3],
                                # [1,2,5],
                                # [1,3,2],
                                # [4,3,5]
                                ],
                    "Sionna-Ray-Tracing":[
                        #[1,2],[1,3],[4,3]
                        [1,2]
                        ]
                },
                "PN-Sequence": {
                    "OTA-Lab": [],
                    "OTA-Dense": []
                },
                "deltaPulse": {
                    "OTA-Lab": [[5,6,1],[5,6,2],[5,6,3],
                                [5,6,4],[5,6,7],[5,6,8]],
                                # [5,7,1],[5,7,2],[5,7,3],
                                # [5,7,4]
                    "OTA-Dense": [[1,2,3],[1,2,4],[1,3,2],
                                  [1,3,4],[1,4,2],[1,4,3],
                                  [2,3,1],[2,3,4],[2,4,1],
                                  [2,4,3],[3,4,1],[3,4,2]],
                    "Sionna-Ray-Tracing":[]
                }         
    }
    
    test_node_configurations = {
        'OTA-Lab': {
            'dataset_name': 'Key-Generation',
            'config_name': signal_type+'-Powder-OTA-Lab-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': node_Ids[signal_type]["OTA-Lab"]               
        },
        'OTA-Dense': {
            'dataset_name': 'Key-Generation',
            'config_name': signal_type+'-Powder-OTA-Dense-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': node_Ids[signal_type]["OTA-Dense"]
        },
        'Sionna-Ray-Tracing': {
            'dataset_name': 'Key-Generation',
            'config_name': signal_type+'-Sionna-Ray-Tracing-POWDER-OTA-Dense-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': node_Ids[signal_type]["Sionna-Ray-Tracing"]
        }
    }
    
    test_node_configurations["OTA-All"] = {"OTA-Lab": test_node_configurations["OTA-Lab"], 
                                           "OTA-Dense": test_node_configurations["OTA-Dense"],}
                                        #    "Sionna-Ray-Tracing": test_node_configurations["Sionna-Ray-Tracing"]}
    print("Test node configurations: ", test_node_configurations["OTA-All"])
    # Get name of file from command line
    # If there is no command line argument, use the default model name
    data_collection_type_train = "OTA-Lab" # "OTA-Lab", "OTA-Dense"
    data_collection_type_test = "OTA-All" # "OTA-Lab", "OTA-Dense"
    
    # Create a new BDR plot per scenario
    run_nist_test = False
    if run_nist_test:
        scenario1BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Lab-Nodes-123_RNN.csv"
        scenario2BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
        scenario3BDR = "/home/Research/POWDER/Results/results_BDR_models_Sinusoid-Powder-OTA-Dense-Nodes-132_RNN.csv"
        scenario1Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Lab-Nodes-123_RNN.csv"
        scenario2Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Dense-Nodes-123_RNN.csv"
        scenario3Reconciliation = "/home/Research/POWDER/Results/results_reconciliation_models_Sinusoid-Powder-OTA-Dense-Nodes-132_RNN.csv"
        ScenariosBDR = [scenario1BDR, scenario2BDR, scenario3BDR]        
        
        model_name = "Spectrogram_FeatureExtractor_RNN_in256_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_Sinusoid-Powder-OTA-Lab-Nodes_1761276429.h5"
        test_model(model_name, test_node_configurations[data_collection_type_test], home=homeDir, generate_results=True)
        
        exit()
    # exit()
    if len(sys.argv) == 1: 
        # model_name = "FeatureExtractor_512_alpha0.5_beta0.5_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_1758227515"
        # Use the model with the best validation loss
        test_type = "" # "Rec", "BDR", "loss", "plot_BDR", "find_best_model"
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
        elif test_type == "find_best_model":
            print("Finding best model")
            # exit()
            model_name = find_best_model(test_node_configurations[data_collection_type_train], homeDir, test_node_configurations[data_collection_type_test])
            
            # exit()
            # results_reconciliation_file = ResultsDir+"results_reconciliation_models_OTA-All-123_RNN.csv"
            # plot_rec_rate_by_S(csv_path=results_reconciliation_file, title="Reconciliation Rate (A-B) vs S", save_path="rec_rate_by_S_OTA-All-123_RNN"+img_type)
            
            # results_file = ResultsDir+"results_BDR_models_OTA-All-123_RNN.csv"
            # plot_bdr_results_CSV(results_file, "BDR vs L", ResultsDir+"BDR_vs_L_OTA-All-123_RNN"+img_type)
    
            # Copy the file as a best model under models directory
            # shutil.copy(ModelsDir+model_name, ModelsDir+"Best_Model.h5")
        else:
            print("Using default model name")
            # model_name = "Spectrogram_FeatureExtractor_RNN_in2048_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_deltaPulse-Powder-OTA-Lab-Nodes_2_1769443104.h5"
            # model_name = "Spectrogram_FeatureExtractor_RNN_in2048_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_deltaPulse-Powder-OTA-Lab-Nodes_4_1769628988.h5"
            # model_name = "Spectrogram_FeatureExtractor_RNN_QuantizationLayer_in2048_out128_alpha0.5_beta0.5_SGD_lr0.1_deltaPulse-Powder-OTA-Lab-Nodes_1_1769639369.h5"
            # model_name = "Spectrogram_FeatureExtractor_RNN_QuantizationLayer_in2048_out2040_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_deltaPulse-Powder-OTA-Lab-Nodes_1_1770149385.h5"
            # model_name = "Spectrogram_FeatureExtractor_RNN_QuantizationLayerKDR_in2048_out2040_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_deltaPulse-Powder-OTA-Lab-Nodes_1_1770153686.h5"
            # model_name = "Spectrogram_FeatureExtractor_RNN_QuantizationLayerKDR_in2048_out2040_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.001_deltaPulse-Powder-OTA-Lab-Nodes_3_1770221691.h5"
            model_name = "Spectrogram_FeatureExtractor_RNN_QuantizationLayerKDR_in256_out128_alpha0.5_beta0.5_gamma0.1_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_6_1770314341.h5"
    else:
        # model_name = sys.argv[1]
        # 0.2,0.4,0.3,SGD
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
        if optimizer == "RMSprop":
            lr = 0.0001
        elif optimizer == "SGD":
            lr = 0.1
        elif optimizer == "Adam":
            lr = 0.01
        model_name = "Spectrogram_FeatureExtractor_"+network_type \
                +"_in"+input_length+"_out"+output_length \
                +"_alpha"+alpha+"_beta"+beta+("_gamma"+gamma+"_" if gamma is not None else "_") \
                +optimizer+"_lr"+str(lr)+"_"+signal_type
        # Find the file that starts with the model_name
        found = False
        # print("Model name: ", model_name)
        # time.sleep(10)
        for file in os.listdir(ModelsDir):
            if file.startswith(model_name):
                model_name = file
                found = True
                print("Model found: ", model_name)
                # exit()
                break
        if not found:
            print("Model not found")
            print("Model name: ", model_name)
            # print("Files in models directory: ", os.listdir(ModelsDir))
            print("Directory: ", ModelsDir)
            exit()
    
    # model_name = "FeatureExtractor_512_alpha0.5_beta0.5_SGD_lr0.1_Sinusoid-Powder-OTA-Lab-Nodes_1758225804"
    # model_name = "Spectrogram_FeatureExtractor_RNN_in256_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_Sinusoid-Powder-OTA-Lab-Nodes_1761276429.h5"
    
    feature_extractor_name = ModelsDir+model_name
    test_model(model_name, test_node_configurations[data_collection_type_test], home=homeDir, generate_results=True)
    
    # Spectrogram_FeatureExtractor_RNN_in2048_out128_alpha0.5_beta0.5_gamma0.1_RMSprop_lr0.0001_deltaPulse-Powder-OTA-Lab-Nodes_1769122091.h5