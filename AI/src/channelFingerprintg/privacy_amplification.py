import numpy as np
import hashlib
from nistrng import *


class PrivacyAmplification:
    def __init__(self):
        pass

    def str2arr(self, string):
        arr = []
        integer = int(string, 16)
        binary_string = format(integer, '0>42b')
        for i in binary_string:
            arr.append(int(i))
        return arr
    
    def privacyAmplification(self, data):
        # encode the string
        encoded_str = data.encode()
        # create sha3-256 hash objects
        obj_sha3_256 = hashlib.new("sha3_512", encoded_str)
        return(obj_sha3_256.hexdigest())

    #https://github.com/InsaneMonster/NistRng/blob/master/benchmarks/numpy_rng_test.py
    def NIST_RNG_test(self, data):
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
    
if __name__ == "__main__":
    from Quantization import Quantization
    from reconciliation import ReedSolomonReconciliation, BitToSymbolTransformation
    
    # Random feature vector of values between 0 and 1
    features_A = np.random.rand(128)
    features_B = np.random.rand(128)
    print(features_A)
    print(features_B)
    
    # Quantize the features
    quantization = Quantization()
    features_A_quatized = quantization.threshold_quantization(features_A, 0.5, False)
    features_B_quatized = quantization.threshold_quantization(features_B, 0.5, False)
    print(features_A_quatized)
    print(features_B_quatized)
    
    # Bits to symbols transformation
    bits_per_symbol = 2
    k = int(len(features_A_quatized)/bits_per_symbol)
    features_A_symbols = BitToSymbolTransformation().bin_to_2bps(features_A_quatized)
    features_B_symbols = BitToSymbolTransformation().bin_to_2bps(features_B_quatized)
    print(features_A_symbols)
    print(features_B_symbols)
    print("K: ", k)
    L = len(features_A_quatized)
    print("L: ", L)
    Ss = [(2**i)-1 for i in range(1, int(np.log2(k)+1))]
    # Remove any values over 255
    Ss = [S for S in Ss if S < 255]
    for S in Ss:
        N = int(k + S)
        print("K: ", k, "S: ", S, "N: ", N)
        reconciliation = ReedSolomonReconciliation(L, bits_per_symbol, k, N)
        reconciled_data = reconciliation.reconcile(features_A_symbols,features_B_symbols)
        print("Reconciled data: ", reconciled_data)
        
    # Amplify the data
    privacy_amplification = PrivacyAmplification()
    priv_amp_data_A = privacy_amplification.privacyAmplification(features_A_symbols)
    priv_amp_data_B = privacy_amplification.privacyAmplification(features_B_symbols)
    print(priv_amp_data_A)
    print(priv_amp_data_B)
    
    # NIST-SP800-22r1a test
    keys = [priv_amp_data_A, priv_amp_data_B]
    
    # Transform the keys to arrays
    keys = [privacy_amplification.str2arr(key) for key in keys]
    print(keys)
    
    # NIST-SP800-22r1a test
    results_score, results_passed = privacy_amplification.NIST_RNG_test(keys)
    print("Results score: ", results_score)
    print("Results passed: ", results_passed)
    
    