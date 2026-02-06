from math import sqrt
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

class Quantization:
    def __init__(self):
        pass
    
    def feature_normalization(self, features):
        # Normalize features such that the max value is 1 and the min value is 0
        arr = np.asarray(features, dtype=np.float32)
        if arr.size == 0:
            return []
        min_feature = float(np.min(arr))
        max_feature = float(np.max(arr))
        denom = max_feature - min_feature
        if denom == 0.0:
            return [0.0] * int(arr.size)
        normalized = (arr - min_feature) / denom
        return normalized.tolist()
    
    def mean_quantization(self, features):
        mean_features = mean(features)
        # print("Mean features: ", mean_features)
        threshold = mean_features #0
        features_quatized = []
        for i in features:
            if i >= threshold:
                features_quatized.append(1)
            else:
                features_quatized.append(0)
        return features_quatized
    
    def threshold_quantization(self, features, threshold=0.5, normalize=False):
        features_quatized = []
        if normalize:
            features = self.feature_normalization(features)
        # print("Features normalized:", features)
        for i in features:
            if i >= threshold:
                features_quatized.append(1)
            else:
                features_quatized.append(0)
        return features_quatized
    
    def floating_point_quantization(self, features, precision=1, normalize=False):
        features_quatized = []
        # Quantize features from floating point to binary
        if normalize:
            features = self.feature_normalization(features)
        for i in features:
            
            binary_representation = bin(int(i * (2**precision)))
            binary_representation = binary_representation[2:].zfill(precision)
            # features_quatized.append(binary_representation[2:].zfill(precision))
            # Add as many zeros at the beginning of the binary representation as the precision
            # add each of the bits to the features_quatized
            for bit in binary_representation:
                features_quatized.append(int(bit))
            # Check if the length of the features_quatized is equal to the length of the features multiplied by the precision
        if len(features_quatized) != len(features)*precision:
            # Either add zeros or remove bits from the binary representation
            if len(features_quatized) < len(features)*precision:
                features_quatized.extend([0] * (len(features)*precision - len(features_quatized)))
            else:
                features_quatized = features_quatized[:len(features)*precision]
        return features_quatized
    
    def quantized_data_information(self, features_quatized):
        length = len(features_quatized)
        ones = features_quatized.count(1)
        zeros = features_quatized.count(0)
        ones_percentage = ones / length
        zeros_percentage = zeros / length
        print(f"Length: {length}")
        print(f"Ones: {ones}")
        print(f"Zeros: {zeros}")
        print(f"Ones Percentage: {ones_percentage}")
        print(f"Zeros Percentage: {zeros_percentage}")
        
    def L2_threshold_quantization(self, features):
        L = len(features)
        threshold = 1/sqrt(L)
        features_quatized = []
        for i in features:
            if i >= threshold:
                features_quatized.append(1)
            else:
                features_quatized.append(0)
        return features_quatized
    
def plot_features(features, threshold=0.5, L=512):
    plt.scatter(range(L), features)
    # Plot a line by the 0.5 threshold
    plt.axhline(y=threshold, color='r', linestyle='--')
    # Color the features above the threshold in red and the features below the threshold in blue
    
    for i in range(L):
        if features[i] >= threshold:
            plt.scatter(i, features[i], color='red')
        else:
            plt.scatter(i, features[i], color='blue')
    plt.title("Features")
    plt.xlabel("Index")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    quantization = Quantization()
    # Random feature vector with L values between -1 and 1 with fp16 precision
    L = 128
    # Create a feature vector of L ones
    # features = np.ones(L).astype(np.float16)
    features = np.random.rand(L).astype(np.float16) * 2 - 1
    # Plot features on a line plot as scatter plot over a single plane
    # plot_features(features, threshold=0.5, L=L)
    
    r = np.linalg.norm(features)
    u = features / r  
    print("u:", u)
    print("u size:", u.size)
    
    angles = np.linspace(0, 2*np.pi, L, endpoint=False)
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.scatter(angles, u, color='blue', linewidth=1)
    # Plot the radius r
    ax.scatter(0, r, color='red', linewidth=1)
    ax.set_title('Radial (Polar) Representation of L²-Normalized Vector')
    plt.show()
    # exit()
    print("Features:", features[0:10])
    l2_norm = np.linalg.norm(features, ord=2)
    print("L2 norm:", l2_norm)
    print("Sqrt of L:", sqrt(L))
    print("L2 norm average:", l2_norm / L)
    print("L2 norm values expected avg magnitude:", 1/sqrt(L))
    features = features / l2_norm 
    
    print("Features L2 normalized:", features[0:10])
    print("features average:", mean(features))
    print("Features sum:", np.sum(features**2))
    print("Features Max:", np.max(features))
    print("Features Min:", np.min(features))
    # plot_features(features, threshold=1/sqrt(L), L=L)
    features = sigmoid(features)
    # Have features range from 0 to 1
    features = (features - np.min(features)) / (np.max(features) - np.min(features))
    # plot_features(features, threshold=0.5, L=L)
    print("Features sigmoid:", features[0:10])
    print("Features sigmoid average:", mean(features))
    print("Features sigmoid sum:", np.sum(features))
    print("Features sigmoid Max:", np.max(features))
    print("Features sigmoid Min:", np.min(features))    
    # exit()
    # print("Mean Quantization:")
    # features_quatized = quantization.mean_quantization(features)
    # print("features_quatized mean:", features_quatized[0:10])
    # quantization.quantized_data_information(features_quatized)
    # print("Threshold Quantization:")
    # features_quatized = quantization.threshold_quantization(features, 0.5, False)
    # print("features_quatized threshold:", features_quatized[0:10])
    # quantization.quantized_data_information(features_quatized)
    print("Floating Point Quantization:")
    num_bits = 2
    features_quatized = quantization.floating_point_quantization(features, num_bits, False) # Setting to 1 is the same as threshold 0.5
    print("features_quatized floating point:", features_quatized[0:10*num_bits])
    
    print("features length:", len(features))
    print("features_quatized floating point length:", len(features_quatized))
    
    # Plot the features in a plot with the features from [0,1] as the y axis and the index as the x axis
    # plt.plot(features)
    plt.scatter(range(L), features)
    # plt.plot(features_quatized)
    # Add lines along the y axis that mark the bits-1 thresholds for the floating point quantization
    for i in range(num_bits*2-1):
        plt.axhline(y=(i+1)/(num_bits*2), color='r', linestyle='--')
    plt.title("Features and Floating Point Quantization")
    plt.xlabel("Index")
    plt.ylabel("Feature")
    plt.tight_layout()
    # Have the y axis range from 0 to 1
    plt.ylim(0, 1)
    # rename y axis values to "yi*2^num_bits" where yi is the y value and num_bits is the number of bits but keeping the original dimension
    # plt.yticks([i/num_bits*2**num_bits for i in range(num_bits*2)])
    # plt.yticklabels([str(i) for i in range(num_bits*2)])
    # Set ticks at multiples of 1/(2^num_bits) and label as yi*2^num_bits (integer scale), while keeping axis in [0,1]
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(1 / (2**num_bits)))
    def format_func(value, tick_number):
        decimal_value = int(np.floor(value * (2**num_bits)))
        binary_value = bin(decimal_value)[2:].zfill(num_bits)
        return f"{binary_value}"
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    # plt.yscale('log', base=2)
    plt.savefig("Quantization_example_bits_"+str(num_bits)+".png")
    plt.show()
    
    num_bits = 4
    num =0.999999
    decimal_value = int(np.floor(num * (2**num_bits)))
    binary_value = bin(decimal_value)[2:].zfill(num_bits)
    print("feature:", num)
    print("decimal value:", decimal_value)
    print("binary value:", binary_value)
    
    
    # quantization.quantized_data_information(features_quatized)
    # print("L2 Threshold Quantization:")
    # features_quatized = quantization.L2_threshold_quantization(features)
    # print("features_quatized L2 threshold:", features_quatized[0:10])
    # quantization.quantized_data_information(features_quatized)
    # print("Threshold 0.5 and floating point 1 the same:", quantization.threshold_quantization(features, 0.5, False) == quantization.floating_point_quantization(features, 1, False))
    
    