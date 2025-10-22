from math import sqrt
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt

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
    
    def threshold_quantization(self, features, threshold=0.5, normalize=True):
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
    
    def floating_point_quantization(self, features, precision=1, normalize=True):
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
    
if __name__ == "__main__":
    quantization = Quantization()
    # Random feature vector with L values between 0 and 1 with fp16 precision
    L = 128
    # Create a feature vector of L ones
    # features = np.ones(L).astype(np.float16)
    features = np.random.rand(L).astype(np.float16)
    # Plot features on a line plot as scatter plot over a single plane
    plot_features(features, threshold=0.5, L=L)
    
    print("Features:", features[0:10])
    l2_norm = np.linalg.norm(features)
    print("L2 norm:", l2_norm)
    print("L2 norm average:", l2_norm / L)
    print("L2 norm values expected avg magnitude:", 1/sqrt(L))
    features = features / l2_norm 
    print("Features L2 normalized:", features[0:10])
    print("features average:", mean(features))
    print("Features sum:", np.sum(features))
    print("Features Max:", np.max(features))
    print("Features Min:", np.min(features))
    print("Magnitude:", np.abs(features))
    plot_features(features, threshold=1/sqrt(L), L=L)
    # exit()
    print("Mean Quantization:")
    features_quatized = quantization.mean_quantization(features)
    print("features_quatized mean:", features_quatized[0:10])
    quantization.quantized_data_information(features_quatized)
    print("Threshold Quantization:")
    features_quatized = quantization.threshold_quantization(features, 0.5, True)
    print("features_quatized threshold:", features_quatized[0:10])
    quantization.quantized_data_information(features_quatized)
    print("Floating Point Quantization:")
    features_quatized = quantization.floating_point_quantization(features, 1, True) # Setting to 1 is the same as threshold 0.5
    print("features_quatized floating point:", features_quatized[0:10])
    quantization.quantized_data_information(features_quatized)
    print("L2 Threshold Quantization:")
    features_quatized = quantization.L2_threshold_quantization(features)
    print("features_quatized L2 threshold:", features_quatized[0:10])
    quantization.quantized_data_information(features_quatized)
    
    
    print("Threshold 0.5 and floating point 1 the same:", quantization.L2_threshold_quantization(features) == quantization.floating_point_quantization(features, 1))
    
    