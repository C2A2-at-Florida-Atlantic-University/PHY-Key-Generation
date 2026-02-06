import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from deep_learning_models import identity_loss, ResNet_QuadrupletNet_Channel, FeedForward_QuadrupletNet_Channel, RNN_QuadrupletNet_Channel, Transformer_QuadrupletNet_Channel, AE_QuadrupletNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram, ChannelIQ, ChannelPolar

import time
from TestChannelFingerprinting import test_model
import h5py
from collections import Counter
import numpy as np
import os
import random
# Optional: force CPU when CUDA/cuBLAS is misconfigured.
# Set POWDER_FORCE_CPU=1 in the environment to disable GPU.
if os.environ.get("POWDER_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt

def set_reproducible(seed: int, deterministic: bool = True, max_threads: int = 1):
    """Set seeds and deterministic execution for reproducibility.

    Note: Some ops may remain nondeterministic on certain GPUs/versions.
    """
    # Python, NumPy, TensorFlow seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        # Sets TF, NumPy, and Python seeds together (TF 2.9+)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    # Deterministic behavior (TF 2.12+)
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass

    # Constrain parallelism to reduce nondeterminism from threads
    try:
        tf.config.threading.set_intra_op_parallelism_threads(max_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max_threads)
    except Exception:
        pass

def train_channel_feature_extractor(data, labels, train_configurations, model_type, network_type="ResNet", model_name="", seed: int = 42, quantization_layer=False):
    '''
    train_feature_extractor trains an RFF extractor using triplet loss.
    
    INPUT: 
        FILE_PATH is the path of training dataset.
        
        DEV_RANGE is the label range of LoRa devices to train the RFF extractor.
        
        PKT_RANGE is the range of packets from each device to train the RFF extractor.
        
        SNR_RANGE is the SNR range used in data augmentation. 
        
    RETURN:
        FEATURE_EXTRACTOR is the RFF extractor which can extract features from
        channel-independent spectrograms.
    '''
        
    # Ensure reproducibility for any TF/NumPy/Python randomness within training
    set_reproducible(seed)
    
    if train_configurations[model_type]['data_type'] == "IQ":
        ChannelIQObj = ChannelIQ()
        data = ChannelIQObj.channel_iq(data)
    elif train_configurations[model_type]['data_type'] == "Polar":
        ChannelPolarObj = ChannelPolar()
        data = ChannelPolarObj.channel_polar(data)
    elif train_configurations[model_type]['data_type'] == "Spectrogram":
        ChannelSpectrogramObj = ChannelSpectrogram()
        data = ChannelSpectrogramObj.channel_spectrogram(
            data,
            train_configurations[model_type]['fft_len']
        )

    #NetObj =  TripletNet_Channel()
    # NetObj = QuadrupletNet_Channel()
    
    # NetObj = ResNet_QuadrupletNet_Channel()
    output_length = train_configurations[model_type]['output_length']
    # Create an RFF extractor.
    if network_type == "ResNet":
        NetObj = ResNet_QuadrupletNet_Channel()
    elif network_type == "FeedForward":
        NetObj = FeedForward_QuadrupletNet_Channel()
    elif network_type == "RNN":
        NetObj = RNN_QuadrupletNet_Channel()
    elif network_type == "Transformer":
        NetObj = Transformer_QuadrupletNet_Channel()
    elif network_type == "AE":
        NetObj = AE_QuadrupletNet_Channel()
    
    if quantization_layer:
        feature_extractor = NetObj.build_quantized_extractor(data.shape, output_length)
    else:
        feature_extractor = NetObj.feature_extractor(data.shape, output_length)
    
    # Create the quadruplet net using the RFF extractor.
    loss_type = "KDR" if quantization_layer else ""
    net = NetObj.create_quadruplet_net(
        feature_extractor, 
        train_configurations[model_type]['alpha'], 
        train_configurations[model_type]['beta'], 
        train_configurations[model_type]['gamma'] if 'gamma' in train_configurations[model_type] else None,
        loss_type=loss_type
    )

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    patience = train_configurations[model_type]['patience']
    
    early_stop = EarlyStopping('val_loss', 
                                min_delta = 0, 
                                patience = patience
                                )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    min_delta = 0.0001, 
                                    factor = train_configurations[model_type]['factor'], 
                                    patience = int(math.ceil(patience/2)), 
                                    verbose=1
                                    )
    
    ResultsDir = homeDir+"Results/"
    
    # Model checkpoint
    checkpoint_filename = os.path.join(ResultsDir, f"model_checkpoint_{model_name}.h5")
    checkpoint = ModelCheckpoint(checkpoint_filename, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min',
                                 save_weights_only=True)
    
    # callbacks = [early_stop, reduce_lr, checkpoint]
    callbacks = [reduce_lr]
    
    validation_size= train_configurations[model_type]['validation_size']
    
    # Split the dasetset into validation and training sets.
    data_train, data_valid, label_train, label_valid = train_test_split(data, 
                                                                        labels, 
                                                                        test_size=validation_size, 
                                                                        shuffle= False)
    del data, labels
    
    batch_size = train_configurations[model_type]['batch_size']
    # Create the trainining generator.
    train_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_train, 
                                                        label_train,
                                                        seed=seed)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_valid, 
                                                        label_valid,
                                                        seed=seed + 1 if isinstance(seed, int) else None)
    
    # Use the RMSprop optimizer for training.
    LearningRate = train_configurations[model_type]['LearningRate']
    optimizer = train_configurations[model_type]['optimizer']
    if optimizer == "Adam":
        opt = Adam(learning_rate=LearningRate)
    elif optimizer == "RMSprop":
        opt = RMSprop(learning_rate=LearningRate)
    elif optimizer == "SGD":
        opt = SGD(learning_rate=LearningRate)

    net.compile(
        loss = identity_loss,
        #metrics=['accuracy'],
        optimizer = opt)

    print("Training data:", data_train.shape)
    print("Validation data:", data_valid.shape)
    # Start training.
    history = net.fit(train_generator,
                        steps_per_epoch = data_train.shape[0]//batch_size,
                        epochs = train_configurations[model_type]['epochs'],
                        validation_data = valid_generator,
                        validation_steps = data_valid.shape[0]//batch_size,
                        verbose=1, 
                        callbacks = callbacks)
    # Save the history and reproducibility metadata
    with h5py.File(ResultsDir+"History_"+model_name, "w") as f:
                        f.create_dataset("train_loss",data=history.history['loss'])
                        f.create_dataset("validation_loss",data=history.history['val_loss'])
                        f.close()
                        
    timestamp = int(time.time())
    filename = model_name +'_'+str(timestamp)+'.h5'
    
    feature_extractor.save(ModelsDir+filename)
    print("Saving file: ", filename)
    
    return feature_extractor

def plot_data_scenario(data, index_start, filename):
    example_data = {"AB": data[index_start], "AE": data[index_start+1], "BA": data[index_start+2], "BE": data[index_start+3]}
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    
    for i, (label, data) in enumerate(example_data.items()):
        axes[i, 0].plot(data.real, label='Real')
        axes[i, 0].plot(data.imag, label='Imaginary')
        axes[i, 0].set_title("IQ Samples "+label)
        axes[i, 0].set_xlabel("Sample Number")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].legend()
    
    fft_len = 2048 # 2048
    overlap = fft_len/2
    ChannelSpectrogramObj = ChannelSpectrogram()
    for i, (label, data) in enumerate(example_data.items()):
        spectrogram = ChannelSpectrogramObj._gen_single_channel_spectrogram(data, fft_len, overlap)
        arr = np.asarray(spectrogram)
        axes[i, 1].imshow(arr, aspect='auto', cmap='jet')
        axes[i, 1].set_title("Spectrogram "+label)
        axes[i, 1].set_xlabel("Frequency")
        axes[i, 1].set_ylabel("Time")

    plt.tight_layout()
    plt.show()
    plt.savefig(filename)
    
# Plot the avg power of the data for each scenario
def plot_avg_power_scenario(data, ScenarioStartIndex, DataPerScenario, filename):
    # Get entire data greatest and lowest values
    greatest_value = np.max(data)
    lowest_value = np.min(data)
    avg_power = {"AB": [], "AE": [], "BA": [], "BE": []}
    i = 0
    data = data[ScenarioStartIndex:ScenarioStartIndex+DataPerScenario]
    while i < data.shape[0]:
        avg_power["AB"].append(np.mean(np.abs(data[i])**2))
        avg_power["AE"].append(np.mean(np.abs(data[i+1])**2))
        avg_power["BA"].append(np.mean(np.abs(data[i+2])**2))
        avg_power["BE"].append(np.mean(np.abs(data[i+3])**2))
        i += 4
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    for i, (label, power) in enumerate(avg_power.items()):
        axes[i].plot(power, label=label)
        axes[i].set_title("Avg Power of "+label)
        axes[i].set_xlabel("Sample Number")
        axes[i].set_ylabel("Avg Power (dB)")
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(filename+".png")
    
def plot_avg_min_max_power_scenario(data, ScenarioStartIndex, DataPerScenario, filename):
    # Get entire data greatest and lowest values
    greatest_value = np.max(data)
    lowest_value = np.min(data)
    avg_power = {"AB": [], "AE": [], "BA": [], "BE": []}
    i = 0
    data = data[ScenarioStartIndex:ScenarioStartIndex+DataPerScenario]
    while i < data.shape[0]:
        # Get the average from the greatest and lowest values of each sample
        avg_power["AB"].append(np.mean(np.abs([np.max(data[i])**2, np.min(data[i])**2])))
        avg_power["AE"].append(np.mean(np.abs([np.max(data[i+1])**2, np.min(data[i+1])**2])))
        avg_power["BA"].append(np.mean(np.abs([np.max(data[i+2])**2, np.min(data[i+2])**2])))
        avg_power["BE"].append(np.mean(np.abs([np.max(data[i+3])**2, np.min(data[i+3])**2])))
        i += 4
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    for i, (label, power) in enumerate(avg_power.items()):
        axes[i].plot(power, label=label)
        axes[i].set_title("Avg Min Max Power of "+label)
        axes[i].set_xlabel("Sample Number")
        axes[i].set_ylabel("Avg Min Max Power (dB)")
        axes[i].legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(filename+".png")

if __name__ == "__main__":
    homeDir = "/home/Research/POWDER/"
    # Central seed for full reproducibility of data shuffling, initialization, and training
    
    signal_type = "Sinusoid" # Sinusoid, PN-Sequence, deltaPulse
    
    node_Ids = {"Sinusoid":
                    {"OTA-Lab": [# [1,2,3],
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
                            [8,5,4]]
                    ,
                    "OTA-Dense": [[1,2,3],
                                [1,2,5],
                                [1,3,2],
                                [4,3,5]]
                    },
                "PN-Sequence": {
                    "OTA-Lab": [],
                    "OTA-Dense": []
                },
                "deltaPulse": {
                    "OTA-Lab": [
                                    # [5,6,1],[5,6,2],[5,6,3],  # Testing scenarios
                                    # [5,6,4],[5,6,7],[5,6,8],  # Testing scenarios
                                    # [5,7,1],[5,7,3],[5,7,2],# May need to remove due to low signal power
                                    # [5,7,4],[5,7,6],[5,7,8],# May need to remove due to low signal power
                                    [5,8,2],[5,8,3],[5,8,4],
                                    [5,8,6],[5,8,7],[6,7,2],
                                    [6,7,3],[6,7,4],[6,7,5],
                                    [6,7,8],
                                    # [6,8,2],[6,8,3],# May need to remove due to low signal power
                                    # [6,8,4],[6,8,5],[6,8,7],# May need to remove due to low signal power
                                    # [7,8,2],[7,8,3],[7,8,4],# May need to remove due to low signal power
                                    # [7,8,5],[7,8,6]# May need to remove due to low signal power
                                    #     #Scenarios avoided due to low signal power
                                ],
                    "OTA-Dense": [
                                    [1,2,3],[1,2,4],[1,3,2],
                                    [1,3,4],[1,4,2],[1,4,3],
                                    [2,3,1],[2,3,4],[2,4,1],
                                    [2,4,3],[3,4,1],[3,4,2]
                                ]
                }
    }
    node_configurations = {
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
        }
    }
    
    configuration = node_configurations['OTA-Lab']
    dataset_name = configuration['dataset_name']
    repo_name = configuration['repo_name']
    node_Ids = configuration['node_Ids']
    
    for idx, node_ids in enumerate(node_Ids):
        config_name = configuration['config_name']+"-"+"".join(str(node) for node in node_ids)
        print("Config name: ", config_name)
        if idx == 0:
            dataset = DatasetHandler(dataset_name, config_name, repo_name)
            # Exit the loop after the first iteration
            # break
        else:
            dataset.add_dataset(dataset_name, config_name, repo_name)
        dataset.get_dataframe_Info()
    # Set seeds and determinism prior to any TF ops
    REPRO_SEED = 42
    set_reproducible(REPRO_SEED)
    data, labels, rx, tx = dataset.load_data(shuffle=True, seed=REPRO_SEED)
    
    # For every example in the data lets take the average number of samples per example
    
    print("First example number of samples: ", data[0].shape[0])
    print("Last example number of samples: ", data[-1].shape[0])
    avg_num_samples = np.mean([example.shape[0] for example in data])
    print("Average number of samples per example: ", avg_num_samples)
    # Find the example with the least number of samples
    min_num_samples = np.min([example.shape[0] for example in data])
    print("Example with the least number of samples: ", min_num_samples)
    # Find the example with the most number of samples
    max_num_samples = np.max([example.shape[0] for example in data])
    print("Example with the most number of samples: ", max_num_samples)
    min_num_samples = 8192
    # Resize the data in examples to the least number of samples
    data = np.array([example[-min_num_samples:] for example in data])
    print("Data shape: ", data.shape)
    print("Labels shape: ", labels.shape)
    print("RX shape: ", rx.shape)
    print("TX shape: ", tx.shape)
    avg_num_samples = np.mean([example.shape[0] for example in data])
    print("Average number of samples per example: ", avg_num_samples)
        
    print("Data shape: ", data.shape)
    print("Labels shape: ", labels.shape)
    print("RX shape: ", rx.shape)
    print("TX shape: ", tx.shape)

    # Get a single example of the data
    # For every scenario there are 400 examples where i (AB), i+1 (AE), i+2 (BA), i+3 (BE)
    DataPerScenario = 400
    TotalData = data.shape[0]
    numberOfScenarios = TotalData//DataPerScenario
    ScenarioToTest = 0
    print("Scenario to test: ", ScenarioToTest)
    print("Number of scenarios: ", numberOfScenarios)
    print("Data per scenario: ", DataPerScenario)
    print("Total data: ", TotalData)
    print("Data per scenario: ", DataPerScenario)
    
    for i in range(numberOfScenarios):
        ScenarioStartIndex = int(i*DataPerScenario)
        plot_data_scenario(data, ScenarioStartIndex, "XTEST_IQ_Spectrogram_scenario_"+str(i)+".png")
        # plot_avg_power_scenario(data, ScenarioStartIndex, DataPerScenario, "XTEST_Avg_Power_scenario_"+str(i)+".png")
        plot_avg_min_max_power_scenario(data, ScenarioStartIndex, DataPerScenario, "XTEST_Avg_Min_Max_Power_scenario_"+str(i)+".png")
    # exit()
    # Locate locations for delta pulses
    # def locate_delta_pulse(example_data):
    #     """
    #     Locate the positions of delta pulses in IQ data.
        
    #     Parameters:
    #     -----------
    #     example_data : np.ndarray
    #         Complex-valued IQ signal data
            
    #     Returns:
    #     --------
    #     np.ndarray
    #         Array of indices where delta pulses are located
    #     """
    #     # Calculate magnitude for complex IQ data
        
    #     magnitude = np.abs(example_data)
        
    #     # Use percentile-based threshold to detect spikes (delta pulses)
    #     # Delta pulses are typically much higher than the median/background
    #     # Using 95th percentile as threshold to catch significant spikes
    #     threshold = np.percentile(magnitude, 99.95)
        
    #     # Alternative: Use median + multiple of MAD (Median Absolute Deviation) for robustness
    #     median = np.median(magnitude)
    #     mad = np.median(np.abs(magnitude - median))
    #     print("Median: ", median)
    #     print("MAD: ", mad)
    #     print("Magnitude: ", magnitude)
    #     # Get the max value of the magnitude
    #     max_value = np.max(magnitude)
    #     print("Max value: ", max_value)
        
    #     # get the 20 max values of the magnitude
    #     max_values = np.sort(magnitude)[-20:]
    #     print("Max values: ", max_values)
        
    #     threshold = magnitude #median + 1 * mad
    #     print("Threshold: ", threshold)
        
    #     # Find indices where magnitude exceeds threshold
    #     delta_pulse_indices = np.where(magnitude > threshold)[0]
        
    #     return delta_pulse_indices
    # # Print number of delta pulses
    
    # def locate_delta_pulse_energy(iq_samples, threshold_factor=7, window_size=256):
    #     # 1. Energy
    #     energy = np.abs(iq_samples)**2

    #     # 2. Smooth energy
    #     window = window_size
    #     energy_smooth = np.convolve(
    #         energy,
    #         np.ones(window)/window,
    #         mode='same'
    #     )

    #     # 3. Threshold
    #     noise_floor = np.median(energy_smooth)
    #     threshold = noise_floor * threshold_factor

    #     # 4. Detection
    #     is_signal = energy_smooth > threshold

    #     # 5. Rising edges = burst starts
    #     edges = np.diff(is_signal.astype(int))
    #     start_indices = np.where(edges == 1)[0]
    #     # get the end indices of the pulses
    #     end_indices = np.where(edges == -1)[0]
    #     # get the duration of the pulses
    #     # pulse_durations = start_indices - end_indices
    #     # print("Pulse durations: ", pulse_durations)
    #     # # get the average pulse duration
    #     # average_pulse_duration = np.mean(pulse_durations)
    #     # print("Average pulse duration: ", average_pulse_duration)
    #     # # get the median pulse duration
    #     # median_pulse_duration = np.median(pulse_durations)
    #     # print("Median pulse duration: ", median_pulse_duration)

    #     print("Detected burst start indices:")
    #     print(start_indices)
    #     return start_indices
    # window_size = 128
    # threshold_factor = 5
    # # print("Number of delta pulses: ", len(locate_delta_pulse(example_data)))
    # pulse_indices = locate_delta_pulse_energy(example_data, threshold_factor=threshold_factor, window_size=window_size)
    # print("Number of delta pulses: ", len(pulse_indices))
    
    
    
    # plt.imshow(arr, aspect='auto', cmap='jet')
    # plt.title("Spectrogram")
    # plt.xlabel("Frequency")
    # plt.ylabel("Time")
    # plt.colorbar()
    # plt.show()
    # plt.savefig("XTEST_IQ_Spectrogram_class.png")
    
    # Plot each of the  pulses in the time domain
    # Each pulse will have N samples and the pulse will be at the center of the N samples
    # Create a figure with D delta pulses
    # D = len(pulse_indices)
    # fig, axes = plt.subplots(D, 1, figsize=(10, 10))
    # pulses = []
    # # window_size = 256
    # for i, pulse_index in enumerate(pulse_indices):
    #     x_start = pulse_index
    #     x_end = pulse_index+window_size
    #     pulse = example_data[int(x_start):int(x_end)]
    #     pulses.append(pulse)
    #     axes[i].plot(pulse.real, label='Real')
    #     axes[i].plot(pulse.imag, label='Imaginary')
    #     # Mark the pulse index with a vertical line
    #     # Change the x-axis to the pulse index
    #     # axes[i].set_xlim(x_start, x_end)
    #     # axes[i].axvline(x=pulse_index, color='red', linestyle='--')
    #     axes[i].set_title("Pulse")
    #     axes[i].set_xlabel("Sample Number")
    #     axes[i].set_ylabel("Amplitude")
    #     # Set x axis ticks to start and end of the pulse
    #     # axes[i].set_xticks([int(x_start), int(x_end)])
    #     xticks = np.linspace(int(x_start), int(x_end))
    #     # axes[i].set_xticks(xticks)
    #     axes[i].set_xticklabels([int(x) for x in xticks])
    #     # mark the pulse index with a vertical line
    #     # indexMarker = window_size/2
    #     # print("Index marker: ", indexMarker)
    #     # axes[i].axvline(x=indexMarker, color='red', linestyle='--')
    #     # axes[i].legend()
    # # Set independent x axis for each subplot
    # plt.subplots_adjust(hspace=1)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("XTEST_IQ_Pulses.png")
    
    # Create a spectrogram of the pulses
    # Apply FFT to each pulse in pulses
    # Create a new figure with single plot
    
    # spectrograms = []
    # for pulse in pulses:
    #     spectrogram = np.abs(np.fft.fftshift(np.fft.fft(pulse, axis=0), axes=0))
    #     spectrograms.append(spectrogram)
    # # Plot the spectrograms
    # fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    # plt.imshow(spectrograms, aspect='auto', cmap='jet')
    # plt.title("Spectrograms")
    # plt.xlabel("Frequency")
    # plt.ylabel("Time")
    # plt.colorbar()
    # plt.savefig("XTEST_IQ_Pulses_Spectrograms_per_pulse.png")
    # exit()
    
    batch_size = 128 
    fft_len = int(256) #256
    patience = 100
    maxEpochs = 200
    val_size = 0.1
    factor = 0.1
    alphas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    betas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    gammas = [0.1] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    FFT_lengths = [fft_len]
    output_lengths = [128] # Need to define the output length and why based on 255 constraint (bytes)
    optimizers = ["SGD"] # "RMSprop", "SGD", "Adam"
    network_types = ["RNN"] # ["ResNet", "FeedForward", "RNN", "Transformer", "AE"]
    # data_types = ["IQ", "Polar", "Spectrogram"] # ["IQ", "Polar", "Spectrogram"]
    data_type = "Spectrogram" 
    quantization_layer = True
    for fft_len in FFT_lengths:
        for output_length in output_lengths:
            for network_type in network_types:
                for a in alphas:
                    for b in betas:
                        for g in gammas:
                            for o in optimizers:
                                # b = a
                                if o == "RMSprop":
                                    lr = 0.0001
                                elif o == "SGD":
                                    lr = 0.1
                                elif o == "Adam":
                                    lr = 0.001
                                print("Alpha: ", a, "Beta: ", b, "Gamma: ", g, "Optimizer: ", o, "Network Type: ", network_type)
                                train_configurations = {
                                    "QuadrupletNet": {
                                        "alpha": a,
                                        "beta": b,
                                        "gamma": g,
                                        "data_type": data_type,
                                        "fft_len": fft_len,
                                        "output_length": output_length,
                                        "batch_size": batch_size,
                                        "validation_size": val_size,
                                        "LearningRate": lr,
                                        "epochs": maxEpochs,
                                        "patience": patience,
                                        "factor": factor,
                                        "optimizer": o,
                                    },
                                    "TripletNet": {
                                        "alpha": a,
                                        "beta": b,
                                        "data_type": data_type,
                                        "fft_len": fft_len,
                                        "output_length": output_length,
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
                                
                                filename_start = train_configurations[model_type]['data_type'] \
                                        +'_FeatureExtractor_'+network_type \
                                        +('_QuantizationLayerKDR' if quantization_layer else "") \
                                        +'_in'+str(train_configurations[model_type]["fft_len"]) \
                                        +'_out'+str(train_configurations[model_type]["output_length"]) \
                                        +'_alpha'+str(train_configurations[model_type]["alpha"]) \
                                        +'_beta'+str(train_configurations[model_type]["beta"]) \
                                        +('_gamma'+str(train_configurations[model_type]["gamma"]) if "gamma" in train_configurations[model_type] else "") \
                                        +'_'+train_configurations[model_type]['optimizer'] \
                                        +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                                        +'_'+configuration["config_name"]+'_'+str(6)
                                
                                ModelsDir = homeDir+"Models/"
                                
                                model_exisits = False
                                # Check if there is a file that starts with the same name
                                for file in os.listdir(ModelsDir):
                                    if file.startswith(filename_start):
                                        model_exisits = True
                                        filename = file
                                        break
                                if model_exisits:
                                    print("Model already exists")
                                    print("Skipping: ", filename)
                                    # Plot the history of the model located in results directory
                                    history_file = homeDir+"Results/History_"+filename
                                    history = h5py.File(history_file, "r")
                                    train_loss = history["train_loss"][:]
                                    validation_loss = history["validation_loss"][:]
                                    plt.plot(train_loss)
                                    plt.plot(validation_loss)
                                    plt.title(filename+" Model History")
                                    plt.ylabel("Loss")
                                    plt.xlabel("Epoch")
                                    plt.legend(["Train", "Validation"], loc="upper left")
                                    plt.show()
                                    plt.savefig(homeDir+"Results/"+filename_start+"_history.png")
                                    continue
                                else:
                                    print("Training model: ", filename_start)
                                    
                                    feature_extractor = train_channel_feature_extractor(
                                        data,
                                        labels,
                                        train_configurations,
                                        model_type,
                                        network_type,
                                        filename_start,
                                        seed=REPRO_SEED,
                                        quantization_layer=quantization_layer
                                        
                                    )