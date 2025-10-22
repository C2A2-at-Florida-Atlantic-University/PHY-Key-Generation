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
import tensorflow as tf

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

def train_channel_feature_extractor(data, labels, train_configurations, model_type, network_type="ResNet", model_name="", seed: int = 42):
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
    feature_extractor = NetObj.feature_extractor(data.shape, output_length)
    
    # Create the quadruplet net using the RFF extractor.
    net = NetObj.create_quadruplet_net(
        feature_extractor, 
        train_configurations[model_type]['alpha'], 
        train_configurations[model_type]['beta'], 
        train_configurations[model_type]['gamma'] if 'gamma' in train_configurations[model_type] else None
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

if __name__ == "__main__":
    homeDir = "/home/Research/POWDER/"
    # Central seed for full reproducibility of data shuffling, initialization, and training
    
    node_configurations = {
        'OTA-Lab': {
            'dataset_name': 'Key-Generation',
            'config_name': 'Sinusoid-Powder-OTA-Lab-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': [
                # [1,2,3],
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
                [1,2,3],
                [1,2,5],
                [1,3,2],
                [4,3,5]
            ]
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
        else:
            dataset.add_dataset(dataset_name, config_name, repo_name)
        dataset.get_dataframe_Info()
    # Set seeds and determinism prior to any TF ops
    REPRO_SEED = 42
    set_reproducible(REPRO_SEED)
    data, labels, rx, tx = dataset.load_data(shuffle=True, seed=REPRO_SEED)
    
    batch_size = 128
    fft_len = 256
    patience = 100
    maxEpochs = 300
    lr = 0.0001
    val_size = 0.10
    factor = 0.1
    optimizer = "Adam" # "RMSprop", "SGD", "Adam"

    alphas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    betas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    gammas = [0.1] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    FFT_lengths = [256]
    output_lengths = [128, 256, 512]
    optimizers = ["Adam"] # "RMSprop", "SGD", "Adam"
    network_types = ["RNN"] # ["ResNet", "FeedForward", "RNN", "Transformer", "AE"]
    data_types = ["IQ", "Polar", "Spectrogram"]
    data_type = data_types[1]
    for fft_len in FFT_lengths:
        for output_length in output_lengths:
            for network_type in network_types:
                for a in alphas:
                    for b in betas:
                        for g in gammas:
                            for o in optimizers:
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
                                
                                filename_start = train_configurations[model_type]['data_type']+'_FeatureExtractor_'+network_type+'_in'+str(train_configurations[model_type]["fft_len"]) \
                                        +'_out'+str(train_configurations[model_type]["output_length"]) \
                                        +'_alpha'+str(train_configurations[model_type]["alpha"]) \
                                        +'_beta'+str(train_configurations[model_type]["beta"]) \
                                        +('_gamma'+str(train_configurations[model_type]["gamma"]) if "gamma" in train_configurations[model_type] else "") \
                                        +'_'+train_configurations[model_type]['optimizer'] \
                                        +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                                        +'_'+configuration["config_name"]
                                
                                ModelsDir = homeDir+"Models/"
                                
                                model_exisits = False
                                # Check if there is a file that starts with the same name
                                for file in os.listdir(ModelsDir):
                                    if file.startswith(filename_start):
                                        model_exisits = True
                                        break
                                if model_exisits:
                                    print("Model already exists")
                                    print("Skipping: ", filename_start)
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
                                    )
                                    