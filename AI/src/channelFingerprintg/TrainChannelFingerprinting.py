import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from deep_learning_models import identity_loss, QuadrupletNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram

import time
from TestChannelFingerprinting import test_model

def train_channel_feature_extractor(data, labels, train_configurations, model_type):
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
        
    ChannelSpectrogramObj = ChannelSpectrogram()
    
    # Convert time-domain IQ samples to channel-independent spectrograms.
    data = ChannelSpectrogramObj.channel_spectrogram(
        data,
        train_configurations[model_type]['fft_len']
    )

    #NetObj =  TripletNet_Channel()
    NetObj = QuadrupletNet_Channel()
    
    # Create an RFF extractor.
    feature_extractor = NetObj.feature_extractor(data.shape)

    feature_extractor.summary()
    
    # Create the quadruplet net using the RFF extractor.
    net = NetObj.create_quadruplet_net(
        feature_extractor, 
        train_configurations[model_type]['alpha'], 
        train_configurations[model_type]['beta'], 
        train_configurations[model_type]['gamma']
    )

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    patience = train_configurations[model_type]['patience']
    
    early_stop = EarlyStopping('val_loss', 
                                min_delta = 0, 
                                patience = patience
                                )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    min_delta = 0, 
                                    factor = train_configurations[model_type]['factor'], 
                                    patience = int(math.ceil(patience/2)), 
                                    verbose=1
                                    )
    callbacks = [early_stop, reduce_lr]

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
                                                        label_train)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_valid, 
                                                        label_valid)
    
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
    
    return feature_extractor

if __name__ == "__main__":
    
    node_configurations = {
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
                # [4,3,5]
            ]
        }
    }
    
    batch_size = 64
    fft_len = 512
    patience = 30
    maxEpochs = 1000
    lr = 0.5
    val_size = 0.15
    factor = 0.5
    optimizer = "SGD" # "RMSprop", "SGD", "Adam"
    train_configurations = {
        "QuadrupletNet": {
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 0.1,
            "fft_len": fft_len,
            "batch_size": batch_size,
            "validation_size": val_size,
            "LearningRate": lr,
            "epochs": maxEpochs,
            "patience": patience,
            "factor": factor,
            "optimizer": optimizer
        },
        "TripletNet": {
            "alpha": 0.5,
            "beta": 0.5,
            "fft_len": fft_len,
            "batch_size": batch_size,
            "validation_size": val_size,
            "LearningRate": lr,
            "epochs": maxEpochs,
            "patience": patience,
            "factor": factor,
            "optimizer": optimizer
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
    data, labels = dataset.load_data()
    
    model_type = "QuadrupletNet"
    feature_extractor = train_channel_feature_extractor(data, labels, train_configurations, model_type)
    
    timestamp = int(time.time())
    filename = 'FeatureExtractor_'+str(train_configurations[model_type]["fft_len"]) \
                +'_alpha'+str(train_configurations[model_type]["alpha"]) \
                +'_beta'+str(train_configurations[model_type]["beta"]) \
                +'_'+train_configurations[model_type]['optimizer'] \
                +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                +'_'+configuration["config_name"] \
                +'_'+str(timestamp) \
                +'.h5'
    
    feature_extractor.save(filename)
    print("Saving file: ", filename)
    
    # TESTING
    test = True
    if test:
        test_node_configurations = {
            'OTA-Lab': {
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
                    [1,2,3],
                    [1,2,5],
                    [1,3,2],
                    [4,3,5]
                ]
            }
        }
        test_model(filename, test_node_configurations['OTA-Dense'])