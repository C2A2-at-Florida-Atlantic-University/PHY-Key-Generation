from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop

# from dataset_preparation import ChannelSpectrogram, LoadDatasetChannels
from deep_learning_models import identity_loss, QuadrupletNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram

import time

def train_channel_feature_extractor(dataset_name, config_name, repo_name, epochs=1000):
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
    
    dataset_handler = DatasetHandler(dataset_name, config_name, repo_name)
    dataset_handler.get_dataframe_Info()
    
    # Load preamble IQ samples and labels.
    data,label = dataset_handler.load_data()
    
    # Add additive Gaussian noise to the IQ samples.
    #data = awgn(data, snr_range)
    
    ChannelSpectrogramObj = ChannelSpectrogram()
    
    # Convert time-domain IQ samples to channel-independent spectrograms.
    fft_len = 256
    data = ChannelSpectrogramObj.channel_spectrogram(data,fft_len)

    print("len: ", len(data))
    print("shape: ", data.shape)

    data_length = len(data)
    alpha = 0.5
    beta = 0

    batch_size = 64
    patience = 20

    #NetObj =  TripletNet_Channel()
    NetObj = QuadrupletNet_Channel()
    
    # Create an RFF extractor.
    feature_extractor = NetObj.feature_extractor(data.shape)

    feature_extractor.summary()
    
    # Create the quadruplet net using the RFF extractor.
    #net = NetObj.create_triplet_net(feature_extractor, alpha1)
    net = NetObj.create_quadruplet_net(feature_extractor, alpha, beta)

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    early_stop = EarlyStopping('val_loss', 
                               min_delta = 0, 
                               patience = 
                               patience)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  min_delta = 0, 
                                  factor = 0.1, 
                                  patience = 10, 
                                  verbose=1
                                  )
    callbacks = [early_stop, reduce_lr]

    validation_size=0.1
    # Split the dasetset into validation and training sets.
    data_train, data_valid, label_train, label_valid = train_test_split(data, 
                                                                        label, 
                                                                        test_size=validation_size, 
                                                                        shuffle= False)
    del data, label
    
    # Create the trainining generator.
    train_generator = NetObj.create_generator_channel(batch_size, 
                                                     data_train, 
                                                     label_train)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                     data_valid, 
                                                     label_valid)
    
    # Use the RMSprop optimizer for training.
    LearningRate = 1e-3
    opt = RMSprop(learning_rate=LearningRate)

    net.compile(
        loss = identity_loss,
        #metrics=['accuracy'],
        optimizer = opt)

    # Start training.
    history = net.fit(train_generator,
                              steps_per_epoch = data_train.shape[0]//batch_size,
                              epochs = epochs,
                              validation_data = valid_generator,
                              validation_steps = data_valid.shape[0]//batch_size,
                              verbose=1, 
                              callbacks = callbacks)

    timestamp = time.time()
    
    savedFile = 'QExtractor2_'+str(fft_len)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_batch'+str(batch_size)+'_val'+str(validation_size)+'_RMS'+str(LearningRate)+'_DSsin2.4dev1278-'+str(data_length)+'_'+str(timestamp)+'.h5'

    feature_extractor.save(savedFile)
    
    print("Saving file: ", savedFile)
    
    return feature_extractor

run_for = 'Train Channel Fingerprinting'

if run_for == 'Train Channel Fingerprinting':

    dataset_name = "Key-Generation"
    config_name = "Sinusoid-Powder-OTA-Dense" #"Sinusoid-Powder-OTA-Lab"
    repo_name="CAAI-FAU"
    # Train an RFF extractor.
    # Save the trained model.
    feature_extractor = train_channel_feature_extractor(dataset_name, config_name, repo_name, epochs=1000)