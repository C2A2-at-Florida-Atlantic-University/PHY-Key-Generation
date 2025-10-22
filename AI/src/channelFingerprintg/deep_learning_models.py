import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, ReLU, Add, Dense, Conv2D, Conv2DTranspose, Flatten, AveragePooling2D, Dropout, BatchNormalization, Reshape, Permute, GlobalAveragePooling1D, Bidirectional, GRU, MultiHeadAttention, LayerNormalization, Concatenate
from tensorflow.keras.regularizers import l2

def divisible_random(a, b, n, rng: random.Random = None):
    """Random integer in [a, b] divisible by n. Uses provided RNG if given.

    Providing an RNG enables deterministic sampling tied to a seed.
    """
    if b - a < n:
        raise Exception('{} is too big'.format(n))
    _rng = rng if rng is not None else random
    result = _rng.randint(a, b)
    while result % n != 0:
        result = _rng.randint(a, b)
    return result

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)           

class TripletNet_Channel():
    def __init__(self):
        pass
        
    def create_triplet_net(self, embedding_net, alpha):
        self.alpha = alpha
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N = embedding_net(input_3)
        loss = Lambda(self.triplet_loss)([A, P, N]) 
        model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
        return model

    def triplet_loss_KDR(self,x):
        # Triplet Loss function.
        anchor,positive,negative = x
        anchor = self.quantization_layer(anchor)
        positive = self.quantization_layer(positive)
        negative = self.quantization_layer(negative)
        # Creating bool feature vectors for logical XOR
        zeros = tf.zeros_like(anchor)
        ones = tf.ones_like(anchor)
        #GET BOOLEAN VALUES       
        anchor_bool = tf.greater_equal(anchor,1)
        positive_bool = tf.greater_equal(positive,1)
        negative_bool = tf.greater_equal(negative,1)
        #XOR(ANCHOR,POSITIVE) , XOR(ANCHOR,NEGATIVE)
        pos_xor_bool = tf.math.logical_xor(anchor_bool,positive_bool)
        pos_xor_not_bool = tf.math.logical_not(pos_xor_bool)
        neg_xor_bool = tf.math.logical_xor(anchor_bool,negative_bool)
        neg_xor_not_bool = tf.math.logical_not(neg_xor_bool)
        #TRANSFORM BOOLEAN TO BINARY
        pos_xor = tf.where(pos_xor_bool, anchor, zeros)
        pos_xor = tf.where(pos_xor_not_bool, pos_xor, ones)
        neg_xor  = tf.where(neg_xor_bool, anchor, zeros)
        neg_xor = tf.where(neg_xor_not_bool, neg_xor, ones)
        #GET POSITIVE AND NEGATIVE MEAN (SUM(VALUES)/LENGTH(VALUES))
        pos_kdr = K.mean(pos_xor,axis=1)
        neg_kdr = K.mean(neg_xor,axis=1)
        #Calculate Loss
        basic_loss = (pos_kdr-neg_kdr+self.alpha)
        loss = K.maximum(basic_loss,0.0)
        return loss

    def triplet_loss(self,x):
        # Triplet Loss function.
        anchor,positive,negative = x
        # K.l2_normalize
        # distance between the anchor and the positive
        pos_dist = K.sum(K.square(anchor-positive),axis=1)
        # distance between the anchor and the negative
        neg_dist = K.sum(K.square(anchor-negative),axis=1)
        basic_loss = (pos_dist-neg_dist) + self.alpha
        loss = K.maximum(basic_loss,0.0)
        return loss  

    def quantization_layer(self,x):
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        x_mean = K.mean(x)
        x_less = K.less(x,x_mean)
        x_greater = K.greater_equal(x,x_mean)
        x_q = tf.where(x_greater, x, ones)
        x_q = tf.where(x_less, x_q, zeros)
        return x_q
    
    def resblock(self, x, kernelsize, filters, first_layer = False):
        reg = l2(0.001)  # Define L2 regularizer
        if first_layer:
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
            fx = BatchNormalization()(fx)
            x = Conv2D(filters, 1, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            out = Add()([x,fx])
            out = ReLU()(out)
        else:
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
            fx = BatchNormalization()(fx)
            out = Add()([x,fx])
            out = ReLU()(out)

        return out 
    
    def feature_extractor(self, datashape):
        self.datashape = datashape
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same')(inputs)
        x =  Dropout(0.3)(x)
        x = self.resblock(x, 3, 32)
        x = self.resblock(x, 3, 32)
        #x = resblock(x, 3, 32)
        x = self.resblock(x, 3, 64, first_layer = True)
        x = self.resblock(x, 3, 64)
        #x = resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(512)(x)
        #x = Dense(512,kernel_regularizer='l1_l2')(x)
        outputs = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model             
    
    def create_generator_channel(self, batchsize, data, label, seed: int = None):
        """Generate a triplets generator for training."""
        self.data = data
        self.label = label
        local_rng = random.Random(seed) if seed is not None else None
        while True:
            list_a = []
            list_p = []
            list_n = []
            idx = divisible_random(0, len(data)-4, 4, rng=local_rng)
            #batchsize_limit = batchsize+idx-1
            #print("batchsize_limit",batchsize_limit)
            #print("idx",idx)
            batch = 0
            while batch < batchsize:
                decision = (bool(local_rng.getrandbits(1)) if local_rng is not None else bool(random.getrandbits(1)))
                if decision:
                    a =data[idx]
                    n =data[idx+1]
                    p =data[idx+2]
                    list_a.append(a)
                    list_p.append(p)
                    list_n.append(n)
                else:
                    a =data[idx+2]
                    n =data[idx+3]
                    p =data[idx]
                    list_a.append(a)
                    list_p.append(p)
                    list_n.append(n)
                idx = divisible_random(0, len(data)-4, 4, rng=local_rng)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N = np.array(list_n, dtype='float32')
            label = np.ones(int(batchsize))
            #print("label",label.shape)
            #print("A",A.shape)
            yield [A, P, N], label 

class QuadrupletNet():
    def __init__(self):
        pass
        
    def create_quadruplet_net(self, embedding_net, alpha, beta, gamma=None):
        
        # embedding_net = encoder()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_3 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_4 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        N1 = embedding_net(input_3)
        N2 = embedding_net(input_4)
        
        loss = Lambda(self.quadruplet_loss)([A, P, N1, N2]) 
        return Model(inputs=[input_1, input_2, input_3, input_4], outputs=loss)

    # Quadruplet Loss function.
    def quadruplet_loss(self,x):
        # getting Quadruplet
        anchor,positive,negative1,negative2 = x
        # distance between the anchor and the positive features (Alice-Bob & Bob-Alice)
        D1 = K.sum(K.square(anchor-positive),axis=1)
        # distance between the anchor and negative 1 features (Alice-Bob & Alice-Eve)
        D2 = K.sum(K.square(anchor-negative1),axis=1)
        # distance between the anchor and negative 2 features (Bob-Alice & Bob-Eve)
        D3 = K.sum(K.square(positive-negative2),axis=1)
        # distance between the negative 1 and negative 2 features (Alice-Eve & Bob-Eve)
        D4 = K.sum(K.square(negative1-negative2),axis=1)
        # Loss taking into account all distances
        if self.gamma is not None:
            loss = K.maximum(D1 - D2 + self.alpha,0.0) + K.maximum(D1 - D3 + self.beta,0.0) + K.maximum(D1 - D4 + self.gamma,0.0)
        else:
            # Loss taking into account only the distances between the anchor and the positive and the negative 1
            loss = K.maximum(D1 - D2 + self.alpha,0.0) + K.maximum(D1 - D3 + self.beta,0.0)
        return loss        

    def quantization_layer(self,x):
        ones = tf.ones_like(x)
        zeros = tf.zeros_like(x)
        x_mean = K.mean(x)
        x_less = K.less(x,x_mean)
        x_greater = K.greater_equal(x,x_mean)
        x_q = tf.where(x_greater, x, ones)
        x_q = tf.where(x_less, x_q, zeros)
        return x_q           
            
    def create_generator_channel(self, batchsize, data, label, seed: int = None):
        """Generate a quadruplets generator for training."""
        self.data = data
        self.label = label
        local_rng = random.Random(seed) if seed is not None else None
        while True:
            list_a = []
            list_p = []
            list_n1 = []
            list_n2 = []
            idx = divisible_random(0, len(data)-4, 4, rng=local_rng)
            #batchsize_limit = batchsize+idx-1
            #print("batchsize_limit",batchsize_limit)
            #print("idx",idx)
            batch = 0
            while batch < batchsize:
                a =data[idx]
                n1 =data[idx+1]
                p =data[idx+2]
                n2 =data[idx+3]
                list_a.append(a)
                list_p.append(p)
                list_n1.append(n1)
                list_n2.append(n2)
                idx = divisible_random(0, len(data)-4, 4, rng=local_rng)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N1 = np.array(list_n1, dtype='float32')
            N2 = np.array(list_n2, dtype='float32')
            label = np.ones(int(batchsize))
            #print("label",label.shape)
            #print("A",A.shape)
            yield [A, P, N1, N2], label 

class ResNet_QuadrupletNet_Channel(QuadrupletNet):
    def __init__(self):
        pass
    
    def resblock(self, x, kernelsize, filters, first_layer = False):
        reg = l2(0.001)  # Define L2 regularizer
        if first_layer:
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
            fx = BatchNormalization()(fx)
            x = Conv2D(filters, 1, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            out = Add()([x,fx])
            out = ReLU()(out)
        else:
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(x)
            fx = BatchNormalization()(fx)
            fx = ReLU()(fx)
            fx = Conv2D(filters, kernelsize, padding='same', kernel_regularizer=reg)(fx)
            fx = BatchNormalization()(fx)
            out = Add()([x,fx])
            out = ReLU()(out)

        return out 

    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        reg = l2(0.001)  # Define L2 regularizer
        inputs = Input(shape=([self.datashape[1],self.datashape[2],self.datashape[3]]))
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
        x =  Dropout(0.3)(x)
        x = self.resblock(x, 3, 32, first_layer = True)
        for _ in range(2):
            x = self.resblock(x, 3, 32)
        x = self.resblock(x, 3, 64, first_layer = True)
        for _ in range(2):
            x = self.resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(output_length)(x)
        x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        # x = Lambda(lambda x: tf.cast(x > 0.5, dtype=tf.float32))(x) # Quantization layer
        # x = Dense(units=output_length, activation='sigmoid', kernel_initializer="lecun_normal")(x)
        model = Model(inputs=inputs, outputs=x)
        return model
    
class FeedForward_QuadrupletNet_Channel(QuadrupletNet):
    def __init__(self, embed_dim=512, width=[2048, 1024], dropout=0.3, l2_w=1e-3):
        self.embed_dim = embed_dim
        self.width = width
        self.dropout = dropout
        self.l2_w = l2_w
    
    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        reg = l2(self.l2_w)
        
        def mlp_layer(y, units):
            # y = Dense(units, kernel_regularizer=reg)(y)
            y = Conv2D(units, 1, padding='same', kernel_regularizer=reg)(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            return y

        def mlp_block(y, units):
            # Two-layer block mirroring a ResNet block (without residual Add)
            y = mlp_layer(y, units)
            y = mlp_layer(y, units)
            return y

        inputs = Input(shape=(datashape[1], datashape[2], datashape[3]))
        # x = Flatten()(inputs)
        x = Conv2D(32, 7, strides = 2, activation='relu', padding='same', kernel_regularizer=reg)(inputs)
        x = Dropout(self.dropout)(x)
        # Stage 1: first-layer block + 3 blocks
        x = mlp_block(x, 32)
        x = mlp_block(x, 32)
        x = mlp_block(x, 64)
        x = mlp_block(x, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)  # unit-norm embeddings
        # x = Dense(units=output_length, activation='sigmoid', kernel_initializer="lecun_normal")(x)

        return Model(inputs, x, name="MLP_Encoder")

# Recurrent Neural Network (RNN) model
class RNN_QuadrupletNet_Channel(QuadrupletNet):
    """
    GRU over time axis:
      - Treat spectrogram time (W) as sequence; features per step = F*C
      - Bi-GRU -> pooling -> 512-D embedding.
    """
    def __init__(self,
                 embed_dim=512,
                 gru_units=384,
                 td_width=256,
                 dropout=0.3,
                 recurrent_dropout=0.0,
                 l2_w=1e-3,
                 bidirectional=True):
        self.embed_dim = embed_dim
        self.gru_units = gru_units
        self.td_width = td_width
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.l2_w = l2_w
        self.bidirectional = bidirectional

    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        # F: Frequency, T: Time, C: Channels
        F, T, C = datashape[1], datashape[2], datashape[3]
        reg = l2(self.l2_w)
        
        def mlp_layer(y, units):
            # y = Dense(units, kernel_regularizer=reg)(y)
            # y = Conv2D(units, 1, padding='same', kernel_regularizer=reg)(y)
            y = Dense(units, kernel_regularizer=reg)(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)
            return y

        def mlp_block(y, units):
            # Two-layer block mirroring a ResNet block (without residual Add)
            y = mlp_layer(y, units)
            y = mlp_layer(y, units)
            return y

        inputs = Input(shape=(F, T, C))
        # Treat time as sequence; per-step features = F*C
        x = Permute((2,1,3))(inputs)                      # (T, F, C)
        x = Reshape((T, F*C))(x)                          # (T, F*C)
        x = Dense(output_length, activation='relu', kernel_regularizer=reg)(x)
        x = LayerNormalization(epsilon=1e-5)(x)
        x = Dropout(self.dropout)(x)

        # Single GRU layer (optionally bidirectional)
        if self.bidirectional:
            x = Bidirectional(GRU(self.gru_units, return_sequences=True, dropout=self.dropout))(x)
        else:
            x = GRU(self.gru_units, return_sequences=True, dropout=self.dropout)(x)

        # Normalize sequence features before pooling
        x = LayerNormalization(epsilon=1e-5)(x)

        # Projection head -> L2-normalized embedding
        # x = Dense(output_length, kernel_regularizer=reg)(pooled)
        # x = BatchNormalization()(x)
        # add two mlp blocks
        x = mlp_block(x, int(np.ceil(output_length/2)))
        x = mlp_block(x, output_length)
        # Average pooling
        x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)

        return Model(inputs, x, name="GRU_Encoder")
    
class RNN_QuadrupletNet_Channel2(QuadrupletNet):
    """
    GRU over time axis:
      - Treat spectrogram time (W) as sequence; features per step = F*C
      - Bi-GRU -> pooling -> 512-D embedding.
    """
    def __init__(self,
                 embed_dim=512,
                 gru_units=384,
                 td_width=256,
                 dropout=0.3,
                 recurrent_dropout=0.0,
                 l2_w=1e-3,
                 bidirectional=True,
                 num_gru_layers=1,
                 use_attention=False,
                 n_heads=4,
                 ff_multiplier=2):
        self.embed_dim = embed_dim
        self.gru_units = gru_units
        self.td_width = td_width
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.l2_w = l2_w
        self.bidirectional = bidirectional
        self.num_gru_layers = num_gru_layers
        self.use_attention = use_attention
        self.n_heads = n_heads
        self.ff_multiplier = ff_multiplier

    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        # F: Frequency, T: Time, C: Channels
        F, T, C = datashape[1], datashape[2], datashape[3]
        reg = l2(self.l2_w)

        inputs = Input(shape=(F, T, C))
        # Treat time as sequence; per-step features = F*C
        x = Permute((2,1,3))(inputs)                      # (T, F, C)
        x = Reshape((T, F*C))(x)                          # (T, F*C)
        x = Dense(output_length, activation='relu', kernel_regularizer=reg)(x)
        x = LayerNormalization(epsilon=1e-5)(x)
        x = Dropout(self.dropout)(x)

        # Stacked GRU blocks with optional attention and residual connections
        d_model = int(self.gru_units * (2 if self.bidirectional else 1))
        for _ in range(self.num_gru_layers):
            gru_layer = GRU(self.gru_units, return_sequences=True,
                            dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)
            y = Bidirectional(gru_layer)(x) if self.bidirectional else gru_layer(x)
            y = LayerNormalization(epsilon=1e-5)(y)
            # Residual connection if dimensions align
            if y.shape[-1] == x.shape[-1]:
                y = Add()([x, y])
            x = y

            if self.use_attention:
                attn = MultiHeadAttention(num_heads=self.n_heads,
                                          key_dim=max(1, d_model // self.n_heads),
                                          dropout=self.dropout)(x, x)
                x = Add()([x, Dropout(self.dropout)(attn)])
                x = LayerNormalization(epsilon=1e-5)(x)

            ff = Dense(self.ff_multiplier * d_model, activation='relu')(x)
            ff = Dropout(self.dropout)(ff)
            ff = Dense(d_model)(ff)
            x = Add()([x, Dropout(self.dropout)(ff)])
            x = LayerNormalization(epsilon=1e-5)(x)
        # Average pooling
        x = GlobalAveragePooling1D()(x)
        x = Flatten()(x)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)

        return Model(inputs, x, name="GRU_Encoder")
    
# Transformer model
class Transformer_QuadrupletNet_Channel(QuadrupletNet):
    """
    Tiny Transformer encoder over time tokens:
      - Tokens = time steps (W); features per token = F*C projected to d_model
      - 2 encoder blocks -> pooled -> 512-D embedding
    """
    def __init__(self, embed_dim=512, d_model=256, n_heads=4, mlp_dim=512, depth=2, dropout=0.1, l2_w=1e-3, max_len=2048):
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.depth = depth
        self.dropout = dropout
        self.l2_w = l2_w
        self.max_len = max_len  # for learned positional embeddings

    def _encoder_block(self, x):
        # Multi-Head Self-Attention
        attn = MultiHeadAttention(num_heads=self.n_heads, key_dim=self.d_model//self.n_heads, dropout=self.dropout)(x, x)
        x = Add()([x, Dropout(self.dropout)(attn)])
        x = LayerNormalization(epsilon=1e-5)(x)

        # Feed-forward
        ff = Dense(self.mlp_dim, activation='relu')(x)
        ff = Dropout(self.dropout)(ff)
        ff = Dense(self.d_model)(ff)
        x = Add()([x, Dropout(self.dropout)(ff)])
        x = LayerNormalization(epsilon=1e-5)(x)
        return x

    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        F, T, C = datashape[1], datashape[2], datashape[3]
        reg = l2(self.l2_w)

        inputs = Input(shape=(F, T, C))
        # Tokens = time steps; features = F*C
        x = Permute((2,1,3))(inputs)             # (T, F, C)
        x = Reshape((T, F*C))(x)                 # (T, F*C)

        # Linear projection to d_model
        x = Dense(self.d_model, activation=None, kernel_regularizer=reg)(x)

        # Learnable positional embeddings (length up to max_len)
        # Build positions [0..T-1] -> (batch, T)
        pos_idx = Lambda(lambda t: tf.tile(tf.expand_dims(tf.range(tf.shape(t)[1]), axis=0),
                                           [tf.shape(t)[0], 1]))(x)
        pos_embed = tf.keras.layers.Embedding(input_dim=self.max_len, output_dim=self.d_model)(pos_idx)
        x = Add()([x, pos_embed])

        # Stacked encoder blocks
        for _ in range(self.depth):
            x = self._encoder_block(x)

        # Pool tokens
        x = GlobalAveragePooling1D()(x)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)

        return Model(inputs, x, name="Transformer_Encoder")
    
class AE_QuadrupletNet_Channel(QuadrupletNet):
    """
    Convolutional Autoencoder (CAE):
      - Encoder: 2D conv downsamples spectrogram to a compact embedding (default 512-D, L2-normalized)
      - Decoder: symmetric deconvs for reconstruction pretraining (optional)
      - Use encoder output for quadruplet training with your [A, P, N1, N2] dataset.
    """
    def __init__(self,
                 embed_dim=512,
                 chans=(32, 64, 128),
                 dropout=0.2,
                 l2_w=1e-3,
                 recon_activation='linear'  # 'sigmoid' if inputs scaled to [0,1]
                 ):
        self.embed_dim = embed_dim
        self.chans = chans
        self.dropout = dropout
        self.l2_w = l2_w
        self.recon_activation = recon_activation

    # --- small conv block helper
    def _conv_block(self, x, filters, stride=2, reg=None):
        x = Conv2D(filters, 3, strides=stride, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x); x = ReLU()(x)
        return x

    # --- decoder block helper
    def _deconv_block(self, x, filters, stride=2, reg=None):
        x = Conv2DTranspose(filters, 3, strides=stride, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x); x = ReLU()(x)
        return x

    # ------------------- Encoder only (for quadruplet training) ----------------
    def feature_extractor(self, datashape, output_length):
        self.datashape = datashape
        reg = l2(self.l2_w)

        inputs = Input(shape=(datashape[1], datashape[2], datashape[3]))
        x = inputs
        # (Optionally) denoising during training—keep mild for small data
        x = Dropout(self.dropout)(x)

        # Downsampling conv tower
        shapes = []  # keep shapes only if you later want to attach skip connections
        for f in self.chans:
            x = self._conv_block(x, f, stride=2, reg=reg)
            shapes.append(tf.shape(x))  # not used but left as doc

        # Bottleneck
        x = Conv2D(self.chans[-1], 3, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x); x = ReLU()(x)

        # Projection to embedding
        x = Flatten()(x)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = Lambda(lambda t: K.l2_normalize(t, axis=1))(x)  # 512-D unit-norm

        return Model(inputs, x, name="CAE_Encoder")

    # --------------- Full AE for (optional) reconstruction pretraining --------
    def build_autoencoder_pretrain(self, datashape, output_length):
        """Returns (encoder, autoencoder). Train AE with MSE (or SSIM) before quadruplet."""
        self.datashape = datashape
        reg = l2(self.l2_w)

        # Encoder
        enc_in = Input(shape=(datashape[1], datashape[2], datashape[3]))
        x = Dropout(self.dropout)(enc_in)
        conv_shapes = []  # store spatial dims for decoder
        for f in self.chans:
            x = self._conv_block(x, f, stride=2, reg=reg)
            conv_shapes.append(x.shape[1:3])  # (H, W)

        x = Conv2D(self.chans[-1], 3, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x); x = ReLU()(x)
        # Save spatial size & channels for decoder seed
        h_last, w_last = x.shape[1], x.shape[2]
        c_last = x.shape[-1]

        flat = Flatten()(x)
        emb  = Dense(output_length, kernel_regularizer=reg)(flat)
        emb  = BatchNormalization()(emb)
        emb_n= Lambda(lambda t: K.l2_normalize(t, axis=1), name="embedding")(emb)

        encoder = Model(enc_in, emb_n, name="CAE_Encoder")

        # Decoder (mirror)
        dec_in = Input(shape=(output_length,))
        x = Dense(int(h_last)*int(w_last)*int(c_last), kernel_regularizer=reg)(dec_in)
        x = BatchNormalization()(x); x = ReLU()(x)
        x = Reshape((int(h_last), int(w_last), int(c_last)))(x)

        # Reverse conv stack
        for f in reversed(self.chans):
            x = self._deconv_block(x, f, stride=2, reg=reg)

        # Final reconstruction to input channels
        recon = Conv2D(datashape[3], 3, padding='same', activation=self.recon_activation)(x)
        decoder = Model(dec_in, recon, name="CAE_Decoder")

        # Wire encoder+decoder into pretrain model
        ae_out = decoder(emb_n)
        autoencoder = Model(enc_in, ae_out, name="CAE_Pretrain")

        return encoder, autoencoder