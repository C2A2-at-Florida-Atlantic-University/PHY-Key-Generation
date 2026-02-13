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


@tf.custom_gradient
def straight_through_quantize(x, threshold=0.5):
    """Hard quantization with straight-through estimator for gradients.
    
    Forward pass: returns hard binary values (0 or 1)
    Backward pass: passes gradients through as if it were identity (STE)
    
    This allows the network to learn while still producing binary outputs.
    """
    # Forward: hard threshold
    y = tf.cast(x > threshold, dtype=tf.float32)
    
    def grad(dy):
        # Backward: pass gradients straight through (identity)
        # Optionally clip to region where sigmoid gradient would be non-zero
        # This helps focus learning on values near the threshold
        return dy, None  # None for threshold gradient
    
    return y, grad


def ste_quantize_layer(threshold=0.5):
    """Returns a Lambda layer that applies straight-through quantization."""
    return Lambda(
        lambda t: straight_through_quantize(t, threshold),
        name="quantization_ste"
    )


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
            yield (A, P, N), label 

class QuadrupletNet():
    def __init__(self):
        pass
        
    def create_quadruplet_net(self, embedding_net, alpha, beta, gamma=None, loss_type=""):
        
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
        
        if loss_type == "KDR":
            # loss = Lambda(self.quadruplet_loss_KDR)([A, P, N1, N2]) 
            print("Using targeted KDR quadruplet loss")
            loss = Lambda(self.targeted_kdr_quadruplet_loss)([A, P, N1, N2]) 
        else:
            # loss = Lambda(self.quadruplet_loss)([A, P, N1, N2]) 
            print("Using quadruplet target band loss")
            loss = Lambda(self.quadruplet_target_band_loss)([A, P, N1, N2]) 
            
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
    
    def quadruplet_target_band_loss(self, x, pos_max=0.1, neg_min=0.5, w_neg=1.0, w_nn=0.2, eps=1e-12):
        anchor, positive, negative1, negative2 = x

        # squared L2 distances
        D1 = K.sum(K.square(anchor - positive), axis=1)  # a-p
        D2 = K.sum(K.square(anchor - negative1), axis=1) # a-n1
        D3 = K.sum(K.square(positive - negative2), axis=1) # p-n2
        D4 = K.sum(K.square(negative1 - negative2), axis=1) # n1-n2

        # convert target thresholds to squared-distance thresholds
        pos_max2 = pos_max ** 2
        neg_min2 = neg_min ** 2

        pos_term = K.square(K.maximum(D1 - pos_max2, 0.0))               # penalize if too far
        neg_term = K.square(K.maximum(neg_min2 - D2, 0.0)) + K.square(K.maximum(neg_min2 - D3, 0.0))
        nn_term  = K.square(K.maximum(neg_min2 - D4, 0.0))

        loss = pos_term + w_neg * neg_term + w_nn * nn_term
        return loss
    
    def kdr_xor(self, a, b):
        # a,b in [0,1]; expected bit disagreement per bit
        # XOR for Bernoulli bits: a + b - 2ab
        return tf.reduce_mean(a + b - 2.0*a*b, axis=1)

    def quadruplet_loss_KDR(self, x, entropy_weight=0.5, diversity_weight=0.5):
        """Simple KDR-based quadruplet loss.
        """
        
        def kdr(a, b):
            return self.kdr_xor(a, b)
        
        # Unpack
        anchor, positive, negative1, negative2 = x
        
        # KDR calculations
        KDR_AP = kdr(anchor, positive)       # Should be LOW (~0)
        KDR_AN = kdr(anchor, negative1)      # Should be HIGH (~0.5)
        KDR_PN = kdr(positive, negative2)    # Should be HIGH (~0.5)
        KDR_NN = kdr(negative1, negative2)   # Should be HIGH (~0.5)
        
        # =====================================================================
        # LOSS 1: Similarity - minimize KDR between anchor and positive
        # =====================================================================
        # similarity_loss = tf.reduce_mean(KDR_AP)
        if self.gamma is not None:
            loss = tf.maximum(KDR_AP - KDR_AN + self.alpha, 0.0) + tf.maximum(KDR_AP - KDR_PN + self.beta, 0.0) + tf.maximum(KDR_AP - KDR_NN + self.gamma, 0.0)
        else:
            loss = tf.maximum(KDR_AP - KDR_AN + self.alpha, 0.0) + tf.maximum(KDR_AP - KDR_PN + self.beta, 0.0)
        
        
        
        # =====================================================================
        # LOSS 3: Entropy - encourage ~50% ones (balanced bits)
        # =====================================================================
        all_emb = tf.concat([anchor, positive, negative1, negative2], axis=0)
        bit_means = tf.reduce_mean(all_emb, axis=1)
        entropy_loss = tf.reduce_mean(tf.square(bit_means - 0.5))
        # Calculate entropy of each embedding to have a balanced bit distribution
        # anchor_entropy = tf.reduce_mean(tf.square(anchor - 0.5))
        # positive_entropy = tf.reduce_mean(tf.square(positive - 0.5))
        # negative1_entropy = tf.reduce_mean(tf.square(negative1 - 0.5))
        # negative2_entropy = tf.reduce_mean(tf.square(negative2 - 0.5))
        # entropy_loss = tf.reduce_mean(anchor_entropy + positive_entropy + negative1_entropy + negative2_entropy)
        
        # =====================================================================
        # LOSS 4: Diversity - different samples should produce different embeddings
        # =====================================================================
        anchor_shifted = tf.roll(anchor, shift=1, axis=0)
        within_kdr = kdr(anchor, anchor_shifted)
        diversity_loss = tf.reduce_mean(tf.maximum(0.1 - within_kdr, 0.0))
        # Check diversity by calculating entropy across all embeddings in the batch
        # all_emb = tf.concat([anchor, positive, negative1, negative2], axis=0)
        # bit_means = tf.reduce_mean(all_emb, axis=1)
        # diversity_loss = tf.reduce_mean(tf.square(bit_means - 0.5))
        
        # =====================================================================
        # COMBINED LOSS
        # =====================================================================
        # loss = (
        #     loss                       # Push Alice-Bob KDR → 0
        #     + entropy_weight * entropy_loss         # Balanced bits
        #     # + diversity_weight * diversity_loss       # Different inputs → different outputs
        # )
        
        return loss
    
    def targeted_kdr_quadruplet_loss(self,x,pos_max=0.10,   # want KDR_AP <= 0.10
                                            neg_min=0.50,   # want KDR_AN, KDR_PN >= 0.40
                                            w_nn=0.0,       # optional: enforce neg1-neg2 also far
                                            nn_min=0.20,
                                            squared=False):
        anchor, positive, neg1, neg2 = x
        KDR_AP = self.kdr_xor(anchor, positive)
        KDR_AN = self.kdr_xor(anchor, neg1)
        KDR_PN = self.kdr_xor(positive, neg2)
        KDR_NN = self.kdr_xor(neg1, neg2)

        # violations
        v_pos = tf.nn.relu(KDR_AP - pos_max)         # positive too large
        v_an  = tf.nn.relu(neg_min - KDR_AN)         # negative too small
        v_pn  = tf.nn.relu(neg_min - KDR_PN)         # negative too small
        v_nn  = tf.nn.relu(nn_min  - KDR_NN)         # optional

        if squared:
            base = v_pos**2 + v_an**2 + v_pn**2 + w_nn*(v_nn**2)
        else:
            base = v_pos + v_an + v_pn + w_nn*v_nn

        loss = tf.reduce_mean(base)

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
        # Keep only complete consecutive quadruplets: [AB, AE, BA, BE]
        usable_len = (len(data) // 4) * 4
        if usable_len < 4:
            raise ValueError("Dataset must contain at least one full quadruplet (4 samples).")
        data_usable = data[:usable_len]
        quad_starts = np.arange(0, usable_len, 4, dtype=np.int32)
        while True:
            list_a = []
            list_p = []
            list_n1 = []
            list_n2 = []
            batch = 0
            while batch < batchsize:
                if local_rng is not None:
                    idx = int(quad_starts[local_rng.randrange(len(quad_starts))])
                    swap_direction = bool(local_rng.getrandbits(1))
                else:
                    idx = int(random.choice(quad_starts))
                    swap_direction = bool(random.getrandbits(1))

                # Expected order within each consecutive quadruplet:
                # idx: AB, idx+1: AE, idx+2: BA, idx+3: BE
                # Randomly swap anchor/positive direction to balance AB->BA and BA->AB.
                if swap_direction:
                    a = data_usable[idx]      # AB
                    p = data_usable[idx + 2]  # BA
                    n1 = data_usable[idx + 1] # AE
                    n2 = data_usable[idx + 3] # BE
                else:
                    a = data_usable[idx + 2]  # BA
                    p = data_usable[idx]      # AB
                    n1 = data_usable[idx + 3] # BE
                    n2 = data_usable[idx + 1] # AE

                list_a.append(a)
                list_p.append(p)
                list_n1.append(n1)
                list_n2.append(n2)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            N1 = np.array(list_n1, dtype='float32')
            N2 = np.array(list_n2, dtype='float32')
            label = np.ones(int(batchsize))
            #print("label",label.shape)
            #print("A",A.shape)
            yield (A, P, N1, N2), label

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
        for _ in range(1):
            x = self.resblock(x, 3, 32)
        x = self.resblock(x, 3, 64, first_layer = True)
        for _ in range(1):
            x = self.resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(output_length)(x)
        x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        # x = Lambda(lambda x: tf.cast(x > 0.5, dtype=tf.float32))(x) # Quantization layer
        x = Dense(units=output_length, activation='sigmoid', kernel_initializer="lecun_normal")(x)
        model = Model(inputs=inputs, outputs=x)
        return model
    
    def build_quantized_extractor(self, datashape, output_length, threshold=0.5):
        """Builds a feature extractor with STE quantization (differentiable)."""
        base = self.feature_extractor(datashape, output_length)
        q_out = ste_quantize_layer(threshold)(base.output)
        return Model(base.input, q_out, name="ResNet_QuadrupletNet_Channel_Quantized")
    
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
        x = Dense(units=output_length, activation='sigmoid', kernel_initializer="lecun_normal")(x)

        return Model(inputs, x, name="MLP_Encoder")
    
    def build_quantized_extractor(self, datashape, output_length, threshold=0.5):
        """Builds a feature extractor with STE quantization (differentiable)."""
        base = self.feature_extractor(datashape, output_length)
        q_out = ste_quantize_layer(threshold)(base.output)
        return Model(base.input, q_out, name="MLP_Encoder_Quantized")

# Recurrent Neural Network (RNN) model
class RNN_QuadrupletNet_Channel(QuadrupletNet):
    """
    Simple RNN for learning time-frequency channel features from spectrograms.
    
    Architecture:
        Input: Spectrogram (F, T, C) - Frequency bins x Time steps x Channels
        
        1. Permute to (T, F, C) - treat time as sequence dimension
        2. Reshape to (T, F*C) - each time step has all frequency info
        3. GRU processes sequence - learns temporal evolution of frequency features
        4. Final hidden state → Dense → Sigmoid → Feature vector [0, 1]
        5. Optional: STE quantization for binary output {0, 1}
    
    The RNN sees frequency content at each time step and learns how
    the channel's frequency response evolves over time.
    """
    def __init__(self,
                 gru_units=256,
                 dropout=0.3,
                 recurrent_dropout=0.1,
                 bidirectional=True,
                 num_layers=2):
        self.gru_units = gru_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.bidirectional = bidirectional
        self.num_layers = num_layers

    def feature_extractor(self, datashape, output_length):
        """
        Build the RNN feature extractor.
        
        Args:
            datashape: Tuple (batch, F, T, C) - spectrogram dimensions
            output_length: Size of output feature vector
            
        Returns:
            Keras Model: Input spectrogram → Output feature vector [0, 1]
        """
        self.datashape = datashape  # Store for create_quadruplet_net
        F, T, C = datashape[1], datashape[2], datashape[3]
        print(f"Building RNN: F={F} (freq), T={T} (time), C={C} (channels)")
        print(f"GRU units={self.gru_units}, layers={self.num_layers}, bidirectional={self.bidirectional}")
        
        # Input: spectrogram (F, T, C)
        inputs = Input(shape=(F, T, C))
        
        # Step 1: Permute to (T, F, C) - time becomes sequence dimension
        x = Permute((2, 1, 3))(inputs)
        
        # Step 2: Reshape to (T, F*C) - flatten frequency & channels per timestep
        x = Reshape((T, F * C))(x)
        
        # Step 3: Stacked GRU layers
        for i in range(self.num_layers):
            return_sequences = (i < self.num_layers - 1)  # Only last layer returns final state
            
            gru = GRU(
                self.gru_units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                # recurrent_dropout=self.recurrent_dropout,
                name=f"gru_{i+1}"
            )
            
            if self.bidirectional:
                x = Bidirectional(gru, name=f"bi_gru_{i+1}")(x)
            else:
                x = gru(x)
            
            # Add layer norm between GRU layers (not after last)
            if return_sequences:
                x = LayerNormalization()(x)
                
        
        # Step 4: Project to output dimension
        hidden_dim = self.gru_units * 2 if self.bidirectional else self.gru_units
        x = Dense(hidden_dim, activation='relu', name="projection")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        # Step 5: Output layer with sigmoid for [0, 1] values
        x = Dense(output_length, activation='sigmoid', name="output")(x)
        
        return Model(inputs, x, name="RNN_ChannelEncoder")

    def build_quantized_extractor(self, datashape, output_length, threshold=0.5):
        """
        Build RNN with STE quantization layer for binary {0, 1} output.
        
        Uses Straight-Through Estimator for differentiable hard quantization:
        - Forward: Hard threshold to {0, 1}
        - Backward: Gradients pass through as identity
        """
        base = self.feature_extractor(datashape, output_length)
        q_out = ste_quantize_layer(threshold)(base.output)
        return Model(base.input, q_out, name="RNN_ChannelEncoder_Quantized")
    
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
        F, T, C = datashape[1], datashape[2], datashape[3]
        reg = l2(self.l2_w)
        
        inputs = Input(shape=(F, T, C))
        x = Permute((2, 1, 3))(inputs)           # (T, F, C)
        x = Reshape((T, F * C))(x)               # (T, F*C)
        
        # Project to working dimension
        x = Dense(self.embed_dim, kernel_regularizer=reg)(x)
        x = LayerNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        # GRU layer - use the __init__ parameter!
        gru = GRU(self.gru_units, return_sequences=False,  # Only need final state
                dropout=self.dropout, 
                recurrent_dropout=self.recurrent_dropout)
        if self.bidirectional:
            x = Bidirectional(gru)(x)
        else:
            x = gru(x)
        
        # Projection head (after pooling, much cheaper)
        x = Dense(output_length, kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dense(output_length, activation='sigmoid', 
                kernel_initializer="lecun_normal")(x)
        
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
        # Sigmoid activation
        x = Lambda(lambda t: K.sigmoid(t))(x)

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

class SiammeseHashNet():
    
    def __init__(self):
        pass
    
    def siammese_net(self, embedding_net, m=0.2):
        
        self.m = m # Margin threshold
        
        input_1 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        input_2 = Input([self.datashape[1],self.datashape[2],self.datashape[3]])
        label_1 = Input(shape=(1,))
        label_2 = Input(shape=(1,))
        
        A = embedding_net(input_1)
        P = embedding_net(input_2)
        
        
        loss = Lambda(self.siammese_hamming_loss)([A, P, label_1, label_2])
        
        return Model(inputs=[input_1, input_2], outputs=loss)

    # Siammese Hamming Loss function.
    def siammese_hamming_loss(self,x):
        # getting Quadruplet
        anchor,positive,label_1,label_2 = x
        # Hamming distance between the anchor and positive features
        # Loss is 1/2 * (1 - y) D(anchor,positive) + 1/2 * y * max(m-D(anchor,positive),0)
        # y is set to 0 if label_1 == label_2, otherwise 1
        y = 1.0 - K.cast(label_1 == label_2, tf.float32)
        # D is the Hamming distance between the anchor and positive features
        D = K.sum(K.square(anchor-positive),axis=1)
        # Loss is 1/2 * (1 - y) D + 1/2 * y * max(m-D,0)
        loss = 0.5 * (1 - y) * D + 0.5 * y * K.maximum(self.m-D,0)
        return loss
    
    def create_generator_channel(self, batchsize, data, label, seed: int = None):
        """Generate a siammese pairs generator for training."""
        self.data = data
        self.label = label
        local_rng = random.Random(seed) if seed is not None else None
        # Keep only complete consecutive quadruplets: [AB, AE, BA, BE]
        usable_len = (len(data) // 4) * 4
        if usable_len < 4:
            raise ValueError("Dataset must contain at least one full quadruplet (4 samples).")
        data_usable = data[:usable_len]
        quad_starts = np.arange(0, usable_len, 4, dtype=np.int32)
        while True:
            list_a = []
            list_p = []
            label_1 = []
            label_2 = []
            batch = 0
            while batch < batchsize:
                if local_rng is not None:
                    idx = int(quad_starts[local_rng.randrange(len(quad_starts))])
                    swap_direction = bool(local_rng.getrandbits(1))
                else:
                    idx = int(random.choice(quad_starts))
                    swap_direction = bool(random.getrandbits(1))

                # Expected order within each consecutive quadruplet:
                # idx: AB, idx+1: AE, idx+2: BA, idx+3: BE
                # Randomly swap anchor/positive direction to balance AB->BA and BA->AB.
                if swap_direction:
                    a = data_usable[idx]      # AB
                    p = data_usable[idx + 2]  # BA
                    n1 = data_usable[idx + 1] # AE
                    n2 = data_usable[idx + 3] # BE
                else:
                    a = data_usable[idx + 2]  # BA
                    p = data_usable[idx]      # AB
                    n1 = data_usable[idx + 3] # BE
                    n2 = data_usable[idx + 1] # AE

                list_a.append(a)
                label_1.append(1)
                list_p.append(p)
                label_2.append(1)
                list_a.append(a)
                label_1.append(1)
                list_p.append(n1)
                label_2.append(0)
                list_a.append(p)
                label_1.append(1)
                list_p.append(n2)
                label_2.append(0)
                batch = batch + 1
            A = np.array(list_a, dtype='float32')
            P = np.array(list_p, dtype='float32')
            label_1 = np.array(label_1, dtype='float32')
            label_2 = np.array(label_2, dtype='float32')
            yield (A, P, label_1, label_2), np.ones(int(batchsize))

class ResNet_HashNet_Channel(SiammeseHashNet):
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
        for _ in range(1):
            x = self.resblock(x, 3, 32)
        x = self.resblock(x, 3, 64, first_layer = True)
        for _ in range(1):
            x = self.resblock(x, 3, 64)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(output_length)(x)
        x = Lambda(lambda  x: K.l2_normalize(x,axis=1))(x)
        # x = Lambda(lambda x: tf.cast(x > 0.5, dtype=tf.float32))(x) # Quantization layer
        x = Dense(units=output_length, activation='sigmoid', kernel_initializer="lecun_normal")(x)
        
        # Lets add a quantization layer that sets outputs to +1 and -1
        x = Lambda(lambda x: tf.cast(x > 0.5, dtype=tf.float32))(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model

if __name__ == "__main__":
    
    
    
    def diff_xor(a, b):
        return a + b - 2.0 * a * b
    
    # A = [0, 1, 0, 1, 1]
    # A = tf.cast(A, tf.float32)
    A = tf.cast(tf.random.uniform((1, 5)) > 0.5, tf.float32)
    # B = [1, 0, 1, 0, 1]
    # B = tf.cast(B, tf.float32)
    B = tf.cast(tf.random.uniform((1, 5)) > 0.5, tf.float32)
    print("A:", A)
    print("B:", B)
    xor_AB = diff_xor(A, B)
    print("xor_AB:", xor_AB)
    
    # calculate KDR
    KDR = tf.reduce_mean(xor_AB)
    print("KDR:", KDR)
    exit()
    # Test RNN with optional quantization and loss behavior
    F = 2048
    T = 32
    C = 1
    datashape = (1, F, T, C)
    output_length = 128  # Smaller for quick testing
    batch_size = 4
    
    def diff_xor(a, b):
        """Differentiable XOR for values in [0,1] or binary {0,1}."""
        return a + b - 2.0 * a * b
    
    def compute_kdr(a, b):
        """Compute KDR = mean(XOR) between two tensors."""
        return tf.reduce_mean(diff_xor(a, b))
    
    def compute_loss(kdr_ap, kdr_an, kdr_bp, kdr_bn, alpha, beta, gamma):
        """Compute quadruplet loss from KDR values."""
        loss = (
            tf.maximum(kdr_ap - kdr_an + alpha, 0.0)
            + tf.maximum(kdr_ap - kdr_bp + beta, 0.0)
            + tf.maximum(kdr_ap - kdr_bn + gamma, 0.0)
        )
        return loss
    
    print("\n" + "="*70)
    print("KDR AND MARGIN PARAMETER ANALYSIS")
    print("="*70)
    
    # Test 1: Simulate expected KDR values from real channel data
    print("\n--- Test 1: Simulated Binary Embeddings ---")
    
    # Simulate embeddings (binary vectors of length output_length)
    # Anchor: random binary
    emb_A = tf.cast(tf.random.uniform((batch_size, output_length)) > 0.5, tf.float32)
    
    # Positive: similar to Anchor (only ~10% bits different - reciprocal channel)
    flip_mask_P = tf.cast(tf.random.uniform((batch_size, output_length)) < 0.10, tf.float32)
    emb_P = tf.abs(emb_A - flip_mask_P)  # XOR with flip mask
    
    # Negatives: ~50% different from Anchor (independent Eve channels)
    flip_mask_N1 = tf.cast(tf.random.uniform((batch_size, output_length)) < 0.50, tf.float32)
    emb_N1 = tf.abs(emb_A - flip_mask_N1)
    flip_mask_N2 = tf.cast(tf.random.uniform((batch_size, output_length)) < 0.50, tf.float32)
    emb_N2 = tf.abs(emb_P - flip_mask_N2)
    
    # Compute KDR for each pair
    kdr_ap = compute_kdr(emb_A, emb_P)
    kdr_an = compute_kdr(emb_A, emb_N1)
    kdr_bp = compute_kdr(emb_P, emb_N2)
    kdr_bn = compute_kdr(emb_N1, emb_N2)
    
    print(f"KDR_AP (Alice-Bob, should be LOW):    {kdr_ap.numpy():.4f}")
    print(f"KDR_AN (Alice-Eve, should be ~0.5):   {kdr_an.numpy():.4f}")
    print(f"KDR_BP (Bob-Eve, should be ~0.5):     {kdr_bp.numpy():.4f}")
    print(f"KDR_BN (Eve-Eve, should be ~0.5):     {kdr_bn.numpy():.4f}")
    
    # Test different margin values
    print("\n--- Test 2: Loss for Different Margin Values ---")
    print(f"{'Alpha':>8} {'Beta':>8} {'Gamma':>8} | {'Loss':>10} | Notes")
    print("-" * 60)
    
    margin_configs = [
        (0.1, 0.1, 0.05, "Conservative margins"),
        (0.2, 0.2, 0.1,  "Moderate margins (recommended)"),
        (0.3, 0.3, 0.15, "Aggressive margins"),
        (0.4, 0.4, 0.2,  "Very aggressive (may not converge)"),
        (0.5, 0.5, 0.5,  "Too aggressive (loss always > 0)"),
    ]
    
    for alpha, beta, gamma, note in margin_configs:
        loss = compute_loss(kdr_ap, kdr_an, kdr_bp, kdr_bn, alpha, beta, gamma)
        print(f"{alpha:>8.2f} {beta:>8.2f} {gamma:>8.2f} | {loss.numpy():>10.4f} | {note}")
    
    # Dynamic margin suggestion based on expected KDR gap
    print("\n--- Test 3: Dynamic Margin Calculation ---")
    expected_kdr_ap = 0.10  # What we expect for good Alice-Bob pairs
    expected_kdr_eve = 0.50  # What we expect for Eve pairs
    gap = expected_kdr_eve - expected_kdr_ap
    
    # Set margins to ~50-70% of the gap (leave room for learning)
    suggested_alpha = gap * 0.5
    suggested_beta = gap * 0.5
    suggested_gamma = gap * 0.25  # Smaller for Eve-Eve comparison
    
    print(f"Expected gap (KDR_eve - KDR_ap): {gap:.2f}")
    print(f"Suggested alpha: {suggested_alpha:.3f} (50% of gap)")
    print(f"Suggested beta:  {suggested_beta:.3f} (50% of gap)")
    print(f"Suggested gamma: {suggested_gamma:.3f} (25% of gap)")
    
    # Test with actual model
    print("\n--- Test 4: Full Model Forward Pass ---")
    
    alpha, beta, gamma = suggested_alpha, suggested_beta, suggested_gamma
    print(f"Using margins: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")
    
    rnn = RNN_QuadrupletNet_Channel(embed_dim=output_length)
    embed_net_q = rnn.build_quantized_extractor(datashape, output_length, threshold=0.5)
    quad_loss_model_q = rnn.create_quadruplet_net(
        embed_net_q, alpha=alpha, beta=beta, gamma=gamma, loss_type="KDR"
    )
    
    # Random input spectrograms
    A = tf.random.uniform((batch_size, datashape[1], datashape[2], datashape[3]))
    P = tf.random.uniform((batch_size, datashape[1], datashape[2], datashape[3]))
    N1 = tf.random.uniform((batch_size, datashape[1], datashape[2], datashape[3]))
    N2 = tf.random.uniform((batch_size, datashape[1], datashape[2], datashape[3]))
    
    # Get embeddings
    emb_A_model = embed_net_q(A)
    emb_P_model = embed_net_q(P)
    emb_N1_model = embed_net_q(N1)
    emb_N2_model = embed_net_q(N2)
    
    print(f"Embedding sample (first 16 bits): {emb_A_model[0, :16].numpy()}")
    print(f"Unique values in embedding: {tf.unique(tf.reshape(emb_A_model, [-1]))[0].numpy()}")
    
    # Compute KDR on model outputs
    kdr_ap_model = compute_kdr(emb_A_model, emb_P_model)
    kdr_an_model = compute_kdr(emb_A_model, emb_N1_model)
    print(f"Model KDR_AP (random inputs, untrained): {kdr_ap_model.numpy():.4f}")
    print(f"Model KDR_AN (random inputs, untrained): {kdr_an_model.numpy():.4f}")
    
    # Forward pass through loss
    loss_q = quad_loss_model_q([A, P, N1, N2])
    print(f"Quadruplet loss (untrained model): {tf.reduce_mean(loss_q).numpy():.4f}")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print(f"  alpha = {suggested_alpha:.2f}")
    print(f"  beta  = {suggested_beta:.2f}")
    print(f"  gamma = {suggested_gamma:.2f}")
    print("="*70)
    
    exit()