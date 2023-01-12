# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf
import keras.utils


def get_angles(pos, i, d_model):
    """Getting the angle rates of the position of the model
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """Retrieving the positional encodings of the model which means where the data points exist
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculating the attention weights where q is the query shape, k is the key shape and v is the value shape
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # matmul_dk is being scaled
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Adding the mask 
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # Having the weights being normalized by softmax
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  

    output = tf.matmul(attention_weights, v) 

    return output, attention_weights


@keras.utils.register_keras_serializable
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Heads are being split. Result is transposed. 
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        #Getting the batch size, sequence length and input length of the model
        q = self.wq(q)  
        k = self.wk(k)  
        v = self.wv(v)  

        #Splitting the heads of the data
        q = self.split_heads(q, batch_size)  
        k = self.split_heads(k, batch_size)  
        v = self.split_heads(v, batch_size)  

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        #Transpose the scaled attention
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  
        #Reshape the scaled attention
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  

        output = self.dense(concat_attention)  

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """A sequential forward network with two different dense layers
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  
            tf.keras.layers.Dense(d_model),  
        ]
    )


@keras.utils.register_keras_serializable
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        #Implement the multiheadattention class 
        self.mha = MultiHeadAttention(d_model, num_heads)
        #Implement the sequential forward network
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        #Implement two normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        #Implement two dropout layers 
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  

        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(
            out1 + ffn_output
        )  

        return out2


@keras.utils.register_keras_serializable
class Encoder(tf.keras.layers.Layer):
    """This Encoder class will encode the data
    """
    def __init__(
        self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None, mask=None):
        seq_len = tf.shape(x)[1]
        # Transform the data mathematicall
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        #Add the position of the data to the data
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
