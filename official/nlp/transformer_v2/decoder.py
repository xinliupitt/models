# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer decoder that mimics a BERT encoder, to load BERT checkpoints."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import math
import tensorflow as tf
from official.modeling import tf_utils
from official.nlp.modeling import layers
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.layers import transformer
# from official.nlp.nhnet import multi_channel_attention
from official.nlp.transformer import model_utils as transformer_utils

class TransformerDecoderBlock(tf.keras.layers.Layer):
  """Single transformer layer for decoder.

  It has three sub-layers:
  (1) a multi-head self-attention mechanism.
  (2) a encoder-decoder attention.
  (3) a positionwise fully connected feed-forward network.
  """

  def __init__(self,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="relu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               multi_channel_cross_attention=False,
               **kwargs):
    super(TransformerDecoderBlock, self).__init__(**kwargs)
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.multi_channel_cross_attention = multi_channel_cross_attention
    self._kernel_initializer = tf.keras.initializers.get("glorot_uniform")
    # self._kernel_initializer = tf.keras.initializers.TruncatedNormal(
    #     stddev=initializer_range)
    self._bias_initializer = tf.keras.initializers.get("zeros")
    if self.multi_channel_cross_attention:
      self._cross_attention_cls = multi_channel_attention.MultiChannelAttention
    else:
      self._cross_attention_cls = layers.MultiHeadAttention

    if self.hidden_size % self.num_attention_heads != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_size, self.num_attention_heads))
    self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

  def build(self, input_shape):
    # Self attention.
    def _glorot_initializer(fan_in, fan_out):
      limit = math.sqrt(6.0 / (fan_in + fan_out))
      return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)
    attention_initializer = _glorot_initializer(input_shape[0].as_list()[-1],
                                                self.hidden_size)
    self.self_attention = layers.CachedAttention(
        num_heads=self.num_attention_heads,
        key_size=self.attention_head_size,
        dropout=self.attention_probs_dropout_prob,
        kernel_initializer=self._kernel_initializer,
        name="self_attention")
    self.self_attention_output_dense = layers.DenseEinsum(
        output_shape=self.hidden_size,
        num_summed_dimensions=2,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name="self_attention_output")
    self.self_attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.self_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm", axis=-1, epsilon=1e-12))
    # Encoder-decoder attention.
    self.encdec_attention = self._cross_attention_cls(
        num_heads=self.num_attention_heads,
        key_size=self.attention_head_size,
        dropout=self.attention_probs_dropout_prob,
        output_shape=self.hidden_size,
        kernel_initializer=self._kernel_initializer,
        name="attention/encdec")

    self.encdec_attention_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob)
    self.encdec_attention_layer_norm = (
        tf.keras.layers.LayerNormalization(
            name="attention/encdec_output_layer_norm", axis=-1, epsilon=1e-12))

    # Feed-forward projection.
    self.intermediate_dense = layers.DenseEinsum(
        output_shape=self.intermediate_size,
        activation=None,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name="intermediate")
    self.intermediate_activation_layer = tf.keras.layers.Activation(
        self.intermediate_activation)
    output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
    self.output_dense = layers.DenseEinsum(
        output_shape=self.hidden_size,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name="output")
    self.output_dropout = tf.keras.layers.Dropout(rate=self.hidden_dropout_prob)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="output_layer_norm", axis=-1, epsilon=1e-12)
    super(TransformerDecoderBlock, self).build(input_shape)

  def common_layers_with_encoder(self):
    """Gets layer objects that can make a Transformer encoder block."""
    return [
        self.self_attention, self.self_attention_layer_norm,
        self.intermediate_dense, self.output_dense, self.output_layer_norm
    ]

  def call(self, inputs, cache=None, decode_loop_step=None):
    if self.multi_channel_cross_attention:
      if len(inputs) != 5:
        raise ValueError(
            "TransformerDecoderBlock must have 5 inputs, when it uses "
            "multi_channel_cross_attention. But it got: %d" % len(inputs))
    elif len(inputs) != 4:
      raise ValueError(
          "TransformerDecoderBlock must have 4 inputs, but it got: %d" %
          len(inputs))
    input_tensor, memory, attention_mask, self_attention_mask = inputs[:4]
    self_attention_inputs = [input_tensor, input_tensor]
    self_attention_output, cache = self.self_attention(
        self_attention_inputs,
        attention_mask=self_attention_mask,
        cache=cache,
        decode_loop_step=decode_loop_step)
    self_attention_output = self.self_attention_dropout(self_attention_output)
    self_attention_output = self.self_attention_layer_norm(
        input_tensor + self_attention_output)

    cross_attn_inputs = [self_attention_output, memory]
    if self.multi_channel_cross_attention:
      # Accesses the 5-th input tensor for the doc-attention probabilities.
      cross_attn_inputs.append(inputs[-1])
    attention_output = self.encdec_attention(cross_attn_inputs, attention_mask)
    attention_output = self.encdec_attention_dropout(attention_output)
    attention_output = self.encdec_attention_layer_norm(self_attention_output +
                                                        attention_output)

    intermediate_output = self.intermediate_dense(attention_output)
    intermediate_output = self.intermediate_activation_layer(
        intermediate_output)
    layer_output = self.output_dense(intermediate_output)
    layer_output = self.output_dropout(layer_output)
    layer_output = self.output_layer_norm(layer_output + attention_output)
    return layer_output, cache



def get_attention_bias(input_tensor,
                       bias_type,
                       padding_value=0,
                       max_length=None):
  """A helper function to get various attention bias tensors."""
  if bias_type not in ("single_cross", "multi_cross", "decoder_self"):
    raise ValueError("Invalid attention bias type: %s" % bias_type)
  if bias_type == "single_cross":
    length = tf_utils.get_shape_list(input_tensor, expected_rank=2)[1]
    bias = transformer_utils.get_padding_bias(
        input_tensor, padding_value=padding_value)
  elif bias_type == "multi_cross":
    length = tf_utils.get_shape_list(input_tensor, expected_rank=3)[2]
    padding = transformer_utils.get_padding(
        input_tensor, padding_value=padding_value)
    bias = padding * -1e9
  else:
    if max_length is not None:
      length = max_length
    else:
      length = tf_utils.get_shape_list(input_tensor, expected_rank=2)[1]
    bias = transformer_utils.get_decoder_self_attention_bias(length)

  return tf.where(bias < 0, tf.zeros_like(bias), tf.ones_like(bias))


class AttentionBias(tf.keras.layers.Layer):

  def __init__(self, bias_type, **kwargs):
    super(AttentionBias, self).__init__(**kwargs)
    self.bias_type = bias_type

  def call(self, inputs):
    return get_attention_bias(inputs, self.bias_type)


class TransformerDecoder(tf.keras.layers.Layer):
  """Transformer decoder stack."""

  def __init__(self,
               num_hidden_layers=6,
               hidden_size=768,
               num_attention_heads=12,
               intermediate_size=3072,
               intermediate_activation="relu",
               hidden_dropout_prob=0.0,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               attend_to_last_layer=True,
               multi_channel_cross_attention=False,
               transformer=False,
               embedding_lookup=None,
               config=None,
               **kwargs):
    super(TransformerDecoder, self).__init__(**kwargs)
    self.config = config
    self.num_hidden_layers = num_hidden_layers
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.intermediate_activation = tf_utils.get_activation(
        intermediate_activation)
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.initializer_range = initializer_range
    self.attend_to_last_layer = attend_to_last_layer
    self.multi_channel_cross_attention = multi_channel_cross_attention
    self.transformer = transformer
    self.embedding_lookup = embedding_lookup
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.hidden_size)
    self.output_layer_norm = tf.keras.layers.LayerNormalization(
        name="layer_norm", axis=-1, epsilon=1e-12, dtype=tf.float32)
    print ('self.hidden_dropout_prob', self.hidden_dropout_prob)
    self.output_dropout = tf.keras.layers.Dropout(
        rate=self.hidden_dropout_prob, dtype=tf.float32)


  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.layers = []
    for i in range(self.num_hidden_layers):
      self.layers.append(
          transformer.TransformerDecoderLayer(
              # hidden_size=self.hidden_size,
              num_attention_heads=self.num_attention_heads,
              intermediate_size=self.intermediate_size,
              intermediate_activation=self.intermediate_activation,
              dropout_rate=self.hidden_dropout_prob,
              attention_dropout_rate=self.attention_probs_dropout_prob,
              # initializer_range=self.initializer_range,
              multi_channel_cross_attention=self.multi_channel_cross_attention,
              name=("layer_%d" % i)))
    super(TransformerDecoder, self).build(unused_input_shapes)

  def call(self, inputs, cache=None, decode_loop_step=None, padded_decode=False, mode="predict"):
    """Return the output of the decoder layer stacks.

    Args:
      inputs: A dictionary of inputs. `decoder_inputs` is a tf.int32 tensor for
        input ids. `encoder_outputs` is a list of tensors with shape
        [batch_size, input_length, hidden_size]. `self_attention_mask` is the
        bias for decoder self-attention layer. [1, 1, target_length,
        target_length]. `attention_mask` is the bias for encoder-decoder
        attention layer, [batch_size, 1, 1, input_length].
      cache: A dictionary of cache tensors, including key & value attentions.
      decode_loop_step: an integer to indicate the step inside a decoding loop.

    Returns:
      Output of decoder layer stack.
      float32 tensor with shape [batch_size, target_length, hidden_size]
    """

    # encoder_input_ids = inputs["encoder_input_ids"]
    if "all_encoder_outputs" in inputs:
      encoder_outputs = inputs["all_encoder_outputs"]
    else:
      encoder_outputs = inputs["encoder_outputs"]
    target_ids = inputs["target_ids"]

    # cross_attention_bias = AttentionBias(bias_type="single_cross")(
    #     encoder_input_ids)
    # self_attention_bias = AttentionBias(bias_type="decoder_self")(
    #     target_ids)

    # self_attention_mask = self_attention_bias
    # attention_mask = cross_attention_bias

    self_attention_mask = inputs["self_attention_bias"]
    attention_mask = inputs["attention_bias"]

    if self.transformer:

      if not isinstance(encoder_outputs, list):
        encoder_outputs = [encoder_outputs]

      target_embeds = self.embedding_lookup(target_ids)
      if mode == "train":
        # Shift targets to the right, and remove the last element
        target_embeds = tf.pad(target_embeds,
                                [[0, 0], [1, 0], [0, 0]])[:, :-1, :]


      # Sin cos relative position embedding
      if mode == "train":
        pos_encoding = self.position_embedding(inputs=target_embeds)
        pos_encoding = tf.cast(pos_encoding, target_embeds.dtype)
      elif decode_loop_step is None:
        pos_encoding = self.position_embedding(inputs=target_embeds)
        pos_encoding = tf.cast(pos_encoding, target_embeds.dtype)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
      else:
        pos_encoding = self.position_embedding(
            inputs=None, length=self.config["decode_max_length"])
        # if length=self.config.max_position_embeddings+1, the pos_encoding shape becomes (129, 16)
        pos_encoding = tf.cast(pos_encoding, target_embeds.dtype)
        # pos_encoding = tf.gather(pos_encoding, [decode_loop_step])
        if self.config["padded_decode"]:
          pos_encoding_shape = pos_encoding.shape.as_list()
          pos_encoding = tf.slice(pos_encoding, [decode_loop_step, 0], [1, pos_encoding_shape[1]])
        else:
          pos_encoding = tf.gather(pos_encoding, [decode_loop_step])
          # pos_encoding = pos_encoding[decode_loop_step:decode_loop_step + 1]
        # pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        # Broadcasts to all sequences inside a batch.

      target_embeds = target_embeds + pos_encoding
      target_embeds = self.output_layer_norm(target_embeds)
      # if mode == "train":
      if True:
        target_embeds = self.output_dropout(target_embeds)

      decoder_inputs = target_embeds

      if not padded_decode:
        decode_loop_step = None
    else:
      decoder_inputs = inputs["decoder_inputs"]


    decoder_shape = tf_utils.get_shape_list(decoder_inputs, expected_rank=3)
    batch_size = decoder_shape[0]
    decoder_length = decoder_shape[1]

    def _to_bert_self_attention_mask(matrix):
      """[1, 1, target_len, target_len] -> [bs, target_len, target_len]."""
      matrix = tf.squeeze(matrix, axis=[1])
      matrix = tf.tile(matrix, [batch_size, 1, 1])
      return matrix

    def _to_bert_encdec_attention_mask(matrix):
      """[bs, 1, 1, input_len] -> [bs, target_len, input_len]."""
      if self.multi_channel_cross_attention:
        matrix = tf.expand_dims(matrix, axis=2)
        matrix = tf.tile(matrix, [1, 1, decoder_length, 1])
      else:
        matrix = tf.squeeze(matrix, axis=[1])
        matrix = tf.tile(matrix, [1, decoder_length, 1])
      return matrix

    attention_mask = _to_bert_encdec_attention_mask(attention_mask)
    self_attention_mask = _to_bert_self_attention_mask(self_attention_mask)

    output_tensor = decoder_inputs
    for layer_idx in range(self.num_hidden_layers):
      if self.attend_to_last_layer:
        memory = encoder_outputs[-1]
      else:
        memory = encoder_outputs[layer_idx]
      if self.multi_channel_cross_attention:
        transformer_inputs = [
            output_tensor, memory, attention_mask, self_attention_mask,
            inputs["doc_attention_probs"]
        ]
      else:
        transformer_inputs = [
            output_tensor, memory, attention_mask, self_attention_mask
        ]
      # Gets the cache for decoding.
      if cache is None:
        output_tensor, _ = self.layers[layer_idx](transformer_inputs)
      else:
        cache_layer_idx = str(layer_idx)
        output_tensor, cache[cache_layer_idx] = self.layers[layer_idx](
            transformer_inputs,
            cache=cache[cache_layer_idx],
            decode_loop_step=decode_loop_step)
      # output_tensor = self.output_layer_norm(output_tensor)
    return output_tensor
