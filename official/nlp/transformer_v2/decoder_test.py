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
"""Tests for nlp.nhnet.decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from official.nlp.modeling import layers
from official.nlp.transformer_v2 import configs
from official.nlp.transformer_v2 import decoder
from official.nlp.transformer_v2 import utils


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
  return {
      "key":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32),
      "value":
          tf.zeros([batch_size, init_decode_length, num_heads, head_size],
                   dtype=tf.float32)
  }


class DecoderTest(tf.test.TestCase):

  def setUp(self):
    super(DecoderTest, self).setUp()
    self._config = utils.get_test_params()

  # def test_transformer_decoder(self):
  #   decoder_block = decoder.TransformerDecoder(
  #       num_hidden_layers=self._config.num_hidden_layers,
  #       hidden_size=self._config.hidden_size,
  #       num_attention_heads=self._config.num_attention_heads,
  #       intermediate_size=self._config.intermediate_size,
  #       intermediate_activation=self._config.hidden_act,
  #       hidden_dropout_prob=self._config.hidden_dropout_prob,
  #       attention_probs_dropout_prob=self._config.attention_probs_dropout_prob,
  #       initializer_range=self._config.initializer_range)
  #   decoder_block.build(None)
  #   self.assertEqual(len(decoder_block.layers), self._config.num_hidden_layers)

  def test_decoder_block_with_cache(self):
    decoder_block = decoder.TransformerDecoderBlock(
        hidden_size=self._config.hidden_size,
        num_attention_heads=self._config.num_attention_heads,
        intermediate_size=self._config.intermediate_size,
        intermediate_activation=self._config.hidden_act,
        hidden_dropout_prob=self._config.hidden_dropout_prob,
        attention_probs_dropout_prob=self._config.attention_probs_dropout_prob,
        initializer_range=self._config.initializer_range)
    # Forward path.
    dummy_tensor = tf.zeros([2, 4, self._config.hidden_size], dtype=tf.float32)
    dummy_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
    inputs = [dummy_tensor, dummy_tensor, dummy_mask, dummy_mask]
    cache = _create_cache(
        2, 0, self._config.num_attention_heads,
        self._config.hidden_size // self._config.num_attention_heads)
    output, cache = decoder_block(inputs, cache)
    self.assertEqual(output.shape, (2, 4, self._config.hidden_size))
    self.assertEqual(cache["value"].shape, (2, 4, 2, 8))

  def test_bert_decoder(self):
    seq_length = 10
    # encoder_input_ids = tf.keras.layers.Input(
    #     shape=(seq_length,), name="encoder_input_ids", dtype=tf.int32)
    # target_ids = tf.keras.layers.Input(
    #     shape=(seq_length,), name="target_ids", dtype=tf.int32)
    # encoder_outputs = tf.keras.layers.Input(
    #     shape=(seq_length, self._config.hidden_size),
    #     name="encoder_outputs",
    #     dtype=tf.float32)

    encoder_input_ids = np.zeros((2, seq_length), dtype=np.int32)
    target_ids = np.zeros((2, seq_length), dtype=np.int32)
    encoder_outputs = np.zeros((2, seq_length, 16), dtype=np.float32)


    embedding_lookup = layers.OnDeviceEmbedding(
        vocab_size=self._config.vocab_size,
        embedding_width=self._config.hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=self._config.initializer_range),
        name="word_embeddings")
    # cross_attention_bias = decoder.AttentionBias(bias_type="single_cross")(
    #     encoder_input_ids)
    # self_attention_bias = decoder.AttentionBias(bias_type="decoder_self")(
    #     target_ids)

    config = self._config

    decoder_layer = decoder.TransformerDecoder(
        num_hidden_layers=config.num_decoder_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_decoder_attn_heads,
        intermediate_size=config.decoder_intermediate_size,
        intermediate_activation=config.hidden_act,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        initializer_range=config.initializer_range,
        multi_channel_cross_attention=False,
        transformer=True,
        embedding_lookup=embedding_lookup,
        config=config,
        name="decoder")


    fake_inputs = {
        "encoder_input_ids": encoder_input_ids,
        "target_ids": target_ids,
        "encoder_outputs": encoder_outputs,
        # "self_attention_mask": self_attention_bias,
        # "attention_mask": cross_attention_bias,
    }
    output_tensor, _ = decoder_layer(fake_inputs)


    # inputs = dict(
    #     attention_mask=cross_attention_bias,
    #     self_attention_mask=self_attention_bias,
    #     target_ids=target_ids,
    #     encoder_outputs=encoder_outputs)

    # decoder_layer = decoder.Decoder(self._config, embedding_lookup)
    # outputs = decoder_layer(inputs)
    # model_inputs = dict(
    #     encoder_input_ids=encoder_input_ids,
    #     target_ids=target_ids,
    #     all_encoder_outputs=encoder_outputs)
    # model = tf.keras.Model(inputs=model_inputs, outputs=outputs, name="test")
    # self.assertLen(decoder_layer.trainable_weights, 30)
    # Forward path.


    self.assertEqual(output_tensor.shape, (2, 10, 16))


if __name__ == "__main__":
  tf.test.main()
