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
"""tf.keras Models for NHNet."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import logging
import gin
import tensorflow as tf
from typing import Optional, Text

from official.modeling import tf_utils
from official.modeling.hyperparams import params_dict
from official.nlp.modeling import layers
from official.nlp.modeling import networks
from official.nlp.modeling.layers import position_embedding
from official.nlp.modeling.networks import encoder_scaffold
from official.nlp.transformer_v2 import configs
from official.nlp.transformer_v2 import decoder
from official.nlp.transformer_v2 import multi_channel_attention
from official.nlp.transformer_v2 import utils
from official.nlp.transformer import beam_search


def embedding_linear(embedding_matrix, x):
  """Uses embeddings as linear transformation weights."""
  with tf.name_scope("presoftmax_linear"):
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    hidden_size = tf.shape(x)[2]
    vocab_size = tf.shape(embedding_matrix)[0]

    x = tf.reshape(x, [-1, hidden_size])
    logits = tf.matmul(x, embedding_matrix, transpose_b=True)

    return tf.reshape(logits, [batch_size, length, vocab_size])


def _add_sos_to_seq(seq, start_token_id):
  """Add a start sequence token while keeping seq length."""
  batch_size = tf.shape(seq)[0]
  seq_len = tf.shape(seq)[1]
  sos_ids = tf.ones([batch_size], tf.int32) * start_token_id
  targets = tf.concat([tf.expand_dims(sos_ids, axis=1), seq], axis=1)
  targets = targets[:, :-1]
  tf.assert_equal(tf.shape(targets), (batch_size, seq_len))
  return targets


def remove_sos_from_seq(seq, pad_token_id):
  """Remove the start sequence token while keeping seq length."""
  batch_size, seq_len = tf_utils.get_shape_list(seq, expected_rank=2)
  # remove <s>
  targets = seq[:, 1:]
  # pad
  pad_ids = tf.ones([batch_size], tf.int64) * pad_token_id
  targets = tf.concat([targets, tf.expand_dims(pad_ids, axis=1)], axis=1)
  tf.assert_equal(tf.shape(targets), (batch_size, seq_len))
  return targets


class Seq2SeqTransformer(tf.keras.Model):
  """Seq2SeqTransformer encoder decoder model for training."""

  def __init__(self, params, name=None):
    super(Seq2SeqTransformer, self).__init__(name=name)

    # self.params is BERT2BERTConfig(dictionary content)
    self.params = params
    # print ('self.params', self.params)

    # bert_config is a official.nlp.bert.configs.BertConfig object at memory address XXX
    # bert_config = utils.get_bert_config_from_params(self.params)
    self.position_embedding = position_embedding.RelativePositionEmbedding(
        hidden_size=self.params["hidden_size"])

    input_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name="input_ids")
    embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=self.params["vocab_size"],
        embedding_width=self.params["hidden_size"],
        # initializer=tf.keras.initializers.TruncatedNormal(stddev=self.params["hidden_size"]**-0.5),
        initializer=tf.random_normal_initializer(
              mean=0., stddev=self.params["hidden_size"]**-0.5),
        name="word_embeddings")
    word_embeddings = embedding_layer(input_ids)
    pos_encoding = self.position_embedding(inputs=word_embeddings)
    pos_encoding = tf.cast(pos_encoding, word_embeddings.dtype)
    word_embeddings = word_embeddings + pos_encoding

    mask_processed = tf.cast(tf.not_equal(input_ids, 0), tf.float32)
    attention_mask = layers.SelfAttentionMask()([word_embeddings, mask_processed])
    network = tf.keras.Model(input_ids,
                             [word_embeddings, attention_mask])

    hidden_cfg = {
        "num_attention_heads":
            self.params["num_heads"],
        "intermediate_size":
            self.params["intermediate_size"],
        "intermediate_activation":
            tf_utils.get_activation(self.params["hidden_act"]),
        "dropout_rate":
            self.params["relu_dropout"],
        "attention_dropout_rate":
            self.params["attention_dropout"],
        "kernel_initializer":
            tf.random_normal_initializer(
              mean=0., stddev=self.params["hidden_size"]**-0.5),
    }

    # Create a encoder network.
    encoder_network = encoder_scaffold.EncoderScaffold(
        num_hidden_instances=self.params["num_hidden_layers"],
        pooled_output_dim=self.params["hidden_size"],
        pooler_layer_initializer=tf.random_normal_initializer(
              mean=0., stddev=self.params["hidden_size"]**-0.5),
        hidden_cfg=hidden_cfg,
        embedding_cls=network,
        embedding_data=embedding_layer.embeddings)

    # Create the inputs (note that the first dimension is implicit).
    # encoder_outputs, _ = encoder_network(input_ids)

    # config = params
    decoder_layer = decoder.TransformerDecoder(
        num_hidden_layers=self.params["num_decoder_layers"],
        hidden_size=self.params["hidden_size"],
        num_attention_heads=self.params["num_decoder_attn_heads"],
        intermediate_size=self.params["decoder_intermediate_size"],
        intermediate_activation=self.params["hidden_act"],
        hidden_dropout_prob=self.params["relu_dropout"],
        attention_probs_dropout_prob=self.params["attention_dropout"],
        initializer_range=self.params["hidden_size"]**-0.5,
        multi_channel_cross_attention=False,
        transformer=True,
        embedding_lookup=embedding_layer,
        config=self.params,
        name="decoder")


    # decoder_layer = decoder.Decoder(params, embedding_layer)
    # pylint: enable=protected-access


    # if not encoder_network.built:
    #   raise ValueError("bert_layer should be built.")
    # if not decoder_layer.built:
    #   raise ValueError("decoder_layer should be built.")
    self.bert_layer = encoder_network
    self.decoder_layer = decoder_layer

  def get_config(self):
    return {"params": self.params.as_dict()}

  def get_decode_logits(self,
                        decoder_inputs,
                        ids,
                        decoder_self_attention_bias,
                        step,
                        cache=None):
    if cache:
      if self.params.get("padded_decode", False):
        bias_shape = decoder_self_attention_bias.shape.as_list()
        self_attention_bias = tf.slice(
            decoder_self_attention_bias, [0, 0, step, 0],
            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])
        # print ('cache decoder_self_attention_bias', decoder_self_attention_bias)
        # print ('cache self_attention_bias', self_attention_bias)
      else:
        self_attention_bias = decoder_self_attention_bias[:, :, step:step +
                                                          1, :step + 1]

      # Sets decoder input to the last generated IDs.
      decoder_input = ids[:, -1:]
    else:
      self_attention_bias = decoder_self_attention_bias[:, :, :step + 1, :step +
                                                        1]
      decoder_input = ids
      # print ('regular self_attention_bias', self_attention_bias)
    decoder_inputs["target_ids"] = decoder_input
    decoder_inputs["self_attention_bias"] = self_attention_bias
    if cache:
      decoder_outputs = self.decoder_layer(
          decoder_inputs,
          cache,
          decode_loop_step=step,
          padded_decode=self.params.get("padded_decode", False))
    else:
      decoder_outputs = self.decoder_layer(decoder_inputs)
    # logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
    #                           decoder_outputs[:, -1:, :])
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decoder_outputs)
    logits = tf.squeeze(logits, axis=[1])
    return logits

  def _get_symbols_to_logits_fn(self, max_decode_length):
    """Returns a decoding function that calculates logits of the next tokens."""
    # Max decode length should be smaller than the positional embedding max
    # sequence length.
    decoder_self_attention_bias = decoder.get_attention_bias(
        input_tensor=None,
        bias_type="decoder_self",
        max_length=max_decode_length)

    def _symbols_to_logits_fn(ids, i, cache):
      """Generate logits for next candidate IDs.

      Args:
        ids: Current decoded sequences. int tensor with shape [batch_size *
          beam_size, i + 1]
        i: Loop index
        cache: dictionary of values storing the encoder output, encoder-decoder
          attention bias, and previous decoder attention values.

      Returns:
        Tuple of
          (logits with shape [batch_size * beam_size, vocab_size],
           updated cache values)
      """
      decoder_inputs = dict(
          # encoder_input_ids=self._encoder_input_ids,
          all_encoder_outputs=cache["all_encoder_outputs"],
          attention_bias=cache["attention_bias"])
      logits = self.get_decode_logits(
          decoder_inputs,
          ids,
          decoder_self_attention_bias,
          step=i,
          cache=cache if self.params["use_cache"] else None)
      return logits, cache

    return _symbols_to_logits_fn

  def train_decode(self, decode_outputs):
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decode_outputs)
    decode_output_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    output_log_probs = tf.nn.log_softmax(logits, axis=-1)
    return logits, decode_output_ids, output_log_probs

  def predict_decode(self, start_token_ids, cache):
    # print ('decode_max_length', self.params["decode_max_length"])
    symbols_to_logits_fn = self._get_symbols_to_logits_fn(self.params["decode_max_length"])
    # Use beam search to find the top beam_size sequences and scores.
    # print ('end_token_id', self.params["end_token_id"])
    decoded_ids, scores = beam_search.sequence_beam_search(
        symbols_to_logits_fn=symbols_to_logits_fn,
        initial_ids=start_token_ids,
        initial_cache=cache,
        vocab_size=self.params["vocab_size"],
        beam_size=self.params["beam_size"],
        alpha=self.params["alpha"],
        max_decode_length=self.params["decode_max_length"],
        padded_decode=self.params.get("padded_decode", False),
        eos_id=self.params["end_token_id"])
    # print ('decoded_ids', decoded_ids)
    # print ('scores', scores)
    return decoded_ids, scores

  def _get_logits_for_decode_ids(self, decoder_inputs, top_decoded_ids):
    """Returns the log probabilities for ids."""
    target_ids = _add_sos_to_seq(top_decoded_ids, self.params["start_token_id"])
    decoder_inputs["self_attention_bias"] = decoder.get_attention_bias(
        target_ids, bias_type="decoder_self")
    decoder_inputs["target_ids"] = target_ids
    decoder_outputs = self.decoder_layer(decoder_inputs)
    logits = embedding_linear(self.decoder_layer.embedding_lookup.embeddings,
                              decoder_outputs)
    return logits

  def _init_cache(self, batch_size):
    num_heads = self.params["num_decoder_attn_heads"]
    dim_per_head = self.params["hidden_size"] // num_heads
    init_decode_length = (
        self.params["decode_max_length"] if self.params.get("padded_decode", False) else 0)
    cache = {}
    for layer in range(self.params["num_decoder_layers"]):
      cache[str(layer)] = {
          "key":
              tf.zeros(
                  [batch_size, init_decode_length, num_heads, dim_per_head],
                  dtype=tf.float32),
          "value":
              tf.zeros(
                  [batch_size, init_decode_length, num_heads, dim_per_head],
                  dtype=tf.float32)
      }
    return cache

  def call(self, inputs, mode="train"):
    """Implements call().

    Args:
      inputs: a dictionary of tensors.
      mode: string, an enum for mode, train/eval.

    Returns:
      logits, decode_output_ids, output_log_probs for training. top_decoded_ids
      for eval.
    """
    if isinstance(inputs, dict):
      input_ids = inputs["input_ids"]
      target_ids = inputs["target_ids"]
    else:
      if len(inputs) == 2:
        input_ids, target_ids = inputs[0], inputs[1]
      else:
        # Decoding path.
        input_ids, target_ids = inputs[0], None
        if self.params["padded_decode"]:
          if not self.params["num_replicas"]:
            raise NotImplementedError(
                "Padded decoding on CPU/GPUs is not supported.")
          decode_batch_size = int(self.params["decode_batch_size"] /
                                  self.params["num_replicas"])
          input_ids.set_shape([
              decode_batch_size, self.params["decode_max_length"]
          ])

    with tf.name_scope("encode"):
      all_encoder_outputs = self.bert_layer(input_ids)

    if mode not in ("train", "eval", "predict"):
      raise ValueError("Invalid call mode: %s" % mode)
    encoder_decoder_attention_bias = decoder.get_attention_bias(
        input_ids,
        bias_type="single_cross",
        padding_value=self.params["pad_token_id"])
    if mode == "train":
      self_attention_bias = decoder.get_attention_bias(
          target_ids, bias_type="decoder_self")
      decoder_inputs = dict(
          encoder_input_ids=input_ids,
          self_attention_bias=self_attention_bias,
          attention_bias=encoder_decoder_attention_bias,
          encoder_outputs=all_encoder_outputs,
          target_ids=target_ids)
      with tf.name_scope("decode"):
        decoder_outputs = self.decoder_layer(decoder_inputs, mode="train")

      # this "if" is to provide output to old transformer's tpu training
      if self.params["static_batch"]:
        logits_decode, _, _, = self.train_decode(decoder_outputs)
        return logits_decode
      else:
        return self.train_decode(decoder_outputs)

    batch_size = tf.shape(input_ids)[0]
    # print ('start_token_ids', self.params["start_token_id"])
    start_token_ids = tf.ones([batch_size],
                              tf.int32) * self.params["start_token_id"]
    # Add encoder output and attention bias to the cache.
    if self.params["use_cache"]:
      cache = self._init_cache(batch_size)
    else:
      cache = {}
    cache["all_encoder_outputs"] = all_encoder_outputs
    cache["attention_bias"] = encoder_decoder_attention_bias
    decoded_ids, scores = self.predict_decode(start_token_ids, cache)
    if mode == "predict":
      return decoded_ids[:, 0, 1:], scores[:, 0]

    decoder_inputs = dict(
        encoder_input_ids=input_ids,
        target_ids=target_ids,
        attention_bias=encoder_decoder_attention_bias,
        encoder_outputs=all_encoder_outputs)
    top_decoded_ids = decoded_ids[:, 0, 1:]
    return self._get_logits_for_decode_ids(decoder_inputs, top_decoded_ids)


def create_transformer_model(params,
                             init_checkpoint: Optional[Text] = None
                            ) -> tf.keras.Model:
  """A helper to create Transformer model."""
  model = Seq2SeqTransformer(
      params=params,
      # bert_layer=bert_layer,
      # decoder_layer=decoder_layer,
      name="transformer")

  if init_checkpoint:
    logging.info(
        "Checkpoint file %s found and restoring from "
        "initial checkpoint.", init_checkpoint)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(init_checkpoint).expect_partial()

  return model


@gin.configurable
def get_model_params(model: Optional[Text] = "bert2bert",
                     config_class=None) -> params_dict.ParamsDict:
  """Helper function to convert config file to ParamsDict."""
  if model == "transformer":
    return configs.BERT2BERTConfig()
  elif model == "nhnet":
    return configs.NHNetConfig()
  elif config_class:
    return config_class()
  else:
    raise KeyError("The model type is not defined: %s" % model)


@gin.configurable
def create_model(params,
                 init_checkpoint: Optional[Text] = None):
  """A factory function to create different types of models."""
  return create_transformer_model(
      params, init_checkpoint=init_checkpoint)
