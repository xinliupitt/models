# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test Transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.nlp.modeling.models import seq2seq_transformer
from official.nlp.transformer import metrics
from official.nlp.transformer import model_params

def create_model(params, is_train):
  """Creates transformer model."""
  with tf.name_scope("model"):
    if is_train:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
      internal_model = seq2seq_transformer.Seq2SeqTransformer(params, name="transformer_v2")
      logits = internal_model([inputs, targets], training=is_train)
      # vocab_size = params["vocab_size"]
      # label_smoothing = params["label_smoothing"]
      # if params["enable_metrics_in_training"]:
      #   logits = metrics.MetricLayer(vocab_size)([logits, targets])
      # logits = tf.keras.layers.Lambda(lambda x: x, name="logits",
      #                                 dtype=tf.float32)(logits)
      model = tf.keras.Model([inputs, targets], logits)
      # loss = metrics.transformer_loss(
      #     logits, targets, label_smoothing, vocab_size)
      # model.add_loss(loss)
      return model

    else:
      inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
      internal_model = seq2seq_transformer.Seq2SeqTransformer(params, name="transformer_v2")
      ret = internal_model([inputs], training=is_train)
      outputs, scores = ret["outputs"], ret["scores"]
      return tf.keras.Model(inputs, [outputs, scores])

class TransformerV2Test(tf.test.TestCase):

  def setUp(self):
    self.params = params = model_params.TINY_PARAMS
    params["batch_size"] = params["default_batch_size"] = 16
    params["use_synthetic_data"] = True
    params["hidden_size"] = 12
    params["num_hidden_layers"] = 2
    params["filter_size"] = 14
    params["num_heads"] = 2
    params["vocab_size"] = 41
    params["extra_decode_length"] = 2
    params["beam_size"] = 3
    params["dtype"] = tf.float32

  def test_create_model_train(self):
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    internal_model = seq2seq_transformer.Seq2SeqTransformer(self.params, name="transformer_v2")
    logits = internal_model([inputs, targets], training=True)
    model = tf.keras.Model([inputs, targets], logits)
    inputs, outputs = model.inputs, model.outputs
    self.assertEqual(len(inputs), 2)
    self.assertEqual(len(outputs), 1)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(inputs[1].shape.as_list(), [None, None])
    self.assertEqual(inputs[1].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None, 41])
    self.assertEqual(outputs[0].dtype, tf.float32)

  def test_create_model_not_train(self):
    model = create_model(self.params, False)
    inputs, outputs = model.inputs, model.outputs
    self.assertEqual(len(inputs), 1)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(inputs[0].shape.as_list(), [None, None])
    self.assertEqual(inputs[0].dtype, tf.int64)
    self.assertEqual(outputs[0].shape.as_list(), [None, None])
    self.assertEqual(outputs[0].dtype, tf.int32)
    self.assertEqual(outputs[1].shape.as_list(), [None])
    self.assertEqual(outputs[1].dtype, tf.float32)


if __name__ == "__main__":
  tf.test.main()
