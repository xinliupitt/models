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

import numpy as np

import tensorflow as tf

from official.nlp.modeling.models import seq2seq_transformer
from official.nlp.transformer import model_params
from official.nlp.transformer import transformer

class TransformerV2Test(tf.test.TestCase):

  def setUp(self):
    self.params = params = model_params.TINY_PARAMS
    params["batch_size"] = params["default_batch_size"] = 16
    params["hidden_size"] = 12
    params["num_hidden_layers"] = 3
    params["filter_size"] = 14
    params["num_heads"] = 2
    params["vocab_size"] = 41
    params["extra_decode_length"] = 2
    params["beam_size"] = 3
    params["dtype"] = tf.float32

  def test_forward_pass_train(self):
    # Set input_len different from target_len
    inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
    targets = np.asarray(
        [[4, 3, 4, 0], [13, 19, 17, 8], [20, 14, 1, 2], [5, 7, 3, 0]])

    # src_model is the original model before refactored.
    src_model = transformer.create_model(self.params, True)
    src_num_weights = _count_params(src_model)
    src_weights = src_model.get_weights()
    src_model_output = src_model([inputs, targets], training=True)

    # dest_model is the refactored model. Please create it to be different from
    # src_model
    dest_model = transformer.create_model(self.params, True)
    dest_num_weights = _count_params(dest_model)
    if src_num_weights != dest_num_weights:
      raise ValueError("Source weights can't be set to destination model due to"
                       "different number of weights.")
    dest_model.set_weights(src_weights)
    dest_model_output = dest_model([inputs, targets], training=True)

    # If src_model and dest_model contains dropout layers, they can't have the
    # same output. Please disable all dropout layers before test.
    self.assertAllEqual(src_model_output, dest_model_output)



  def test_forward_pass_not_train(self):
    inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])

    # src_model is the original model before refactored.
    src_model = transformer.create_model(self.params, False)
    src_num_weights = _count_params(src_model)
    src_weights = src_model.get_weights()
    src_model_output = src_model([inputs], training=False)

    # dest_model is the refactored model. Please create it to be different from
    # src_model
    # dest_model = transformer.create_model(self.params, False)
    dest_model = seq2seq_transformer.create_model(self.params, False)
    dest_num_weights = _count_params(dest_model)
    if src_num_weights != dest_num_weights:
      raise ValueError("Source weights can't be set to destination model due to"
                       "different number of weights.")
    dest_model.set_weights(src_weights)
    dest_model_output = dest_model([inputs], training=False)

    # If src_model and dest_model contains dropout layers, they can't have the
    # same output. Please disable all dropout layers before test.
    self.assertAllEqual(src_model_output[0], dest_model_output[0])
    self.assertAllEqual(src_model_output[1], dest_model_output[1])

def _count_params(layer, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return layer.count_params()
  else:
    return int(
        np.sum([
            tf.keras.backend.count_params(p) for p in layer.trainable_weights
        ]))

if __name__ == "__main__":
  tf.test.main()
