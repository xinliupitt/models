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

from official.nlp.transformer import model_params
from official.nlp.transformer import transformer

is_train = True

def _count_params(self, layer, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return layer.count_params()
  else:
    return int(
        np.sum([
            tf.keras.backend.count_params(p) for p in layer.trainable_weights
        ]))

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

  if is_train:
    def test_create_model_train(self):
      model = transformer.create_model(self.params, True)
      inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
      targets = np.asarray([[4, 3, 0], [13, 19, 17], [20, 14, 1], [5, 7, 0]])
      model([inputs, targets], training=True)
      print ("new params count", self._count_params(model))
      w = model.get_weights()
      print ('weights', w[0])

  if not is_train:
    def test_create_model_not_train(self):
      model = transformer.create_model(self.params, False)
      inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
      model([inputs], training=False)



if __name__ == "__main__":
  tf.test.main()
