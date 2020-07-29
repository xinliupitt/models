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
from numpy import load
from numpy import save

import tensorflow as tf

from official.nlp.transformer import model_params
from official.nlp.transformer import transformer

is_train = True
get_weights_flag = False

def _count_params(layer, trainable_only=True):
  """Returns the count of all model parameters, or just trainable ones."""
  if not trainable_only:
    return layer.count_params()
  else:
    return int(
        np.sum([
            tf.keras.backend.count_params(p) for p in layer.trainable_weights
        ]))

# def _count_params_model(model_name):
#   total_parameters = 0
#   for variable in model_name.trainable_variables:
#       # shape is an array of tf.Dimension
#       shape = variable.get_shape()
#       # print(shape)
#       # print(len(shape))
#       variable_parameters = 1
#       for dim in shape:
#           # print(dim)
#           variable_parameters *= dim
#       # print(variable_parameters)
#       total_parameters += variable_parameters
#   print(total_parameters)
#   return total_parameters

class TransformerV2Test(tf.test.TestCase):

  def setUp(self):
    self.params = params = model_params.TINY_PARAMS
    params["batch_size"] = params["default_batch_size"] = 16
    params["use_synthetic_data"] = True
    params["hidden_size"] = 12
    params["num_hidden_layers"] = 1
    params["filter_size"] = 14
    params["num_heads"] = 2
    params["vocab_size"] = 41
    params["extra_decode_length"] = 2
    params["beam_size"] = 3
    params["dtype"] = tf.float32

  if is_train:
    if get_weights_flag:
      def test_a_get_weights(self):
        model = transformer.create_model(self.params, True)
        w = model.get_weights()
        save('w.npy', w)

    def test_create_model_train(self):
      model = transformer.create_model(self.params, True)
      inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
      targets = np.asarray([[4, 3, 0], [13, 19, 17], [20, 14, 1], [5, 7, 0]])
      w = load('w.npy', allow_pickle=True)
      model.set_weights(w)
      print ('Model begins!')
      print ('Model begins!')
      model([inputs, targets], training=True)
      # print ("new params count", self._count_params(model))

      # print ('w[0]', w[0])
      # print ('weight count', len(w))
      # print ("new params count", len(model.trainable_variables))
      # print ('model params', _count_params_model(model))
      print ('model params', _count_params(model))


  if not is_train:
    if get_weights_flag:
      def test_a_get_weights_eval(self):
        model = transformer.create_model(self.params, False)
        w_eval = model.get_weights()
        save('w_eval.npy', w_eval)
        print ('model params', _count_params(model))

    def test_create_model_not_train(self):
      model = transformer.create_model(self.params, False)
      inputs = np.asarray([[5, 2, 1], [7, 5, 0], [1, 4, 0], [7, 5, 11]])
      w_eval = load('w_eval.npy', allow_pickle=True)
      model.set_weights(w_eval)
      model([inputs], training=False)



if __name__ == "__main__":
  tf.test.main()
