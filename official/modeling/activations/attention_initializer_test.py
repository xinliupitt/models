# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for attention initializer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling.activations import attention_initializer
from official.nlp.modeling import layers

@keras_parameterized.run_all_keras_modes
class AttentionInitializerTest(keras_parameterized.TestCase):

  def test_attention_initializer(self):
    query_dense_layer = layers.DenseEinsum(
        output_shape=(2, 4),
        kernel_initializer=attention_initializer.attention_initializer(16))
    weights = query_dense_layer._kernel
    print ('weights', weights)

if __name__ == '__main__':
  tf.test.main()
