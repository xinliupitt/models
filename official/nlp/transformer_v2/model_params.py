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
"""Defines Transformer model parameters."""

from collections import defaultdict


BASE_PARAMS = defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=2048,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=32768,
    max_length=256,  # Maximum number of tokens per example.

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=33708,  # Number of tokens defined in the vocabulary file.
    hidden_size=512,  # Model dimension in the hidden layers.

    filter_size=2048,  # Inner layer dimension in the feedforward network.

    intermediate_size = 4096,
    decoder_intermediate_size = 4096,

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,


    # params names from bert
    # under similar name, they should be the same as values above


    hidden_act = "relu",
    # hidden_dropout_prob = 0.1,
    # attention_probs_dropout_prob = 0.1,
    max_position_embeddings = 512,
    type_vocab_size = 2,
    initializer_range = 0.02,

    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_decoder_layers = 6,

    num_heads=8,  # Number of heads to use in multi-headed attention.
    num_decoder_attn_heads = 8,


    # label_smoothing = 0.1,
    # learning_rate = 0.05,
    # learning_rate_warmup_steps = 20000,
    optimizer = "Adam",
    adam_beta1 = 0.9,
    adam_beta2 = 0.997,
    adam_epsilon = 1e-09,

    # predict params
    # beam_size = 5,
    # alpha = 0.6,
    # initializer_gain = 1.0,
    use_cache = True,

    # input params
    input_sharding = False,
    input_data_not_padded = False,
    pad_token_id = 0,
    end_token_id = 1,
    start_token_id = 0,



)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,
    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,

    filter_size=4096,

    num_heads=16,
    num_decoder_attn_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=1024,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)
