# Lint as: python3
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
"""Run NHNet model training and eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import app
from absl import flags
from absl import logging
from six.moves import zip
import tensorflow as tf
from official.modeling import performance
from official.modeling.hyperparams import params_dict
from official.nlp.transformer_v2 import compute_bleu
from official.nlp.transformer_v2 import data_pipeline
from official.nlp.transformer_v2 import evaluation
from official.nlp.transformer_v2 import metrics
from official.nlp.transformer_v2 import misc
from official.nlp.transformer_v2 import model_params
from official.nlp.transformer_v2 import input_pipeline
from official.nlp.transformer_v2 import models
from official.nlp.transformer import optimizer as optimizer_v1
from official.nlp.transformer_v2 import optimizer
from official.nlp.transformer_v2 import tokenizer
from official.nlp.transformer_v2 import translate
from official.nlp.transformer import metrics as transformer_metrics
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import keras_utils

INF = int(1e9)
BLEU_DIR = "bleu"
_SINGLE_SAMPLE = 1

FLAGS = flags.FLAGS
PARAMS_MAP = {
    'tiny': model_params.TINY_PARAMS,
    'base': model_params.BASE_PARAMS,
    'big': model_params.BIG_PARAMS,
}

def define_flags():
  """Defines command line flags used by NHNet trainer."""

  # distribution strategy activation and necessary flags
  misc.define_transformer_flags()

  ## Required parameters
  flags.DEFINE_enum("mode", "train", ["train", "eval", "train_and_eval"],
                    "Execution mode.")
  flags.DEFINE_string("train_file_pattern", "", "Train file pattern.")
  flags.DEFINE_string("eval_file_pattern", "", "Eval file pattern.")
  # flags.DEFINE_string(
  #     "model_dir", None,
  #     "The output directory where the model checkpoints will be written.")

  # Model training specific flags.
  # flags.DEFINE_enum(
  #     "distribution_strategy", "mirrored", ["tpu", "mirrored"],
  #     "Distribution Strategy type to use for training. `tpu` uses TPUStrategy "
  #     "for running on TPUs, `mirrored` uses GPUs with single host.")
  # flags.DEFINE_string("tpu", "", "TPU address to connect to.")
  flags.DEFINE_string(
      "init_checkpoint", None,
      "Initial checkpoint (usually from a pre-trained BERT model).")
  # flags.DEFINE_integer("train_steps", 100000, "Max train steps")
  # flags.DEFINE_integer("eval_steps", 32, "Number of eval steps per run.")
  flags.DEFINE_integer("eval_timeout", 3000, "Timeout waiting for checkpoints.")
  flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
  flags.DEFINE_integer("eval_batch_size", 4, "Total batch size for evaluation.")
  # flags.DEFINE_integer(
  #     "steps_per_loop", 1000,
  #     "Number of steps per graph-mode loop. Only training step "
  #     "happens inside the loop.")
  flags.DEFINE_integer("checkpoint_interval", 2000, "Checkpointing interval.")
  # flags.DEFINE_integer("len_title", 15, "Title length.")
  # flags.DEFINE_integer("len_passage", 200, "Passage length.")
  # flags.DEFINE_integer("hidden_size", 512,
  #                      "Number of hidden layers.")
  flags.DEFINE_integer("num_encoder_layers", 12,
                       "Number of hidden layers of encoder.")
  flags.DEFINE_integer("num_decoder_layers", 12,
                       "Number of hidden layers of decoder.")
  # flags.DEFINE_string("model_type", "transformer",
  #                     "Model type to choose a model configuration.")
  # flags.DEFINE_integer(
  #     "num_nhnet_articles", 5,
  #     "Maximum number of articles in NHNet, only used when model_type=nhnet")
  flags.DEFINE_string(
      "params_override",
      default=None,
      help=("a YAML/JSON string or a YAML file which specifies additional "
            "overrides over the default parameters"))
  flags.DEFINE_boolean(
      name='use_tpu', default=False,
      help='Whether to use TPU.')
  # flags.DEFINE_integer("max_position_embeddings", 97,
  #                      "maximal position embeddings length.")



def load_weights_if_possible(model, init_weight_path=None, use_tpu=False,
                             params=None):
  """Loads model weights when it is provided."""
  if init_weight_path:
    logging.info("Load weights: {}".format(init_weight_path))
    # TODO(b/139414977): Having the same variable restoring method for both
    # TPU and GPU.
    if use_tpu:
      print ('use tpu weights')
      checkpoint = tf.train.Checkpoint(
          model=model, optimizer=create_optimizer(params))
      # checkpoint.restore(init_weight_path).expect_partial()
      checkpoint.restore(
        init_weight_path).assert_existing_objects_matched().expect_partial()
    else:
      # model.load_weights(init_weight_path)
      pass
  else:
    logging.info("Weights not loaded from path:{}".format(init_weight_path))

def translate_and_compute_bleu(model,
                               params,
                               subtokenizer,
                               bleu_source,
                               bleu_ref,
                               distribution_strategy=None):
  """Translate file and report the cased and uncased bleu scores.
  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    subtokenizer: A subtokenizer object, used for encoding and decoding source
      and translated lines.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.
  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  # Create temporary file to store translation.
  tmp = tempfile.NamedTemporaryFile(delete=False)
  tmp_filename = tmp.name

  translate.translate_file(
      model,
      params,
      subtokenizer,
      bleu_source,
      output_file=tmp_filename,
      print_all_translations=False,
      distribution_strategy=distribution_strategy,
      model_dir=FLAGS.model_dir)

  # Compute uncased and cased bleu scores.
  uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
  cased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, True)
  os.remove(tmp_filename)
  return uncased_score, cased_score

def create_optimizer(params):
  """Creates optimizer."""
  # params = self.params
  lr_schedule = optimizer_v1.LearningRateSchedule(
      params["learning_rate"], params["hidden_size"],
      params["learning_rate_warmup_steps"])
  opt = tf.keras.optimizers.Adam(
      lr_schedule if FLAGS.use_tpu else params["learning_rate"],
      params["optimizer_adam_beta1"],
      params["optimizer_adam_beta2"],
      epsilon=params["optimizer_adam_epsilon"])

  opt = performance.configure_optimizer(
      opt,
      use_float16=params["dtype"] == tf.float16,
      use_graph_rewrite=FLAGS.fp16_implementation == "graph_rewrite",
      loss_scale=flags_core.get_loss_scale(
          FLAGS, default_for_fp16="dynamic"))

  return opt

def evaluate_and_log_bleu(model,
                          params,
                          bleu_source,
                          bleu_ref,
                          vocab_file,
                          distribution_strategy=None):
  """Calculate and record the BLEU score.
  Args:
    model: A Keras model, used to generate the translations.
    params: A dictionary, containing the translation related parameters.
    bleu_source: A file containing source sentences for translation.
    bleu_ref: A file containing the reference for the translated sentences.
    vocab_file: A file containing the vocabulary for translation.
    distribution_strategy: A platform distribution strategy, used for TPU based
      translation.
  Returns:
    uncased_score: A float, the case insensitive BLEU score.
    cased_score: A float, the case sensitive BLEU score.
  """
  subtokenizer = tokenizer.Subtokenizer(vocab_file)

  uncased_score, cased_score = translate_and_compute_bleu(
      model, params, subtokenizer, bleu_source, bleu_ref, distribution_strategy)

  logging.info("Bleu score (uncased): %s", uncased_score)
  logging.info("Bleu score (cased): %s", cased_score)
  return uncased_score, cased_score

# pylint: disable=protected-access


class Trainer(tf.keras.Model):
  """A training only model."""

  def __init__(self, model, params):
    super(Trainer, self).__init__()
    self.model = model
    self.params = params
    self._num_replicas_in_sync = tf.distribute.get_strategy(
    ).num_replicas_in_sync

  def call(self, inputs, mode="train"):
    return self.model(inputs, mode)

  def train_step(self, inputs):
    """The logic for one training step."""
    with tf.GradientTape() as tape:
      logits, _, _ = self(inputs, mode="train", training=True)
      targets = models.remove_sos_from_seq(inputs[1],
                                           self.params["pad_token_id"])
      loss = transformer_metrics.transformer_loss(logits, targets,
                                                  self.params["label_smoothing"],
                                                  self.params["vocab_size"])
      # Scales the loss, which results in using the average loss across all
      # of the replicas for backprop.
      scaled_loss = loss / self._num_replicas_in_sync

    tvars = self.trainable_variables
    grads = tape.gradient(scaled_loss, tvars)
    self.optimizer.apply_gradients(list(zip(grads, tvars)))
    return {
        "training_loss": loss,
        "learning_rate": self.optimizer._decayed_lr(var_dtype=tf.float32)
    }

def train(params, strategy, dataset=None):
  """Runs training."""


  if FLAGS.distribution_strategy=="tpu":
    distribution_strategy = strategy

    params["num_replicas"] = distribution_strategy.num_replicas_in_sync
    if distribution_strategy:
      logging.info("For training, using distribution strategy: %s",
                   distribution_strategy)
    performance.set_mixed_precision_policy(
        params["dtype"],
        flags_core.get_loss_scale(FLAGS, default_for_fp16="dynamic"))

    keras_utils.set_session_config(enable_xla=FLAGS.enable_xla)

    def _ensure_dir(log_dir):
      """Makes log dir if not existed."""
      if not tf.io.gfile.exists(log_dir):
        tf.io.gfile.makedirs(log_dir)

    _ensure_dir(FLAGS.model_dir)

    with distribution_utils.get_strategy_scope(distribution_strategy):
      model = models.create_model(params, init_checkpoint=FLAGS.init_checkpoint)

      opt = create_optimizer(params)

      current_step = 0

      model.global_step = opt.iterations
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      checkpoint_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=FLAGS.model_dir,
          max_to_keep=10,
          step_counter=model.global_step,
          checkpoint_interval=FLAGS.checkpoint_interval)
      if checkpoint_manager.restore_or_initialize():
        logging.info("Training restored from the checkpoints in: %s",
                     FLAGS.model_dir)


      # checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      # latest_checkpoint = tf.train.latest_checkpoint(FLAGS.model_dir)
      # if latest_checkpoint:
      #   checkpoint.restore(latest_checkpoint)
      #   logging.info("Loaded checkpoint %s", latest_checkpoint)
      #   current_step = opt.iterations.numpy()

      train_loss_metric = tf.keras.metrics.Mean(
          "training_loss", dtype=tf.float32)
      if params["enable_tensorboard"]:
        summary_writer = tf.compat.v2.summary.create_file_writer(
            FLAGS.model_dir)
      else:
        summary_writer = tf.compat.v2.summary.create_noop_writer()
      train_metrics = [train_loss_metric]
      if params["enable_metrics_in_training"]:
        train_metrics = train_metrics + model.metrics

    params["batch_size"] /= distribution_strategy.num_replicas_in_sync

    print ('error params')
    for k, v in params.items():
      print (k, v)


    train_ds = (
        distribution_strategy
        .experimental_distribute_datasets_from_function(
            lambda ctx: data_pipeline.train_input_fn(params, ctx)))
    train_ds_iterator = iter(train_ds)

    def create_callbacks(cur_log_dir, init_steps, params):
      """Creates a list of callbacks."""
      sfunc = optimizer_v1.LearningRateFn(params["learning_rate"],
                                       params["hidden_size"],
                                       params["learning_rate_warmup_steps"])
      scheduler_callback = optimizer_v1.LearningRateScheduler(sfunc, init_steps)
      callbacks = misc.get_callbacks()
      callbacks.append(scheduler_callback)
      if params["enable_checkpointing"]:
        ckpt_full_path = os.path.join(cur_log_dir, "cp-{epoch:04d}.ckpt")
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_full_path, save_weights_only=True))
      return callbacks

    callbacks = create_callbacks(FLAGS.model_dir, 0, params)

    # Only TimeHistory callback is supported for CTL
    callbacks = [cb for cb in callbacks
                 if isinstance(cb, keras_utils.TimeHistory)]

    # TODO(b/139418525): Refactor the custom training loop logic.
    @tf.function
    def train_steps(iterator, steps):
      """Training steps function for TPU runs.
      Args:
        iterator: The input iterator of the training dataset.
        steps: An integer, the number of training steps.
      Returns:
        A float, the loss value.
      """

      def _step_fn(inputs):
        """Per-replica step function."""
        inputs, targets = inputs
        with tf.GradientTape() as tape:
          logits = model([inputs, targets], training=True)
          loss = metrics.transformer_loss(logits, targets,
                                          params["label_smoothing"],
                                          params["vocab_size"])
          # Scales the loss, which results in using the average loss across all
          # of the replicas for backprop.
          scaled_loss = loss / distribution_strategy.num_replicas_in_sync

        # De-dupes variables due to keras tracking issues.
        tvars = list({id(v): v for v in model.trainable_variables}.values())
        grads = tape.gradient(scaled_loss, tvars)
        opt.apply_gradients(zip(grads, tvars))
        # For reporting, the metric takes the mean of losses.
        train_loss_metric.update_state(loss)

      for _ in tf.range(steps):
        train_loss_metric.reset_states()
        distribution_strategy.run(
            _step_fn, args=(next(iterator),))

    cased_score, uncased_score = None, None
    cased_score_history, uncased_score_history = [], []
    while current_step < FLAGS.train_steps:
      remaining_steps = FLAGS.train_steps - current_step
      train_steps_per_eval = (
          remaining_steps if remaining_steps < FLAGS.steps_between_evals
          else FLAGS.steps_between_evals)
      current_iteration = current_step // FLAGS.steps_between_evals

      logging.info(
          "Start train iteration at global step:{}".format(current_step))
      history = None
      if not FLAGS.use_tpu:
        raise NotImplementedError(
            "Custom training loop on GPUs is not implemented.")

      # Runs training steps.
      with summary_writer.as_default():
        for cb in callbacks:
          cb.on_epoch_begin(current_iteration)
          cb.on_batch_begin(0)

        train_steps(
            train_ds_iterator,
            tf.convert_to_tensor(train_steps_per_eval, dtype=tf.int32))
        current_step += train_steps_per_eval
        train_loss = train_loss_metric.result().numpy().astype(float)
        logging.info("Train Step: %d/%d / loss = %s", current_step,
                     FLAGS.train_steps, train_loss)

        for cb in callbacks:
          cb.on_batch_end(train_steps_per_eval - 1)
          cb.on_epoch_end(current_iteration)

        if params["enable_tensorboard"]:
          for metric_obj in train_metrics:
            tf.compat.v2.summary.scalar(metric_obj.name, metric_obj.result(),
                                        current_step)
            summary_writer.flush()

      for cb in callbacks:
        cb.on_train_end()

      if FLAGS.enable_checkpointing:
        # avoid check-pointing when running for benchmarking.
        checkpoint_name = checkpoint.save(
            os.path.join(FLAGS.model_dir,
                         "ctl_step_{}.ckpt".format(current_step)))
        logging.info("Saved checkpoint to %s", checkpoint_name)

    stats = ({
        "loss": train_loss
    } if history is None else {})
    misc.update_stats(history, stats, callbacks)
    if uncased_score and cased_score:
      stats["bleu_uncased"] = uncased_score
      stats["bleu_cased"] = cased_score
      stats["bleu_uncased_history"] = uncased_score_history
      stats["bleu_cased_history"] = cased_score_history
    return stats



  else:
    if not dataset:
      train_ds = data_pipeline.train_input_fn(params)
      map_data_fn = data_pipeline.map_data_for_transformer_fn
      # train_ds = train_ds.map(
      #     map_data_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = train_ds
    with strategy.scope():
      model = models.create_model(params, init_checkpoint=FLAGS.init_checkpoint)
      opt = optimizer.create_optimizer(params)
      trainer = Trainer(model, params)
      model.global_step = opt.iterations

      trainer.compile(
          optimizer=opt,
          experimental_steps_per_execution=FLAGS.steps_per_loop)
      summary_dir = os.path.join(FLAGS.model_dir, "summaries")
      summary_callback = tf.keras.callbacks.TensorBoard(
          summary_dir, update_freq=max(100, FLAGS.steps_per_loop))
      checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
      checkpoint_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=FLAGS.model_dir,
          max_to_keep=10,
          step_counter=model.global_step,
          checkpoint_interval=FLAGS.checkpoint_interval)
      if checkpoint_manager.restore_or_initialize():
        logging.info("Training restored from the checkpoints in: %s",
                     FLAGS.model_dir)
      checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    # Trains the model.
    steps_per_epoch = min(FLAGS.train_steps, FLAGS.checkpoint_interval)
    epochs = FLAGS.train_steps // steps_per_epoch
    history = trainer.fit(
        x=dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[summary_callback, checkpoint_callback],
        verbose=2)
    train_hist = history.history
    # Gets final loss from training.
    stats = dict(training_loss=float(train_hist["training_loss"][-1]))
  return stats


def eval(distribution_strategy=None,
         params=None,
         predict_model=None):
  """Evaluates the model."""
  distribution_strategy = distribution_strategy if FLAGS.use_tpu else None

  # We only want to create the model under DS scope for TPU case.
  # When 'distribution_strategy' is None, a no-op DummyContextManager will
  # be used.
  with distribution_utils.get_strategy_scope(distribution_strategy):
    if not predict_model:
      params["layer_postprocess_dropout"] = 0
      params["attention_dropout"] = 0
      params["relu_dropout"] = 0
      predict_model = models.create_model(params,
                                          init_checkpoint=FLAGS.init_checkpoint)
    load_weights_if_possible(
      predict_model,
      tf.train.latest_checkpoint(FLAGS.model_dir),
      FLAGS.use_tpu,
      params)
    # self.predict_model.summary()
  return evaluate_and_log_bleu(
      predict_model, params, FLAGS.bleu_source,
      FLAGS.bleu_ref, FLAGS.vocab_file,
      distribution_strategy)


def run():
  """Runs NHNet using Keras APIs."""

  # Add flag-defined parameters to params object
  num_gpus = flags_core.get_num_gpus(FLAGS)
  params = misc.get_model_params(FLAGS.param_set, num_gpus)
  params["num_gpus"] = FLAGS.num_gpus

  if FLAGS.distribution_strategy=="tpu":
    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=FLAGS.distribution_strategy,
        num_gpus=params["num_gpus"],
        all_reduce_alg=FLAGS.all_reduce_alg,
        num_packs=FLAGS.num_packs,
        tpu_address=FLAGS.tpu or "")
  else:
    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy=FLAGS.distribution_strategy, tpu_address=FLAGS.tpu)
  if strategy:
    logging.info("***** Number of cores used : %d",
                 strategy.num_replicas_in_sync)


  # regular_params = {}
  params["use_ctl"] = FLAGS.use_ctl
  params["data_dir"] = FLAGS.data_dir
  params["model_dir"] = FLAGS.model_dir
  params["static_batch"] = FLAGS.static_batch
  params["max_length"] = FLAGS.max_length
  params["decode_batch_size"] = FLAGS.decode_batch_size
  params["decode_max_length"] = FLAGS.decode_max_length
  params["padded_decode"] = FLAGS.padded_decode
  params["max_io_parallelism"] = (
      FLAGS.num_parallel_calls or tf.data.experimental.AUTOTUNE)

  params["use_synthetic_data"] = FLAGS.use_synthetic_data
  params["batch_size"] = FLAGS.train_batch_size # or params["default_batch_size"]
  params["repeat_dataset"] = None
  params["dtype"] = flags_core.get_tf_dtype(FLAGS)
  params["enable_tensorboard"] = FLAGS.enable_tensorboard
  params["enable_metrics_in_training"] = FLAGS.enable_metrics_in_training
  params["steps_between_evals"] = FLAGS.steps_between_evals
  params["enable_checkpointing"] = FLAGS.enable_checkpointing
  params["use_cache"] = FLAGS.use_cache
  # params["hidden_size"] = FLAGS.hidden_size
  params["use_tpu"] = FLAGS.use_tpu

  stats = 0

  if "train" in FLAGS.mode:
    stats = train(params, strategy)
  if "eval" in FLAGS.mode:
    timeout = 0 if FLAGS.mode == "train_and_eval" else FLAGS.eval_timeout
    # Uses padded decoding for TPU. Always uses cache.
    params["padded_decode"] = False
    if FLAGS.distribution_strategy=="tpu":
      padded_decode = isinstance(strategy, tf.distribute.experimental.TPUStrategy)
      # params.override({
      #     "padded_decode": padded_decode,
      # }, is_strict=False)
      params["padded_decode"] = padded_decode
      params["num_replicas"] = strategy.num_replicas_in_sync
      stats = eval(distribution_strategy=strategy,
           params=params,
           predict_model=None)
    else:
      stats = eval(distribution_strategy=strategy,
           params=params,
           predict_model=None)
  return stats


def main(_):
  stats = run()
  if stats:
    logging.info("Stats:\n%s", stats)

if __name__ == "__main__":
  define_flags()
  app.run(main)
