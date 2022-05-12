from tensorflow.python.util import nest
import tfutils as tfu
from tfutils import log_tensor, log, log_warn
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from typing import Callable
import os
import md_malconv_ds as dps
from collections import namedtuple
# from ipdb import set_trace


class MalConv(tfu.ModelBase):

    def __init__(self, md_folder, preprocessor: dps.MalConvPreprocessr,
                 embed_dim=8,
                 filter=128,
                 kernel=500,
                 stride=500,
                 affine=128,
                 md_suffix='',
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9,
                 overwrite_model=False,
                 learning_rate_decay_ep=-1,
                 rsk_limit=150,
                 *args, **kwargs):
        super(MalConv, self).__init__(
            md_folder, md_suffix, learning_rate, learning_rate_decay_factor,
            learning_rate_decay_ep,
            overwrite_model, 'seq_')
        log('unused CLI arguments to build the model: {}, {}', args, kwargs)
        self.ps = preprocessor
        if rsk_limit > 0:
            self.rsk_limit = min(rsk_limit, len(self.ps.rsk_id2rsk))
        else:
            self.rsk_limit = len(self.ps.rsk_id2rsk)
        for i in range(self.rsk_limit):
            key = self.ps.rsk_id2rsk[i]
            print('{} {} {}'.format(
                key, self.ps.rsk_id2frq[i], self.ps.rsk_rsk2dsp[key]))

        self.ly_embd = tfu.Embedding(256, embedding_dim=embed_dim)
        self.ly_gconvd = tfu.GatedConvMxPool1D(kernel, filter, stride)
        self.ly_linear = tfu.Linear(128, False)
        self.ly_pred = tfu.Linear(self.rsk_limit, False)
        self.optimizer = tf.train.AdamOptimizer(self.lr.value())

    def _encode(self, bat, use_pool=False):
        x = bat['byte']
        embd_x = self.ly_embd(x)
        gated_max_pool = self.ly_gconvd(embd_x)
        affined = self.ly_linear(gated_max_pool)
        return affined 

    def _cal_loss_and_pred(self, bat, encoded=None):
        if encoded is None:
            encoded = self._encode(bat)
        logits = self.ly_pred(encoded)
        batch_size = tfu.get_shape(logits, 1)
        logits = tf.reshape(
            logits, [-1])
        log_tensor(logits)
        confs = tf.sigmoid(logits)
        log_tensor(confs)

        if 'rsk' in bat:
            labels = tf.not_equal(bat['rsk'], 0)
            labels = labels[:, :self.rsk_limit]
            labels = tf.reshape(labels, [-1])
            labels = tf.cast(labels, tf.float32)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
            metrics = tfu.classification_binary_metrics(
                confs, labels, name='cf')
            _, metrics['conf_entropy'] = tf.metrics.mean(loss)
            confs = tf.reshape(confs, [batch_size, -1])
            return confs, loss, metrics
        else:
            confs = tf.reshape(confs, [batch_size, -1])
            return confs

    def get_loss(self, bat):
        _, loss, metrics = self._cal_loss_and_pred(bat)
        return loss, metrics


def set_flags(parser):
    parser = parser.add_argument_group('model configuraion')
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=8,
        metavar="",
        help="Embedding dim for single byte.")
    parser.add_argument(
        "--filter",
        type=int,
        default=128,
        metavar="",
        help="Number of filters used in conv1d.")
    parser.add_argument(
        "--kernel",
        type=int,
        default=500,
        metavar="",
        help="Kernel size for conv1d.")
    parser.add_argument(
        "--stride",
        type=int,
        default=500,
        metavar="",
        help="Stride used in conv1d.")
    parser.add_argument(
        "--affine",
        type=int,
        default=128,
        metavar="",
        help="Linear transformantion before generating logotis.")
    parser.add_argument(
        "--md_suffix",
        type=str,
        default='',
        metavar="",
        help="The suffix to store the model "
        "(distingushed different hyperparams).")
    parser.add_argument(
        "--md_folder",
        type=str,
        default="model",
        metavar="",
        help="Base directory to model files and saved logs.")
    parser.add_argument(
        "--rsk_limit",
        type=int,
        default=150,
        metavar="",
        help="Rsks to be considered.")
