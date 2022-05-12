from tensorflow.python.util import nest
import tfutils as tfu
from tfutils import log_tensor, log, log_warn
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from typing import Callable
import os
import md_raven_ds as dps
from collections import namedtuple
import numpy as np
from md_raven_train import main


class MultiHeadAttnLayer(tf.keras.layers.Layer):

    def __init__(self, attn_size, num_heads=1,
                 name=None, out_lin=True, **kwargs):
        super(MultiHeadAttnLayer, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.attn_size = attn_size
        self.lin_mem = None
        self.mem_length = None
        self.ws = None
        self.bs = None
        self.vs = None
        self.lin_out = tfu.Linear(attn_size)
        self.out_lin = out_lin
        tfu.track_dependent_layers(self)

    def atnn(self, memory):
        # memory [batch, time, depth]
        in_dim = tfu.get_shape(memory, -1)
        attn_size = self.attn_size
        num_heads = self.num_heads

        if self.vs is None:
            self.vs = [
                self.add_variable(
                    'v' + str(i), [1, 1, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                for i in range(num_heads)
            ]
        if self.ws is None:
            self.ws = [
                self.add_variable(
                    'w' + str(i), [in_dim, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
                for i in range(num_heads)
            ]
        if self.bs is None:
            self.bs = [self.add_variable(
                'b' + str(i), [attn_size],
                dtype=tf.float32,
                initializer=tf.zeros_initializer)
                for i in range(num_heads)]

        probs = []
        attns = []
        log('building {} heads...', num_heads)
        for hd in range(num_heads):
            lin1 = (tfu.t_dot(memory, self.ws[hd]) + self.bs[hd]) * self.vs[hd]
            prob = tf.nn.softmax(tf.reduce_sum(lin1, -1))
            attn = tf.reduce_max(tf.expand_dims(prob, -1) * memory, 1)
            probs.append(prob)
            attns.append(attn)

        probs = tf.transpose(tf.stack(probs), [1, 0, 2])
        attns = tf.transpose(tf.stack(attns), [1, 0, 2])

        attns = tf.reduce_max(attns, axis=1)

        # attns = tf.reshape(
        #     attns,
        #     [tfu.get_shape(attns, 0), -1]
        # )
        # if self.out_lin:
        #     attns = self.lin_out(attns)

        log_tensor(probs)
        log_tensor(attns)
        return attns, probs


class RavenVConv3(tfu.ModelBase):

    def __init__(self, md_folder, preprocessor: dps.RavenPreprocessor,
                 dat_embd_dim=8,
                 encoder_dim=128,
                 md_suffix='',
                 attn_hidden=128,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9,
                 overwrite_model=False,
                 learning_rate_decay_ep=-1,
                 rsk_limit=150,
                 use_pretain_asm_embd=True,
                 obj_lookup_fn: Callable=None,
                 *args, **kwargs):
        super(RavenVConv3, self).__init__(
            md_folder, md_suffix,
            learning_rate, learning_rate_decay_factor,
            learning_rate_decay_ep,
            overwrite_model, 'seq_')
        log('unused CLI arguments to build the model: {}, {}', args, kwargs)
        self.ps = preprocessor
        self.obj_lookup_fn = obj_lookup_fn
        if rsk_limit > 0:
            self.rsk_limit = min(rsk_limit, len(self.ps.rsk_id2rsk))
        else:
            self.rsk_limit = len(self.ps.rsk_id2rsk)
        for i in range(self.rsk_limit):
            key = self.ps.rsk_id2rsk[i]
            print('{} {} {}'.format(
                key, self.ps.rsk_id2frq[i], self.ps.rsk_rsk2dsp[key]))

        self.rsk_keys = tf.range(rsk_limit, dtype=tf.int32)

        self.dat_embd = tfu.Embedding(256, dat_embd_dim)
        self.dat_conv1d1024 = tfu.ConvMxPool1D(
            kernel_width=1024,
            filter_size=encoder_dim,
            stride=1024,
            max_pool=True,
            name='conv1024'
        )
        self.dat_conv1d512 = tfu.ConvMxPool1D(
            kernel_width=512,
            filter_size=encoder_dim,
            stride=512,
            max_pool=True,
            name='conv512'
        )
        self.dat_conv1d256 = tfu.ConvMxPool1D(
            kernel_width=256,
            filter_size=encoder_dim,
            stride=256,
            max_pool=True,
            name='conv256'
        )

        self.gen_attn = MultiHeadAttnLayer(
            attn_hidden,
            num_heads=1,
            name='attn_generation')

        self.ly_linear = tfu.Linear(encoder_dim)
        self.ly_pred = tfu.Linear(self.rsk_limit)

        self.optimizer = tf.train.AdamOptimizer(self.lr.value())

    def _encode(self, bat, use_pool=False):

        print('starting encoding')

        data = self.dat_embd(bat['byte'])

        d1024 = self.dat_conv1d1024(
            data
        )
        d512 = self.dat_conv1d512(
            data
        )
        d256 = self.dat_conv1d256(
            data
        )

        log_tensor(d1024)
        log_tensor(d512)
        log_tensor(d256)

        merged = tf.concat([
            tf.expand_dims(d1024, 1),
            tf.expand_dims(d512, 1),
            tf.expand_dims(d256, 1)
        ],
            axis=1)
        log_tensor(merged)
        # merged = encoded_data

        attned, prob = self.gen_attn.atnn(merged)
        return attned, prob
        # return tf.reduce_max(merged, axis=1), None
        # return encoded_data, None

    def _cal_loss_and_pred(self, bat, attned=None):
        if attned is None:
            attned, _ = self._encode(bat)
        log_tensor(attned)
        batch_size = tf.shape(attned)[0]
        z = self.ly_linear(attned)
        log_tensor(z)
        logits = self.ly_pred(z)
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

    def sampling(self, bat):
        attned, prob = self._encode(bat)
        confs, *_ = self._cal_loss_and_pred(bat, attned)
        log_tensor(prob)
        return (bat['rsk'][:, :self.rsk_limit],
                confs, prob, bat['sha256'], bat['blk_ord'])

    def sampling_visualization(self, data, ep, gp):
        with open(os.path.join(
                self.md_folder, 'sampling-gen-' + gp + '-' + str(ep) + '.txt'),
                'w', encoding='utf-8', errors='ignore') as outf:
            for batch in data:
                for (rsks, confs, probs, sha256, blk_ord) in zip(*batch):
                    for ind in range(rsks.shape[0]):
                        rsk = rsks[ind]
                        prob = probs[ind]
                        conf = confs[ind]
                        rsk_key = self.ps.rsk_id2rsk[ind]
                        desp = self.ps.rsk_rsk2dsp[rsk_key]
                        desp = "{}: {}".format(rsk_key, desp)
                        if rsk == 0:
                            continue
                        outf.write('## sha456 {}\n'.format(sha256[0]))
                        outf.write('   desp:  {}\n'.format(desp))
                        outf.write('   conf: {} val: {}\n'.format(conf, rsk))
                        self._write_attn_as_text(
                            sha256[0], prob, blk_ord, outf)

    def sampling_json(self, data, ep, gp, input_obj=None, save_to_file=True):
        objs = []
        for batch in data:
            obj = {}
            objs.append(obj)

            for (rsks, confs, probs, sha256, blk_ord) in zip(*batch):
                obj['sha256'] = sha256[0].decode()
                obj['rsk'] = []
                for ind in range(rsks.shape[0]):
                    rsk = rsks[ind]
                    prob = probs[ind]
                    conf = confs[ind]
                    rsk_key = self.ps.rsk_id2rsk[ind]
                    desp = self.ps.rsk_rsk2dsp[rsk_key]
                    attns = self._get_attn(
                        sha256[0], prob, blk_ord, obj=input_obj)
                    obj_rsk = {
                        'truth': rsk,
                        'conf': conf,
                        'rsk_key': rsk_key,
                        'rsk_desp': desp,
                        'attns': attns,
                    }
                    obj['rsk'].append(obj_rsk)
        if not save_to_file:
            return objs

        with open(os.path.join(
                self.md_folder,
                'sampling-gen-' + gp + '-' + str(ep) + '.json'),
                'w', encoding='utf-8', errors='ignore') as outf:
            tfu.json_dump({'reports': objs, 'ep': ep, 'gp': gp}, outf)
            return objs

    def _get_attn(self, sha256, prob, blk_ord, obj=None):
        obj = obj if obj is not None else self.obj_lookup_fn(sha256)
        eles, weights = self.ps.get_attn(obj, blk_ord, prob)
        attns = []
        for ele, w in zip(eles, weights):
            if isinstance(ele, dict) and 'name' in ele:
                attns.append((w, ele['name'], ele['ins']))
            else:
                attns.append((w, ele.replace('\n', '<lb/>')))
        return attns

    def _write_attn_as_text(self, sha256, prob, blk_ord, outf):
        attns = self._get_attn(sha256, prob, blk_ord)
        for a in attns:
            if len(a) == 3:
                outf.write("     {:.4f}, {}\n".format(a[0], a[1]))
                for i in a[2]:
                    outf.write("             {}\n".format(' '.join(i)))
            elif len(a) == 2:
                outf.write("     {:.4f}, {}\n".format(a[0], a[1]))
            else:
                log_warn("Unknow attention type {}", a)

    @staticmethod
    def set_flags(parser):
        parser = parser.add_argument_group('model configuraion')
        # parser.add_argument(
        #     "--encoder_dim",
        #     type=int,
        #     default=128,
        #     metavar="",
        #     help="Encoder hidden units.")
        # parser.add_argument(
        #     "--attn_hidden",
        #     type=int,
        #     default=128,
        #     metavar="",
        #     help="Hidden units in the attention layer.")
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
            "--topk",
            type=int,
            default=512,
            metavar="",
            help="Topk for the self-attiontion selection.")
        parser.add_argument(
            "--rsk_limit",
            type=int,
            default=150,
            metavar="",
            help="Rsks to be considered.")


if __name__ == '__main__':
    main(RavenVConv3)
