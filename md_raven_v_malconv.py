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


class ResetCell(rnn.LayerRNNCell):

    def __init__(self, inner_cell, reuse=None, name=None):
        super(ResetCell, self).__init__(_reuse=reuse, name=name)
        self.inner_cell = inner_cell
        tfu.track_dependent_layers(self)

    @property
    def state_size(self):
        return self.inner_cell.state_size

    @property
    def output_size(self):
        return self.inner_cell.output_size

    def compute_output_shape(self, input_shape):
        return self.output_size

    def zero_state(self, batch_size, dtype):
        return self.inner_cell.zero_state(batch_size, dtype)

    def call(self, inputs, *args):
        output, inner_s = self.inner_cell(inputs, args[0])
        rest = tf.cast(tf.expand_dims(tf.equal(inputs[:, 0], 0), -1),
                       tf.float32)
        next_s = nest.map_structure(lambda v: rest * v, inner_s)
        return output, next_s


class ExpandedQuasiResetableRNN(tf.keras.layers.Layer):

    def __init__(self, cell_dim, neighbours,
                 reset_at_zero,
                 name=None, **kwargs):
        super(ExpandedQuasiResetableRNN, self).__init__(
            name=name, **kwargs)
        self.cell_dim = cell_dim
        self.neighbours = neighbours
        self.reset_at_zero = reset_at_zero
        self.f_z = None
        self.f_f = None

    def build(self, input_shape):
        if self.f_z is None:
            self.f_z = self.add_variable(
                name=self.name+'_f_z',
                shape=[self.neighbours, 1, input_shape[-1].value,
                       self.cell_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )
        if self.f_f is None:
            self.f_f = self.add_variable(
                name=self.name+'_f_f',
                shape=[self.neighbours, 1, input_shape[-1].value,
                       self.cell_dim],
                initializer=tf.contrib.layers.xavier_initializer(),
                dtype=tf.float32
            )

    def call(self, inputs, **_):
        keep = tf.cast(tf.not_equal(
            inputs[:, :, 0], 0), tf.float32)

        inputs = tf.expand_dims(inputs, 2)
        # [batch, seq, 1, channel]
        zl = tf.squeeze(tf.nn.conv2d(
            inputs, self.f_z, [1, 1, 1, 1], 'SAME'
        ), 2)
        fl = tf.squeeze(tf.nn.conv2d(
            inputs, self.f_f, [1, 1, 1, 1], 'SAME'
        ), 2)

        h0 = tf.zeros(
            shape=[tfu.get_shape(inputs, 0), self.cell_dim],
            dtype=tf.float32)

        z = tf.tanh(zl)
        f = tf.sigmoid(fl)

        outputs_a = tf.TensorArray(
            dtype=tf.float32,
            size=tf.shape(keep)[1])
        batch_size = tfu.get_shape(inputs, 0)

        def lp_fn(t, ht_1, clt):
            m = tf.expand_dims(keep[:, t], -1)
            ft = f[:, t, :]
            zt = z[:, t, :]
            ht = (ft * ht_1 + (1 - ft) * zt) * m
            clt = clt.write(t, ht)
            return t+1, ht, clt

        *_, outputs_a = tf.while_loop(
            lambda t, *_: t < tf.shape(keep)[1],
            lp_fn,
            (0, h0, outputs_a))
        return tf.transpose(outputs_a.stack(), [1, 0, 2])


class AttnLayer(tf.keras.layers.Layer):

    def __init__(self, attn_size, num_heads=1,
                 top_k=1024, name=None, **kwargs):
        super(AttnLayer, self).__init__(name=name, **kwargs)
        self.attn_size = attn_size
        self.num_heads = num_heads
        self.lin_mem = None
        self.mem_length = None
        self.ws = None
        self.bs = None
        self.vs = None
        self.w = None
        self.b = None
        self.v = None
        self.top_k = top_k
        tfu.track_dependent_layers(self)

    def atnn(self, memory, dynamic=False):
        # memory [batch, time, depth]
        num_heads = self.num_heads
        in_dim = tfu.get_shape(memory, -1)
        attn_size = self.attn_size

        probs = None
        attns = None
        if not dynamic:
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
            if self.v is None:
                self.v = self.add_variable(
                    'v', [1, 1, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            if self.w is None:
                self.w = self.add_variable(
                    'w', [in_dim, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            if self.b is None:
                self.b = self.add_variable(
                    'b', [attn_size],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)
            log_tensor(memory)
            lin0 = (tfu.t_dot(memory, self.w) + self.b) * self.v
            log_tensor(lin0)
            if self.top_k > 0:
                probs = tf.nn.softmax(tf.reduce_sum(lin0, -1))
                topk = tf.minimum(self.top_k, tfu.get_shape(lin0, 1))
                _, indices = tf.nn.top_k(probs, k=topk, sorted=False)
                bat_ind = tf.reshape(
                    tf.range(tfu.get_shape(lin0, 0)), [-1, 1])
                bat_ind = tf.tile(bat_ind, [1, topk])
                indices = tf.concat(
                    [tf.expand_dims(bat_ind, -1),
                    tf.expand_dims(indices, -1)], -1)
                vals = tf.gather_nd(lin0, indices)
            else:
                vals = lin0
            log_tensor(vals)

            probs = []
            attns = []
            for hd in range(num_heads):
                lin1 = (tfu.t_dot(vals, self.ws[hd]) + self.bs[hd]) * self.vs[hd]
                prob = tf.nn.softmax(tf.reduce_sum(lin1, -1))
                attn = tf.reduce_sum(tf.expand_dims(prob, -1) * vals, 1)
                probs.append(prob)
                attns.append(attn)

            probs = tf.transpose(tf.stack(probs), [1, 0, 2])
            attns = tf.transpose(tf.stack(attns), [1, 0, 2])
        else:
            if self.vs is None:
                self.vs = self.add_variable(
                    'v', [num_heads, 1, 1, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            if self.ws is None:
                self.ws = self.add_variable(
                    'w', [num_heads, 1, in_dim, attn_size],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            if self.bs is None:
                self.bs = self.add_variable(
                    'b', [num_heads, 1, attn_size],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)
            probs = tf.TensorArray(dtype=tf.float32, size=num_heads)
            attns = tf.TensorArray(dtype=tf.float32, size=num_heads)

            def lp_fn(hd, l_probs: tf.TensorArray, l_attns: tf.TensorArray):
                lin = (memory @ self.ws[hd] + self.bs[hd]) * self.vs[hd]
                prob = tf.nn.softmax(tf.reduce_sum(lin, -1))
                attn = tf.reduce_sum(tf.expand_dims(prob, -1) * memory, 1)
                l_probs.write(hd, prob)
                l_attns.write(hd, attn)
                return hd+1, l_probs, l_attns

            _, probs, attns = tf.while_loop(
                cond=lambda hd, *_: hd < num_heads,
                body=lp_fn, loop_vars=(0, probs, attns))
            probs = tf.transpose(probs.stack(), [1, 0, 2])
            attns = tf.transpose(attns.stack(), [1, 0, 2])

        log_tensor(probs)
        log_tensor(attns)
        return attns, probs


AttnCellState = namedtuple(
    'AttnCellState',
    ['inner_state', 'alignment_history'])


class MemInputCell(rnn.LayerRNNCell):

    def __init__(self, inner_cell, reuse=None, name='mem_cell'):
        super(MemInputCell, self).__init__(_reuse=reuse, name=name)
        self.inner_cell = inner_cell
        tfu.track_dependent_layers(self)
        self.memory = None

    @property
    def state_size(self):
        return self.inner_cell.state_size

    @property
    def output_size(self):
        return self.inner_cell.output_size

    def compute_output_shape(self, input_shape):
        return self.output_size

    def zero_state(self, batch_size, dtype):
        return self.inner_cell.zero_state(batch_size, dtype)

    def set_memory(self, memory, condition=None, keep_attn_history=False):
        self.memory = memory

    def call(self, inputs, *args):
        inner_state = args[0]
        if self.memory is None:
            log_warn('Memory is not set for the mem-appended cell')
            return self.inner_cell(inputs, inner_state)
        return self.inner_cell(
            tf.concat([inputs, self.memory], axis=-1),
            inner_state)


class RavenVMalConv(tfu.ModelBase):

    def __init__(self, md_folder, preprocessor: dps.RavenPreprocessor,
                 encoder_dim=200,
                 md_suffix='',
                 attn_hidden=128,
                 kernel=500,
                 stride=500,
                 topk=1024,
                 learning_rate=0.001,
                 learning_rate_decay_factor=0.9,
                 overwrite_model=False,
                 learning_rate_decay_ep=-1,
                 rsk_limit=150,
                 use_pretain_asm_embd=True,
                 obj_lookup_fn: Callable=None,
                 *args, **kwargs):
        super(RavenVMalConv, self).__init__(
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

        if use_pretain_asm_embd:
            self.ps.asm_embd[0] = np.zeros([self.ps.asm_embd.shape[-1]])
            self.asm_embd = tf.constant(self.ps.asm_embd, name='asm_embd')
        else:
            log('Training new embedding layer')
            self.asm_embd = tfu.Embedding(
                len(self.ps.asm_id2tkn), 10, name='asm_embd')

        self.qrnn = ExpandedQuasiResetableRNN(
            encoder_dim, 1, True)
        self.seg_str_linear = tfu.Linear(encoder_dim)
        self.seg_imp_linear = tfu.Linear(encoder_dim)

        self.dat_embd = tfu.Embedding(256, 8)
        # self.dat_conv1d = tfu.ConvMxPool1D(
        #     kernel_width=kernel,
        #     filter_size=encoder_dim,
        #     stride=stride,
        #     max_pool=False
        # )
        self.dat_conv1d = tfu.GatedConvMxPool1D(
            kernel, encoder_dim, stride, max_pool=True)

        # self.gen_attn = AttnLayer(
        #     attn_hidden, 1,
        #     name='attn_generation', top_k=-1)

        # self.cate_linear0 = tfu.Linear(attn_hidden)
        # self.cate_linear1 = tfu.MultiLabelIndependentLinear(
        #     self.rsk_limit, True)
        
        self.ly_linear = tfu.Linear(128)
        self.ly_pred = tfu.Linear(self.rsk_limit)

        self.optimizer = tf.train.AdamOptimizer(self.lr.value())

    def _add_cell(self, cell_type, name, dim, depth=1, resetable=False):
        cells = []
        for i in range(depth):
            cname = "cell-{}-{}".format(name, i)
            cells.append(cell_type(dim, name=cname))
            setattr(self, cname, cells[-1])
        cell = rnn.MultiRNNCell(cells) if len(cells) > 1 else cells[0]
        return ResetCell(cell) if resetable else cell

    def _qrnn(self, bat):
        x = bat['asm_vec']
        log_tensor(x)

        last_dim = tfu.get_shape(x, -1)
        if not isinstance(last_dim, int):
            # they are ids. we need to lookup.
            if isinstance(self.asm_embd, tf.Tensor):
                x = tf.nn.embedding_lookup(self.asm_embd, x)
            else:
                x = self.asm_embd(x)
            x = tf.reduce_sum(x, -2)
            x = tf.stop_gradient(x)
            log_tensor(x)

        batch_size, trunk_size, seq_max = tfu.get_shape(
            x, 0), tfu.get_shape(x, 1), tfu.get_shape(x, 2)
        rep_dim = x.shape[3].value
        x = tf.reshape(x, [batch_size * trunk_size, seq_max, rep_dim])
        xl = tf.reshape(bat['asm_vec_len'], [batch_size * trunk_size])
        log_tensor(x)

        outputs = self.qrnn(x)

        outputs = tf.reshape(
            outputs,
            [batch_size, trunk_size, seq_max, outputs.shape[-1].value])
        log_tensor(outputs)

        bat_ind = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
        trk_ind = tf.reshape(tf.range(trunk_size), [1, trunk_size, 1])
        inds_size = tf.shape(bat['inds'])[-1]
        gat_ind = tf.concat([
            tf.expand_dims(tf.tile(bat_ind, [1, trunk_size, inds_size]), - 1),
            tf.expand_dims(tf.tile(trk_ind, [batch_size, 1, inds_size]), - 1),
            tf.expand_dims(bat['inds'], -1)
        ], -1)
        masked = tf.expand_dims(
            tf.sequence_mask(bat['inds_len'], dtype=tf.float32),
            -1) * tf.gather_nd(outputs, gat_ind)
        masked = tf.reshape(
            masked,
            [batch_size, trunk_size * inds_size, outputs.shape[-1].value])
        bat['blk_ord'] = tf.reshape(
            bat['blk_ord'], [batch_size, trunk_size * inds_size]
        )
        return masked

    def _encode(self, bat, use_pool=False):

        print('starting encoding')

        # masked = self._qrnn(bat)
        # bat['imp'] = tf.Print(bat['imp'], [tf.reduce_any(tf.is_nan(bat['imp']))])
        encoded_data = self.dat_conv1d(
            self.dat_embd(bat['byte'])
            )
        
        # encoded_str = self.seg_str_linear(bat['str'])
        # encoded_imp = self.seg_imp_linear(bat['imp'])

        # merged = tf.concat([
        #     # masked,
        #     encoded_str,
        #     encoded_imp,
        #     encoded_data
        # ],
        #     axis=1)
        # log_tensor(merged)

        # attned, prob = self.gen_attn.atnn(merged)
        # return attned, prob
        # return tf.reduce_max(merged, axis=1), None
        return encoded_data, None

    def _cal_loss_and_pred(self, bat, attned=None):
        if attned is None:
            attned, _ = self._encode(bat)
        log_tensor(attned)
        batch_size = tf.shape(attned)[0]
        # attned = tf.reshape(attned, [-1, tfu.get_shape(attned, -1)])
        z = self.ly_linear(attned)
        # z = self.cate_linear0(attned)
        log_tensor(z)
        # logits = self.cate_linear1(z, batch_size=batch_size)
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
    main(RavenVMalConv)