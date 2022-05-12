import tensorflow.contrib.eager as tfe
from tensorflow.python.client import device_lib
import tensorflow as tf
import shutil
import os
import json
from time import time
from tensorflow.python.util import nest
import inspect
import numpy as np
from tensorflow.python.client import timeline
from contextlib import contextmanager
from colorama import Fore, init
from datetime import datetime
import pickle
from sklearn.metrics import roc_curve

init(autoreset=True)

LOG_VERBOSITY = 1


class ModelBase(tf.keras.Model):

    def __init__(self, md_folder, md_suffix='',
                 learning_rate=0.001, decay_factor=0.9,
                 learning_rate_decay_ep=-1, overwrite_model=False,
                 tag_prefix=None):
        super(ModelBase, self).__init__()
        md_folder = os.path.join(md_folder, self.__class__.__name__)
        if len(md_suffix.strip()) > 0:
         md_folder = md_folder + '_' + md_suffix
        self.md_folder = md_folder
        if overwrite_model and os.path.exists(self.md_folder):
            shutil.rmtree(self.md_folder)
            os.mkdir(self.md_folder)

        self.lr = LearningRateDecay(learning_rate, decay_factor)
        self.lr_decay_ep = learning_rate_decay_ep

        self.summary_trn_dir = os.path.join(md_folder, 'summary-train/')
        self.summary_vld_dir = os.path.join(md_folder, 'summary-validation/')
        self.summary_tst_dir = os.path.join(md_folder, 'summary-test/')
        self.optimizer = None

        self.step_counter = tf.train.get_or_create_global_step()

        if tf.executing_eagerly():
            self.steps_per_epoch = tfe.Variable(
                initial_value=0, trainable=False, dtype=tf.int64,
                name='steps_per_epoch')

            def __set_step_fn():
                self.steps_per_epoch = tf.assign(
                    self.steps_per_epoch, self.step_counter)

            self._set_steps_per_epoch_by_global_step = __set_step_fn
        else:
            self.steps_per_epoch = tf.Variable(
                initial_value=0, trainable=False, expected_shape=(),
                dtype=tf.int64, name='steps_per_epoch')
            self._set_steps_per_epoch_by_global_step = tf.assign(
                self.steps_per_epoch, self.step_counter)
        self.checkpoint = None
        self._restored = False
        self.summary_trn = TFSummary(
            self.summary_trn_dir, tag_prefix)
        self.summary_vld = TFSummary(
            self.summary_vld_dir, tag_prefix)
        self.summary_tst = TFSummary(
            self.summary_tst_dir, tag_prefix)

    def _get_check_point_(self):
        if self.checkpoint is None:
            self.checkpoint = tfe.Checkpoint(
                model=self, optimizer=self.optimizer,
                step_counter=self.step_counter)
        return self.checkpoint

    def restore(self):
        self._do_restore()

    def _do_restore(self):
        latest = tf.train.latest_checkpoint(self.md_folder)
        chkpt = self._get_check_point_().restore(latest)
        if latest is not None:
            log('Restoring from', latest)
            chkpt.run_restore_ops()
            self._restored = True

    def _save(self, dry_loop=False):
        log('Creating a new check points..')
        if not dry_loop:
            checkpoint_prefix = os.path.join(self.md_folder, 'ckpt')
            if not os.path.exists(checkpoint_prefix):
                os.makedirs(checkpoint_prefix)
            self._get_check_point_().save(checkpoint_prefix)

    def train_basic_graph(self, trn, vld=None, tst=None, epoch=10,
                          sampling_epoch_interval=1, sampling_steps=5,
                          log_step_interval=20, gradient_clip=5,
                          profile_steps=None,
                          dry_loop=False, loss_fn=None, loss_optm_name=None,
                          sampling_fn=None, sampling_v_fn=None,
                          save_roc=False,
                          eval_only=False,
                          **kwargs):
        if profile_steps and profile_steps > 0:
            log("Start profiling the graph model using {} mini-batches...",
                profile_steps)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, trn.output_types, trn.output_shapes)
        ele = iterator.get_next()
        log('batch_shape:', ele)
        shapes = {e: tf.shape(ele[e]) for e in ele}
        lsv, all_metrics = loss_fn(ele)
        grads = self.optimizer.compute_gradients(lsv)
        if gradient_clip and gradient_clip > 0:
            gradients, variables = zip(*grads)
            clipped, _ = tf.clip_by_global_norm(
                gradients, gradient_clip)
            grads = zip(clipped, variables)
        train_opr = self.optimizer.apply_gradients(
            grads, global_step=self.step_counter)
        sample_opr = sampling_fn(ele) \
            if sampling_steps and sampling_fn is not None else None
        log('local variables:', tf.local_variables())

        last_best_valiation = 1e10
        last_best_ep = -1

        log('start training ....')
        sess = tf.get_default_session()
        self.restore()
        if not self._restored:  # finish building graphs
            tf.global_variables_initializer().run()
        if not sess.run(tf.is_variable_initialized(self.lr.value())):
            sess.run(self.lr.value().initializer)
        if LOG_VERBOSITY > 1:
            profile_model_vars(sess)
        for ep in range(epoch):
            if (sample_opr and
                    sampling_epoch_interval and
                    ep % sampling_epoch_interval == 0):
                log('Validation set sampling ....')
                iterator = vld.make_one_shot_iterator()
                handle_v = sess.run(iterator.string_handle())
                res = [sess.run(sample_opr, feed_dict={handle: handle_v})
                       for _ in range(sampling_steps)]
                sampling_v_fn(res, sess.run(self.checkpoint.save_counter),
                              'validation')
                log('Training set sampling ....')
                iterator = trn.make_one_shot_iterator()
                handle_v = sess.run(iterator.string_handle())
                res = [sess.run(sample_opr, feed_dict={handle: handle_v})
                       for _ in range(sampling_steps)]
                sampling_v_fn(res, sess.run(self.checkpoint.save_counter),
                              'training')
            if save_roc:
                roc_path_vld = os.path.join(
                    self.md_folder, 'roc_vld.csv')
                iterator = vld.make_one_shot_iterator()
                handle_v = sess.run(iterator.string_handle())
                res = [sess.run(sample_opr, feed_dict={handle: handle_v})
                       for _ in range(sampling_steps)]
                generate_roc(res, roc_path_vld)
                roc_path_tst = os.path.join(
                    self.md_folder, 'roc_tst.csv')
                iterator = tst.make_one_shot_iterator()
                handle_v = sess.run(iterator.string_handle())
                res = [sess.run(sample_opr, feed_dict={handle: handle_v})
                       for _ in range(sampling_steps)]
                generate_roc(res, roc_path_tst)
                return 
            if not eval_only:
                with self.summary_trn.writer() as writer:
                    self._cal_loss_all_graph(
                        trn, 'trn_ds', sess, shapes, handle, train_opr,
                        all_metrics,
                        writer=writer,
                        log_interval=log_step_interval,
                        profile_steps=profile_steps,
                        dry_loop=dry_loop)
            if not eval_only and vld and (not profile_steps or profile_steps < 1):
                with self.summary_vld.writer() as writer:
                    mets = self._cal_loss_all_graph(
                        vld, 'vld_ds', sess, shapes, handle, None,
                        all_metrics,
                        writer=writer,
                        log_interval=log_step_interval, dry_loop=dry_loop)
                    if self.lr_decay_ep > 0:
                        log("validation best {} vs current {}",
                            last_best_valiation, mets[loss_optm_name])
                        if mets[loss_optm_name] <= last_best_valiation:
                            last_best_valiation = mets[loss_optm_name]
                            last_best_ep = ep
                        else:
                            if ep - last_best_ep >= self.lr_decay_ep:
                                self.lr.decay(sess)
            if tst and (not profile_steps or profile_steps < 1):
                with self.summary_tst.writer() as writer:
                    self._cal_loss_all_graph(
                        tst, 'tst_ds', sess, shapes, handle, None,
                        all_metrics,
                        writer=None if eval_only else writer,
                        # eval_only only shows the result
                        log_interval=log_step_interval, dry_loop=dry_loop)

            if sess.run(self.checkpoint.save_counter) == 0:
                sess.run(self._set_steps_per_epoch_by_global_step)
            self._save(dry_loop)

    def _cal_loss_all_graph(self, ds, ds_name, sess, shapes, ds_handle,
                            trn_op,
                            all_metrics, writer=None, log_interval=-1,
                            profile_steps=None, dry_loop=False):
        tf.local_variables_initializer().run()
        iterator = ds.make_one_shot_iterator()
        handle_v = sess.run(iterator.string_handle())
        method = 'Train' if trn_op is not None else 'Test'
        start_e = time()
        run_metadata = None
        if profile_steps and profile_steps > 0:
            run_metadata = tf.RunMetadata()
            run_option = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE,
                report_tensor_allocations_upon_oom=True)
        else:
            run_option = tf.RunOptions()
            # report_tensor_allocations_upon_oom=True)

        ind = 0
        covered = 0
        start_b = time()
        step = None
        metric_vals = None
        try:
            while True:
                if profile_steps and ind > profile_steps > 0:
                    break
                if dry_loop:
                    metric_vals = {'metric_skip': 0}
                    sp, step, lr, ep = sess.run(
                        [shapes, self.step_counter, self.lr.value(),
                         self.checkpoint.save_counter],
                        feed_dict={ds_handle: handle_v},
                        run_metadata=run_metadata, options=run_option)
                elif trn_op is not None:
                    metric_vals, sp, _, step, lr, ep = sess.run(
                        [all_metrics, shapes, trn_op, self.step_counter,
                         self.lr.value(), self.checkpoint.save_counter],
                        feed_dict={ds_handle: handle_v},
                        run_metadata=run_metadata,
                        options=run_option)
                    if writer:
                        writer.log_scalar('training/lr', lr, step)
                        writer.log_scalar('training/ep', ep, step)
                        for met in metric_vals:
                            writer.log_scalar('training/step-' + met,
                                              metric_vals[met], step)
                else:
                    metric_vals, sp, step, lr = sess.run(
                        [all_metrics, shapes, self.step_counter,
                         self.lr.value()],
                        feed_dict={ds_handle: handle_v},
                        run_metadata=run_metadata,
                        options=run_option)
                ind += 1
                covered += next(iter(sp.values()))[0]
                sp = [(k, sp[k].tolist()) for k in sp]
                sp.sort(key=lambda b: np.prod(b[1]), reverse=True)
                if log_interval > 0 and ind % log_interval == 0:
                    rate = covered / (time() - start_b)
                    mets = format_metric(metric_vals)
                    log(
                        '{}ing #{} {}/{} {} ({:.2f} sams/sec'
                        ') lr {:.5f} largest: {}',
                        method, step, covered, sess.run(self.steps_per_epoch),
                        mets, rate, lr, sp[0])
        except tf.errors.OutOfRangeError:
            mets = format_metric(metric_vals)
            log('----> {}ing time for {} ep #{} ({} steps): {:.2f} \t'
                '{}',
                method,
                ds_name,
                sess.run(self.checkpoint.save_counter) + 1,
                sess.run(self.step_counter),
                time() - start_e,
                mets)

        if writer and step:
            for met in metric_vals:
                writer.log_scalar('evaluation/' + met, metric_vals[met],
                                  sess.run(self.step_counter))
        if profile_steps and profile_steps > 0:
            log('Profiling memory..')
            profile_memory(run_metadata, sess)
            log('Profiling model..')
            profile_model(run_metadata, sess)
            log('Profiling as timelines..')
            profile_timeline(os.path.join(self.md_folder, 'profiler'),
                             run_metadata, sess)

        return metric_vals

    def get_serving_fn_graph(
            self, sampling_fn, placeholders, single_element=True, **kwargs):
        input_tensors = {}
        if single_element:
            for k in placeholders:
                input_tensors[k] = tf.expand_dims(placeholders[k], 0)
        else:
            input_tensors = placeholders
        sample_opr = sampling_fn(input_tensors)
        sess = tf.get_default_session()
        self.restore()
        if not self._restored:  # finish building graphs
            log_err(
                'The model is not trained yet. Cannot run inference. Returning NONE')
            return None

        def run(input):
            input = {placeholders[k]: pad_to_dense(input[k]) for k in placeholders}
            sess.run(sample_opr, feed_dict=input)

        return run
    

    def get_serving_fn_graph_tfr(
            self, sampling_fn, 
            ds: tf.data.Dataset, 
            tfrs_holder: tf.Tensor):
        ite = ds.make_initializable_iterator()
        ele = ite.get_next()
        infer_opr = sampling_fn(ele) 
        sess = tf.get_default_session()
        self.restore()
        if not self._restored:  # finish building graphs
            log_err(
                'The model is not trained yet. Cannot run inference. Returning NONE')
            return None

        def run(tfrs):
            if not isinstance(tfrs, list):
                tfrs = [tfrs]
            tfrs = [tfr.SerializeToString() for tfr in tfrs]
            sess.run(ite.initializer, feed_dict={tfrs_holder: tfrs})
            res = []
            log('start inferring...')
            for _ in range(len(tfrs)):
                res.append(sess.run(infer_opr))
            log('done')
            return res

        return run

    def train_basic_eager(self, trn, vld=None, tst=None, epoch=10,
                          sampling_epoch_interval=1, sampling_steps=5,
                          log_step_interval=20, gradient_clip=5,
                          dry_loop=False,
                          loss_fn=None, loss_optm_name=None, sampling_fn=None,
                          sampling_v_fn=None,
                          **kwargs):
        log('start training ....')

        last_best_valiation = 1e10
        last_best_ep = -1

        for ep in range(epoch):
            if sampling_fn is not None and sampling_epoch_interval \
                    and ep % sampling_epoch_interval == 0:
                res = [sampling_fn(bat) for _, bat
                       in zip(range(sampling_steps), tfe.Iterator(vld))]
                res = nest.map_structure(
                    lambda x: x.numpy() if tf.contrib.framework.is_tensor(
                        x) and tf.executing_eagerly() else x,
                    res)
                sampling_v_fn(
                    res, self.checkpoint.save_counter.numpy(),
                    'validation')
            with self.summary_trn.writer() as writer:
                self._cal_loss_all_eager(
                    trn, 'trn_ds', loss_fn,
                    writer=writer,
                    gradient_clip=gradient_clip,
                    train=True,
                    log_interval=log_step_interval,
                    dry_loop=dry_loop)
            if vld:
                with self.summary_vld.writer() as writer:
                    mets = self._cal_loss_all_eager(
                        vld, 'vld_ds', loss_fn,
                        writer=writer,
                        gradient_clip=gradient_clip,
                        train=False,
                        log_interval=log_step_interval,
                        dry_loop=dry_loop)
                if self.lr_decay_ep > 0:
                    log("validation best {} vs current {}",
                        last_best_valiation, mets[loss_optm_name])
                    if mets[loss_optm_name].numpy() <= last_best_valiation:
                        last_best_valiation = mets[loss_optm_name].numpy()
                        last_best_ep = ep
                    else:
                        if ep - last_best_ep >= self.lr_decay_ep:
                            self.lr.decay()
            if tst:
                with self.summary_tst.writer() as writer:
                    self._cal_loss_all_eager(
                        tst, 'tst_ds', loss_fn,
                        writer=writer,
                        gradient_clip=gradient_clip,
                        train=False,
                        log_interval=log_step_interval,
                        dry_loop=dry_loop)

            if self.checkpoint.save_counter.numpy() == 0:
                self._set_steps_per_epoch_by_global_step()
            self._save(dry_loop)

    def _cal_loss_all_eager(self, ds, ds_name, loss_func, gradient_clip=5,
                            writer=None,
                            train=False, log_interval=-1, dry_loop=False):

        method = 'Train' if train else 'Test'
        start_e = time()
        covered = 0
        metric_vals = {}

        def exec_fun(data):
            ls_objective, metrics = loss_func(data)
            for k in metrics:
                if k not in metric_vals:
                    metric_vals[k] = tfe.metrics.Mean(k)
                metric_vals[k](metrics[k])
            return ls_objective

        for ind, bat in enumerate(tfe.Iterator(ds)):
            if dry_loop:
                metric_vals['Metric skip'] = tfe.metrics.Mean('Metric skip')
            elif train:
                _, grads = tfe.implicit_value_and_gradients(
                    exec_fun)(bat)
                if gradient_clip and gradient_clip > 0:
                    gradients, variables = zip(*grads)
                    clipped, _ = tf.clip_by_global_norm(
                        gradients, gradient_clip)
                    grads = zip(clipped, variables)
                self.optimizer.apply_gradients(
                    grads, global_step=self.step_counter)
                if writer:
                    for met in metric_vals:
                        writer.log_scalar('training/step-' + met,
                                          metric_vals[met].result(),
                                          self.step_counter)
                    writer.log_scalar('training/lr', self.lr.value().numpy(),
                                      self.step_counter)
                    writer.log_scalar('training/ep',
                                      self.checkpoint.save_counter.numpy(),
                                      self.step_counter)
            else:
                exec_fun(bat)
            shapes = [(k, tf.shape(bat[k]).numpy().tolist()) for k in bat]
            shapes.sort(
                key=lambda b: np.prod(b[1]), reverse=True)
            covered += shapes[0][1][0]
            if log_interval > 0 and ind % log_interval == 0:
                mets = format_metric(metric_vals)
                rate = covered / (time() - start_e)
                log(
                    '{}ing #{} {}/{} {} ({:.0f} sams/sec) '
                    'lr {:.5f} largest: {}',
                    method, self.step_counter.numpy(), covered,
                    self.steps_per_epoch.numpy(),
                    mets, rate, self.lr.value().numpy(), shapes[0])
        mets = format_metric(metric_vals)
        log('----> {}ing time for {} ep #{} ({} steps): {:.2f} \t'
            '{}',
            method,
            ds_name,
            self.checkpoint.save_counter.numpy() + 1,
            self.step_counter.numpy(),
            time() - start_e,
            mets)
        if writer:
            for met in metric_vals:
                writer.log_scalar('evaluation/' + met,
                                  metric_vals[met].result(),
                                  self.step_counter)
        return metric_vals

    def add_variable(self, name, shape, dtype=None, initializer=None,
                     regularizer=None, trainable=True, constraint=None):
        raise Exception("Not supported.")


def tile_batch(t, multiplier):
    return nest.map_structure(lambda x: _tile_batch(x, multiplier), t)


def _tile_batch(t, multiplier):
    """Updated version to accommodate eager mode"""
    if type(t) is tf.TensorArray:
        return t
    t = tf.convert_to_tensor(t, name="t")
    shape_t = tf.shape(t)
    if t.shape.ndims is None or t.shape.ndims < 1:
        raise ValueError("t must have statically known rank")
    tiling = [1] * (t.shape.ndims + 1)
    tiling[1] = multiplier
    tiled = tf.tile(tf.expand_dims(t, 1), tiling)
    tiled = tf.reshape(tiled,
                       tf.concat(
                           ([shape_t[0] * multiplier], shape_t[1:]), 0))
    if not tf.executing_eagerly() and isinstance(multiplier, int):
        tiled_static_batch_size = get_shape(t, 0)
        if isinstance(tiled_static_batch_size, int):
            tiled_static_batch_size *= multiplier
            tiled.set_shape(
                tf.TensorShape([tiled_static_batch_size]).concatenate(
                    t.shape[1:]))
    return tiled


def generate_roc(res, path):
    truths = np.reshape(np.stack(res[0]), -1)
    confs = np.reshape(np.stack(res[1]), -1)
    fpr, tpr, thresholds = roc_curve(
        truths, confs)
    with open(path, 'r') as of:
        for f, t, r in zip(fpr, tpr, thresholds):
            of.wirte('{},{},{}\n'.format(
                f, t, r
                ))



def sequence_loss(logits: tf.Tensor, labels: tf.Tensor,
                  seq_len, avg_across_time=False):
    batch_dim = tf.shape(logits)[0]
    seq_dim = tf.shape(logits)[1]
    cls_dim = logits.shape[-1].value
    logits = tf.reshape(logits, [-1, cls_dim])
    labels = tf.reshape(labels, [-1])
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    cost = tf.reshape(cost, [batch_dim, seq_dim])
    log_tensor(cost)
    mask = tf.sequence_mask(
        seq_len,
        dtype=tf.float32,
        maxlen=seq_dim)
    cost = cost * mask
    # Warning: may cause NaN or Inf
    cost = tf.reduce_sum(cost, axis=1)
    if avg_across_time:
        cost = cost / tf.reduce_sum(mask, axis=1)
    cost = tf.reduce_mean(cost, axis=0)
    return cost


def classification_metrics(pred_label, label, name='cls'):
    """
    warning: we only use the update opr. so make sure the returned 
    dict only run ONCE per-batch! At the end of epoch we just show
    the result from the last updat opr (last batch). 
    """
    _, acc = tf.metrics.accuracy(
        labels=label, predictions=pred_label)
    _, prc = tf.metrics.precision(
        labels=label, predictions=pred_label)
    _, rcl = tf.metrics.recall(
        labels=label, predictions=pred_label)
    _, fpc = tf.metrics.false_positives(
        labels=label, predictions=pred_label)
    _, tnc = tf.metrics.true_negatives(
        labels=label, predictions=pred_label)
    _, fnc = tf.metrics.false_negatives(
        labels=label, predictions=pred_label)
    _, tpc = tf.metrics.true_positives(
        labels=label, predictions=pred_label)
    fpr = fpc / (fpc + tnc)
    fnr = fnc / (fnc + tpc)

    return {name+'_acc': acc, name+'_prc': prc,
            name+'_rcl': rcl, name+'_fpr': fpr,
            name+'_fnr':fnr}


def classification_binary_metrics(
        pred_prob, label, threshold=0.5, name='conf'):
    """
    warning: we only use the update opr. so make sure the returned 
    dict only run ONCE per-batch! At the end of epoch we just show
    the result from the last updat opr (last batch). 
    """
    _, auc = tf.metrics.auc(
        labels=label, predictions=pred_prob)
    _, brf = tf.metrics.mean_squared_error(
        labels=label, predictions=pred_prob)
    pred_lbl = tf.greater(pred_prob, threshold)
    lbl_metrics = classification_metrics(pred_lbl, label, name)
    return {**lbl_metrics, name+'_auc': auc, name+'_brf': brf}


def sequence_metrics(
        pred_ids, truth_ids, pred_len, truth_len, name='seq_'):
    def mp_reshape(ele):
        if len(ele.shape) > 2:
            return tf.reshape(ele, [-1, tf.shape(ele)[-1]])
        return ele
    p = mp_reshape(pred_ids)
    t = mp_reshape(truth_ids)
    p_l = tf.reshape(pred_len, [-1])
    t_l = tf.reshape(truth_len, [-1])
    m_l = tf.maximum(p_l, t_l)
    m, p, t = pad_to_compatible(p, t)
    mask = tf.sequence_mask(
        lengths=m_l,
        maxlen=m)
    mask = tf.reshape(mask, [-1])
    merged = tf.concat(
        [tf.reshape(p, [-1, 1]), tf.reshape(t, [-1, 1])],
        axis=1)
    vals = tf.boolean_mask(
        merged, mask)
    return classification_metrics(vals[:, 0], vals[:, 1], name=name)


def pad_to(a, b):
    diff = tf.maximum(tf.shape(b)[-1] - tf.shape(a)[-1], 0)
    return tf.pad(a, [[0, 0], [0, diff]])


def pad_to_compatible(a, b):
    m = tf.maximum(tf.shape(a)[-1], tf.shape(b)[-1])
    a = pad_to(a, b)
    b = pad_to(b, a)
    return m, a, b


def get_shape(t, dim):
    if t.shape[dim].value is not None:
        return t.shape[dim].value
    return tf.shape(t)[dim]


def t_dot(a, b):
    shape = a.get_shape().as_list()
    return tf.tensordot(a, b, [[len(shape) - 1], [0]]) \
    if len(shape) > 2 else tf.matmul(a, b)


def pad_to_dense2d(M):
    if isinstance(M, list) and len(M)>0 and isinstance(M[0], list):
        maxlen = max(len(r) for r in M)
        Z = np.zeros((len(M), maxlen))
        for enu, row in enumerate(M):
            Z[enu, :len(row)] += row 
        return Z
    return M


def log_tensor(*ts):
    if LOG_VERBOSITY > 0:
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        args = string[string.find('(') + 1:-1].split(',')

        names = []
        for i in args:
            if i.find('=') != -1:
                names.append(i.split('=')[1].strip())

            else:
                names.append(i)
        for t, n in zip(ts, names):
            log('T:{} S:{}', n, tf.shape(t) if tf.executing_eagerly() else t)


def time_str():
    return datetime.now().strftime('%m-%d-%H-%M-%S')


def log(msg, *params):
    if not isinstance(msg, str):
        msg = str(msg)
    if '{' not in msg:
        msg = ' '.join([msg] + [str(p) for p in params])
    else:
        msg = msg.format(*params)
    print("{} {} INF {} {}".format(Fore.LIGHTCYAN_EX, time_str(),
                                   Fore.LIGHTGREEN_EX, msg))


def log_err(msg, *params):
    if not isinstance(msg, str):
        msg = str(msg)
    if '{' not in msg:
        msg = ' '.join([msg] + [str(p) for p in params])
    else:
        msg = msg.format(*params)
    print("{} {} ERR {} {}".format(Fore.LIGHTRED_EX, time_str(),
                                   Fore.LIGHTRED_EX, msg))


def log_warn(msg, *params):
    if not isinstance(msg, str):
        msg = str(msg)
    if '{' not in msg:
        msg = ' '.join([msg] + [str(p) for p in params])
    else:
        msg = msg.format(*params)
    print("{} {} WRN {} {}".format(Fore.LIGHTCYAN_EX, time_str(),
                                   Fore.LIGHTYELLOW_EX, msg))


def log_highlight(msg, *params):
    if not isinstance(msg, str):
        msg = str(msg)
    if '{' not in msg:
        msg = ' '.join([msg] + [str(p) for p in params])
    else:
        msg = msg.format(*params)
    print("{} {} HIG {} {}".format(Fore.LIGHTWHITE_EX, time_str(),
                                   Fore.LIGHTWHITE_EX, msg))


def format_metric(metric_vals):
    if tf.executing_eagerly():
        return ' '.join(
            ['{} {}:{} {:.4f}'.format(Fore.LIGHTYELLOW_EX, met, Fore.GREEN,
                                      metric_vals[met].result())
             for met in metric_vals])
    else:
        return ' '.join(
            ['{} {}:{} {:.4f} '.format(Fore.LIGHTYELLOW_EX, met, Fore.GREEN,
                                       metric_vals[met])
             for met in metric_vals])


class TFSummary(object):

    def __init__(self, log_dir, tag_prefix=None):
        self.tag_prefix = tag_prefix
        self.log_dir = log_dir
        if tf.executing_eagerly():
            self._writer = tf.contrib.summary.create_file_writer(
                log_dir, flush_millis=10000)
        else:
            self._writer = tf.summary.FileWriter(log_dir)

    @contextmanager
    def writer(self):
        if tf.executing_eagerly():
            with self._writer.as_default():
                with tf.contrib.summary.always_record_summaries():
                    yield self
        else:
            yield self

    def log_scalar(self, tag, value, step):
        if self.tag_prefix is not None:
            tag = self.tag_prefix + tag
        if tf.executing_eagerly():
            tf.contrib.summary.scalar(tag, value, step=step)
        else:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self._writer.add_summary(summary, step)

    def log_htmls(self, tag, step, a_generator):
        if self.tag_prefix is not None:
            tag = self.tag_prefix + tag
        html_folder = os.path.join(self.log_dir, 'docs')
        if not os.path.exists(html_folder):
            os.makedirs(html_folder)
        file = os.path.join(html_folder, tag + '-' + step + '.html')
        body = ['<div>\n' + obj + '\n</div>' for obj in a_generator]
        content = '<!DOCTYPE html>\
               <html lang="en">\
               <head>\
               <meta charset="utf-8">\
               <title>title</title>\
               <script src="script.js"></script>\
               </head>\
               <body>\
               {}\
               </body>\
               </html>'.format(body)
        with open(file, 'w', encoding='utf8') as of:
            of.write(content)


def get_available_gpus():
    # from mrry
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def profile_timeline(folder, run_metadata, sess):
    if not os.path.exists(folder):
        os.makedirs(folder)
    # ecluded `graph`. too slow.
    for cmd in ['scope', 'code', 'graph']:
        builder = tf.profiler.ProfileOptionBuilder
        opts = builder(builder.time_and_memory()).with_timeline_output(
            os.path.join(
                folder, 'testing_profiler_' + cmd + '.json')).build()

        tf.profiler.profile(
            sess.graph,
            run_meta=run_metadata,
            options=opts,
            cmd=cmd
        )
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(os.path.join(
            folder, 'testing_profiler_step_stats.json'), 'w') as f:
        f.write(chrome_trace)


def profile_memory(run_metadata, sess):
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory(min_bytes=100 * 1024)) \
        .select(['bytes']) \
        .with_node_names(show_name_regexes=['.*']) \
        .order_by('bytes').build()
    tf.profiler.profile(
        sess.graph,
        run_meta=run_metadata,
        options=opts,
        cmd='scope'
    )


def profile_model_vars(sess):
    builder = tf.profiler.ProfileOptionBuilder
    tf.profiler.profile(
        sess.graph,
        options=builder.trainable_variables_parameter())


def profile_model(run_metadata, sess):
    profile_model_vars(sess)
    tf.profiler.advise(sess.graph, run_metadata)


def track_dependent_layers(obj: tf.keras.layers.Layer):
    mp = obj.__dict__
    tps = [(k, mp[k]) for k in mp]
    for k, v in tps:
        if isinstance(v, tf.keras.layers.Layer):
            # noinspection PyProtectedMember
            obj._track_checkpointable(v, v.name)


class ConvMxPool1D(tf.keras.layers.Layer):

    def __init__(self, kernel_width, filter_size, stride, name=None,
                 max_pool=True, **kwargs):
        super(ConvMxPool1D, self).__init__(name=name, **kwargs)
        self.kernel_width = kernel_width
        self.filter_size = filter_size
        self.stride = stride
        self.filters = None
        self.max_pool = max_pool

    def build(self, input_shape):
        self.filters = self.add_variable(
            name=self.name + '-filters',
            shape=[self.kernel_width, 1, input_shape[-1].value,
                   self.filter_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

    def call(self, inputs, **_):
        inputs = tf.expand_dims(inputs, 2)
        # [batch, seq, 1, channel]
        convd = tf.nn.conv2d(inputs, self.filters, [1, self.stride, 1, 1],
                             'SAME')
        outputs = tf.nn.relu(convd)
        outputs = tf.squeeze(outputs, 2)
        if self.max_pool:
            outputs = tf.reduce_max(outputs, axis=1)
        return outputs


class GatedConvMxPool1D(tf.keras.layers.Layer):

    def __init__(self, kernel_width, filter_size, stride, name=None, max_pool=True,
                 **kwargs):
        super(GatedConvMxPool1D, self).__init__(name=name, **kwargs)
        self.kernel_width = kernel_width
        if filter_size % 2 != 0:
            log_err(
                'ERROR. Filter size has to be an even number but {} given.',
                filter_size)
        self.filter_size = filter_size
        self.stride = stride
        self.filters = None
        self.max_pool = max_pool

    def build(self, input_shape):
        self.filters = self.add_variable(
            name=self.name + '-filters',
            shape=[self.kernel_width, 1, input_shape[-1].value,
                   self.filter_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            dtype=tf.float32
        )

    def call(self, inputs, **_):
        inputs = tf.expand_dims(inputs, 2)
        # [batch, seq, 1, channel]
        convd = tf.nn.conv2d(inputs, self.filters, [1, self.stride, 1, 1],
                             'SAME')
        # split to two parts
        a, b = tf.split(axis=-1, value=convd, num_or_size_splits=2)
        gate = tf.sigmoid(a)
        gated = gate * b
        log_tensor(gated)
        outputs = tf.squeeze(gated, 2)
        if self.max_pool:
            outputs = tf.reduce_max(outputs, axis=1)
        return outputs


class Embedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size=None, embedding_dim=None,
                 initial_values=None, trainable=True, name=None, **kwargs):
        super(Embedding, self).__init__(name=name, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = None
        self.initial_values = initial_values
        self.trainable = trainable

    def build(self, _):
        if self.initial_values is None:
            self.embedding = self.add_variable(
                self.name + "-embedding_kernel",
                shape=[self.vocab_size, self.embedding_dim],
                dtype=tf.float32,
                initializer=tf.random_uniform_initializer(-0.1, 0.1),
                trainable=self.trainable)
        else:
            self.embedding = self.add_variable(
                self.name + "-embedding_kernel",
                dtype=tf.float32,
                initializer=self.initial_values,
                shape=None,
                trainable=self.trainable)

    def call(self, x, **_):
        return tf.nn.embedding_lookup(self.embedding, x)


class Linear(tf.keras.layers.Layer):

    def __init__(self, hidden_units, bias=False, name=None, **kwargs):
        super(Linear, self).__init__(name=name, **kwargs)
        self.hidden_units = hidden_units
        self.bias = bias
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_variable(
            "w",
            shape=[input_shape[-1].value, self.hidden_units],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=self.trainable)
        if self.bias:
            self.b = self.add_variable(
                "b",
                shape=[self.hidden_units],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=self.trainable)

    def call(self, inputs, **_):
        if inputs.dtype is not tf.float32:
            inputs = tf.cast(inputs, tf.float32)
        shape = inputs.get_shape().as_list()
        r = tf.tensordot(inputs, self.w, [[len(shape) - 1], [0]]) \
            if len(shape) > 2 else tf.matmul(inputs, self.w)
        return r + self.b if self.b is not None else r

    def compute_output_shape(self, input_shape):
        in_sp = list(input_shape)
        in_sp[-1] = tf.Dimension(self.hidden_units)
        return tf.TensorShape(tuple(in_sp))


class MultiLabelIndependentLinear(tf.keras.layers.Layer):

    def __init__(self, num_labels, bias=False, name=None, **kwargs):
        super(MultiLabelIndependentLinear, self).__init__(name=name, **kwargs)
        self.num_labels = num_labels
        self.bias = bias
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_variable(
            "w",
            shape=[self.num_labels, input_shape[-1].value],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=self.trainable)
        if self.bias:
            self.b = self.add_variable(
                "b",
                shape=[self.num_labels],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                trainable=self.trainable)

    def compute_output_shape(self, input_shape):
        return [input_shape[0] * self.num_labels, 1]

    def call(self, inputs, batch_size=None, **_):
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
            inputs = tile_batch(inputs, self.num_labels)
        # [batch * num_labels, input_last_dim]
        w = tf.tile(self.w, [batch_size, 1])
        # [batch * num_labels, 1]
        b = tf.tile(self.b, [batch_size]) if self.b is not None else None

        # [batch * num_labels, 1]
        r = tf.reduce_sum(tf.multiply(inputs, w), -1)
        r = r + b if self.b is not None else r
        # return tf.reshape(r, [batch_size, self.num_labels])
        return r


class LearningRateDecay(tf.keras.layers.Layer):

    def __init__(self, initial_lr, decay_factor=0.9, name='lr_decay'):
        super(LearningRateDecay, self).__init__(name=name)
        self.lr = self.add_variable(
            name=name,
            shape=(),
            dtype=tf.float32,
            initializer=tf.constant_initializer(initial_lr),
            trainable=False)
        self.decay_factor = decay_factor
        if not tf.executing_eagerly():
            self.decay_opr = tf.assign(self.lr, self.lr * self.decay_factor)

    def compute_output_shape(self, input_shape):
        return ()

    def value(self):
        return self.lr

    def decay(self, sess=None):
        if sess is not None:
            sess.run(self.decay_opr)
        else:
            tf.assign(self.lr, self.lr * self.decay_factor)


def map_nested_list(map_fn, ls0, other_lists_with_same_struct):
    if isinstance(ls0, np.ndarray):
        ls0 = ls0.tolist()
    others = [ls.tolist() if isinstance(ls, np.ndarray) else ls for ls in
              other_lists_with_same_struct]
    if isinstance(ls0[0], list):
        return [map_nested_list(map_fn, sl0, sl1) for sl0, *sl1 in
                zip(ls0, *others)]
    return map_fn(ls0, others)


def convert_nested_id_to_str(
        ids, id2word=None, strip=True, join=' ', remove_prefix=None,
        padding_value=0):
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = [ids]
    if len(ids) < 1:
        return 'EMPTY SEQ'
    if isinstance(ids[0], list):
        return [convert_nested_id_to_str(
            sublist, id2word, strip, join) for sublist in ids]
    if id2word is None:
        return join.join([str(v) for v in ids])
    if strip:
        if id2word[0] == padding_value:
            return 'PADDING SEQ'
        return join.join(
            [id2word[i] if remove_prefix is None else id2word[i][
             id2word[i].index(remove_prefix) + 1:]
                for i in
             ids[0:(ids.index(padding_value))
                 if padding_value in ids else len(ids)]])
    else:
        return join.join([id2word[i] for i in ids])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.isscalar(obj):
            return np.asscalar(obj)
        return json.JSONEncoder.default(self, obj)


def json_dump(obj, fobj):
    if isinstance(fobj, str):
        with open(fobj, 'w') as fo:
            json.dump(obj, fo, cls=NumpyEncoder)
    else:
        json.dump(obj, fobj, cls=NumpyEncoder)


def json_load(fobj):
    if isinstance(fobj, str):
        with open(fobj, 'r') as fo:
            return json.load(fo)
    else:
        return json.load(fobj)


def pkl_load(fobj):
    if isinstance(fobj, str):
        with open(fobj, 'rb') as inf:
            return pickle.load(inf)
    else:
        return pickle.load(fobj)


def pkl_dump(obj, fobj):
    if isinstance(fobj, str):
        with open(fobj, 'wb') as of:
            pickle.dump(obj, of, pickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(obj, fobj, pickle.HIGHEST_PROTOCOL)
