import json
from sklearn.decomposition import TruncatedSVD as tsvd
import os
import numpy as np
from collections import Counter, namedtuple
from tqdm import tqdm
from tfutils import log, log_warn, json_dump, json_load
from tfutils import pkl_dump, pkl_load
import tensorflow as tf
import pickle
from tensorflow.contrib.tensorboard.plugins import projector
from google.protobuf import text_format
import contextlib
import tempfile
import shutil
import subprocess
from tensorflow.python.lib.io import file_io
import re
import sys
import tfr
import random
from typing import Optional
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from scipy.sparse import csr_matrix as csr
from scipy.sparse import vstack as vs
from typing import List, Dict, NamedTuple


Protocol = NamedTuple(
    'Protocol',
    [('rsk_id2rsk', Dict[int, str]),
     ('rsk_id2frq', Dict[int, int]),
     ('rsk_rsk2id', Dict[str, int]),
     ('rsk_rsk2dsp', Dict[str, str]),
     ('trn', List[str]),
     ('vld', List[str]),
     ('tst', List[str])
     ])


def read_tsv(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.strip().split('\t') for line in lines if
            len(line.strip()) > 0]


def save_tsv(file, values, headers=None):
    with open(file, 'w') as f:
        if headers is not None:
            f.write('{}\n'.format('\t'.join(headers)))
        for tp in tqdm(values):
            f.write('{}\n'.format('\t'.join([str(v) for v in tp])))


__keywords = (
    'loc_', 'sub_', 'var_', 'arg_', 'word_', 'off_', 'locret_', 'flt_',
    'dbl_')


def tokenize_ins(ins, dat=None):
    tkns = [par.lower() for tkn in ins[1:] for par in
            re.split(r'[+\-*\\\[\]:()\s]', tkn)]
    if dat is not None:
        tkns += [dat]
    return list(
        filter(lambda x: len(x) > 0 and not x.startswith(__keywords),
               tkns))


def tokenize_asm(obj):
    for blk in obj['asm']:
        for ind in range(len(blk['ins'])):
            blk['ins'][ind] = tokenize_ins(blk['ins'][ind])


def stats_rsk_tkn_file(file):
    obj = read_tagged_json(file)
    if obj is None:
        return {}
    return obj['rsk']


def _build_rsk_vocab(training_files, min_freq=-1):
    ls = None
    log('Building rsk vocabs...')
    rsk2frq = Counter()
    rsk2dsp = {}
    with ProcessPoolExecutor() as e:
        ite = e.map(stats_rsk_tkn_file, training_files)
        for rsks in tqdm(ite, total=len(training_files)):
            for rsk, dsp in rsks.items():
                rsk2frq[rsk] += 1
                rsk2dsp[rsk] = dsp
    ls = list(sorted(
        rsk2frq.items(), key=lambda v: (v[1], v[0]), reverse=True))
    ls = [(v[0], v[1], rsk2dsp[v[0]]) for v in ls]
    ls = [tp for tp in ls if tp[1] > min_freq]

    rsk_id2rsk = [tp[0] for tp in ls]
    rsk_id2frq = [tp[1] for tp in ls]
    rsk_rsk2id = {rsk_id2rsk[ind]: ind for ind in range(len(rsk_id2rsk))}
    rsk2dsp = {tp[0]: tp[2] for tp in ls}
    log('Total rsk indicators: {}', len(rsk_id2rsk))
    return rsk_id2rsk, rsk_id2frq, rsk_rsk2id, rsk2dsp


def transform_rsk_to_id(obj, rsk_id2rsk):
    obj['truth'] = obj['rsk']
    obj['rsk'] = [1 if rsk_id2rsk[i] in obj['rsk'] else 0
                  for i in range(len(rsk_id2rsk))]


def read_tagged_json(file):
    try:
        with open(file, encoding='utf-8') as zf:
            try:
                obj = json.load(zf)
                return obj
            except Exception:
                with open(file, 'r') as zf1:
                    try:
                        obj = json.load(zf1)
                    except Exception as e0:
                        log_warn("Failed to load file {} - {}", file, e0)
                        return None
    except Exception as e1:
        log_warn("Failed to load file {} -{}", file, e1)
        return None


def read_tagged_jsons(files, show_progress=True):
    if show_progress:
        files = tqdm(files)
    for file in files:
        obj = read_tagged_json(file)
        if obj is not None:
            yield obj


def write_objs_as_tfrds(self, objs, file):
    def gen():
        for obj in tqdm(objs):
            modeled = self.model_obj(obj, as_tfr=True)
            if modeled is not None:
                yield modeled

    log('Generating tfr for {}.', file)
    tfr.write_tfr(gen(), file)


def read_tfrds(self, folder, map_parallelism=8):
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    log('dataset will be loaded from:')
    for f in files:
        log('    {}'.format(f))
    return tfr.read_objs_trf_ds(
        files, meta=self.ds_meta, parallelism=map_parallelism)


def convert_objs_to_ele(self, objs, sess=None, map_parallelism=8):
    tfrs = [self.model_obj(obj, as_tfr=True) for obj in objs]
    tfrs = [r for r in tfrs if r is not None]
    tfrs = tf.convert_to_tensor(tfrs)
    if tf.executing_eagerly():
        ds = tf.data.Dataset.from_tensor_slices(tfrs)
        ds = ds.map(lambda x: tfr.parse_tfr(x, self.ds_meta, True),
                    num_parallel_calls=map_parallelism)
        return ds.make_one_shot_iterator()
    else:
        if self.tfr_placeholder is None:
            self.tfr_placeholder = tf.placeholder(dtype=tf.string,
                                                  shape=[None])
            ds = tf.data.Dataset.from_tensor_slices(
                self.tfr_placeholder)
            ds.map(
                lambda x: tfr.parse_tfr(x, self.ds_meta, True),
                num_parallel_calls=map_parallelism)
            self.tfr_ite = ds.make_initializable_iterator()
        sess.run(self.tfr_ite.initializer,
                 feed_dict={self.tfr_placeholder: tfrs})
        return self.tfr_ite


def test_tfds(ds, sess):
    def cal_density(val):
        val = np.array(val)
        return np.prod(val.shape), np.count_nonzero(val) / np.prod(
            val.shape)

    ite = ds.make_one_shot_iterator()
    nxt = ite.get_next()
    for k in nxt:
        log(' {} of shape {}', k, nxt[k].shape)
    ind = 0
    while True:
        ind += 1
        try:
            bat = sess.run(nxt)
            for k in sorted(bat):
                log('   ', ind, k, bat[k].shape, bat[k].dtype,
                    cal_density(bat[k]))
        except tf.errors.OutOfRangeError:
            break


def _wraper_ps_convert_file_to_trf(model_obj_fn, inputs_and_outputs):
    files, tfr_file = inputs_and_outputs
    log('Saving TFRs to {}', tfr_file)
    with tfr.get_writer(tfr_file) as writer:
        for ind, file in enumerate(files):
            r = model_obj_fn(file, True)
            if r is not None:
                writer.write(r.SerializeToString())
            else:
                log_warn('{} cannot be processed. None instead!', file)
            log('Processed {}/{}', ind, len(files))


def write_files_as_tfr_trnks(files, tfr_folder, model_obj_fn, trunk_size=100):
    if not os.path.exists(tfr_folder):
        os.makedirs(tfr_folder)
        trunks = [files[i:i + trunk_size] for i in
                  range(0, len(files), trunk_size)]
        rfiles = [os.path.join(tfr_folder, 'part-{:04d}.tfr').format(i) for i in
                  range(len(trunks))]
        log('Writing {} trunks to folder {}', len(rfiles), tfr_folder)
        params = [tp for tp in zip(trunks, rfiles)]
        log('Saving TFRs to trunk of {} size. File folder is {}.', len(trunks),
            tfr_folder)
        with ProcessPoolExecutor() as e:
            list(e.map(partial(_wraper_ps_convert_file_to_trf, model_obj_fn), params))
    else:
        log_warn(
            'Folder {} exists. Skipped for now. Delete it and run again if intended.',
            tfr_folder)


def split_and_label(ds_folder, rsk_min_freq=20) -> Protocol:
    meta_file = os.path.join(
        ds_folder,
        '..',
        os.path.basename(ds_folder))
    json_file = meta_file + '.json'
    stats_file = meta_file + '_rsk.tsv'
    meta_file = meta_file + '.pk'
    if not os.path.exists(meta_file):
        files = [os.path.join(ds_folder, fs) for fs in
                 os.listdir(ds_folder) if
                 fs.endswith('.tagged.json')]
        random.shuffle(files)
        trn, vld, tst = [prt.tolist() for prt in np.split(
            np.array(files),
            [int(.80 * len(files)), int(.90 * len(files))])]
        rsk_id2rsk, rsk_id2frq, rsk_rsk2id, rsk2dsp = _build_rsk_vocab(
            trn, rsk_min_freq)
        proto = Protocol(
            rsk_id2rsk, rsk_id2frq, rsk_rsk2id, rsk2dsp, trn, vld, tst)
        pkl_dump(proto, meta_file)
        lines = [(rsk_id2rsk[i], rsk_id2frq[i], rsk2dsp[rsk_id2rsk[i]])
                 for i in range(len(rsk_id2rsk))]
        save_tsv(stats_file, lines)
        json_dump(proto, json_file)
    else:
        proto = pkl_load(meta_file)
    return proto


def set_flags(oparser):
    parser = oparser.add_argument_group('dataset global setting')
    parser.add_argument(
        "--rsk_min_freq",
        type=int,
        default="20",
        metavar="",
        help="Mimimum rsk frequency.")
