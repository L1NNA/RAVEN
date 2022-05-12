import json
from sklearn.decomposition import TruncatedSVD as tsvd
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
from tfutils import log, log_warn, json_dump
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
import dps
from nltk import ngrams


class MalConvPreprocessr:

    def __init__(self, ps_folder,
                 training_files, cache_folder,
                 protocol: dps.Protocol):

        self.rsk_id2rsk = protocol.rsk_id2rsk
        self.rsk_id2frq = protocol.rsk_id2frq
        self.rsk_rsk2id = protocol.rsk_rsk2id
        self.rsk_rsk2dsp = protocol.rsk_rsk2dsp
        self.ds_meta = None

        if tf.executing_eagerly():
            raise Exception('Eager mode is not supported.')

        if not os.path.exists(ps_folder):
            os.makedirs(ps_folder)

        log('loading training set to build preprocessor...')

        log('Infering shapes [loading objects]....')
        objs = [self.model_obj(file) for file in tqdm(training_files[:20])]
        log('Infering shapes [calculating]....')
        self.ds_meta = tfr.infer_shape(objs, show_progress=True)
        log('Inferred shape:')
        for k in self.ds_meta['shapes']:
            log('   {} {} {}',
                k, self.ds_meta['shapes'][k], self.ds_meta['types'][k])

        log('Saving preprocessor...')
        with open(os.path.join(ps_folder, 'model.pkl'),
                  'wb') as of:
            pickle.dump(self, of, pickle.HIGHEST_PROTOCOL)

    def __inplace_transform(self, obj):
        dps.transform_rsk_to_id(obj, self.rsk_id2rsk)
        return obj

    def model_obj(self, obj, as_tfr=False):
        if isinstance(obj, str):
            obj = dps.read_tagged_json(obj)
        if obj is None:
            return None
        self.__inplace_transform(obj)

        # *_sparse will keep sparse (no padding for performance concern)
        dat = {'rsk': obj['rsk'],
               'byte': obj['byte'],
               'byte_len': len(obj['byte']),
               'sha256': str(obj['sha256'])}
        return dat if not as_tfr else tfr.convert_to_tfr(dat, self.ds_meta)


def build_pps(ps_folder, training_files, cache_folder, protocol) -> MalConvPreprocessr:
    ps = get_pps(ps_folder)
    if ps is None:
        from md_malconv_ds import MalConvPreprocessr
        return MalConvPreprocessr(
            ps_folder, training_files, cache_folder,
            protocol)
    else:
        log('Loaded the existing preprocessor from {} instead.', ps_folder)
    return ps


def get_pps(ps_folder: str) -> MalConvPreprocessr:
    if os.path.exists(os.path.join(ps_folder, 'model.pkl')):
        from md_malconv_ds import MalConvPreprocessr
        main_module = sys.modules['__main__']
        setattr(main_module, 'MalConvPreprocessor', MalConvPreprocessr)
        with open(os.path.join(ps_folder, 'model.pkl'), 'rb') as inf:
            return pickle.load(inf)
    return None


def build_training_data_and_ps(
        ds_folder, overwrite_preprocessor_ignore_cache=False,
        rsk_min_freq=20, **kwargs):

    protocol = dps.split_and_label(ds_folder, rsk_min_freq)

    tfr_folder = os.path.join(
        ds_folder, '..', '{}_malconv'.format(os.path.basename(ds_folder)))
    ps_folder = os.path.join(tfr_folder, 'preprocessor')

    if overwrite_preprocessor_ignore_cache and os.path.exists(ps_folder):
        shutil.rmtree(ps_folder)
    cache_folder = os.path.join(tfr_folder, "cache")
    if overwrite_preprocessor_ignore_cache and os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)

    file_trn = os.path.join(tfr_folder, "trn")
    file_vld = os.path.join(tfr_folder, "vld")
    file_tst = os.path.join(tfr_folder, "tst")

    trn = protocol.trn
    vld = protocol.vld
    tst = protocol.tst

    log('Generating preprocesor. Will be saved to {}', ps_folder)
    ps = build_pps(ps_folder, trn, cache_folder, protocol)
    # for d in (file_trn, file_vld, file_tst):
    #    if os.path.exists(d):
    #        shutil.rmtree(d)

    dps.write_files_as_tfr_trnks(vld, file_vld, ps.model_obj)
    dps.write_files_as_tfr_trnks(tst, file_tst, ps.model_obj)
    dps.write_files_as_tfr_trnks(trn, file_trn, ps.model_obj)
    log('done.')
    return load_training_data(ds_folder)


def load_training_data(ds_folder, map_parallelism=8, **kwargs):

    tfr_folder = os.path.join(
        ds_folder, '..', '{}_malconv'.format(os.path.basename(ds_folder)))
    ps_folder = os.path.join(tfr_folder, 'preprocessor')
    ps = get_pps(ps_folder)

    file_trn = os.path.join(tfr_folder, "trn")
    file_vld = os.path.join(tfr_folder, "vld")
    file_tst = os.path.join(tfr_folder, "tst")

    trn = tfr.read_objs_trf_ds(
        file_trn, map_parallelism, meta=ps.ds_meta).tfds
    vld = tfr.read_objs_trf_ds(
        file_vld, map_parallelism, meta=ps.ds_meta).tfds
    tst = tfr.read_objs_trf_ds(
        file_tst, map_parallelism, meta=ps.ds_meta).tfds

    return trn, vld, tst, ps


def set_flags(oparser):
    parser = oparser.add_argument_group('data source')
    parser.add_argument(
        "--ds_folder",
        type=str,
        default="data/small",
        metavar="",
        help="The data directory containing .tagged.json files.")


def prepare(FLAGS):
    trn, vld, tst, ps = build_training_data_and_ps(**vars(FLAGS))
    if FLAGS.verbose > 0:
        with tf.Session() as sess:
            log('##### Training set:')
            ps.test_ds(trn, sess)
            log('##### Validation set:')
            ps.test_ds(vld, sess)
            log('##### Testing set:')
            ps.test_ds(tst, sess)


def main():
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    main_parser.add_argument(
        "--help",
        action='store_true',
        help='display the usage.')
    main_parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        metavar="",
        help="Verbosity for testing. 0 skips the test run.")
    dps.set_flags(main_parser)
    set_flags(main_parser)
    gparser = main_parser.add_argument_group('preprocessor generation')
    gparser.add_argument(
        "--overwrite_preprocessor_ignore_cache",
        action='store_true',
        help="Ignore existing preprocessor and any cached data.")
    FLAGS, unparsed = main_parser.parse_known_args()
    if FLAGS.help:
        main_parser.print_help()
    else:
        if len(unparsed) > 0:
            log_warn("Uknown arguments: {}", unparsed)
        prepare(FLAGS)


if __name__ == '__main__':
    main()
