import json
from sklearn.decomposition import TruncatedSVD as tsvd
import os
import numpy as np
from collections import Counter, OrderedDict
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


ds_suffix = 'byte'


@contextlib.contextmanager
def tempdir(prefix='tmp'):
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def __read_floats(file):
    return np.fromfile(file, dtype='>f')


def __check_embedding_and_vocab(data_folder):
    return os.path.exists(
        os.path.join(data_folder, 'embd.tsv')) and os.path.exists(
        os.path.join(data_folder, 'embd.bin'))


def __read_embeding_and_vocab_list(data_folder):
    ws = dps.read_tsv(os.path.join(data_folder, 'embd.tsv'))
    arr = __read_floats(os.path.join(data_folder, 'embd.bin'))
    arr = np.reshape(arr, (len(ws) - 1, -1))
    arr = arr.astype(np.float32)
    return ws, arr


def __tokenize_asm_file(cache_folder, file):
    obj = dps.read_tagged_json(file)
    if obj is not None:
        dps.tokenize_asm(obj)
        del obj['str']
        del obj['byte']
        del obj['data']
        f = os.path.join(
            cache_folder, '{}.tagged.json'.format(obj['sha256']))
        with open(f, 'w') as f:
            json.dump(obj, f)


def _build_asm_tkn_embd(files, kam1n0exp_jar, cache_folder):
    if cache_folder is None:
        cache_folder = tempdir('tmp_train_embd_')
    cache_folder = os.path.abspath(cache_folder)

    if not __check_embedding_and_vocab(cache_folder):
        if not os.path.exists(cache_folder):
            log('tokenizing and saving files to the folder: {}...',
                cache_folder)
            os.makedirs(cache_folder)
            with ProcessPoolExecutor() as e:
                list(tqdm(
                    e.map(partial(__tokenize_asm_file, cache_folder), files),
                    total=len(files)
                ))
        cmd = [
            'java',
            '-Xmx300G',
            '-cp',
            kam1n0exp_jar,
            'ca.mcgill.sis.dmas.nlp.model.rsk.PrepareEmbeddingLayer',
            cache_folder
        ]
        log('Executing {}', cmd)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1)
        for line in iter(p.stdout.readline, b''):
            print(line)
        p.stdout.close()
        p.wait()
    return __read_embeding_and_vocab_list(cache_folder)


def _transform_asm(obj, tkn2id, embd=None):
    dps.tokenize_asm(obj)
    for blk in obj['asm']:
        asms = []
        for ind in range(len(blk['ins'])):
            tkns = blk['ins'][ind]
            ids = [tkn2id[tkn] for tkn in tkns if tkn in tkn2id]
            if embd is not None:
                vec = np.zeros(embd.shape[1], dtype=np.float32)
                for tid in ids:
                    vec = np.add(
                        vec, embd[tid]) if tid != 0 else vec
                asms.append((ids, vec.tolist()))
            else:
                asms.append(ids[:5])
        blk['ins'] = asms
    sorted(obj['asm'], key=lambda x: len(x), reverse=True)


def __tokenize_str(val):
    val = val.lower()
    grams = []
    for n in [2, 3]:
        grams.extend(ngrams(val, n))
    return ['_'.join(g) for g in grams]


def __uniqe_token_str_file(field, file):
    obj = dps.read_tagged_json(file)
    if obj is None:
        return {}
    return {gram for val in obj[field] for gram in __tokenize_str(val)}


def __str_feat_embed(val, features):
    vec = np.zeros(len(features), dtype=np.float32)
    for gram in __tokenize_str(val):
        if gram in features:
            vec[features[gram]] += 1
    return csr(vec)


def _build_str_matrix(training_files, field, cache_folder=None,
                      n_components=50) -> (dict, tsvd):
    cache_file = os.path.join(cache_folder, "svd-{}.pkl".format(
        field)) if cache_folder is not None else None
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as inf:
            features, model = pickle.load(inf)
            log('loaded previously built transformer for {}', field)
            return features, model
    log('Building matrix for {}', field)
    with ProcessPoolExecutor() as e:
        features = set()
        ite = e.map(partial(
            __uniqe_token_str_file, field), training_files)
        for st in tqdm(ite, total=len(training_files)):
            features.update(st)
    features = list(sorted(features))
    log('Total {} features. Preparing matrix for SVD.', len(features))
    features = {features[ind]: ind for ind in range(len(features))}
    with ProcessPoolExecutor() as e:
        vecs = {}
        ite = e.map(partial(_transform_str_to_vec, field, features),
                    training_files)
        for st, _ in tqdm(ite, total=len(training_files)):
            vecs.update(st)

    mat = vs([v[1] for v in vecs.items()])
    model = tsvd(n_components=n_components, n_iter=10)
    model.fit(mat)
    if cache_file is not None:
        with open(cache_file, 'wb') as of:
            pickle.dump((features, model), of, pickle.HIGHEST_PROTOCOL)
    return features, model


def _transform_str_to_vec(fields, features, obj, model=None):
    if isinstance(obj, str):
        obj = dps.read_tagged_json(obj)
    if not isinstance(fields, list):
        fields = [fields]
    if not isinstance(features, list):
        features = [features]
    if obj is None:
        if len(fields) == 1:
            return {}, {}
        else:
            return [({},{}) for _ in range(len(fields))]

    counters = []
    uniques = []
    for field, feature in zip(fields, features):
        unique = OrderedDict()
        counter = Counter()
        counters.append(counter)
        uniques.append(unique)
        for val in obj[field]:
            key = val.lower()[:100]
            counter[key] += 1
            unique[key] = None
        if len(unique) < 1:
            obj[field] = []
            continue 
        feats = []
        for k in unique:
            feat = __str_feat_embed(k, feature)
            feats.append(feat)
        if model is not None:
            feats = vs(feats)
            feats = model.transform(feats)
        for i,k in enumerate(unique):
            if len(k.strip()) > 0:
                vec = feats[i]
                unique[k] = vec
                # unique[k] = vec[
                #     np.logical_not(np.isnan(vec))]
            else:
                unique[k] = np.zeros_like(feats[i])
        obj[field] = [(val, unique[val.lower()[:100]]) for val in obj[field]]
        

    if len(uniques) == 1:
        return uniques[0], counters[0]
    else:
        return uniques, counters


class RavenPreprocessor:

    def __init__(self, ps_folder, kam1n0exp_jar,
                 training_files, cache_folder,
                 protocol: dps.Protocol,
                 n_components=50,
                 topk_tfboard=50000):

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
        ws, arr = _build_asm_tkn_embd(
            training_files, kam1n0exp_jar, cache_folder)
        # first one never got trained (for '</s>')
        if str(round(arr[0][0], 8)) != '0.00400269' or str(
                round(arr[0][-1], 8)) != '0.00305939':
            log_warn(
                'WARNING. The embedding values may not be expected.'
                ' Please check Kam1n0 output. {} and {}.',
                str(round(arr[0][0], 8)), str(round(arr[0][-1], 8)))
        self.asm_embd = arr
        self.asm_id2tkn = [w[-1] for w in ws[1:]]
        self.asm_tkn2id = {self.asm_id2tkn[ind]: ind for
                           ind in range(len(self.asm_id2tkn))}

        asm_tsv_path = os.path.join(ps_folder, 'asm_embd.tsv')
        with open(asm_tsv_path, 'w') as of:
            lines = ['\t'.join(w) + '\n' for w in ws][:topk_tfboard+1]
            of.writelines(lines)

        self.str_feat, self.__str_mod = _build_str_matrix(
            training_files, 'str', cache_folder, n_components)

        self.imp_feat, self.__imp_mod = _build_str_matrix(
            training_files, 'imp_f', cache_folder, n_components)

        skip_tensorboard = True
        if skip_tensorboard:
            log('Skipping Tensorboard checkout for embedding visualization.')
        else:
            log('Checking out embeddings for tensorboard...')
            with tf.name_scope('embds_visualization'):
                log('Caculating and Deduplicating embeddings...')
                tps_cache_file = os.path.join(cache_folder, 'tps_str_imp.obj')
                if os.path.exists(tps_cache_file):
                    with open(tps_cache_file, 'rb') as inf:
                        str_tps, imp_tps, str_cnts, imp_cnts = pickle.load(inf)
                else:
                    str_tps = {}
                    str_cnts = Counter()
                    imp_tps = {}
                    imp_cnts = Counter()

                    with ProcessPoolExecutor() as e:
                        ite = e.map(partial(
                            _transform_str_to_vec,
                            ['str', 'imp_f'],
                            [self.str_feat, self.imp_feat]
                        ), training_files)
                        for st, cnt in tqdm(ite, total=len(training_files)):
                            str_tps.update(st[0])
                            imp_tps.update(st[1])
                            str_cnts = str_cnts + cnt[0]
                            imp_cnts = imp_cnts + cnt[1]

                    with open(tps_cache_file, 'wb') as of:
                        pickle.dump(
                            (str_tps, imp_tps, str_cnts, imp_cnts), of, pickle.HIGHEST_PROTOCOL)

                log('Pikcing top {}', topk_tfboard)
                str_cnts = sorted(str_cnts.items(), key=lambda _: (
                    _[1], _[0]), reverse=True)[:topk_tfboard]
                imp_cnts = sorted(imp_cnts.items(), key=lambda _: (
                    _[1], _[0]), reverse=True)[:topk_tfboard]

                var_str = vs([str_tps[w] for w, _ in str_cnts])
                var_str = self.__str_mod.transform(var_str)
                var_imp = vs([imp_tps[w] for w, _ in imp_cnts])
                var_imp = self.__imp_mod.transform(var_imp)

                log('Writing...')
                str_tsv_path = os.path.join(ps_folder, 'str_embd.tsv')
                with open(str_tsv_path, 'w', encoding='utf-8', errors='ignore') as of:
                    of.writelines(
                        [w[0].replace('\n', '[lb]') + '\n' for w in str_cnts])
                imp_tsv_path = os.path.join(ps_folder, 'imp_embd.tsv')
                with open(imp_tsv_path, 'w', encoding='utf-8', errors='ignore') as of:
                    of.writelines([w[0] + '\n' for w in imp_cnts])

                tboard_asm = arr[:topk_tfboard]
                var_asm = tf.get_variable(
                    initializer=tboard_asm,
                    dtype=tf.float32,
                    name="asm_embeding")

                var_str = tf.get_variable(
                    initializer=var_str.astype(np.float32),
                    dtype=tf.float32,
                    name="str_embeding")
                var_imp = tf.get_variable(
                    initializer=var_imp.astype(np.float32),
                    dtype=tf.float32,
                    name="imp_embeding")

                saver = tf.train.Saver(var_list=[var_asm, var_str, var_imp])
                with tf.Session() as sess:
                    tf.global_variables_initializer().run()
                    writer = tf.summary.FileWriter(ps_folder)
                    ph = tf.summary.scalar('preprecessor/placeholder', 0)
                    writer.add_summary(sess.run(ph), global_step=0)
                    saver.save(sess, os.path.join(ps_folder, 'ckpt'))
                    config = projector.ProjectorConfig()
                    embedding = config.embeddings.add()
                    embedding.tensor_name = var_asm.name
                    embedding.metadata_path = os.path.abspath(asm_tsv_path)
                    embedding = config.embeddings.add()
                    embedding.tensor_name = var_str.name
                    embedding.metadata_path = os.path.abspath(str_tsv_path)
                    embedding = config.embeddings.add()
                    embedding.tensor_name = var_imp.name
                    embedding.metadata_path = os.path.abspath(imp_tsv_path)
                    config_pbtxt = text_format.MessageToString(config)
                    file_io.write_string_to_file(
                        os.path.join(ps_folder, 'projector_config.pbtxt'),
                        config_pbtxt)
                    log('Dumped checkpoint to {}', ps_folder)

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

    def __inplace_transform(self, obj, pre_cal_asm_embd=True):
        dps.transform_rsk_to_id(obj, self.rsk_id2rsk)

        _transform_asm(
            obj, self.asm_tkn2id, self.asm_embd if pre_cal_asm_embd else None)

        _transform_str_to_vec('str', self.str_feat, obj, self.__str_mod)

        _transform_str_to_vec('imp_f', self.imp_feat, obj, self.__imp_mod)

        return obj

    def model_obj(self, obj, as_tfr=False, pre_cal_asm_embd=False):
        if isinstance(obj, str):
            obj = dps.read_tagged_json(obj)
        if obj is None:
            return None
        # self.__inplace_transform(obj, pre_cal_asm_embd)
        # mx = -1
        # tt = 0
        # for blk in obj['asm']:
        #     mx = len(blk['ins']) if len(blk['ins']) > mx else mx
        #     tt += len(blk['ins'])

        # trunk_size = tt // mx + 1
        # trunk_size = 64 if trunk_size > 64 else trunk_size
        # vec_indx = [list() for _ in
        #             range(trunk_size)]  # type: List[List[int]]
        # vecs = [list() for _ in
        #         range(trunk_size)]  # type: List[List[Iterable]]
        # blk_ids = [list() for _ in
        #            range(trunk_size)]  # type: List[List[int]]
        # for ctn, blk in enumerate(
        #         sorted(obj['asm'], key=lambda x: len(x), reverse=True)):
        #     trk = min(range(trunk_size), key=lambda ind: len(vecs[ind]))
        #     if pre_cal_asm_embd:
        #         for _, vec in blk['ins']:
        #             vecs[trk].append(vec)
        #     else:
        #         for ids in blk['ins']:
        #             vecs[trk].append(ids)
        #     blk_ids[trk].append(ctn)
        #     vec_indx[trk].append(len(vecs[trk]) - 1)
        #     if pre_cal_asm_embd:
        #         vecs[trk].append(
        #             np.zeros(self.asm_embd.shape[1], dtype=np.float32).tolist())
        #     else:
        #         vecs[trk].append([0])

        # str_vals = [tp[1].tolist() for tp in obj['str']]
        # imp_vals = [tp[1].tolist() for tp in obj['imp_f']]
        # dat_vals = [bt for k in obj['data'] for bt in obj['data'][k] 
        # if k.strip().lower() in ('.data', '.rdata', '.rsrc')]
        # if len(dat_vals) < 1:
        #     dat_vals = [0]

        # *_sparse will keep sparse (no padding for performance concern)
        dat = {
            # 'asm_vec': vecs,
            #    'asm_vec_len': [len(ls) for ls in vecs],
            #    'inds': vec_indx,
            #    'inds_len': [len(ls) for ls in vec_indx],
            #    'blk_ord': blk_ids,
            #    'rsk': obj['rsk'],
            #    'str': str_vals,
            #    'imp': imp_vals,
            #    'data': dat_vals,
            #    'data_len': len(dat_vals),
               'byte': obj['byte'],
               'byte_len': len(obj['byte']),
               'sha256': str(obj['sha256'])}
        return dat if not as_tfr else tfr.convert_to_tfr(dat, self.ds_meta)

    @staticmethod
    def get_attn(obj, blk_ord, probs, topk=10):
        men_len = len(blk_ord) + len(obj['str']) + len(obj['imp_f'])
        if len(probs) != men_len:
            log_warn("Unmatched attention dim for {} - {} vs {} ",
                     obj['sha256'], len(probs), men_len)
        idx = np.argsort(-probs, axis=-1)[..., :topk]
        weights = probs[idx]
        eles = []
        for ele_id in idx:
            if ele_id < len(blk_ord):
                eles.append(obj['asm'][blk_ord[ele_id]])
            elif ele_id < len(blk_ord) + len(obj['str']):
                eles.append(obj['str'][ele_id - len(blk_ord)])
            else:
                eles.append(
                    obj['imp_f'][ele_id - len(blk_ord) - len(obj['str'])])
        return eles, weights


def build_pps(ps_folder, training_files, cache_folder, kam1n0exp_jar, protocol,
              n_components=50) -> RavenPreprocessor:
    ps = get_pps(ps_folder)
    if ps is None:
        from md_raven_ds_byte import RavenPreprocessor
        return RavenPreprocessor(
            ps_folder, kam1n0exp_jar, training_files, cache_folder,
            protocol, n_components)
    else:
        log('Loaded the existing preprocessor from {} instead.', ps_folder)
    return ps


def get_pps(ps_folder: str) -> RavenPreprocessor:
    if os.path.exists(os.path.join(ps_folder, 'model.pkl')):
        from md_raven_ds_byte import RavenPreprocessor
        main_module = sys.modules['__main__']
        setattr(main_module, 'RavenPreprocessor', RavenPreprocessor)
        with open(os.path.join(ps_folder, 'model.pkl'), 'rb') as inf:
            return pickle.load(inf)
    return None


def build_training_data_and_ps(
        ds_folder, kam1n0exp_jar, n_components=50,
        overwrite_preprocessor_ignore_cache=False,
        rsk_min_freq=20, ds_suffix='', **kwargs):

    protocol = dps.split_and_label(ds_folder, rsk_min_freq)

    tfr_folder = os.path.join(
        ds_folder, '..', '{}_raven{}'.format(
            os.path.basename(ds_folder), ds_suffix))
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
    ps = build_pps(ps_folder, trn, cache_folder, kam1n0exp_jar,
                   protocol, n_components)
    # for d in (file_trn, file_vld, file_tst):
    #    if os.path.exists(d):
    #        shutil.rmtree(d)

    dps.write_files_as_tfr_trnks(vld, file_vld, ps.model_obj)
    dps.write_files_as_tfr_trnks(tst, file_tst, ps.model_obj)
    dps.write_files_as_tfr_trnks(trn, file_trn, ps.model_obj)
    log('done.')
    return load_training_data(ds_folder)


def load_training_data(
    ds_folder, map_parallelism=8, ds_suffix='', **kwargs):

    tfr_folder = os.path.join(
        ds_folder, '..', '{}_raven{}'.format(
            os.path.basename(ds_folder), ds_suffix))
    log('loading from {}', tfr_folder)
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
    parser.add_argument(
        "--ds_suffix",
        type=str,
        default="byte",
        metavar="",
        help="suffix to be applied (variants of the dataset)")


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
        "--n_components",
        type=int,
        default=50,
        metavar="",
        help="Number of components for string-based SVD embedding.")
    gparser.add_argument(
        "--kam1n0exp_jar",
        type=str,
        default="kam1n0-server.jar",
        metavar="",
        help="The path of embedding learner (for generation only).")
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
