import dps
import sklearn
from sklearn import metrics
import argparse
import os
import functools
import sys
import pickle
import md_bytegram_ds
from md_bytegram_ds import load_raw_ds
from dps import tokenize_ins, read_tagged_json
from collections import Counter, defaultdict
from nltk import ngrams
import numpy as np
from tfutils import log_err, log, log_warn
import tfutils as tfu
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, Trials, space_eval, hp
from concurrent.futures import ProcessPoolExecutor 
from functools import partial
from math import ceil
import sqlite3
from queue import Queue
import threading
# from sqlite3worker import Sqlite3Worker


class pCounter:
    def __init__(self, file_path, overwrite=True):
        if overwrite and os.path.exists(file_path):
            os.remove(file_path)
        self.cnn = sqlite3.connect(file_path)
        self.cnn.execute(
            'create table counter (key blob primary key, val int)')
        self.cnn.execute(
            'create index counter_val on counter (val)')
        self.cnn.commit()
        # self.cnt = shelve.open(file_path, flag='c')

    def inc(self, key, val=1):
        p_val = val
        item = self.cnn.execute(
            'select val from counter where key = ?',
            (key,)).fetchone()
        if item is not None and len(item) > 0:
            p_val = item[0] + val
        self.cnn.execute(
            'replace into counter (key, val) values (?, ?)',
            (key, p_val))
        # p_val = val
        # if key in self.cnt:
        #     p_val = self.cnt[key] + p_val
        # self.cnt[key] = p_val

    def topk(self, k):
        items = self.cnn.execute(
            'select key, val from counter order by val desc limit {}'.format(k)
            ).fetchall()
        return list(items)
        # simple brute force approach to reduce memory usage
        # eles = set()
        # for _ in tqdm(k):
        #     km = None
        #     kv = -1
        #     for k in self.cnt:
        #         if k not in eles:
        #             if self.cnt[k] > kv:
        #                 km = k
        #     if km is not None:
        #         eles.add(km)

        # return list(eles)

    def commit(self):
        self.cnn.commit()
        # pass

    def close(self):
        self.cnn.close()


def _feat_byte_ngram(obj, ns=None):
    if ns is None:
        ns = [6]  # best in the paper.
    for n in ns:
        for gram in ngrams(obj['byte'], n):
            yield bytes(gram)


def __build_feature_subdict(objs):
    tf = dict()
    if not isinstance(objs, list):
        objs = [objs]
    for obj in objs:
        if isinstance(obj, str):
            obj = read_tagged_json(obj)
        if obj is None:
            continue
        for fea in _feat_byte_ngram(obj):
            if fea in tf:
                tf[fea] += 1
            else:
                tf[fea] = 1
    for k in list(tf.keys()):
        if tf[k] < len(objs):
            del tf[k]
    return tf


def _build_feature_dict(objs, folder, topk=200000):
    log('building features...')

    dict_file = os.path.join(
        folder, './feat_db.sqlite')
    tfc = pCounter(dict_file, True)

    N = 50 
    subList = [objs[n:n+N] for n in range(0, len(objs), N)]
    with ProcessPoolExecutor(10) as e:
        ite = e.map(
            __build_feature_subdict,
           subList 
        )
        for i,tf in enumerate(ite):
            print('Processing {}/{}'.format(i,len(subList)))
            if tf is not None:
                for k in tqdm(tf):
                    tfc.inc(k, tf[k])
            tfc.commit()

    # N = 20
    # subList = [objs[n:n+N] for n in range(0, len(objs), N)]
    # for i,ls in enumerate(subList):
    #     print('Processing {}/{}'.format(i,len(subList)))
    #     tf = __build_feature_subdict(ls)
    #     if tf is not None:
    #         for k in tqdm(tf):
    #             if tf[k] > N:
    #                 tfc.inc(k, tf[k])
    #     tfc.commit()

    # q = Queue()
    # l = threading.Lock()
    # def worker():
    #     while True:
    #         item = q.get()
    #         if item is None:
    #             break
    #         tf = __build_feature_subdict(item)
    #         print(
    #             'processed {}/{}. updating.'.format(
    #                 q.qsize() , len(objs)))
    #         if tf is not None:
    #             for i,k in enumerate(tf):
    #                 print('updated {}/{}'.format(i, len(tf)))
    #                 tfc.inc(k, tf[k])
    #             tfc.commit()
            
    #         q.task_done()
    # threads = []
    # for i in range(5):
    #     t = threading.Thread(target=worker)
    #     t.start()
    #     threads.append(t)
    # N = 5
    # subList = [objs[n:n+N] for n in range(0, len(objs), N)]
    # for ls in subList:
    #     q.put(ls)
    # q.join()

    # for i in range(5):
    #     q.put(None)
    # for t in threads:
    #     t.join()

    features = tfc.topk(topk)
    id2fea = [x[0] for x in features]
    fea2id = {fea: ind for ind, fea in enumerate(id2fea)}
    return id2fea, fea2id


def _embed_obj(
        obj,
        fea2id=None,
        rsk2id=None):
    if isinstance(obj, str):
        obj = read_tagged_json(obj)
    if obj is None:
        return None
    x = np.zeros([len(fea2id)])
    for fea in _feat_byte_ngram(obj):
        if fea in fea2id:
            x[fea2id[fea]] += 1
    y = np.zeros(len(rsk2id))
    for rsk in obj['rsk']:
        if rsk in rsk2id:
            y[rsk2id[rsk]] = 1
    return x, y


def _fpr_flatten(y, yp):
    y = np.array(y).flatten()
    yp = np.array(yp).flatten()
    fp = 0
    for ind in range(len(y)):
        if y[ind] == 0 and yp[ind] == 1:
            fp += 1
    return fp / len(y)


def _embed_objs(objs, fea2id, rsk2id):
    log('embedding objects into feature space...')
    xs = []
    ys = []

    with ProcessPoolExecutor() as e:
        ite = e.map(
            partial(
                _embed_obj,
                fea2id=fea2id,
                rsk2id=rsk2id),
            objs
        )
        for re in tqdm(ite, total=len(objs)):
            if re is not None:
                x, y = re
                xs.append(x)
                ys.append(y)
    return np.stack(xs), np.stack(ys)


def train(FLAGS):
    log('startup argvs:')
    fdict = vars(FLAGS)
    for k in sorted(fdict):
        log('  {} -- {}', k, fdict[k])

    trn, vld, tst, rsk2id = load_raw_ds(
        FLAGS.ds_folder, FLAGS.rsk_min_freq)
    ds_id = "{}-byte-n-grams".format(
        os.path.basename(FLAGS.ds_folder)
    )
    cache_folder = os.path.join(
        FLAGS.ds_folder, '..', os.path.basename(FLAGS.ds_folder) + '_bytegram')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    ds_cache = os.path.join(cache_folder, ds_id + '.ds.pk')
    if os.path.exists(ds_cache):
        with open(ds_cache, 'rb') as inf:
            log('loading previously built dataset from {}',
                ds_cache)
            (xs_trn, ys_trn, xs_vld, ys_vld,
             xs_tst, ys_tst) = pickle.load(inf)
    else:
        feat_cache = os.path.join(
            cache_folder,
            '{}.{}.fe.pk'.format(ds_id, FLAGS.topk))
        if os.path.exists(feat_cache):
            log('loading pre-learned features from {}',
                feat_cache)
            with open(feat_cache, 'rb') as inf:
                fea2id = pickle.load(inf)
        else:
            _, fea2id = _build_feature_dict(
                trn, cache_folder, FLAGS.topk)
            with open(feat_cache, 'wb') as of:
                pickle.dump(
                    fea2id, of, pickle.HIGHEST_PROTOCOL)
        xs_trn, ys_trn = _embed_objs(
            trn,
            fea2id,
            rsk2id)
        xs_vld, ys_vld = _embed_objs(
            vld,
            fea2id,
            rsk2id)
        xs_tst, ys_tst = _embed_objs(
            tst,
            fea2id,
            rsk2id)
        with open(ds_cache, 'wb') as of:
            pickle.dump(
                (xs_trn, ys_trn, xs_vld, ys_vld, xs_tst, ys_tst),
                of, pickle.HIGHEST_PROTOCOL)

    log('tuning params for the baselines...')
    param = {
        'C': hp.loguniform('C', -5, 5)
    }
    if FLAGS.rsk_limit > 0:
        rsk_limit = min(FLAGS.rsk_limit, len(rsk2id))
    else:
        rsk_limit = len(rsk2id)
    ys_vld = ys_vld[:, :rsk_limit]
    ys_trn = ys_trn[:, :rsk_limit]
    ys_tst = ys_tst[:, :rsk_limit]
    # import ipdb
    # ipdb.set_trace()

    def get_model(params):
        model = OneVsRestClassifier(
            LogisticRegression(
                verbose=FLAGS.verbose,
                **params),
            n_jobs=6)
        return model

    def eval(model, x_t, y_t):
        prob_tst = model.predict_proba(x_t)
        pred_tst = model.predict(x_t)

        y_t_roc = y_t[:,~np.all(y_t == 0, axis=0)]
        prob_tst_roc = prob_tst[:,~np.all(y_t == 0, axis=0)]
        auroc = metrics.roc_auc_score(y_t_roc, prob_tst_roc)

        acc = metrics.accuracy_score(y_t, pred_tst)
        prc = metrics.precision_score(
            y_t, pred_tst, average='macro')
        rcl = metrics.recall_score(
            y_t, pred_tst, average='macro')
        fpr = _fpr_flatten(y_t, pred_tst)
        brf = metrics.mean_squared_error(
            y_t, prob_tst)
        log('evaluation score: roc {}', auroc)
        log('evaluation score: acc {}', acc)
        log('evaluation score: prc {}', prc)
        log('evaluation score: rcl {}', rcl)
        log('evaluation score: fpr {}', fpr)
        log('evaluation score: brf {}', brf)
        return auroc

    def objective(params):
        model = get_model(params)
        log('Fitting mode of {}', params)
        model.fit(xs_trn, ys_trn)
        log('Testing trained model of {}', params)
        auroc = eval(model, xs_vld, ys_vld)
        log('topk {} Tested {}. AUROC: {}', FLAGS.topk, params, auroc)
        return -1 * auroc

    #best = fmin(objective, param, algo=tpe.rand.suggest, max_evals=10)
    #best_param = space_eval(param, best)
    best_param = {'C': 0.029040282291397785}
    log('Best param is {}', best_param)
    for p in [0.2, 0.4, 0.6, 0.8]:
        log('Evaluating {}', p)
        model = get_model(best_param)
        cp = int(len(xs_trn) * p)
        model.fit(xs_trn[:cp,:], ys_trn[:cp,:])
        log('Validation set {}', 1)
        eval(model, xs_vld, ys_vld)
        log('Testing set {}', 1)
        eval(model, xs_tst, ys_tst)


def set_flags(parser):
    parser = parser.add_argument_group("training configuration")
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output.")
    parser.add_argument(
        "--topk",
        type=int,
        default=200000,
        metavar="",
        help="Topk features to be used.")
    parser.add_argument(
        "--rsk_limit",
        type=int,
        default=150,
        metavar="",
        help="Rsks to be considered.")
    return parser


def main():
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    set_flags(main_parser)
    md_bytegram_ds.set_flags(main_parser)
    main_parser.add_argument(
        "--help",
        action='store_true',
        help='display the usage.')

    FLAGS, unparsed = main_parser.parse_known_args()
    tfu.LOG_VERBOSITY = FLAGS.verbose
    if FLAGS.help:
        main_parser.print_help()
    else:
        if len(unparsed) > 0:
            log_warn("Uknown arguments: {}", unparsed)
        train(FLAGS)


if __name__ == '__main__':
    main()
