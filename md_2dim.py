import dps
import sklearn
from sklearn import metrics
import argparse
import os
import functools
import sys
import pickle
import md_2dim_ds
from md_2dim_ds import load_raw_ds
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
from collections import Counter
from scipy import stats
from math import floor
import math
import hashlib
import pefile
import array
import re
from sklearn.neural_network import MLPClassifier


def _is_number(num):
    if not isinstance(num, str):
        num = str(num)
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


def _hash(val, bits=8):
    l = bits // 4
    return int(
        hashlib.md5(val.encode(errors='ignore')).hexdigest()[:l],
        16)


def _embed_obj(
        obj,
        rsk2id=None):
    if isinstance(obj, str):
        obj = read_tagged_json(obj)
    if obj is None:
        return None

    # byte/entropy histogram
    seq = obj['byte']
    histo = np.zeros([16, 16])
    for i in range(0, len(seq), 256):
        win = seq[i:i+1024]
        counter = Counter(win)
        b2ent = stats.entropy(
            [x for x in counter.values()], base=2)
        b2ent = floor(min(b2ent, 8)/(8/15))
        for k, v in counter.items():
            histo[floor(k/16)][b2ent] += v
    histo = histo.flatten()

    # PE import features
    imp = np.zeros([256])
    for imp_m in obj['imp_m']:
        imp[_hash(imp_m)] += 1
    for imp_f in obj['imp_f']:
        imp[_hash(imp_f)] += 1

    # PE header:
    binary = array.array(
        'B', obj['byte']).tostring()
    pef = np.zeros([256])
    try:
        # may fail
        pe = pefile.PE(data=binary)
        for header in [pe.DOS_HEADER,
                       pe.FILE_HEADER,
                       pe.NT_HEADERS,
                       pe.OPTIONAL_HEADER,
                       pe.RICH_HEADER]:
            if header is not None:
                for attr, value in header.__dict__.items():
                    if _is_number(value):
                        pef[_hash(attr)] += 1
                        pef[_hash(str(value))] += 1
    except Exception as e:
        log_err('Failed to parse PE header. {}', str(e))

    # String
    stf = np.zeros([16, 16])
    for st in obj['str']:
        if len(st) > 5:
            b0 = _hash(st, 4)
            b1 = math.log(len(st), 1.25)
            if b1 < 8:
                b1 = 0
            elif b1 > 200:
                b1 = 15
            else:
                b1 = 1 + b1 // ((200-8)/13)
            b1 = math.floor(b1)
            stf[b0][b1] += 1
    stf = stf.flatten()

    # x:
    x = np.concatenate([histo, stf, imp, pef])

    # label
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


def _embed_objs(objs, rsk2id):
    log('embedding objects into feature space...')
    xs = []
    ys = []

    with ProcessPoolExecutor(1) as e:
        ite = e.map(
            partial(
                _embed_obj,
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
    ds_id = "{}-2dims".format(
        os.path.basename(FLAGS.ds_folder)
    )
    cache_folder = os.path.join(
        FLAGS.ds_folder, '..', os.path.basename(FLAGS.ds_folder) + '_2dim')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    ds_cache = os.path.join(cache_folder, ds_id + '.ds.pk')
    if os.path.exists(ds_cache):
        with open(ds_cache, 'rb') as inf:
            log('loading previously built dataset from {}', ds_cache)
            (xs_trn, ys_trn, xs_vld, ys_vld,
             xs_tst, ys_tst) = pickle.load(inf)
    else:
        xs_trn, ys_trn = _embed_objs(
            trn,
            rsk2id)
        xs_vld, ys_vld = _embed_objs(
            vld,
            rsk2id)
        xs_tst, ys_tst = _embed_objs(
            tst,
            rsk2id)
        with open(ds_cache, 'wb') as of:
            pickle.dump(
                (xs_trn, ys_trn, xs_vld, ys_vld, xs_tst, ys_tst),
                of, pickle.HIGHEST_PROTOCOL)

    log('tuning params for the baselines...')
    param = {
        'learning_rate_init': hp.uniform(
            'learning_rate_init', 0.001, 0.1)
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
        return OneVsRestClassifier(
            MLPClassifier(
                hidden_layer_sizes=(1024, 1024, 1024),
                activation='relu',
                max_iter=10,
                # verbose=True,
                **params
            ), n_jobs=-1)

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
        brf = metrics.mean_squared_error(y_t, prob_tst)
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
        log('Tested {}. AUROC: {}', params, auroc)
        return -1 * auroc

    # best = fmin(objective, param, algo=tpe.rand.suggest, max_evals=10)
    # best_param = space_eval(param, best)
    best_param = {'learning_rate_init': 0.1}
    model = get_model(best_param)
    log('Best param is {}', best_param)
    model.fit(xs_trn, ys_trn)
    eval(model, xs_tst, ys_tst)


def set_flags(parser):
    parser = parser.add_argument_group("training configuration")
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output.")
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
    md_2dim_ds.set_flags(main_parser)
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
