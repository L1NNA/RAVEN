import sklearn
from sklearn import metrics
import argparse
import os
import functools
import sys
import pickle
import md_gist_ds
from md_gist_ds import load_raw_ds
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import fmin, tpe, Trials, space_eval, hp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import math
# Sorry, only works on Linux/Mac:
from colorama import Fore, init
from PIL import Image
import leargist
from datetime import datetime
import json


def read_tagged_json(file):
    try:
        with open(file) as zf:
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


def _to_image(obj):
    byte_arr = obj['byte']
    byte_arr_len = len(obj['byte'])
    width = 1024
    if byte_arr_len <= 10000:
        width = 32
    elif byte_arr_len <= 30000:
        width = 64
    elif byte_arr_len <= 60000:
        width = 128
    elif byte_arr_len <= 100000:
        width = 256
    elif byte_arr_len <= 200000:
        width = 512
    elif byte_arr_len <= 500000:
        width = 768
    height = math.ceil(byte_arr_len * 1.0 / width)
    arr = np.zeros([height * width])
    arr[:byte_arr_len] = byte_arr
    arr = np.reshape(arr, [height, width])
    # 8-bit pixel, black-and-white
    return Image.fromarray(arr, 'L')


def _embed_obj(
        obj,
        rsk2id=None):
    if isinstance(obj, unicode):
        obj = obj.encode('ascii','ignore')
        obj = obj.replace('\\','/')
    if isinstance(obj, str):
        obj = read_tagged_json(obj)
    if obj is None:
        return None
    img = _to_image(obj)
    x = leargist.color_gist(img)
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

    with ProcessPoolExecutor() as e:
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
        FLAGS.ds_folder)
    ds_id = "{}-byte-n-grams".format(
        os.path.basename(FLAGS.ds_folder)
    )
    cache_folder = os.path.join(
        FLAGS.ds_folder, '..', os.path.basename(FLAGS.ds_folder) + '_gist')
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
        'n_neighbors': hp.quniform('n_neighbors', 3, 8, 1)
    }
    if FLAGS.rsk_limit > 0:
        rsk_limit = min(FLAGS.rsk_limit, len(rsk2id))
    else:
        rsk_limit = len(rsk2id)
    ys_vld = ys_vld[:, :rsk_limit]
    ys_trn = ys_trn[:, :rsk_limit]
    ys_tst = ys_tst[:, :rsk_limit]
    print(ys_vld.shape)
    print(ys_trn.shape)
    print(ys_tst.shape)

    def get_model(params):
        params['n_neighbors'] = int(params['n_neighbors'])
        model = OneVsRestClassifier(
            KNeighborsClassifier(algorithm='brute', **params),
            n_jobs=-1)
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

    best = fmin(objective, param, algo=tpe.rand.suggest, max_evals=6)
    best_param = space_eval(param, best)
    # best_param = {
    #     'n_neighbors': 1 
    # }
    model = get_model(best_param)
    log('Best param is {}; Fitting.', best_param)
    model.fit(xs_trn, ys_trn)
    log('Testing...')
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
    md_gist_ds.set_flags(main_parser)
    main_parser.add_argument(
        "--help",
        action='store_true',
        help='display the usage.')

    FLAGS, unparsed = main_parser.parse_known_args()
    if FLAGS.help:
        main_parser.print_help()
    else:
        if len(unparsed) > 0:
            log_warn("Uknown arguments: {}", unparsed)
        train(FLAGS)


if __name__ == '__main__':
    main()
