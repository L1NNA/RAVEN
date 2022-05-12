import dps
import sklearn
from sklearn import metrics
import argparse
import os
import functools
import sys
import pickle
import md_tkn_ds
from md_tkn_ds import load_raw_ds
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


def _feat_asm_ngram(obj, ns=None):
    if ns is None:
        ns = [2, 3]
    op_seq = [tokenize_ins(ins)[0]
              for blk in obj['asm'] for ins in blk['ins']]
    for n in ns:
        for gram in ngrams(op_seq, n):
            yield '-'.join(gram)


def _feat_str(obj):
    for str_val in obj['str']:
        yield str_val


def _feat_imp(obj):
    for imp_val in obj['imp_f']:
        yield imp_val


def _feat(obj, strv=False, impv=False):
    for asmv in _feat_asm_ngram(obj):
        yield asmv
    if strv:
        for strvv in _feat_str(obj):
            yield strvv
    if impv:
        for impvv in _feat_imp(obj):
            yield impvv


def __build_feature_subdict(
        param, strv=False, impv=False):
    ind, obj = param
    if isinstance(obj, str):
        obj = read_tagged_json(obj)
    if obj is None:
        return None
    tf = Counter()
    for fea in _feat(obj, strv, impv):
        tf[fea] += 1
    return tf, ind


def _build_feature_dict(
        objs, strv=False, impv=False, max_vocab=None):
    log('building features...')
    tf = Counter()
    df = defaultdict(set)

    params = [tp for tp in enumerate(objs)]
    with ProcessPoolExecutor() as e:
        ite = e.map(
            partial(
                __build_feature_subdict,
                strv=strv,
                impv=impv),
            params)
        for re in tqdm(ite, total=len(params)):
            if re is not None:
                stf, sind = re
                for fea in stf:
                    tf[fea] += stf[fea]
                    df[fea].add(sind)

    tfidf = {
        key: tf[key]/len(df[fea]) for key in tf
    }
    tfidf = sorted(
        tfidf.items(),
        key=lambda x: (x[1], x[0]),
        reverse=True)[:max_vocab]
    id2fea = [x[0] for x in tfidf]
    fea2id = {fea: ind for ind, fea in enumerate(id2fea)}
    weights = np.array([len(df[fea]) for fea in id2fea])
    return id2fea, fea2id, weights


def _embed_obj(
        obj,
        fea2id=None,
        weights=None,
        rsk2id=None,
        strv=False,
        impv=False):
    if isinstance(obj, str):
        obj = read_tagged_json(obj)
    if obj is None:
        return None
    x = np.zeros([len(fea2id)])
    for fea in _feat(obj, strv, impv):
        if fea in fea2id:
            x[fea2id[fea]] += 1
    x = x / weights
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


def _embed_objs(objs, fea2id, weights, rsk2id, strv=False, impv=False):
    log('embedding objects into feature space...')
    xs = []
    ys = []

    with ProcessPoolExecutor() as e:
        ite = e.map(
            partial(
                _embed_obj,
                fea2id=fea2id,
                weights=weights,
                rsk2id=rsk2id,
                strv=strv,
                impv=impv),
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
    ds_id = "{}-str-{}-imp-{}-maxvocab-{}".format(
        os.path.basename(FLAGS.ds_folder),
        1 if FLAGS.feature_str else 0,
        1 if FLAGS.feature_imp else 0,
        FLAGS.max_vocab
    )
    cache_folder = os.path.join(
        FLAGS.ds_folder, '..', os.path.basename(FLAGS.ds_folder) + '_tkn')
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    ds_cache = os.path.join(cache_folder, ds_id + '.ds.pk')
    if os.path.exists(ds_cache):
        with open(ds_cache, 'rb') as inf:
            log('loading previously built dataset from {}', ds_cache)
            (xs_trn, ys_trn, xs_vld, ys_vld,
             xs_tst, ys_tst) = pickle.load(inf)
    else:
        feat_cache = os.path.join(cache_folder, ds_id + '.fe.pk')
        if os.path.exists(feat_cache):
            log('loading pre-learned features from {}', feat_cache)
            with open(feat_cache, 'rb') as inf:
                fea2id, weights = pickle.load(inf)
        else:
            _, fea2id, weights = _build_feature_dict(
                trn,
                strv=FLAGS.feature_str,
                impv=FLAGS.feature_imp,
                max_vocab=FLAGS.max_vocab)
            with open(feat_cache, 'wb') as of:
                pickle.dump(
                    (fea2id, weights), of, pickle.HIGHEST_PROTOCOL)
        xs_trn, ys_trn = _embed_objs(
            trn,
            fea2id,
            weights,
            rsk2id,
            strv=FLAGS.feature_str,
            impv=FLAGS.feature_imp)
        xs_vld, ys_vld = _embed_objs(
            vld,
            fea2id,
            weights,
            rsk2id,
            strv=FLAGS.feature_str,
            impv=FLAGS.feature_imp)
        xs_tst, ys_tst = _embed_objs(
            tst,
            fea2id,
            weights,
            rsk2id,
            strv=FLAGS.feature_str,
            impv=FLAGS.feature_imp)
        with open(ds_cache, 'wb') as of:
            pickle.dump(
                (xs_trn, ys_trn, xs_vld, ys_vld, xs_tst, ys_tst),
                of, pickle.HIGHEST_PROTOCOL)

    log('tuning params for the baselines...')
    param = {
        'gamma': hp.loguniform('gamma', -5, 5),
        'C': hp.loguniform('C', -5, 5)
    } if not FLAGS.use_logistic else {
        'C': hp.loguniform('C', -5, 5)
    }
    if FLAGS.rsk_limit > 0:
        rsk_limit = min(FLAGS.rsk_limit, len(rsk2id))
    else:
        rsk_limit = len(rsk2id)
    xs_trn = xs_trn[:, :FLAGS.topk]
    xs_vld = xs_vld[:, :FLAGS.topk]
    xs_tst = xs_tst[:, :FLAGS.topk]
    ys_vld = ys_vld[:, :rsk_limit]
    ys_trn = ys_trn[:, :rsk_limit]
    ys_tst = ys_tst[:, :rsk_limit]
    # import ipdb
    # ipdb.set_trace()

    def get_model(params):
        if FLAGS.use_logistic:
            model = OneVsRestClassifier(
                LogisticRegression(
                    verbose=FLAGS.verbose,
                    **params),
                n_jobs=-1)
        else:
            model = OneVsRestClassifier(
                SVC(
                    kernel='linear',
                    probability=True,
                    verbose=FLAGS.verbose,
                    **params),
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
        log('topk {} Tested {}. AUROC: {}', FLAGS.topk, params, auroc)
        return -1 * auroc

    best = fmin(objective, param, algo=tpe.rand.suggest, max_evals=10)
    best_param = space_eval(param, best)
    model = get_model(best_param)
    log('Best param is {}', best_param)
    model.fit(xs_trn, ys_trn)
    eval(model, xs_tst, ys_tst)


def set_flags(parser):
    parser = parser.add_argument_group("training configuration")
    parser.add_argument(
        "--feature_str",
        action='store_true',
        help="Enable string feature")
    parser.add_argument(
        "--feature_imp",
        action='store_true',
        help="Enable import feature.")
    parser.add_argument(
        "--use_logistic",
        action='store_true',
        help="User logistic regression instead of SVM.")
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose output.")
    parser.add_argument(
        "--max_vocab",
        type=int,
        default=50000,
        metavar="",
        help="Maximum features to be extracted.")
    parser.add_argument(
        "--topk",
        type=int,
        default=50000,
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
    md_tkn_ds.set_flags(main_parser)
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
