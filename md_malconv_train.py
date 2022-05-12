from tensorflow.python.util import nest
import tfutils as tfu
from tfutils import log, log_warn
import md_malconv_ds as dps
import md_malconv
import tensorflow as tf
import argparse
import os
import functools


def make_batch(ds, padded_shape, batch_size=1, map_parallelism=8, cache=False,
               prefech=-1, **kwargs):
    if batch_size and batch_size > 1:
        ds = ds.padded_batch(batch_size, padded_shape)
    else:
        ds = ds.map(
            lambda v: nest.map_structure(
                lambda x: tf.expand_dims(x, 0), v),
            num_parallel_calls=map_parallelism)
    if cache:
        ds = ds.cache()
    elif prefech > 1:
        ds = ds.prefetch(prefech)
    return ds


def train(FLAGS):
    log('startup argvs:')
    fdict = vars(FLAGS)
    for k in sorted(fdict):
        log('  {} -- {}', k, fdict[k])
    if FLAGS.eager_mode:
        tf.enable_eager_execution(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
    trn, vld, tst, ps = dps.load_training_data(**fdict)
    trn, vld, tst = nest.map_structure(
        lambda v: make_batch(v, ps.ds_meta['shapes'], **fdict),
        (trn, vld, tst))

    model = md_malconv.MalConv(
        preprocessor=ps,
        **fdict)

    if FLAGS.dataset_prefetch > 0:
        log("The program will prefetch {} records before training..",
            FLAGS.dataset_prefetch)
    if FLAGS.eager_mode:
        if FLAGS.profile_steps > 0:
            log("Currently eager_mode does not support graph profiling.")
        model.train_basic_eager(
            trn, vld, tst,
            loss_fn=model.get_loss,
            loss_optm_name='cf_entropy',
            epoch=FLAGS.epoch_cls,
            **fdict)
    else:
        model.train_basic_graph(
            trn, vld, tst,
            loss_fn=model.get_loss,
            loss_optm_name='cf_entropy',
            epoch=FLAGS.epoch_cls,
            **fdict)


def set_flags(parser):
    parser = parser.add_argument_group("training configuration")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        metavar="",
        help="Learning rate.")
    parser.add_argument(
        "--learning_rate_decay_ep",
        type=int,
        default=-1,
        metavar="",
        help="The step inverval for learning rate decay ^0.99.")
    parser.add_argument(
        "--learning_rate_decay_factor",
        type=float,
        default=0.9,
        metavar="",
        help="The factor to be decayed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        metavar="",
        help="Batch size.")
    parser.add_argument(
        "--epoch_cls",
        type=int,
        default=100,
        metavar="",
        help="Epoches to be trained.")
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=5,
        metavar="",
        help="Cliping gradient (max) in training.")
    parser.add_argument(
        "--log_step_interval",
        type=int,
        default=1,
        metavar="",
        help="Steps to be logged.")
    parser.add_argument(
        "--eager_mode",
        action='store_true',
        help="Enable eager mode.")
    parser.add_argument(
        "--log_device_placement",
        action='store_true',
        help="Logging opr device placements.")
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=-1,
        metavar="",
        help="Profiling mode enabled when > 0. Steps to profile the model.")
    parser.add_argument(
        "--dry_loop",
        action='store_true',
        help="Looping through batches without running training/testing; "
             "to see I/O bottleneck.")
    parser.add_argument(
        "--overwrite_model",
        action='store_true',
        help="Delete existing logs and checkpoints. Start a new one."
             " Use with caution.")
    parser.add_argument(
        "--dataset_prefetch",
        type=int,
        default=-1,
        metavar="",
        help="Prefetch counts. Will conflict with the dataset_cache option.")
    parser.add_argument(
        "--dataset_cache",
        action='store_true',
        help="Cache dataset in memory during the first epoch"
             " and reuse in the subsequent epoch.")
    parser.add_argument(
        "--map_parallelism",
        type=int,
        default=32,
        metavar="",
        help="Number of concurrent calls to "
             "the map function for dataset generation (CPU-intensive)")
    return parser


def main():
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    md_malconv.set_flags(main_parser)
    set_flags(main_parser)
    dps.set_flags(main_parser)
    main_parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        metavar="",
        help="Verbosity for testing. 0 skips the test run.")
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
        if FLAGS.eager_mode:
            train(FLAGS)
        else:
            with tf.Session(
                    config=tf.ConfigProto(
                        log_device_placement=FLAGS.log_device_placement)):
                train(FLAGS)


if __name__ == '__main__':
    main()
