from tensorflow.python.util import nest
import tfutils as tfu
from tfutils import log, log_warn
from dps import read_tagged_json, split_and_label
import md_raven_ds as dps
import tensorflow as tf
import argparse
import os
import functools
from math import ceil


def make_batch(ds, padded_shape, batch_size=1, map_parallelism=8, cache=False,
               prefech=-1, take_first=-1, **kwargs):
    if batch_size and batch_size > 1:
        ds = ds.padded_batch(batch_size, padded_shape)
    else:
        ds = ds.map(
            lambda v: nest.map_structure(
                lambda x: tf.expand_dims(x, 0), v),
            num_parallel_calls=map_parallelism)
    if take_first > 0:
        ds = ds.take(take_first)
    if cache:
        ds = ds.cache()
    elif prefech > 1:
        ds = ds.prefetch(prefech)
    return ds


def train(cls, FLAGS):
    log('startup argvs:')
    fdict = vars(FLAGS)
    for k in sorted(fdict):
        log('  {} -- {}', k, fdict[k])
    if FLAGS.eager_mode:
        tf.enable_eager_execution(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
    take = FLAGS.take
    if take is not None and take > 0:
        protocol = split_and_label(FLAGS.ds_folder)
        take = ceil(len(protocol.trn) * take)
        
    trn, vld, tst, ps = dps.load_training_data(**fdict)
    trn = make_batch(trn, ps.ds_meta['shapes'], take_first=take,**fdict)
    vld = make_batch(vld, ps.ds_meta['shapes'], take_first=-1,**fdict)
    tst = make_batch(tst, ps.ds_meta['shapes'], take_first=-1,**fdict)

    @functools.lru_cache(maxsize=100)
    def read_tagged_json_cached(sha256):
        return read_tagged_json(
            os.path.join(FLAGS.ds_folder, sha256.decode() + '.tagged.json'))

    model = cls(
        preprocessor=ps,
        obj_lookup_fn=read_tagged_json_cached,
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
            loss_optm_name='conf_entropy',
            sampling_fn=model.sampling,
            sampling_v_fn=model.sampling_json,
            epoch=FLAGS.epoch_cls,
            **fdict)
    else:
        model.train_basic_graph(
            trn, vld, tst,
            loss_fn=model.get_loss,
            loss_optm_name='conf_entropy',
            sampling_fn=model.sampling,
            sampling_v_fn=model.sampling_json,
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
        "--sampling_epoch_interval",
        type=int,
        default=-1,
        metavar="",
        help="Epoch intervals to sample decoder output")
    parser.add_argument(
        "--save_roc",
        action='store_true',
        help="Generate roc file. Skip training.")
    parser.add_argument(
        "--eval_only",
        action='store_true',
        help="Skip training. Only evaluate the testing set.")
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=-1,
        metavar="",
        help="Number of steps to sample decoder.")
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
        default=64,
        metavar="",
        help="Number of concurrent calls to "
             "the map function for dataset generation (CPU-intensive)")
    parser.add_argument(
        "--take",
        type=float,
        default=-1,
        metavar="",
        help="Training ratio.")
    return parser


def main(cls):
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    cls.set_flags(main_parser)
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
                        # operation_timeout_in_ms=60000,
                        log_device_placement=FLAGS.log_device_placement)):
                from tensorflow.python.client import device_lib
                local_device_protos = device_lib.list_local_devices()
                for x in local_device_protos:
                    log('Found device {} of type {}', x.name, x.device_type)
                train(cls, FLAGS)
