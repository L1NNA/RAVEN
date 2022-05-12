import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from collections import namedtuple
from pprint import pprint
import pickle
from tqdm import tqdm
from typing import NamedTuple
import os

DS = NamedTuple("DS", [("tfds", tf.data.Dataset), ("padded_shape", list)])


def _feature(vals, clz):
    if not isinstance(vals, list):
        vals = [vals]
    if clz is int:
        return tf.train.Feature(int64_list=tf.train.Int64List(
            value=[int(v) for v in vals]))
    elif clz is float:
        return tf.train.Feature(float_list=tf.train.FloatList(
            value=[float(v) for v in vals]))
    elif clz is str:
        return tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[str(v).encode() for v in vals]))
    else:
        print('TFR ERROR: uknown value type for {}'.format(clz))
        return None


def _bytes_feature(vals):
    if not isinstance(vals, list):
        vals = [vals]
    return tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[v for v in vals]))


def convert_to_tfr(obj, meta):
    ctx = {}
    seq = {}
    types, shapes = meta['types'], meta['shapes']
    for k in obj:
        val = obj[k] if isinstance(obj[k], list) else [obj[k]]
        if len(shapes[k]) < 2:
            ctx[k] = _feature(val, types[k])
        elif len(shapes[k]) == 2:
            seq[k] = tf.train.FeatureList(
                feature=[_feature(ls, types[k]) for ls in val])
        elif len(shapes[k]) == 3:
            # convert to bytes first:
            dtype = np.float32 if types[k] is float else np.int32
            seq[k] = [_bytes_feature(
                [np.array(ls1).astype(dtype).tobytes()
                 for ls1 in ls0]) for ls0 in val]
            seq[k] = tf.train.FeatureList(feature=seq[k])
        else:
            print('VFR ERROR: unsupported shape', shapes[k], 'for key', k)
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=seq),
        context=tf.train.Features(feature=ctx)
    )


def write_objs_as_tfr(objs, file_path, meta_suffix='.meta',
                      num_samples_to_infer_shape=100):
    """
    Convert a dict of nested list into TFR records. Maximum depth is 3.
    Rank-3 lists, if any, need to have same length (as seqnuence of sequence
    of fixed length vector)
    See below test code for example
    :param objs:
    :param file_path:
    :param meta_suffix:
    :param num_samples_to_infer_shape:
    :return:
    """
    meta = None
    cache_for_shape_inferral = []
    with tf.python_io.TFRecordWriter(file_path) as writer:
        for ind, obj in enumerate(objs):
            if meta is None:
                if ind < num_samples_to_infer_shape:
                    cache_for_shape_inferral.append(obj)
                else:
                    meta = infer_shape(cache_for_shape_inferral)
                    for k in sorted(meta['shapes']):
                        print('INFO: {}\'s inferred shape is {} '
                              'and type is {}'.format(
                                  k, meta['shapes'][k], meta['types'][k]))
                    for c_obj in cache_for_shape_inferral:
                        rd = convert_to_tfr(c_obj, meta)
                        writer.write(rd.SerializeToString())
                    cache_for_shape_inferral.clear()
            else:
                rd = convert_to_tfr(obj, meta)
                writer.write(rd.SerializeToString())
        if meta is None:
            meta = infer_shape(cache_for_shape_inferral)
            for k in sorted(meta['shapes']):
                print(
                    'INFO: {}\'s inferred shape is {} and type is {}'.format(
                        k, meta['shapes'][k], meta['types'][k]))
            for c_obj in cache_for_shape_inferral:
                rd = convert_to_tfr(c_obj, meta)
                writer.write(rd.SerializeToString())
            cache_for_shape_inferral.clear()
    meta['total_len'] = ind + 1
    with open(file_path + meta_suffix, 'wb') as f:
        pickle.dump(meta, f, protocol=0)


def write_tfr(records, file):
    with tf.python_io.TFRecordWriter(file) as writer:
        for rd in records:
            writer.write(rd.SerializeToString())


def get_writer(file):
    return tf.python_io.TFRecordWriter(file)


def parse_tfr(tfr, meta, int64_to_int32=True,
              attr_exclude=None):
    def get_feature_def(clz, shape):
        if len(shape) == 3:
            return tf.VarLenFeature(tf.string)
        elif len(shape) < 3 and clz is str:
            return tf.VarLenFeature(tf.string)
        elif len(shape) < 3 and clz is int:
            return tf.VarLenFeature(tf.int64)
        elif len(shape) < 3 and clz is float:
            return tf.VarLenFeature(tf.float32)
        else:
            print("TFR ERROR: Unsupport configuration ", clz, shape)
        return None

    shapes = meta['shapes']
    types = meta['types']
    ctx_def = {k: get_feature_def(types[k], shapes[k]) for k in shapes if
               len(shapes[k]) < 2}
    seq_def = {k: get_feature_def(types[k], shapes[k]) for k in shapes if
               len(shapes[k]) >= 2}

    def map_fn(k, val):
        if val.dtype == tf.string and types[k] in (int, float):
            npdtype = np.float32 if types[k] is float else np.int32
            tfdtype = tf.float32 if types[k] is float else tf.int32
            if shapes[k][-1]:
                val0 = tf.sparse_tensor_to_dense(
                    val, default_value=np.zeros([shapes[k][-1]],
                                                dtype=npdtype).tobytes())
                actual_shape = [shapes[k][i] or tf.shape(val0)[i] for i in
                                range(len(shapes[k]))]
                val1 = tf.reshape(val0, [-1])
                val = tf.map_fn(lambda x: tf.decode_raw(x, tfdtype), val1,
                                dtype=tfdtype)
                val = tf.reshape(val, actual_shape)
            else:
                val0 = tf.sparse_tensor_to_dense(
                    val, default_value=np.zeros([1], dtype=npdtype).tobytes())
                val1 = tf.reshape(val0, [-1])
                num_eles = tf.shape(val1)[-1]
                arr0 = tf.TensorArray(
                    dtype=tfdtype, size=num_eles, clear_after_read=True,
                    infer_shape=False)
                arr1 = tf.TensorArray(
                    dtype=tfdtype, size=num_eles, clear_after_read=True,
                    infer_shape=True)

                # decode & determine max sequence length to pad
                def fnl1(i, ls, sp):
                    vec = tf.decode_raw(val1[i], tfdtype)
                    return i + 1, ls.write(i, vec), tf.maximum(sp,
                                                               tf.shape(vec)[
                                                                   -1])

                _, arr0, pd = tf.while_loop(lambda i, *_: i < num_eles, fnl1,
                                            [0, arr0, -1])

                # decode
                def fnl2(i, ls1):
                    ele = arr0.read(i)
                    return i + 1, ls1.write(i, tf.pad(ele, [
                        [0, pd - tf.shape(ele)[-1]]]))

                _, arr1 = tf.while_loop(lambda i, *_: i < num_eles, fnl2,
                                        [0, arr1])
                val = arr1.stack()
                actual_shape = [shapes[k][i] or tf.shape(val0)[i] for i in
                                range(len(shapes[k]) - 1)]
                actual_shape.append(pd)
                val = tf.reshape(val, actual_shape)
        elif types[k] is str:
            val = tf.sparse_tensor_to_dense(
                val, default_value=b'')
        else:
            val = tf.sparse_tensor_to_dense(val)
            actual_shape = [shapes[k][i] or tf.shape(val)[i] for i in
                            range(len(shapes[k]))]
            val = tf.reshape(val, actual_shape)
        if int64_to_int32 and val.dtype is tf.int64:
            val = tf.cast(val, tf.int32)
        return val

    ctx_p, seq_p = tf.parse_single_sequence_example(
        serialized=tfr,
        sequence_features=seq_def,
        context_features=ctx_def
    )
    merged = {**ctx_p, **seq_p}
    if attr_exclude is not None:
        for k_id in attr_exclude:
            if k_id in merged:
                del merged[k_id]
    merged = {k: map_fn(k, merged[k]) for k in merged}
    return merged


def read_objs_trf_ds(file_path, parallelism=16, meta_suffix='.meta',
                     attr_exclude=None, meta=None) -> DS:
    if os.path.isdir(file_path):
        file_path = [os.path.join(file_path, f)
                     for f in os.listdir(file_path)]
    if not isinstance(file_path, list):
        file_path = [file_path]
    ds = tf.data.TFRecordDataset(
        file_path, num_parallel_reads=parallelism)
    if meta is None:
        with open(file_path[0] + meta_suffix, 'rb') as f:
            meta = pickle.load(f)

    ds = ds.map(lambda x: parse_tfr(x, meta, True,
                                    attr_exclude),
                num_parallel_calls=parallelism)
    return DS(ds, meta['shapes'])


def read_objs_trf_in_mem(meta, parallelism=16, attr_exclude=None) -> DS:
    tfrs_holder = tf.placeholder(tf.string, shape=[None])
    ds = tf.data.Dataset.from_tensor_slices(tfrs_holder)
    ds = ds.map(lambda x: parse_tfr(x, meta, True,
                                    attr_exclude),
                num_parallel_calls=parallelism)
    return DS(ds, meta['shapes']), tfrs_holder


def infer_shape(objs_to_infer, show_progress=False):
    objs_to_infer = tqdm(objs_to_infer) if show_progress else objs_to_infer
    keys = {k for obj in objs_to_infer for k in obj}
    shapes = {k: [] for k in keys}
    types = {}
    for k in keys:
        vals = [obj[k] if isinstance(obj[k], list) else [obj[k]] for obj in
                objs_to_infer]
        while isinstance(vals[0], list):
            shapes[k].append(-1)
            for ls in vals:
                if len(ls) < 1:
                    continue
                if shapes[k][-1] == -1:
                    shapes[k][-1] = len(ls)
                elif shapes[k][-1] != len(ls):
                    shapes[k][-1] = None
            vals = [ele for val in vals for ele in val]
        types[k] = type(vals[0])
    return {'types': types, 'shapes': shapes}


def meta_to_palceholder(ds_meta, int_as_int32=True):
    phs = {}
    for var in ds_meta['types']:
        var_type = ds_meta['types'][var]
        var_shape = ds_meta['shapes'][var]
        if var_type is int:
            if int_as_int32:
                tf_type = tf.int32
            else:
                tf_type = tf.int64
        elif var_type is float:
            tf_type = tf.float32
        elif var_type is str:
            tf_type = tf.string
        if len(var_shape) == 1 and var_shape[0] == 1:
            var_shape = ()
        phs[var] = tf.placeholder(name=var, shape=var_shape, dtype=tf_type)
    return phs


def test():
    tf.enable_eager_execution()

    objs = [{
        'att1': 0,
        'att2': [1, 2, 3, 4],
        'att3': [[1], [2, 2], [3, 3, 3]],
        'att4': [1.0, 2.0, 3.0, 4.0, 5.0],
        'att5': [[[1, 1, 1]], [[2, 2, 2], [2, 2, 2]],
                 [[3, 3, 3], [3, 3, 3], [3, 3, 3]]],
        'att6': [[[1]], [[2, 2], [2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]],
        'att7': 'abcdefg',
        'att8': ['11111', '22222']
    }, {
        'att1': 1,  # will be detected as fix length vector of [1]
        'att2': [2, 3, 4, 5, 6],
        # will be detected as var length sequence [None]
        'att3': [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]],
        # sequence of sequence [None, None]
        'att4': [1.0, 2.0, 3.0, 4.0, 5.0],
        # will be detected has fixed length vector (same shape for all objects)
        # [5]
        'att5': [[[1, 1, 1]], [[2, 2, 2], [2, 2, 2]]],
        # sequence of sequence of fixed length vector
        'att6': [[[1]], [[2, 2, 2], [2, 2, 2]], [[4, 4, 4, 4]]],
        # sequence of sequence of sequence
        'att7': 'efghijk',
        # support string, sequence of string, sequence of sequence of string
        'att8': ['11111', '22222', '33333333333333']
    }]

    write_objs_as_tfr(objs, 'test.protobuf')
    ds, pd_shapes = read_objs_trf_ds('test.protobuf')
    for ind, sam in enumerate(tfe.Iterator(ds)):
        pprint("### {}".format(ind + 1))
        pprint(sam)

    ds = ds.padded_batch(2, pd_shapes)
    for ind, bat in enumerate(tfe.Iterator(ds)):
        pprint("### bat {}".format(ind + 1))
        pprint(bat)


if __name__ == '__main__':
    test()
