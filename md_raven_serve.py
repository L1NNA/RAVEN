from tensorflow.python.util import nest
import tfutils as tfu
from tfutils import log, log_warn
from dps import read_tagged_json
import md_raven_ds as dps
import md_raven
from md_raven import Raven
import tensorflow as tf
import tfr
import argparse
import os
import functools
from subprocess import call
import sys
from flask import Flask, jsonify, abort, request, make_response, url_for, send_file
from functools import wraps
from werkzeug.utils import secure_filename
from threading import Lock
import hashlib
import datetime
from md_raven_train import make_batch
from tfutils import log, log_err, log_warn
from timeit import default_timer as timer
from flask_basicauth import BasicAuth


FNULL = open(os.devnull, 'w')
SRC = os.path.split(
    os.path.realpath(__file__))[0]
IDA = os.path.join(SRC, 'idas.py')
TMP = os.path.join(SRC, 'tmp')
WEB = os.path.join(SRC, 'web')
SAM = os.path.join(TMP, 'sam')
lock = Lock()
global_serving_fn = None
app = Flask(__name__, static_url_path='', static_folder=WEB)
app.config['BASIC_AUTH_USERNAME'] = 'bb8'
app.config['BASIC_AUTH_PASSWORD'] = 'starwarz'
app.config['BASIC_AUTH_FORCE'] = True
basic_auth = BasicAuth(app)



def prepare_serve(FLAGS):
    log('startup argvs:')
    fdict = vars(FLAGS)
    for k in sorted(fdict):
        log('  {} -- {}', k, fdict[k])

    def process_file_cache(file):
        if file.endswith('.json'):
            return read_tagged_json(file)
        js_file = os.path.join(
            os.path.dirname(file),
            file + '.tagged.json')
        if not os.path.exists(js_file):
            call(
                ['ida64', '-A', '-S{}'.format(IDA),  file],
                stdout=FNULL,
                stderr=sys.stderr)
        # if os.path.exists(file + '.i64'):
        #     os.remove(file + '.i64')
        return read_tagged_json(js_file)

    ps = dps.get_pps(FLAGS.ps_folder)
    ds, tfrs_holder = tfr.read_objs_trf_in_mem(
        ps.ds_meta, attr_exclude=['rsk'])
    ds = make_batch(ds.tfds, ds.padded_shape)
    # placeholders = tfr.meta_to_palceholder(ps.ds_meta)
    # del placeholders['rsk']
    # for pl in placeholders:
    #     log('Serving placeholder: {}:{}', pl, placeholders[pl])

    model = Raven(
        preprocessor=ps,
        obj_lookup_fn=None,
        **fdict)
    sess_run_fn = model.get_serving_fn_graph_tfr(
        model.sampling, ds, tfrs_holder)

    def serving_fn(file, file_name):
        with lock:
            obj = process_file_cache(file)
            tfr = ps.model_obj(obj, as_tfr=True)
            data = sess_run_fn(tfr)
            result = model.sampling_json(
                data, 0, 0, input_obj=obj, save_to_file=False)
            return result

    return serving_fn


def allowed_file(filename):
    if not '.' in filename:
        return True
    if filename.rsplit('.', 1)[1].lower() not in ['html']:
        return True
    return False


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


@app.errorhandler(403)
def unauthorized():
    return make_response(jsonify({'msg': 'Unauthorized access'}), 403)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'msg': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'msg': 'Not found'}), 404)


def err(msg, code=403):
    return make_response(jsonify({'msg': msg}), code)


def ok(msg):
    return make_response(jsonify({'msg': msg}), 200)


@app.route('/analyze', methods=['POST'])
def push():
    # if 'api' not in request.args:
    #     return err('No API key')
    # r_api = request.args['api'].strip()
    # if not r_api == api.strip():
    #     return err('API :(')
    # check if the post request has the file part
    r_api = 'any'

    if 'file' not in request.files:
        return err('No file part')
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        return err('No selected file')
    if file and allowed_file(file.filename):
        sha256 = hashlib.sha256(file.read()).hexdigest()
        file.seek(0)
        filename = secure_filename(file.filename)
        if filename.endswith('.json'):
            fullname = os.path.join(
                TMP,
                filename)
        else:
            fullname = os.path.join(
                TMP,
                sha256)
            if not os.path.exists(os.path.dirname(fullname)):
                os.makedirs(os.path.dirname(fullname))
        file.save(fullname)
        date = datetime.datetime.utcnow()
        log('{} submits {} name {}', r_api, sha256, filename)
        start = timer()
        obj = global_serving_fn(fullname, filename)
        end = timer()
        if len(obj[0]) < 1:
            return err('Analaysis failed. Please contact us.')
        data = {
            'obj': obj[0],
            'time': end-start,
            'name': filename}
        log('{} finishes {} in {}s', r_api, sha256, end-start)
        return jsonify(data)


@app.route('/example', methods=['POST', 'GET'])
def example():
    if request.method == 'POST':
        # if 'api' not in request.args:
        #     return err('No API key')
        # r_api = request.args['api'].strip()
        # if not r_api == api.strip():
        #     return err('API :(')
        # check if the post request has the file part
        r_api = 'any'

        log('here')
        filename = request.json['file_name']
        if filename is None:
            return err('Noe file name found.')
        filename = secure_filename(filename + '.json')
        fullname = os.path.join(SAM, filename)
        if not os.path.exists(fullname):
            return err('Example not found.')
        date = datetime.datetime.utcnow()
        log('{} requests example {}', r_api, filename)
        start = timer()
        obj = global_serving_fn(fullname, filename)
        end = timer()
        if len(obj[0]) < 1:
            return err('Analaysis failed. Please contact us.')
        data = {
            'obj': obj[0],
            'time': end-start,
            'name': filename}
        log('{} finishes {} in {}s', r_api, filename, end-start)
        return jsonify(data)
    else:
        files = os.listdir(SAM)
        files = [os.path.splitext(f)[0] for f in files]
        return jsonify(files)


@app.route('/')
def root():
    return app.send_static_file('querywy.html')


def set_flags(main_parser):
    Raven.set_flags(main_parser)
    main_parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        metavar="",
        help="Verbosity for testing. 0 skips the test run.")
    main_parser.add_argument(
        "--ps_folder",
        type=str,
        metavar="",
        default="serve/preprocessor",
        help="The data directory containing .tagged.json files.")
    main_parser.add_argument(
        "--help",
        action='store_true',
        help='display the usage.')
    main_parser.add_argument(
        "--eager_mode",
        action='store_true',
        help="Enable eager mode.")
    main_parser.add_argument(
        "--port",
        type=int,
        default=8570,
        metavar="",
        help="The binding port.")


def init():
    main_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
    set_flags(main_parser)
    FLAGS, unparsed = main_parser.parse_known_args()
    tfu.LOG_VERBOSITY = FLAGS.verbose
    if FLAGS.help:
        main_parser.print_help()
    else:
        if len(unparsed) > 0:
            log_warn("Uknown arguments: {}", unparsed)
        if FLAGS.eager_mode:
            prepare_serve(FLAGS)
        else:
            with tf.Session():
                global global_serving_fn
                global_serving_fn = prepare_serve(FLAGS)
               
#                 fs = \
#                   ['experiment\\cuckoo\\data\\f951a942b84eaf88d88de2e2b384700fe2551691e564bea4449a1b43f3d4a95a.tagged.json',
#  'experiment\\cuckoo\\data\\3d33693db7e67b6d3937f140d1e09a4d6c19a0266389c89ef63eb68dfeb61821.tagged.json',
#  'experiment\\cuckoo\\data\\7f0c9625f22e06605140ef4a38cd8cf92e7411149fac4a9a647fb34715b1f4c3.tagged.json',
#  'experiment\\cuckoo\\data\\8ae59b74f2fbaf64c3911018652c9dd5ccdf4d3d51c27b8489be15b2c78134ef.tagged.json',
#  'experiment\\cuckoo\\data\\0bb08bb7e645c05c1cdfe0f10c77db928fc935d51d16329a90c71c664c2062e2.tagged.json',
#  'experiment\\cuckoo\\data\\1447486b500a35b9cbd43f98f67b51ab89c1e9aa051c19278494d85f9379016f.tagged.json',
#  'experiment\\cuckoo\\data\\15e7107ce7d21127f948fa93164e773872c3556c295adf7564bd397d6f2308e1.tagged.json',
#  'experiment\\cuckoo\\data\\20c0600db2f6caad8341c90902fe90224e12a0777ff6c2615706eb6a7cbff7a2.tagged.json',
#  'experiment\\cuckoo\\data\\50fb7ad878e6741332576609eda2c8f653f3f1dffe759ea458f838a8a0e2fc80.tagged.json',
#  'experiment\\cuckoo\\data\\5f3598e230c636e06eb292d7cb7180ad3d94d5c96c1f002d2d6c7fde198e0dda.tagged.json',
#  'experiment\\cuckoo\\data\\62b7deb73f1e290a83fdfc2ec86cbea7559fffa0cfdf72c03b1a034c2a8cbf24.tagged.json',
#  'experiment\\cuckoo\\data\\8bb575fca11329fae724b2cabe74d490c1c642be585984fb400cde4e940412c2.tagged.json',
#  'experiment\\cuckoo\\data\\07409049b5043a9a391ddec8a10726041b01eb0d3d41e39b8ca4e8ff3ef6d114.tagged.json',
#  'experiment\\cuckoo\\data\\772f068716bf83e45740a1442617e076796f35e8a12f0bbed042a96edc91c800.tagged.json',
#  'experiment\\cuckoo\\data\\2a72f4eaabf38e3cf1ed0c482c9c78e19b38489c1267ff3a0048e34fcb06178e.tagged.json',
#  'experiment\\cuckoo\\data\\99f48ddda34479425a1629d039636603b49a208aa7d024b202f19df52a224145.tagged.json',
#  'experiment\\cuckoo\\data\\4698b8ab7eff7cacec474d0931feacf2ac93e18b90cde8b25d82301ac05edba9.tagged.json',
#  'experiment\\cuckoo\\data\\6f93a67bdf814632f13d87a52e66e7ab8c9f4f52b4da0afaf5131ebcd89a34ad.tagged.json',
#  'experiment\\cuckoo\\data\\13d0aec15f746f07906002111a91471d2ca0bac4c170fb7c5962488b96f1cebd.tagged.json',
#  'experiment\\cuckoo\\data\\8e3473db0a950e3db37522c1b780fc7d82a9a2d72291782e16bd47842da2095c.tagged.json']
#                 import shutil
#                 for f in fs:
#                     print(f)
#                     r = global_serving_fn(
#                         f,
#                         os.path.basename(f))
#                     shutil.copy(f, os.path.join(SAM, os.path.basename(f)))
#                     for rsk in r[0]['rsk']:
#                         print(rsk['rsk_key'])
                app.run(
                    host= '0.0.0.0',
                    debug=True,
                    port=FLAGS.port,
                    # use_reloader=False
                    )


if __name__ == '__main__':
    init()
