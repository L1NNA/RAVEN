import argparse
from collections import namedtuple
import os
import pickle
import json


def json_load(fobj):
    if isinstance(fobj, str):
        with open(fobj, 'r') as fo:
            return json.load(fo)
    else:
        return json.load(fobj)


def load_raw_ds(ds_folder):
    meta_file = os.path.join(
        ds_folder,
        '..',
        os.path.basename(ds_folder) + '.json')
    
    protocol = json_load(meta_file)
    return protocol[4], protocol[5], protocol[6], protocol[2]


def set_flags(oparser):
    parser = oparser.add_argument_group('data source')
    parser.add_argument(
        "--ds_folder",
        type=str,
        default="data/small",
        metavar="",
        help="The data directory containing .tagged.json files.")
