import argparse
import dps

def load_raw_ds(ds_folder, min_freq):
    protocol = dps.split_and_label(
        ds_folder, rsk_min_freq=min_freq)
    return protocol.trn, protocol.vld, protocol.tst, protocol.rsk_rsk2id


def set_flags(oparser):
    parser = oparser.add_argument_group('data source')
    parser.add_argument(
        "--ds_folder",
        type=str,
        default="data/small",
        metavar="",
        help="The data directory containing .tagged.json files.")
    dps.set_flags(parser)
