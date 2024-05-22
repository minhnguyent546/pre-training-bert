"""
Prepare train and test set for training
"""

import argparse

from datasets import load_dataset

import opts
import utils


def preprocess(data_files, test_size: int, data_save_path: str):
    raw_dataset = load_dataset('json', data_files=data_files)

    dataset = raw_dataset['train'].train_test_split(test_size=test_size, shuffle=True)
    utils.ensure_dir(data_save_path)
    dataset.save_to_disk(data_save_path)

def main():
    parser = argparse.ArgumentParser(
        description='Preparing train and test set for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    opts.prepare_dataset_opts(parser)
    args = parser.parse_args()

    utils.set_random_seed(args.seed)

    preprocess(args.data_file, args.test_size, args.data_save_path)


if __name__ == '__main__':
    main()
