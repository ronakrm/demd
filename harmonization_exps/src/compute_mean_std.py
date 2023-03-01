import argparse
import numpy as np
import os
import pdb
import re
import tensorflow as tf


def get_logfiles(logdir):
    logs = [os.path.join(logdir, log) for log in os.listdir(logdir) if log.startswith('event')]
    return logs


def get_tag_values(logfiles, tagname):
    # Returns the epoch values the tags
    value_dict = {}
    for log in logfiles:
        for event in tf.train.summary_iterator(log):
            for val in event.summary.value:
                if val.tag == tagname:
                    value_dict[int(event.step)] = val.simple_value
    value_list = []
    keys = value_dict.keys()
    sorted(keys)
    for key in keys:
        value_list.append(value_dict[key])
    if len(value_list) == 0:
        return -1
    return value_list[-1]


def get_relevant_dirs(args):
    run_regex = re.compile(args.run_regex)
    relevant_runs = [os.path.join(args.run_dir, run)
                     for run in os.listdir(args.run_dir) if run_regex.match(run)]
    return relevant_runs


def get_mean_std(relevant_runs, tagname):
    arr = []
    for dir in relevant_runs:
        logs = get_logfiles(dir)
        item = get_tag_values(logs, tagname)
        if item > 0:
            print(dir, tagname, item)
            arr.append(item)
    arr = np.array(arr)
    print('{}: Count: {}, Mean: {}, Std: {}'.format(tagname, len(arr), arr.mean(), np.std(arr)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_regex', type=str, default='none.*')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    args = parser.parse_args()

    relevant_runs = get_relevant_dirs(args)
    for tag in args.tags.split(','):
        get_mean_std(relevant_runs, tag)


if __name__ == "__main__":
    main()