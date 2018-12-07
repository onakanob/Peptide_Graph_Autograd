import os
import csv
import random
import autograd.numpy as np
import autograd.numpy.random as rand
import itertools as it


def load_csv_test_split(filename, test_size, rand_seed, input_name, target_name, conditions):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        reader.next()
        for row in reader:
            add_row = True
            for key in conditions.keys():
                if row[key] != conditions[key]:
                    add_row = False
            if add_row:
                data[0].append(row[input_name])
                data[1].append(float(row[target_name]))
    data = np.array(data)
    rand.seed(rand_seed)
    sequence = rand.choice(data[0].size, data[0].size, replace=False)
    testset = (data[0][sequence[:int(data[0].size * test_size)]],
                data[1][sequence[:int(data[0].size * test_size)]])
    trainset = (data[0][sequence[int(data[0].size * test_size):]],
               data[1][sequence[int(data[0].size * test_size):]])
    rand.seed()
    print 'Loaded', trainset[0].size, 'training points;', testset[0].size, 'test points.'
    return trainset, testset


def load_data_csv(filename, sizes, input_name, target_name, conditions):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_from_csv(filename, slices, np.sum(sizes), input_name,
                              target_name, conditions)


def load_from_csv(filename, slices, sub_samples, input_name, target_name,
                       conditions):
    NUM_SAMPLES = 4155          # TODO detect dynamically
    sub_sample = random.sample(range(NUM_SAMPLES), sub_samples)

    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        reader.next()
        for row in reader:
            for key in conditions.keys():
                if row[key] == conditions[key]:
                    data[0].append(row[input_name])
                    data[1].append(float(row[target_name]))
    data = map(np.array, data)
    return [(data[0][sub_sample[s]], data[1][sub_sample[s]]) for s in slices]


def list_concat(lists):
    return list(it.chain(*lists))


def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)


def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)


def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))


def data_dir():
    return os.path.expanduser(safe_get("DATA_DIR"))


def safe_get(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception("%s environment variable not set" % varname)
