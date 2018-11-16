import os
import csv
import random
import numpy as np
import itertools as it

def load_data_csv(filename, sizes, input_name, target_name, conditions):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
#    return load_data_slices(filename, slices, input_name, target_name, conditions)
    return load_from_csv_TEMP(filename, slices, np.sum(sizes), input_name, target_name, conditions)

def load_data_slices(filename, slices, input_name, target_name, conditions):
    # TODO get rid of this slicing nonsense and just select 
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name, conditions)
    return [(data[0][s], data[1][s]) for s in slices]

def read_csv(filename, nrows, input_name, target_name, conditions):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        reader.next()
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def load_from_csv_TEMP(filename, slices, sub_samples, input_name, target_name, conditions):
    NUM_SAMPLES = 4155 # TODO detect dynamically
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

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]


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
