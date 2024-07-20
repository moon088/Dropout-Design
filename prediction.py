from __future__ import print_function

import argparse
import cloudpickle
import csv
import glob
import itertools
import numpy as np
import os
import re
import scipy as sp

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import *
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.engine.base_layer import Layer
from keras.callbacks import Callback
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import training

def read(file):
    with open(file, 'r') as f:
        print(''.join(['read ', str(f)]))
        reader = csv.reader(f)
        for r in reader:
            yield r

def remove(path):
    for f in glob.glob(path):
        print(''.join(['removed ', str(f)]))
        os.remove(f)

def write(file, data):
    with open(file, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        for v in data:
            writer.writerow(v)
    print(''.join(['saved ', str(file)]))

def evaluate_model(h5):
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    models_path = os.path.abspath(src_path + '/models')
    base, ext = os.path.splitext(os.path.basename(h5))
    if ext != '.h5':
        return
    model = keras.models.load_model(h5, custom_objects={'DropoutDesignLayer': training.DropoutDesignLayer})
    print(''.join(['loaded ', str(h5)]))
    _, net, nlayers, nnodes, nalives, dropout, total_epochs, total_exps, dataset_name, nepochs, nexps, _ = tuple(re.split(r'model_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)', base))
    info = [net, nlayers, nnodes, nalives, dropout, total_epochs, total_exps, dataset_name, nepochs, nexps]
    print(', '.join(info))
    _, base_body, _ = tuple(re.split(r'model_(.+)', base))
    write(models_path + '/info.csv', [info])

    # load dataset
    if dataset_name == 'cifar10':
        dataset = training.Cifar10Dataset()
    else:
        print('unsupported dataset:' + dataset_name)
        assert False
            
    if net == 'mlp':
        x_train = dataset.x_train_reshaped
        y_train = dataset.y_train
        x_test = dataset.x_test_reshaped
        y_test = dataset.y_test
    elif net == 'cnn':
        x_train = dataset.x_train
        y_train = dataset.y_train
        x_test = dataset.x_test
        y_test = dataset.y_test
    else:
        print('unsupported network:' + net)
        assert False

    # predict training data
    prediction_training_path = models_path + '/prediction_training_' + base_body + '.csv'
    prediction_training = model.predict(x_train, batch_size=len(x_train), verbose=0)
    remove(prediction_training_path)
    write(prediction_training_path, prediction_training)

    # # evaluate training data
    # evaluations_training = [model.evaluate(x=np.array([x_train[i]]), y=np.array([y_train[i]]), batch_size=1, verbose=0)  for i in range(len(x_train))]
    # evaluation_training_path = models_path + '/evaluation_training_' + base_body + '.csv'
    # remove(evaluation_training_path)
    # write(evaluation_training_path, [evaluations_training])
    # print(model.metrics_names)

    evaluation_training_total = model.evaluate(x=x_train, y=y_train, verbose=0)
    evaluation_training_total_path = models_path + '/evaluation_training_total_' + base_body + '.csv'
    remove(evaluation_training_total_path)
    write(evaluation_training_total_path, [evaluation_training_total])

    # predict test data
    prediction_test = model.predict(x_test, verbose=0)
    prediction_test_path = models_path + '/prediction_test_' + base_body + '.csv'
    remove(prediction_test_path)
    write(prediction_test_path, prediction_test)

    # # evaluate test data
    # evaluations_test = [model.evaluate(x=np.array([x_test[i]]), y=np.array([y_test[i]]), batch_size=1, verbose=0)  for i in range(len(x_test))]
    # evaluation_test_path = models_path + '/evaluation_test_' + base_body + '.csv'
    # remove(evaluation_test_path)
    # write(evaluation_test_path, evaluations_test)
    # print(model.metrics_names)

    evaluations_test_total = model.evaluate(x=x_test, y=y_test, verbose=0)
    evaluation_test_total_path = models_path + '/evaluation_test_total_' + base_body + '.csv'
    remove(evaluation_test_total_path)
    write(evaluation_test_total_path, [evaluations_test_total])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', type=str, default=None, help='model file')
    args = parser.parse_args()
    if args.eval is not None:
        evaluate_model(args.eval)