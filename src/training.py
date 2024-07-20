from __future__ import print_function

from abc import ABCMeta, abstractmethod
import argparse
import cloudpickle
import csv
import glob
import itertools
import json
import numpy as np
import os

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import *
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.engine.base_layer import Layer
from keras.callbacks import Callback
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from keras.wrappers.scikit_learn import KerasClassifier

import PGDesign
from PGDesign import *

class Lazy:
    def __init__(self, getter):
        self.__value = None
        self.__getter = getter
    @property
    def _value(self):
        if self.__value is None:
            self.__value = self.__getter()
        return self.__value

#
# Dropout Design
#

class DesignGenerator(metaclass=ABCMeta):
    @property
    @abstractmethod
    def design(self): pass
    @property
    @abstractmethod
    def dropout_rate(self): pass
    @property
    @abstractmethod
    def keep_rate(self): return self._value[4]
    @property
    @abstractmethod
    def num_batches(self): pass
    @property
    @abstractmethod
    def num_drop(self): pass
    @property
    @abstractmethod
    def num_layers(self): pass
    @property
    @abstractmethod
    def num_units(self): pass

class PGDesignGenerator(DesignGenerator, Lazy):
    '''Dropout Design Generator'''
    @classmethod
    def __make_design(cls, d, p, t):
        _, dd = dropout_design_1_2(d, p, t, True)
        return dropout_design_1_2_to_dropout(list(dd))
    def __init__(self, d, p, t):
        def getter():
            num_units, num_alive, _, num_layers, num_batches = to_params_dropout_design_1_2(d, p, t)
            dropout_rate = min(1., max(0., 1.0 - num_alive / num_units))
            design = self.__make_design(d, p, t)
            return num_units, (num_units - num_alive), num_layers, num_batches, dropout_rate, design
        super(self.__class__, self).__init__(getter)
    @property
    def design(self): return self._value[5]
    @property
    def dropout_rate(self): return self._value[4]
    @property
    def keep_rate(self): return 1.0 - self.dropout_rate
    @property
    def num_batches(self): return self._value[3]
    @property
    def num_drop(self): return self._value[1]
    @property
    def num_layers(self): return self._value[2]
    @property
    def num_units(self): return self._value[0]

class FileDesignGenerator(DesignGenerator, Lazy):
    '''Dropout Design Generator. The design is loaded from the specified design file.'''
    @classmethod
    def __read_design(cls, filename):
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            num_layers = 0
            num_units = cls.__units_num(filename)
            design = list()
            for _, row in enumerate(csvreader):
                layers = [[int(j) for j in r.strip('{}').split(', ')] for r in row]
                design.append(layers)
                if len(layers) > num_layers:
                    num_layers = len(layers)
        return design, len(design), num_layers, num_units, len(design[0][0])
    @classmethod
    def __units_num(cls, filename):
        num_unit = 0
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for _, row in enumerate(csvreader):
                for r in row:
                    units = [int(j) for j in r.strip('{}').split(', ')]
                    nu = max(units)
                    if nu > num_unit:
                        num_unit = nu
        return num_unit + 1
    def __init__(self, filename):
        def getter():
            design, num_batches, num_layers, num_units, num_drop = self.__read_design(filename)
            dropout_rate = min(1., max(0., num_drop / num_units))
            return num_units, num_drop, num_layers, num_batches, dropout_rate, design
        super(self.__class__, self).__init__(getter)
    @property
    def design(self): return self._value[5]
    @property
    def dropout_rate(self): return self._value[4]
    @property
    def keep_rate(self): return 1.0 - self.dropout_rate
    @property
    def num_batches(self): return self._value[3]
    @property
    def num_drop(self): return self._value[1]
    @property
    def num_layers(self): return self._value[2]
    @property
    def num_units(self): return self._value[0]

class DropoutDesign:
    ''' Applying Dropout Design to fully connected layers '''
    def __init__(self, design_generator, update_method):
        self.__design_generator = design_generator
        self.__update_method_name = update_method
        self.__update_method = None
        self.__design = None
        self.__dropout = None
    def __load_design(self):
        if self.__update_method_name == 'shift_var':
            self.__update_method = shift_dropout_varieties
            self.__design = self.__design_generator.design
        elif self.__update_method_name == 'shift_block':
            self.__update_method = shift_dropout_blocks
            self.__design = self.__design_generator.design
        elif self.__update_method_name == 'rand_var':
            self.__update_method = rand_dropout_varieties
            self.__design = self.__update_method(self.__design_generator.design)
        elif self.__update_method_name == 'rand_block':
            self.__update_method = rand_dropout_blocks
            self.__design = self.__update_method(self.__design_generator.design)
        else:
            self.__update_method = lambda a: a
            self.__design = self.__update_method(self.__design_generator.design)
        self.__dropout = self.__to_dropout(self.__design)
    def __to_dropout(self, design):
        dropout = []
        for batches in design:
            layers = []
            for units_per_layer in batches:
                active_units = np.zeros(self.num_units)
                active_units[units_per_layer] = 1
                layers.append(active_units)
            dropout.append(layers)
        return list(zip(*dropout))
    def update(self):
        if self.__design is None:
            self.__load_design()
        self.__design = self.__update_method(self.__design)
        self.__dropout = self.__to_dropout(self.__design)
    def get_layer(self, layer):
        return self._dropout[layer % self.num_layers]
    def get_batch(self, layer, batch):
        return self._dropout[layer % self.num_layers][batch % self.num_batches]
    @property
    def _design(self):
        if self.__design is None:
            self.__load_design()
        return self.__design
    @property
    def _dropout(self):
        if self.__dropout is None:
            self.__load_design()
        return self.__dropout
    @property
    def dropout_rate(self): return self.__design_generator.dropout_rate
    @property
    def keep_rate(self): return self.__design_generator.keep_rate
    @property
    def num_batches(self): return self.__design_generator.num_batches
    @property
    def num_drop(self): return self.__design_generator.num_drop
    @property
    def num_layers(self): return self.__design_generator.num_layers
    @property
    def num_units(self): return self.__design_generator.num_units

#
# Datasets
#

class Cifar10Dataset:
    ''' Cifar-10 Dataset'''
    name = 'cifar10'
    num_classes = 10
    def __init__(self):
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.train_size = len(x_train)
        self.test_size = len(x_test)
        self.x_train = x_train.astype('float32') / 255.0
        self.x_test = x_test.astype('float32') / 255.0
        self.x_train_shape = reduce(lambda x,y:x*y, self.x_train.shape[1:])
        self.x_test_shape = reduce(lambda x,y:x*y, self.x_test.shape[1:])
        print(''.join([str(self.x_train.shape[0]), 'training samples(shape:', str(self.x_train.shape[1]), 'x', str(self.x_train.shape[2]), 'x', str(self.x_train.shape[3]), '=', str(self.x_train_shape)]))
        print(''.join([str(self.x_test.shape[0]), 'test samples(shape:', str(self.x_test.shape[1]), 'x',  str(self.x_test.shape[2]), 'x', str(self.x_test.shape[3]), '=', str(self.x_test_shape)]))
        # convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(y_train, self.num_classes)
        self.y_test = np_utils.to_categorical(y_test, self.num_classes)
        self.y_train_rounded = np.argmax(self.y_train, axis=1)
        self.y_test_rounded = np.argmax(self.y_test, axis=1)
        self.x_train_reshaped = x_train.reshape(x_train.shape[0], self.x_train_shape) / 255.0
        self.x_test_reshaped = x_test.reshape(x_test.shape[0], self.x_test_shape) / 255.0
        #self.batch_size = train_size // num_batches + 1

class UniformDistDataset:
    ''' Dataset generated from uniform distribution.
    The implementation is a customization of testgenerator/test_data.py. \n
    Note: MLPs only support the dataset. '''
    _error = 0.1
    _seed = 7919
    name = 'uniform'
    num_classes = 10
    num_features = 1000
    train_size = 50000
    test_size = 10000
    num_samples = train_size + test_size
    x_train_shape = num_features
    y_train_shape = num_features
    train_shape = (x_train_shape, y_train_shape)
    @classmethod
    def load_data(cls):
        np.random.seed(cls._seed)

        X = np.random.rand(cls.num_samples, cls.num_features)
        W = np.random.normal(0, 1, (cls.num_features, cls.num_classes))
        E = np.random.normal(0, cls._error, (cls.num_samples, cls.num_classes))
        Y = np.dot(X, W) + E
        Y = np.argmax(Y, axis=1)

        x_train = np.empty((cls.train_size, cls.num_features), dtype='float32')
        y_train = np.empty((cls.train_size,), dtype='uint8')
        x_test = np.empty((cls.test_size, cls.num_features), dtype='float32')
        y_test = np.empty((cls.test_size,), dtype='uint8')

        x_train, x_test = np.split(X, [cls.train_size])
        y_train, y_test = np.split(Y, [cls.train_size])
        return (x_train, y_train), (x_test, y_test)
    def __init__(self):
        (self.x_train, y_train), (self.x_test, y_test) = UniformDistDataset.load_data()
        self.y_train = np_utils.to_categorical(y_train, self.num_classes)
        self.y_test = np_utils.to_categorical(y_test, self.num_classes)
        self.y_train_rounded = np.argmax(self.y_train, axis=1)
        self.y_test_rounded = np.argmax(self.y_test, axis=1)
        self.x_train_reshaped = self.x_train
        self.x_test_reshaped = self.x_test

class UniformDistRegressionDataset:
    ''' Regression Dataset generated from uniform distribution.\n
    Note: MLPs only support the dataset. '''
    _error = 0.1
    _seed = 7919
    name = 'uniform-regression'
    num_classes = 1
    num_features = 1000
    train_size = 50000
    test_size = 10000
    num_samples = train_size + test_size
    x_train_shape = num_features
    y_train_shape = num_features
    train_shape = (x_train_shape, y_train_shape)
    @classmethod
    def load_data(cls):
        np.random.seed(cls._seed)

        X = np.random.rand(cls.num_samples, cls.num_features)
        W = np.random.normal(0, 1, (cls.num_features, cls.num_classes))
        E = np.random.normal(0, cls._error, (cls.num_samples, cls.num_classes))
        Y = np.dot(X, W) + E

        x_train = np.empty((cls.train_size, cls.num_features), dtype='float32')
        y_train = np.empty((cls.train_size,), dtype='float32')
        x_test = np.empty((cls.test_size, cls.num_features), dtype='float32')
        y_test = np.empty((cls.test_size,), dtype='float32')

        x_train, x_test = np.split(X, [cls.train_size])
        y_train, y_test = np.split(Y, [cls.train_size])
        x_train = (x_train - np.average(x_train)) / np.std(x_train)
        x_test = (x_test - np.average(x_test)) / np.std(x_test)
        y_train = (y_train - np.average(y_train)) / np.std(y_train)
        y_test = (y_test - np.average(y_test)) / np.std(y_test)
        return (x_train, y_train), (x_test, y_test)
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = UniformDistRegressionDataset.load_data()
        self.y_train_rounded = self.y_train
        self.y_test_rounded = self.y_test
        self.x_train_reshaped = self.x_train
        self.x_test_reshaped = self.x_test

#
# Callbacks
#

class DropoutScheduler(keras.callbacks.Callback):
    def __init__(self, design):
        super(self.__class__, self).__init__()
        self.__design = design
    def on_epoch_begin(self, epoch, logs=None):
        for m in (m for m in self.model.layers if isinstance(m, keras.callbacks.Callback)):
            m.on_epoch_begin(epoch, logs)
    def on_epoch_end(self, epoch, logs={}):
        self.__design.update()
    def on_batch_begin(self, batch, logs=None):
        for m in (m for m in self.model.layers if isinstance(m, keras.callbacks.Callback)):
            m.on_batch_begin(batch, logs)

class DropoutDesignLayer(Layer, keras.callbacks.Callback):
    ''' Dropout layer with Dropout Design'''
    def __init__(self, layer, design, noise_shape=None, seed=None, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.layer = layer
        self.noise_shape = noise_shape
        self.seed = seed
        if type(design) is str:
            with open(design, 'rb') as f:
                self.__design = cloudpickle.loads(f.read())
        else:
            self.__design = design
        self.keep_rate = self.__design.keep_rate
        self.supports_masking = True
        self.__active = K.variable(self.__get_batch(0), name='design') # dropout mask
    def __get_layer(self):
        return self.__design.get_layer(self.layer)
    def __get_batch(self, batch):
        return self.__design.get_batch(self.layer, batch)
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)
    def call(self, inputs, training=None):
        if 0. < self.keep_rate < 1.:
            # noise_shape = self._get_noise_shape(inputs)
            def dropped_inputs():
                '''Applying dropout masks while training!'''
                return math_ops.divide(inputs, self.keep_rate) * self.__active
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        else:
            return inputs
    def get_config(self):
        designfile = "./design.pk"
        with open(designfile, 'wb') as f:
            cloudpickle.dump(self.__design, f)
        config = {'layer': self.layer, 'noise_shape' :self.noise_shape, 'design': designfile,
                  'seed': self.seed}
        base_config = super(DropoutDesignLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    def compute_output_shape(self, input_shape):
        return input_shape
    def on_epoch_begin(self, epoch, logs=None):
        pass
    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.__active, self.__get_batch(batch))

class Logger(keras.callbacks.Callback):
    '''Logging '''
    def __init__(self, filename, data, num, method):
        super(self.__class__, self).__init__()
        self.filename = filename
        self.data = data
        self.num = num
        self.method = method
    def on_epoch_end(self, epoch, logs=None):
        with open(self.filename, 'a') as f:
            writer = csv.writer(f, lineterminator='\n')
            accuracy = None
            if 'acc' in logs:
                accuracy = logs['acc']
            elif 'accuracy' in logs:
                accuracy = logs['accuracy']
            assert accuracy is not None
            val_accuracy = None
            if 'val_acc' in logs:
                val_accuracy = logs['val_acc']
            elif 'val_accuracy' in logs:
                val_accuracy = logs['val_accuracy']
            assert val_accuracy is not None
            writer.writerow([self.data, self.method, self.num+1, epoch+1, logs['loss'], logs['val_loss'], accuracy, val_accuracy])

#
# Models
#

class Classifier:
    class _ConvolutionLayers:
        @classmethod
        def create(cls, model, input_shape):
            model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.25))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.25))
            model.add(Flatten())
            return model
    class _FullyConnectedLayers:
        @classmethod
        def create(cls, model, num_layers, num_units, dropout_rate = 0):
            for _ in range(num_layers):
                model.add(Dense(num_units, activation='relu'))
                if 0 < dropout_rate:
                    model.add(Dropout(dropout_rate))
            return model
        @classmethod
        def create_input(cls, model, num_layers, num_units, dropout_rate, input_shape):
            model.add(Dense(num_units, activation='relu', input_shape=input_shape))
            if 0 < dropout_rate:
                model.add(Dropout(dropout_rate))
            for _ in range(1, num_layers):
                model.add(Dense(num_units, activation='relu', input_shape=input_shape))
                if 0 < dropout_rate:
                    model.add(Dropout(dropout_rate))
            return model
    class _DropoutDesignFullyConnectedLayers:
        @classmethod
        def create(cls, model, num_layers, num_units, design):
            for i in range(num_layers):
                model.add(Dense(num_units, activation='relu'))
                model.add(DropoutDesignLayer(i, design))
            return model
        @classmethod
        def create_input(cls, model, num_layers, num_units, design, input_shape):
            model.add(Dense(num_units, activation='relu', input_shape=input_shape))
            model.add(DropoutDesignLayer(0, design))
            for i in range(1, num_layers):
                model.add(Dense(num_units, activation='relu'))
                model.add(DropoutDesignLayer(i, design))
            return model
    class _OutputLayer:
        @classmethod
        def create(cls, model, units, activation):
            model.add(Dense(units, activation=activation))
            return model
    class ModelFitter(metaclass=ABCMeta):
        @abstractmethod
        def fit(self): pass
        @abstractmethod
        def evaluate(self): pass
    # Multilayer Perceptrons
    class __MlpSimpleModelBuilder:
        def __init__(self, dataset):
            self._dataset = dataset
        def _create(self, num_layers, num_units, optimizer, loss, output_activation, dropout_rate = 0):
            print(''.join(['__MlpSimpleModelBuilder(num_layers=', str(num_layers), ',  num_units=', str(num_units), ', optimizer=', str(optimizer), ', loss=', str(loss), ', dropout_rate=', str(dropout_rate), ')']))
            model = Sequential()
            model = Classifier._FullyConnectedLayers.create_input(model, num_layers, num_units, dropout_rate, (self._dataset.x_train_shape,))
            model = Classifier._OutputLayer.create(model, self._dataset.num_classes, output_activation)
            model.summary()
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model
    class MlpSimpleModelFitter(ModelFitter, __MlpSimpleModelBuilder):
        def __init__(self, dataset, num_layers, num_units, optimizer, loss, output_activation, dropout_rate, epochs, batch_size, shuffle, callbacks):
            super(self.__class__, self).__init__(dataset)
            self.__model = self._create(num_layers, num_units, optimizer, loss, output_activation, dropout_rate=dropout_rate)
            self.__epochs = epochs
            self.__batch_size = batch_size
            self.__shuffle = shuffle
            self.__callbacks = callbacks
        def fit(self):
            print('Not using data augmentation.')
            self.__model.fit(self._dataset.x_train_reshaped, self._dataset.y_train,
                        batch_size=self.__batch_size,
                        epochs=self.__epochs,
                        verbose=1,
                        validation_data=(self._dataset.x_test_reshaped, self._dataset.y_test),
                        shuffle=self.__shuffle,
                        callbacks=self.__callbacks)
        def evaluate(self):
            return self.__model.evaluate(self._dataset.x_test_reshaped, self._dataset.y_test, verbose=1)
    class __MlpDesignModelBuilder:
        def __init__(self, dataset):
            self._dataset = dataset
        def _create(self, design, optimizer, loss, output_activation):
            print(''.join(['__MlpDesignModelBuilder(design=', str(design), ', optimizer=', str(optimizer), ', loss=', str(loss), ', output_activation=', str(output_activation), ')']))
            model = Sequential()
            model = Classifier._DropoutDesignFullyConnectedLayers.create_input(model, design.num_layers, design.num_units, design, (self._dataset.x_train_shape,))
            model = Classifier._OutputLayer.create(model, self._dataset.num_classes, output_activation)
            model.summary()
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model
    class MlpDesignModelFitter(ModelFitter, __MlpDesignModelBuilder):
        def __init__(self, dataset, design, repeat, optimizer, loss, output_activation, epochs, shuffle, callbacks):
            super(self.__class__, self).__init__(dataset)
            self.__model = self._create(design, optimizer, loss, output_activation)
            self.__epochs = epochs
            self.__batch_size = dataset.train_size // (design.num_batches * repeat) + 1
            self.__shuffle = shuffle
            self.__callbacks = [DropoutScheduler(design)] + callbacks
        def fit(self):
            print('Not using data augmentation.')
            self.__model.fit(self._dataset.x_train_reshaped, self._dataset.y_train,
                        batch_size=self.__batch_size,
                        epochs=self.__epochs,
                        verbose=1,
                        validation_data=(self._dataset.x_test_reshaped, self._dataset.y_test),
                        shuffle=self.__shuffle,
                        callbacks=self.__callbacks)
        def evaluate(self):
            return self.__model.evaluate(self._dataset.x_test_reshaped, self._dataset.y_test, verbose=1)
    # Convolutional Neural Networks
    class __CnnSimpleModelBuilder:
        def __init__(self, dataset):
            self._dataset = dataset
        def _create(self, num_layers, num_units, optimizer, loss, output_activation, dropout_rate = 0):
            print(''.join(['__CnnSimpleModelBuilder(num_layers=', str(num_layers), ',  num_units=', str(num_units), ', optimizer=', str(optimizer), ', loss=', str(loss), ', output_activation=', str(output_activation), ', dropout_rate=', str(dropout_rate), ')']))
            model = Sequential()
            model = Classifier._ConvolutionLayers.create(model, self._dataset.x_train.shape[1:])
            model = Classifier._FullyConnectedLayers.create(model, num_layers, num_units, dropout_rate)
            model = Classifier._OutputLayer.create(model, self._dataset.num_classes, output_activation)
            model.summary()
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model
    class CnnSimpleModelFitter(ModelFitter, __CnnSimpleModelBuilder):
        def __init__(self, dataset, num_layers, num_units, optimizer, loss, output_activation, dropout_rate, epochs, batch_size, data_augmentation, shuffle, callbacks):
            super(self.__class__, self).__init__(dataset)
            self.__model = self._create(num_layers, num_units, optimizer, loss, output_activation, dropout_rate=dropout_rate)
            self.__epochs = epochs
            self.__batch_size = batch_size
            self.__data_augmentation = data_augmentation
            self.__shuffle = shuffle
            self.__callbacks = callbacks
        def fit(self):
            if self.__data_augmentation:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = Classifier._create_data_gen()
                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                datagen.fit(self._dataset.x_train)
                # Fit the model on the batches generated by datagen.flow().
                self.__model.fit_generator(
                                    datagen.flow(self._dataset.x_train, self._dataset.y_train, batch_size=self.__batch_size),
                                    steps_per_epoch=int(np.ceil(self._dataset.x_train.shape[0] / float(self.__batch_size))),
                                    epochs=self.__epochs,
                                    validation_data=(self._dataset.x_test, self._dataset.y_test),
                                    workers=4,
                                    shuffle=self.__shuffle,
                                    callbacks=self.__callbacks)
            else:
                print('Not using data augmentation.')
                self.__model.fit(self._dataset.x_train, self._dataset.y_train,
                            batch_size=self.__batch_size,
                            epochs=self.__epochs,
                            verbose=1,
                            validation_data=(self._dataset.x_test, self._dataset.y_test),
                            shuffle=self.__shuffle,
                            callbacks=self.__callbacks)
        def evaluate(self):
            return self.__model.evaluate(self._dataset.x_test, self._dataset.y_test, verbose=1)
    class __CnnDesignModelBuilder:
        def __init__(self, dataset):
            self._dataset = dataset
        def _create(self, design, optimizer, loss, output_activation):
            print(''.join(['__CnnDesignModelBuilder(design=', str(design), ', optimizer=', str(optimizer), ', loss=', str(loss), ', output_activation=', str(output_activation)]))
            model = Sequential()
            model = Classifier._ConvolutionLayers.create(model, self._dataset.x_train.shape[1:])
            model = Classifier._DropoutDesignFullyConnectedLayers.create(model, design.num_layers, design.num_units, design)
            model = Classifier._OutputLayer.create(model, self._dataset.num_classes, output_activation)
            model.summary()
            model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model
    class CnnDesignModelFitter(ModelFitter, __CnnDesignModelBuilder):
        def __init__(self, dataset, design, repeat, optimizer, loss, output_activation, epochs, data_augmentation, shuffle, callbacks):
            super(self.__class__, self).__init__(dataset)
            self.__model = self._create(design, optimizer, loss, output_activation)
            self.__epochs = epochs
            self.__batch_size = dataset.train_size // (design.num_batches * repeat) + 1
            self.__data_augmentation = data_augmentation
            self.__shuffle = shuffle
            self.__callbacks = [DropoutScheduler(design)] + callbacks
        def fit(self):
            if self.__data_augmentation:
                print('Using real-time data augmentation.')
                # This will do preprocessing and realtime data augmentation:
                datagen = Classifier._create_data_gen()
                # Compute quantities required for feature-wise normalization
                # (std, mean, and principal components if ZCA whitening is applied).
                datagen.fit(self._dataset.x_train)
                # Fit the model on the batches generated by datagen.flow().
                self.__model.fit_generator(
                                    datagen.flow(self._dataset.x_train, self._dataset.y_train, batch_size=self.__batch_size),
                                    steps_per_epoch=int(np.ceil(self._dataset.x_train.shape[0] / float(self.__batch_size))),
                                    epochs=self.__epochs,
                                    validation_data=(self._dataset.x_test, self._dataset.y_test),
                                    workers=4,
                                    shuffle=self.__shuffle,
                                    callbacks=self.__callbacks)
            else:
                print('Not using data augmentation.')
                self.__model.fit(self._dataset.x_train, self._dataset.y_train,
                            batch_size=self.__batch_size,
                            epochs=self.__epochs,
                            verbose=1,
                            validation_data=(self._dataset.x_test, self._dataset.y_test),
                            shuffle=self.__shuffle,
                            callbacks=self.__callbacks)
        def evaluate(self):
            return self.__model.evaluate(self._dataset.x_test, self._dataset.y_test, verbose=1)
    @classmethod
    def _create_data_gen(cls):
        '''This will do preprocessing and realtime data augmentation.'''
        return ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=0.,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format="channels_last", #data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)

#
# Input Reader
#
class JsonSettingReader:
    def __init__(self, json_file):
        self.__json_file = json_file
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            # print("{}".format(json.dumps(json_data,indent=4)))
            self.network = json_data["network"]
            dataset = json_data["dataset"]
            assert dataset is None or dataset == 'cifar10' or dataset == 'uniform' or dataset == 'uniform-regression'
            if dataset is None or dataset == 'cifar10':
                self.dataset = Cifar10Dataset()
                self.loss = 'categorical_crossentropy'
                self.ouput_activation = 'softmax'
            elif dataset == 'uniform':
                self.dataset = UniformDistDataset()
                self.loss = 'categorical_crossentropy'
                self.ouput_activation = 'softmax'
            elif dataset == 'uniform-regression':
                self.dataset = UniformDistRegressionDataset()
                self.loss = 'mean_squared_error'
                self.ouput_activation = 'linear'
            assert self.dataset is not None and self.loss is not None
            self.epochs = int(json_data["epochs"])
            self.experiments = int(json_data["experiments"])
            self.shuffle = bool(json_data["shuffle"])
            self.data_augmentation = bool(json_data["data_augmentation"])
            self.checkpoint_path = json_data["checkpoint_path"]

            __design = json_data["design"]
            self.design = None
            self.repeat = None
            self.__design_csv = None
            self.__update_design_method = None
            self.num_units = None
            self.num_layers = None
            self.num_batches = None
            self.dropout_rate = None
            self.num_drop = None
            self.dropout_mode = None
            if __design is None:
                __manual = json_data["manual"]
                assert __manual is not None
                self.num_units = int(__manual["num_units"])
                self.num_layers = int(__manual["num_layers"])
                self.num_batches = int(__manual["num_batches"])
                self.dropout_rate = float(__manual["dropout_rate"])
                self.num_drop = int(self.num_units * self.dropout_rate)
                self.dropout_mode = "non" if self.dropout_rate == 0 else "random"
            else:
                self.__design_csv = __design["design_csv"]
                gen = PGDesignGenerator(int(__design["degrees"]), int(__design["prime_number"]), int(__design["t_flats"])) if self.__design_csv is None else FileDesignGenerator(self.__design_csv)
                self.__update_design_method = __design["update_design_method"] if __design["update_design_method"] in {"shift_var", "shift_block", "rand_var", "rand_block"} else "non_update"
                assert self.__update_design_method is not None
                self.design = DropoutDesign(gen, self.__update_design_method)
                self.num_units = self.design.num_units
                self.num_layers = self.design.num_layers
                self.num_batches = self.design.num_batches
                self.num_drop = self.design.num_drop
                self.repeat = int(__design["repeat"])
                self.dropout_mode = "design" if __design["dropout_mode"] is None else __design["dropout_mode"]
                self.dropout_rate = 0 if self.is_non_dropout() else self.design.dropout_rate
                assert self.is_non_dropout() or self.is_dropout() or self.is_design_dropout()
                assert self.design is not None and self.repeat is not None
            assert self.num_units is not None and self.num_layers is not None and self.num_batches is not None and self.dropout_rate is not None and self.num_drop is not None and self.dropout_mode is not None

            __optimizer = json_data["optimizer"]
            self.optimizer = None
            if __optimizer is None:
                self.optimizer = keras.optimizers.sgd()
            else:
                opt_cls = keras.optimizers.sgd if __optimizer["name"] is None else globals()[__optimizer["name"]]
                if __optimizer["learning_rate"] is None and __optimizer["decay"] is None:
                    self.optimizer = opt_cls()
                elif __optimizer["learning_rate"] is None:
                    self.optimizer = opt_cls(decay = float(__optimizer["decay"]))
                elif __optimizer["decay"] is None:
                    self.optimizer = opt_cls(lr = float(__optimizer["learning_rate"]))
                else:
                    self.optimizer = opt_cls(lr = float(__optimizer["learning_rate"]), decay = float(__optimizer["decay"]))
            assert self.optimizer is not None
    # def __str__(self):
    #     return f'network={self.network},\ndataset={self.dataset.name},\nepochs={self.epochs},\nexperiments={self.experiments},\ndata_augmentation={self.data_augmentation},\ncheckpoint_path={self.checkpoint_path},\nnum_units={self.num_units},\nnum_layers={self.num_layers},\nnum_batches={self.num_batches},\ntotal_num_batch={self.total_num_batches},\ndropout_rate={self.dropout_rate},\nnum_drop={self.num_drop},\noptimizer={self.optimizer},\nloss={self.loss},\nouput_activation={self.ouput_activation},\nlr={K.get_value(self.optimizer.lr)},\ndecay={K.get_value(self.optimizer.decay)},\ndesign={self.design},\nrepeat={self.repeat},\nupdate_design_method={self.__update_design_method},\ndropout_mode={self.dropout_mode},\ndesign_csv={self.__design_csv}'
    @property
    def total_num_batches(self): return self.num_batches * (1 if self.repeat is None else self.repeat)
    def is_design_dropout(self): return self.dropout_mode == "design"
    def is_non_dropout(self): return self.dropout_mode == "non"
    def is_dropout(self): return self.dropout_mode == "random"

def checkpoint(reader, dataset, experiments):
    subdir = 'models'
    prefix = '_'.join(['model', str(reader.network), str(reader.num_layers), str(reader.num_units), str(reader.num_drop), str(reader.dropout_mode), str(reader.experiments), str(reader.epochs), dataset.name, str(experiments+1)])
    base_file = '_'.join([prefix, "{epoch:04d}"]) + '.h5'
    os.makedirs(subdir, exist_ok=True)
    base_file_path = os.path.join(subdir, base_file)
    return keras.callbacks.ModelCheckpoint(base_file_path)

def run(json):
    src_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(src_path)
    reader = JsonSettingReader(json)
    print(reader)
    dataset = reader.dataset
    batch_size = dataset.train_size // reader.total_num_batches + 1
    print(''.join(['batch_size=', str(batch_size)]))
    logfile = '_'.join(['log', str(reader.network), dataset.name, str(reader.num_layers), str(reader.num_units), str(reader.num_drop)]) + '.csv'

    def fit(experiment_count):
        callbacks = [Logger(logfile, dataset.name, experiment_count, reader.dropout_mode), checkpoint(reader, dataset, experiment_count)]
        if reader.checkpoint_path is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(reader.checkpoint_path + '_' + str(experiment_count)))
        print(''.join(['network creating: ', str(reader.network)]))
        fitter = None
        if reader.network == "mlp":
            print('Warning! MLP does not support Data Augmentation')
            if reader.is_non_dropout() or reader.is_dropout():
                fitter = Classifier.MlpSimpleModelFitter(dataset, reader.num_layers, reader.num_units, reader.optimizer, reader.loss, reader.ouput_activation, reader.dropout_rate, reader.epochs, batch_size, reader.shuffle, callbacks)
            elif reader.is_design_dropout():
                fitter = Classifier.MlpDesignModelFitter(dataset, reader.design, reader.repeat, reader.optimizer, reader.loss, reader.ouput_activation, reader.epochs, reader.shuffle, callbacks)
            else:
                print('failed to create fitter')
                assert False
        elif reader.network == "cnn":
            if reader.is_non_dropout() or reader.is_dropout():
                fitter = Classifier.CnnSimpleModelFitter(dataset, reader.num_layers, reader.num_units, reader.optimizer, reader.loss, reader.ouput_activation, reader.dropout_rate, reader.epochs, batch_size, reader.data_augmentation, reader.shuffle, callbacks)
            elif reader.is_design_dropout():
                fitter = Classifier.CnnDesignModelFitter(dataset, reader.design, reader.repeat, reader.optimizer, reader.loss, reader.ouput_activation, reader.epochs, reader.data_augmentation, reader.shuffle, callbacks)
            else:
                print('failed to create fitter')
                assert False
        else:
            print(''.join(['unsupported network:', str(reader.network)]))
            assert False
        assert fitter is not None
        fitter.fit()
        score = fitter.evaluate()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
    for i in range(reader.experiments):
        fit(i)
    print('Finished! See you!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json', type=str, default='../conf/example.json', help='json file')
    args = parser.parse_args()
    run(args.json)