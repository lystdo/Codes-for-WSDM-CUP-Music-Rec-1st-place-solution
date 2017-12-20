import numpy as np
import threading
import warnings

from abc import ABCMeta, abstractmethod
from keras import backend as K

class DataGenerator(object):
    def __init__(self, class_size=None):
        self.class_size = class_size

    def flow(self, x_categorical, x_numberical, y=None, batch_size=32, shuffle=True):
        return NumpyArrayIterator(
            x_categorical, x_numberical, y, self,
            batch_size=batch_size,
            shuffle=shuffle)

class Sequence(object):

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def on_epoch_end(self):
        pass

class Iterator(Sequence):
    def __init__(self, n, batch_size, shuffle):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        raise NotImplementedError

class NumpyArrayIterator(Iterator):
    def __init__(self, x_cat, x_num, y, data_generator,
                 batch_size=32, shuffle=False):
        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.x_cat = x_cat
        self.x_num = x_num
        
        self.data_generator = data_generator
        super(NumpyArrayIterator, self).__init__(len(x_cat[0]), batch_size, shuffle)

    def _get_batches_of_transformed_samples(self, index_array):
        x_batch_deep = [data[index_array] for data in self.x_cat]
        x_batch_deep += [data[index_array] for data in self.x_num]
        
        if self.y is None:
            return x_batch_deep
        else:
            batch_y = self.y[index_array]
            return x_batch_deep, batch_y

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)

