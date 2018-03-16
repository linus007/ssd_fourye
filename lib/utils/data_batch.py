import numpy as np
from datasets import imdb
import threading

from concurrent_queue import  ConcurrentQueue
from config import cfg

class databatch(object):
    def __init__(self, batch_size, cache_size, imdb):
        self._batch_size = batch_size
        self._cache_size = cache_size
        self._cache = ConcurrentQueue(self._cache_size)

        self._imdb = imdb

        self._imdb_inds = np.arange(self._imdb.imdb_size)
        print("imdb_inds:{}".format(len(self._imdb_inds)))
        self.epoch = 0

    def batch_producer(self):
        inds = np.random.choice(self._imdb_inds
            , self._batch_size, replace=False)

        imgs, lables = self._imdb.get_imdb(inds)


        return imgs, lables

    def add_one_batch(self):
        imgs, lables = self.batch_producer()

        self._cache.put([imgs, lables])

    def ready(self):
        print("ready to prepare the data batch")
        for i in range(self._cache_size):
            self.add_one_batch()
        print("data batch preparing done!")

    def next_batch(self):
        imgs, lables = self._cache.get()

        thread = threading.Thread(target=self.add_one_batch)
        thread.start()
        self.epoch += 1
        return imgs, lables
