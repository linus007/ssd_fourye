from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import _init_paths

import os

from datasets.pascal_voc import pascal_voc
from utils.data_batch import databatch
from config import cfg

from net.vgg16 import VGG16
from solver.vgg16_solver import VGG16_Solver

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    imdb = pascal_voc("trainval")
    imdb.prepare()

    batch_size = cfg.BATCH_SIZE
    cache_size = cfg.CACHE_SIZE

    data_batch = databatch(batch_size, cache_size, imdb)
    data_batch.ready()

    print("initializing net...")
    net = VGG16(imdb)
    print("net initializing done.")

    print("initializing solver...")
    solver = VGG16_Solver(data_batch, net, imdb)
    print("solver initializing done")

    print("start training...")
    solver.train()
    print("training done.")


if __name__ == "__main__":
    main()
