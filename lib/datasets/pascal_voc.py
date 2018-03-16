from __future__ import print_function

from datasets.imdb import imdb

import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
import cPickle

class pascal_voc(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, "pascal_voc2007", image_set)
        self._images_classses = ('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

        self._image_set_indxs = self._load_image_set_indxs();
        self._all_iamge_pathes = self._all_image_pathes()
        self._cache_file = self.cache_file

    def all_image_pathes(self):
        return self._all_iamge_pathes

    def _all_image_pathes(self):
        return [osp.join(self.data_path
                , "VOCdevkit2007/VOC2007/JPEGImages"
                , indx + ".jpg")    \
            for indx in self._image_set_indxs]

    def _load_image_set_indxs(self):
        image_set_file = osp.join(self.data_path
            , "VOCdevkit2007/VOC2007/ImageSets/Main"
            , self._image_set + ".txt")
        assert osp.exists(image_set_file)   \
            , "image set file: {} not exists".format(image_set_file)

        with open(image_set_file) as f:
            image_set_indxs = [indx.strip() for indx in f.readlines()]

        return image_set_indxs

    def _load_annotation(self, img_indx):
        file_name = osp.join(self.data_path
            , "VOCdevkit2007/VOC2007/Annotations"
            , img_indx + ".xml")
        assert osp.exists(file_name)    \
            , "file name: {} not exists in {} Annotations"  \
                .format(file_name, self.name)

        tree = ET.parse(file_name)
        objs = tree.findall("object")
        num_objs = len(objs)

        gt_boxes = np.zeros((num_objs, 4))
        gt_classes = np.zeros((num_objs), np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find("bndbox")
            # make pixel indexes 0 based
            x1 = float(bbox.find("xmin").text) - 1.0
            y1 = float(bbox.find("ymin").text) - 1.0
            x2 = float(bbox.find("xmax").text) - 1.0
            y2 = float(bbox.find("ymax").text) - 1.0

            cls = self.class_to_inds[
                obj.find("name").text.lower().strip()]
            gt_boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return gt_boxes, gt_classes

    @property
    def cache_file(self):
        if self._cache_file != None:
            return self._cache_file
        return osp.join(self.cache_dir
            , self.name + "_" + self._image_set + ".pkl")


    def get_gt_infos(self):
        print("cache file")
        print(self.cache_file)
        if osp.exists(self.cache_file):
            with open(self.cache_file) as f:
                print("{} gt info loaded from {}"   \
                    .format(self.name, self.cache_file))
                gt_infos = cPickle.load(f)
                print("gt info loaded done")
                return gt_infos

        gt_boxes = []
        gt_classes = []

        for indx in self._image_set_indxs:
            boxes, clses = self._load_annotation(indx)
            gt_boxes.append(boxes)
            gt_classes.append(clses)

        gt_infos = {"image_pathes": self.all_image_pathes()
            , "boxes": gt_boxes, "classes": gt_classes}

        with open(self.cache_file, "wb") as f:
            print("writing gt info into the {}..."  \
                .format(self.cache_file))
            cPickle.dump(gt_infos, f)
            print("writing done")

        return gt_infos
