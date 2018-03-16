from __future__ import absolute_import
from __future__ import print_function
from __future__ import absolute_import

import config.cfg as cfg
import os
import os.path as osp
import cv2
import numpy as np
import cPickle

CLASSES_INDEX_SHIFT = 1 + 4

class imdb(object):
    def __init__(self, dataset_name, image_set):
        self._dataset_name = dataset_name
        self._data_root = cfg.DATA_ROOT

        self._data_path = osp.join(self._data_root, self._dataset_name)
        assert osp.exists(self._data_path)  \
            , "path of {} not exists".format(self._dataset_name)

        self._catch_dir = osp.join(self._data_path, "cache")
        self._cache_file = None

        if not osp.exists(self._catch_dir):
            os.makedirs(self._catch_dir)

        self._image_set = image_set

        self._img_height = cfg.IMAGE_SIZE
        self._img_width = cfg.IMAGE_SIZE

        self._images_classses = None

        self._all_iamge_pathes = None

        self._class_to_inds = None

        self._lables_file = None

        self._size_imdb = None

        self._lables = None


    @property
    def imdb_size(self):
        if self._size_imdb == None:
            print("type:{}".format(type(self._all_iamge_pathes)))
            self._size_imdb = len(self._all_iamge_pathes)

        return self._size_imdb

    @property
    def data_path(self):
        return self._data_path

    @property
    def name(self):
        return self._dataset_name

    @property
    def cache_dir(self):
        return self._catch_dir

    @property
    def cache_file(self):
        raise NotImplementedError

    @property
    def image_classes(self):
        return self._images_classses

    def all_image_pathes(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self._images_classses)


    def readImage(self, image_path):
        assert osp.exists(image_path)   \
            , "image path: {} not exists".format(image_path)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        ht = img.shape[0]
        wt = img.shape[1]
        return img, wt, ht

    def readImages(self, image_paths):
        imgs = []
        wts = []
        hts = []
        for img_path in image_paths:
            img, wt, ht = self.readImage(img_path)
            imgs.append(img)
            wts.append(wt)
            hts.append(ht)
        return imgs, wts, hts

    def get_gt_infos(self):
        raise NotImplementedError


    def _cal_iou_with_ras(self, gt_box, dg_boxes):
        s1 = (gt_box[2] - gt_box[0])  \
            * (gt_box[3] - gt_box[1])
        s2 = (dg_boxes[:, 2] - dg_boxes[:, 0])  \
            * (dg_boxes[:, 3] - dg_boxes[:, 1])
        x_max = np.maximum(gt_box[0], dg_boxes[:, 0])
        y_max = np.maximum(gt_box[1], dg_boxes[:, 1])
        x_min = np.minimum(gt_box[2], dg_boxes[:, 2])
        y_min = np.minimum(gt_box[3], dg_boxes[:, 3])
        w_inter = np.maximum(x_min - x_max, 0.0)
        h_inter = np.maximum(y_min - y_max, 0.0)
        inter = w_inter * h_inter
        return inter / (s1 + s2 - inter + cfg.ESP)

    def _d_g_mapping(self, bboxes, clses, fp_width, fp_height, s, ras, threshold):
        """
        Args:
            boxes: 2-D array [len(bboxes), 4]
                ===> (x1, y1, x2, y2)(relative to the image)
        Return:
            lables 4-D array [len(ras), fp_height, fp_width, 1 + 4 + num_classes]
                (mask,g_cx, g_cy, g_w, g_h, background + classes)
        """
        assert len(bboxes) == len(clses)

        cx_d = np.array(list(range(fp_width)) * fp_height * len(ras)
            , dtype=np.float32)
        cx_d = np.reshape(cx_d, (len(ras), fp_height, fp_width))
        cx_d = (cx_d + 0.5) / fp_width
        cx_d = np.transpose(cx_d, (1, 2, 0))

        cy_d = np.array(list(range(fp_height)) * fp_width * len(ras)
            , dtype=np.float32)
        cy_d = np.reshape(cy_d, (len(ras), fp_width, fp_height))
        cy_d = (cy_d + 0.5) / fp_height
        cy_d = np.transpose(cy_d, (2, 1, 0))

        w_ds = s * np.sqrt(ras)
        h_ds = s / np.sqrt(ras)

        dg_map = np.zeros((fp_height, fp_width, len(ras), 4))

        # NOTE: whether limit x and y bound?
        dg_map[:, :, :, 0] = cx_d - w_ds / 2
        dg_map[:, :, :, 1] = cy_d - h_ds / 2
        dg_map[:, :, :, 2] = cx_d + w_ds / 2
        dg_map[:, :, :, 3] = cy_d + h_ds / 2

        dg_map = np.transpose(dg_map, (2, 0, 1, 3))

        lables = np.zeros((len(ras), fp_height, fp_width    \
                , 1 + 4 + self.num_classes))

        for box, cls in zip(bboxes, clses):
            x1, y1, x2, y2 = box
            cx = int((x2 + x1) / 2.0 * fp_width)
            cy = int((y2 + y1) / 2.0 * fp_height)
            iou = self._cal_iou_with_ras(box, dg_map[:, cy, cx, :])
            iou_ras_mask = np.where(iou >= threshold)[0]
            if len(iou) > 0:
                for ras_indx in iou_ras_mask:
                    if lables[ras_indx, cy, cx, 0] == 0:
                        _g_cx = (x1 + x2) / 2.0
                        _g_cy = (y1 + y2) / 2.0
                        _g_w = x2 - x1
                        _g_h = y2 - y1
                        d_cx = (dg_map[ras_indx, cy, cx, 0]  \
                            + dg_map[ras_indx, cy, cx, 2]) / 2.0
                        d_cy = (dg_map[ras_indx, cy, cx, 1]
                            + dg_map[ras_indx, cy, cx, 3]) / 2.0
                        d_w = dg_map[ras_indx, cy, cx, 2]   \
                            - dg_map[ras_indx, cy, cx, 0]
                        d_h = dg_map[ras_indx, cy, cx, 3]   \
                            - dg_map[ras_indx, cy, cx, 1]
                        g_cx = (_g_cx - d_cx) / d_w
                        g_cy = (_g_cy - d_cy) / d_h
                        g_w = np.log(_g_w / d_w)
                        g_h = np.log(_g_h / d_h)
                        lables[ras_indx, cy, cx, 0] = 1.0
                        lables[ras_indx, cy, cx, 1:CLASSES_INDEX_SHIFT]     \
                            = [g_cx, g_cy, g_w, g_h]
                        lables[ras_indx, cy, cx, CLASSES_INDEX_SHIFT + cls] = 1.0

        return lables


    def _d_g_map_in_cov(self, boxes, clses
        , fp_width, fp_height, s_ratio_of_1, s_normal, ras, threshold):
        """
        Return:
            lables:4-D ndarray [len(ras), fp_height, fp_width, 1 + 4 + clses]
        """
        lables = None
        if s_ratio_of_1 != None:
            lables = self._d_g_mapping(boxes, clses, fp_width
                , fp_height, s_ratio_of_1, [1.0], threshold)

        lables_2 = None
        if s_normal != None:
            lables_2 = self._d_g_mapping(boxes, clses, fp_width
                , fp_height, s_normal, ras, threshold)

        if type(lables_2) != None and type(lables) != None:
            lables = np.vstack([lables, lables_2])
        elif lables_2 != None:
            lables = lables_2
        return lables

    @property
    def lables_file(self):
        if self._lables_file != None:
            return self._lables_file
        self._lables_file = osp.join(self.cache_dir
            , self._dataset_name + "_" + self._image_set    \
            + "_lables.pkl")
        return self._lables_file

    def _gt_info_process(self, gt_infos):
        all_image_pathes = gt_infos["image_pathes"]
        gt_boxes = gt_infos["boxes"]
        gt_classes = gt_infos["classes"]

        assert len(all_image_pathes) == len(gt_boxes) == len(gt_classes)

        gt_boxes_std = []
        # transform the cordinates into the relative one
        for img_path, boxes in zip(all_image_pathes, gt_boxes):
            _, wt, ht = self.readImage(img_path)
            boxes[:, 0] = boxes[:, 0] / wt
            boxes[:, 2] = boxes[:, 2] / wt
            boxes[:, 1] = boxes[:, 1] / ht
            boxes[:, 3] = boxes[:, 3] / ht
            gt_boxes_std.append(boxes)

        return gt_boxes_std, gt_classes


    def _produce_labels(self, gt_boxes_std, gt_classes):
        """
        Return:
            [len(fp_sizes), len(all_images), len(ras), fp_height, fp_width, 4 + 1 + clses]
        """
        gt_labels = []

        for k in range(cfg.NUM_FEATUE_MAPS_USED):
            lbs = []
            fp_width = cfg.FP_SIZE[k]
            fp_height = cfg.FP_SIZE[k]
            s_ratio_of_1 = cfg.S_RATIO_OF_1[k]
            s = cfg.S[k]
            ras_normal = cfg.RAS_NORMAL[:cfg.NUM_DEFAULT_BOXES[k] - 1]
            dg_threshold = cfg.DG_THRESHOLD
            for boxes, clses in     \
                zip(gt_boxes_std, gt_classes):
                lables = self._d_g_map_in_cov(boxes
                    , clses, fp_width, fp_height
                    , s_ratio_of_1, s
                    , ras_normal
                    , dg_threshold)
                lbs.append(lables)
            lbs = np.array(lbs)
            gt_labels.append(lbs)

        return gt_labels

    def get_imdb(self, inds):
        if self._lables == None:
            self.prepare()
        image_paths = np.array(self._all_iamge_pathes)
        image_paths = image_paths[inds]
        imgs, _, _  = self.readImages(image_paths)

        # img resize
        imgs = [cv2.resize(img, (self._img_height, self._img_width)) for img in imgs]

        lables = []
        for i in range(cfg.NUM_FEATUE_MAPS_USED):
            lables.append(self._lables[i][inds])

        return imgs, lables




    def prepare(self):
        if osp.exists(self.lables_file):
            with open(self.lables_file) as f:
                print("loading lables from the local file...")
                self._lables = cPickle.load(f)
                print("loading done.")
                return self._lables
        gt_infos = self.get_gt_infos()
        print("gt infos processing!")
        gt_boxes_std, gt_classes = self._gt_info_process(gt_infos)

        # NOTE: do much more processing of the image here
        # NOTE: such as flip, crop and so on

        self._lables = self._produce_labels(gt_boxes_std, gt_classes)

        """
        with open(self.lables_file, "wb") as f:
            print("writing the gt lables into the local...")
            cPickle.dump(self._lables, f)
            print("writing done.")
        """
        return self._lables




    @property
    def class_to_inds(self):
        if self._class_to_inds == None:
            self._class_to_inds = dict( \
                zip(self._images_classses, range(self.num_classes)))
        return self._class_to_inds
