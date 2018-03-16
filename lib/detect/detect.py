import tensorflow as tf
import numpy as np
import os
import cv2
from net.vgg16 import VGG16
from utils.timer import Timer
from utils.nms import nms
from config import cfg

class detector(object):
    def __init__(self, net, imdb, modle_file):
        self._net = net
        self._modle_file = modle_file
        self._imdb = imdb
        self._classes = self._imdb.image_classes
        self._num_classes = len(self._classes)

        self._image_size = cfg.IMAGE_SIZE
        self._fp_size = cfg.FP_SIZE
        self._threshold = cfg.THRESHOULD
        self._iou_threshold = cfg.IOU_THRESHOULD

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._scale = cfg.S
        self._scale_ras_of_1 = cfg.S_RATIO_OF_1
        self._ras_normal = cfg.RAS_NORMAL

        self._nums_default_box = cfg.NUM_DEFAULT_BOXES

        print("restore modle from the local file: {}..." \
            .format(self._modle_file))
        self._saver = tf.train.Saver(tf.global_variables())
        self._saver.restore(self._sess, self._modle_file)
        print("restore done.")

    def image_detect(self, img_path, wait=0):
        detect_timer = Timer()
        img = cv2.imread(img_path)

        detect_timer.tic()
        res = self.detect(img)
        detect_timer.toc()

        print("Average detecting time: {:.3f}"  \
            .format(detect_timer.average_time))
        self.draw_result(img, res)
        cv2.imshow("img", img)
        cv2.waitKey(wait)


    def detect(self, img):
        image_size = self._image_size
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.reshape(img, (1, image_size, image_size, 3))
        res = self.detect_from_cvmat(img_w, img_h, img)
        return res

    def draw_result(self, img, res):
        classes = self._classes
        for cls, boxes in res.items():
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                score = box[4]
                cv2.rectangle(img, (x1, y1)
                    , (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, classes[int(cls)] + ": %.2f"%score
                    , (max(x1 + 5, 1), max(y1 - 7, 1)), cv2.FONT_HERSHEY_SIMPLEX
                    , 0.5, (0, 0, 0), 1, cv2.CV_AA)
                print("cls:", cls)
                print("boxes:", box[:4])
                print("score:", score)

    def detect_from_cvmat(self, img_w, img_h, inputs):
        out = self._sess.run(self._net.predicts
            , feed_dict={self._net.images:inputs})
        print("out:",out[1][0, 3, 4, 8, :])
        res = self.interpret_out(out, img_w, img_h)
        return res


    """
    Args:
        img_w: real width of image
        img_h: real height of image
        multi_boxes: list [predicts_1, predicts_2, ..., predicts_6]
        predicts_i: 5-D tensor [1, len(ras), fp_height, fp_width, 4]
    Return:
        2D-tensor [all_predicts, 4]
    """
    def coordinate_transfer(self, img_w, img_h, pre_boxes):
        res = []
        for indx, boxes in enumerate(pre_boxes):
            scale_of_ras_1 = self._scale_ras_of_1[indx]
            scale_normal = self._scale[indx]
            ras = self._ras_normal
            fp_size = self._fp_size[indx]
            num_default_box = self._nums_default_box[indx]

            d = np.zeros_like(boxes)
            d[0, :, :, 2] = scale_of_ras_1
            d[0, :, :, 3] = scale_of_ras_1
            for i in range(num_default_box)[1:]:
                d[i, :, :, 2] = scale_normal * np.sqrt(ras[i - 1])
                d[i, :, :, 3] = scale_normal / np.sqrt(ras[i - 1])

            fp_map_x = np.array(list(range(fp_size)) * fp_size * num_default_box)   \
                .reshape(num_default_box, fp_size, fp_size)
            fp_map_y = np.transpose(fp_map_x, (0, 2, 1))

            d[:, :, :, 0] = (fp_map_x + 0.5) / fp_size
            d[:, :, :, 1] = (fp_map_y + 0.5) / fp_size

            pre = np.zeros_like(boxes)
            pre[:, :, :, 0] = d[:, :, :, 2] * boxes[:, :, :, 0] + d[:, :, :, 0]
            pre[:, :, :, 1] = d[:, :, :, 3] * boxes[:, :, :, 1] + d[:, :, :, 3]
            pre[:, :, :, 2] = np.exp(boxes[:, :, :, 2]) * d[:, :, :, 2]
            pre[:, :, :, 3] = np.exp(boxes[:, :, :, 3]) * d[:, :, :, 3]
            boxes[:, :, :, 0] = pre[:, :, :, 0] - pre[:, :, :, 2] / 2.0 * img_w
            boxes[:, :, :, 1] = pre[:, :, :, 1] - pre[:, :, :, 3] / 2.0 * img_h
            boxes[:, :, :, 2] = pre[:, :, :, 2] + pre[:, :, :, 2] / 2.0 * img_w
            boxes[:, :, :, 3] = pre[:, :, :, 1] + pre[:, :, :, 3] / 2.0 * img_h
            res.append(np.reshape(boxes, [-1, 4]))

        return np.vstack(res)

    def interpret_out(self, out, img_w, img_h):
        num_classes = self._num_classes
        iou_threshould = self._iou_threshold
        threshold = self._threshold

        pre_boxes = [boxes[:,:, :, :, :4] for boxes in out]
        pre_boxes = [np.reshape(boxes, boxes.shape[1:]) \
            for boxes in pre_boxes]
        pre_boxes = self.coordinate_transfer(img_w, img_h, pre_boxes)

        pre_clses = [clses[:, :, :, :, 4:] for clses in out]
        pre_clses = [np.reshape(clses, (-1, clses.shape[-1])) \
            for clses in pre_clses]
        pre_clses = np.vstack(pre_clses)



        res = {}
        assert len(pre_boxes) == len(pre_clses)
        max_inds = np.argmax(pre_clses, axis=1)
        keep_inds = np.where(max_inds != 0)
        pre_boxes = pre_boxes[keep_inds]
        print(keep_inds)
        pre_clses = pre_clses[keep_inds]
        scores = np.exp(pre_clses)  \
            / np.sum(np.exp(pre_clses), axis=1).reshape([-1, 1])
        print("len:", len(pre_boxes))
        for i in range(num_classes)[1:]:
            keep_inds = np.where(scores[:, i] >= threshold)[0]
            print(keep_inds)
            dets = np.hstack([pre_boxes[keep_inds], scores[:, i][keep_inds].reshape([-1, 1])])
            keep_inds = nms(dets, iou_threshould)
            if len(keep_inds) > 0:
                res[str(i)] = dets[keep_inds]
        return res
