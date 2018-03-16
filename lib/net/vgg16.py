from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from net.net import Net
import tensorflow as tf
from config import cfg

VGG_MEANS = [103.939, 116.779, 123.68]
NAME_LOC_LOSS = ["conv4_2_loc_loss", "conv7_loc_loss"
    , "conv8_2_loc_loss", "conv9_2_loc_loss"
    , "conv10_2_loc_loss", "conv11_2_loc_loss"]

NAME_CLS_LOSS = ["conv4_2_cls_loss", "conv7_cls_loss"
    , "conv8_2_cls_loss", "conv9_2_cls_loss"
    , "conv10_2_cls_loss", "conv11_2_cls_loss"]

class VGG16(Net):
    def __init__(self, imdb, is_training=True):
        Net.__init__(self)
        self._imdb = imdb
        self.LOC_SHIFT = 1
        self.CLASS_SHIFT = 5

        self._num_classes = self._imdb.num_classes

        self._batch_size = cfg.BATCH_SIZE

        self._LEN_OF_LABLES_WITH_CLS_LOC = self._num_classes + 4

        self.images = tf.placeholder(tf.float32
            , [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3]
            , name="images")

        self.logits = self.inference(self.images)

        if is_training:
            self.lables = []
            for fp_size, num_boxes  \
                in zip(cfg.FP_SIZE, cfg.NUM_DEFAULT_BOXES):
                lab = tf.placeholder(shape=[self._batch_size, num_boxes, fp_size
                    , fp_size, 1 + 4 + self._num_classes], dtype=tf.float32)
                self.lables.append(lab)
            self.loss_layer(self.logits, self.lables)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar("total_loss", self.total_loss)



    def inference(self, inputs):
        """
        Args:
            inputs 4-D tensor [batch_size, fp_height, fp_width, 3]

        Return:
            predicts list [predicts_1, predicts_2, ..., predicts_6]
            predicts_i: 5-D tensor [batch_size, len(ras), fp_height, fp_width, 4 + num_classes]
        """
        r = inputs[:, :, :, 0] - VGG_MEANS[0]
        g = inputs[:, :, :, 1] - VGG_MEANS[1]
        b = inputs[:, :, :, 2] - VGG_MEANS[2]

        inputs = tf.transpose(tf.stack([r, g, b])
            , (1, 2, 3, 0))

        """
        the base architecture of VGG_16 reduced 3 fc layers
        """
        conv1_1 = self.conv(inputs, "conv1_1", 3, 3, 64, 1, 1
            , pretrainable=False)   # 300 * 300
        tf.summary.histogram("conv1_1", conv1_1)
        conv1_2 = self.conv(conv1_1, "conv1_2", 3, 3, 64, 1, 1
            , pretrainable=False) # 300 * 300
        pool_1 = self.max_pool(conv1_2, 2, 2, 2, 2, "pool_1")   # 150 * 150

        conv2_1 = self.conv(pool_1, "conv2_1", 3, 3, 128, 1, 1
            , pretrainable=False)   # 150 * 150
        conv2_2 = self.conv(conv2_1, "conv2_2", 3, 3, 128, 1, 1
            , pretrainable=False)   # 150 * 150
        pool_2 = self.max_pool(conv2_2, 2, 2, 2, 2, "pool_2")   # 75 * 75

        conv3_1 = self.conv(pool_2, "conv3_1", 3, 3, 256, 1, 1
            , pretrainable=False)   # 75 * 75
        conv3_2 = self.conv(conv3_1, "conv3_2", 3, 3, 256, 1, 1
            , pretrainable=False)   # 75 * 75
        conv3_3 = self.conv(conv3_2, "conv3_3", 3, 3, 256, 1, 1
            , pretrainable=False)   # 75 * 75
        pool_3 = self.max_pool(conv3_3, 2, 2, 2, 2, "pool_3")   # 38 * 38

        conv4_1 = self.conv(pool_3, "conv4_1", 3, 3, 512, 1, 1
            , pretrainable=False)   # 38 * 38
        conv4_2 = self.conv(conv4_1, "conv4_2", 3, 3, 512, 1, 1
            , pretrainable=False)   # 38 * 38
        conv4_3 = self.conv(conv4_2, "conv4_3", 3, 3, 512, 1, 1
            , pretrainable=False)   # 38 * 38
        pool_4 = self.max_pool(conv4_3, 2, 2, 2, 2, "pool_4")   # 19 * 19

        conv5_1 = self.conv(pool_4, "conv5_1", 3, 3, 512, 1, 1
            , pretrainable=False)   # 19 * 19
        conv5_2 = self.conv(conv5_1, "conv5_2", 3, 3, 512, 1, 1
            , pretrainable=False)   # 19 * 19
        conv5_3 = self.conv(conv5_2, "conv5_3", 3, 3, 512, 1, 1
            , pretrainable=False)   # 19 * 19

        """
        additional feature layers on SSD
        """
        tf.summary.histogram("conv5_3", conv5_3)
        conv6 = self.conv(conv5_3, "conv6", 3, 3, 1024, 1, 1
            , is_add_bias=False, pretrainable=False
            , is_xavier=True, is_bn=True)    # 19 * 19

        conv7 = self.conv(conv6, "conv7", 1, 1, 1024, 1, 1
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 19 * 19

        conv8_1 = self.conv(conv7, "conv8_1", 1, 1, 256, 1, 1
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 19 * 19
        #print("conv8_1:", conv8_1.get_shape().as_list())
        conv8_2 = self.conv(conv8_1, "conv8_2", 3, 3, 512, 2, 2
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 10 * 10
        #print("conv8_2:", conv8_2.get_shape().as_list())

        conv9_1 = self.conv(conv8_2, "conv9_1", 1, 1, 128, 1, 1
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 10 * 10
        #print("conv8_2:", conv9_1.get_shape().as_list())
        conv9_2 = self.conv(conv9_1, "conv9_2", 3, 3, 256, 2, 2
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 5 * 5

        conv10_1 = self.conv(conv9_2, "conv10_1", 1, 1, 128, 1, 1
            , is_add_bias=False, pretrainable=False
            ,is_xavier=True,  is_bn=True)    # 5 * 5
        conv10_2 = self.conv(conv10_1, "conv10_2", 3, 3, 256, 1, 1
            , padding="VALID", is_add_bias=False, pretrainable=False
            ,is_xavier=True, is_bn=True)    # 3 *3

        conv11_1 = self.conv(conv10_2, "conv11_1", 1, 1, 128, 1, 1
            , is_add_bias=False, pretrainable=False
            , is_xavier=True, is_bn=True)    # 3 * 3
        conv11_2 = self.conv(conv11_1, "conv11_2", 3, 3, 256, 1, 1
            , padding="VALID", is_add_bias=False, pretrainable=False
            , is_xavier=True, is_bn=True)    # 1 * 1

        layer_boxes = cfg.NUM_DEFAULT_BOXES

        predicts_1 = self.conv(conv4_3, "predicts_1", 3, 3
            , layer_boxes[0] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)
        shape = predicts_1.get_shape().as_list()
        predicts_1 = tf.transpose(
            tf.reshape(predicts_1, [-1, shape[1], shape[2]
                , layer_boxes[0], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])

        predicts_2 = self.conv(conv7, "predicts_2", 3, 3
            , layer_boxes[1] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)
        shape = predicts_2.get_shape().as_list()
        predicts_2 = tf.transpose(
            tf.reshape(predicts_2, [-1, shape[1], shape[2]
                , layer_boxes[1], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])

        predicts_3 = self.conv(conv8_2, "predicts_3", 3, 3
            , layer_boxes[2] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)
        shape = predicts_3.get_shape().as_list()
        predicts_3 = tf.transpose(
            tf.reshape(predicts_3, [-1, shape[1], shape[2]
                , layer_boxes[2], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])


        predicts_4 = self.conv(conv9_2, "predicts_4", 3, 3
            , layer_boxes[3] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)

        shape = predicts_4.get_shape().as_list()
        predicts_4 = tf.transpose(
            tf.reshape(predicts_4, [-1, shape[1], shape[2]
                , layer_boxes[3], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])


        predicts_5 = self.conv(conv10_2, "predicts_5", 3, 3
            , layer_boxes[4] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)
        shape = predicts_5.get_shape().as_list()
        predicts_5 = tf.transpose(
            tf.reshape(predicts_5, [-1, shape[1], shape[2]
                , layer_boxes[4], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])


        predicts_6 = self.conv(conv11_2, "predicts_6", 3, 3
            , layer_boxes[5] * self._LEN_OF_LABLES_WITH_CLS_LOC
            , 1, 1, is_add_bias=False, is_activiation=False
            , is_xavier=True, is_bn=True)
        shape = predicts_6.get_shape().as_list()
        predicts_6 = tf.transpose(
            tf.reshape(predicts_6, [-1, shape[1], shape[2]
                , layer_boxes[5], self._LEN_OF_LABLES_WITH_CLS_LOC])
            , [0, 3, 1, 2, 4])

        self.predicts = [predicts_1, predicts_2, predicts_3
            , predicts_4, predicts_5, predicts_6]

        return self.predicts


    def loss_layer(self, predicts, lables, scope="loss_layer"):
        """
        Args:
            predicts: list [predicts_1, predicts_2, ..., predicts_6]
            predicts_i: 5-D tensor [batch_size, len(ras), fp_height, fp_width, 4 + num_classes]
            lables: list [lab_1, lab_2, ..., lab_6]
            lab1: 5-D tensor [batch_size, len(ras), fp_height, fp_width, 1 + 4 + num_classes]
        """
        with tf.variable_scope(scope):
            total_cls_loss_p = 0.0
            total_cls_loss_n = 0.0
            total_cls_loss = 0.0
            total_loc_loss = 0.0
            for pred, lbs, indx in zip(predicts, lables
                , range(len(cfg.NUM_DEFAULT_BOXES))):
                mask_p = lbs[:, :, :, :, :1]

                loc_train = lbs[:, :, :, :, self.LOC_SHIFT:self.CLASS_SHIFT]
                loc_pred = pred[:, :, :, :, :4]

                # N 2-D tensor [batch_size, num_boxes mapped]
                N = tf.reduce_sum(mask_p, axis=[1, 2, 3, 4])
                # loc_loss 2-D tensor [batch_size, num_boxes mapped]
                loc_loss = tf.reduce_sum(   \
                    self.smooth_l1(mask_p * (loc_pred - loc_train))
                    , axis=[1, 2, 3, 4])

                loc_loss = tf.where(N > 0.0, loc_loss / (N + cfg.ESP), loc_loss - loc_loss)

                cls_mask = lbs[:, :, :, :, self.CLASS_SHIFT:]
                cls_pred = pred[:, :, :, :, 4:]
                cls_soft_max = -tf.log(tf.nn.softmax(cls_pred))
                cls_loss_p = tf.reduce_sum(cls_soft_max * cls_mask, axis=(1, 2, 3, 4))

                mask_n = tf.ones_like(mask_p, dtype=tf.float32) - mask_p
                score_n = cls_soft_max[:, :, :, :, 0] * mask_n[:, :, :, :, 0]
                score_n = tf.reshape(score_n, shape=[score_n.get_shape().as_list()[0], -1])
                num_and_score_n = tf.concat([tf.reshape(N
                        , (N.get_shape().as_list()[0], 1)), score_n]
                    , 1)

                def top_k(input):
                    k = input[0]
                    conf = input[1:]
                    k = tf.cast(k, tf.int32)
                    return tf.cond(tf.greater(k, 0)
                        , lambda: tf.reduce_sum(tf.nn.top_k(conf, k)[0]), lambda: 0.0)

                cls_loss_n = tf.map_fn(top_k, num_and_score_n)


                cls_loss_p = tf.where(N > 0, cls_loss_p / (N + cfg.ESP), cls_loss_p - cls_loss_p)
                cls_loss_n = tf.where(N > 0, cls_loss_n / (N + cfg.ESP), cls_loss_n - cls_loss_n)
                cls_loss = cls_loss_p + cls_loss_n

                tf.summary.scalar(NAME_CLS_LOSS[indx] + "_positive"
                    , tf.reduce_mean(cls_loss_p))
                tf.summary.scalar(NAME_CLS_LOSS[indx] + "_negative"
                    , tf.reduce_mean(cls_loss_n))
                tf.summary.scalar(NAME_CLS_LOSS[indx]
                    , tf.reduce_mean(cls_loss))
                tf.summary.scalar(NAME_LOC_LOSS[indx]
                    , tf.reduce_mean(loc_loss))

                total_cls_loss_p += cls_loss_p
                total_cls_loss_n += cls_loss_n
                total_cls_loss += cls_loss
                total_loc_loss +=  loc_loss


            total_loss = tf.reduce_mean(total_loc_loss + total_cls_loss)
            tf.losses.add_loss(total_loss)

            tf.summary.scalar("cls_loss_p", tf.reduce_mean(total_cls_loss_p))
            tf.summary.scalar("cls_loss_n", tf.reduce_mean(total_cls_loss_n))
            tf.summary.scalar("cls_loss", tf.reduce_mean(total_cls_loss))
            tf.summary.scalar("loc_loss", tf.reduce_mean(total_loc_loss))
