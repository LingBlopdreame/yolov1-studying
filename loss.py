import tensorflow as tf
import numpy as np



class yololoss(tf.keras.losses.Loss):
    def __init__(self, l_coord, l_noobj, image_size):
        super(yololoss, self).__init__()
        self.image_size = image_size
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''
            box1 box2 is [x_min, y_min, x_max, y_max]
            :param box1: bounding boxes, size is [N, 4]
            :param box2: bounding boxes, size is [M, 4]
            :return: iou, size is [N, M]
        '''
        N = box1.shape[0]
        M = box2.shape[0]

        left_top = tf.maximum(
            tf.tile(tf.expand_dims(box1[:, :2], 1), [1, M, 1]),    # shape: [N, 2] -> [N, 1, 2] -> [N, M, 2]
            tf.tile(tf.expand_dims(box2[:, :2], 0), [N, 1, 1])     # shape: [M, 2] -> [1, M, 2] -> [N, M, 2]
        )

        right_bottom = tf.minimum(
            tf.tile(tf.expand_dims(box1[:, :2], 1), [1, M, 1]),   # shape: [N, 2] -> [N, 1, 2] -> [N, M, 2]
            tf.tile(tf.expand_dims(box2[:, :2], 0), [N, 1, 1])    # shape: [M, 2] -> [1, M, 2] -> [N, M, 2]
        )

        wh = right_bottom - left_top    # 计算 w, h , shape: [N, M, 2]
        wh[wh<0] = 0                    # 若box1, box2非交, 令值为0
        inter = wh[:,:,0] * wh[:,:,1]   # 计算交集的面积, shape: [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])   # 计算 box1 的面积, shape: [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])   # 计算 box2 的面积, shape: [M,]
        tf.tile(tf.expand_dims(area1, 1), [1, M])     # shape: [N,] -> [N,M]
        tf.tile(tf.expand_dims(area2, 0), [N, 1])     # shape: [M,] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou



    def call(self, y_true, y_pred):
        '''
        box_number 取2
        :param y_true: 真实y, box表示[x, y, w, h, c], shape: [batch_size, S, S, box_number * 5 + class_number]
        :param y_pred: 预测y, box表示[x, y, w, h, c], shape: [batch_size, S, S, box_number * 5 + class_number]
        :return: loss
        '''
        coo_mask = y_true[:, :, :, 4] > 0
        noo_mask = y_true[:, :, :, 4] == 0
        coo_mask = tf.tile(tf.expand_dims(coo_mask, -1), [1, 1, 1, y_true.shape[-1]])
        noo_mask = tf.tile(tf.expand_dims(noo_mask, -1), [1, 1, 1, y_true.shape[-1]])

        coo_pred = tf.reshape(y_pred[coo_mask], [-1, 30])
        box_pred = tf.reshape(coo_pred[:, :10], [-1, 5])
        class_pred = coo_pred[:, 10:]

        coo_true = tf.reshape(y_true[coo_mask], [-1, 30])
        box_true = tf.reshape(coo_true[:, :10], [-1, 5])
        class_true = coo_true[:, 10:]

        # 计算 not contain object loss
        noo_pred = tf.reshape(y_pred[noo_mask], [-1, 30])
        noo_true = tf.reshape(y_pred[noo_mask], [-1, 30])
        noo_pred_c = tf.stack([noo_pred[:, 4], noo_pred[:, 9]], axis=-1)
        noo_true_c = tf.stack([noo_true[:, 4], noo_true[:, 9]], axis=-1)
        nooobj_loss = tf.keras.losses.mse(y_true=noo_true_c, y_pred=noo_pred_c)

        # 得到best box(iou大的box)以及他的iou
        coo_response_mask = tf.zeros_like(box_true, dtype=tf.float64)
        coo_not_response_mask = tf.zeros_like(box_true, dtype=tf.float64)
        box_true_iou = tf.zeros_like(box_true, dtype=tf.float64)
        for i in range(0, box_true.shape[0], 2):
            box1 = box_pred[i:i+2,:]
            box1_xy_min = (box1[:, :2] - 0.5 * box1[:, 2:4]) / self.image_size
            box1_xy_max = (box1[:, :2] + 0.5 * box1[:, 2:4]) / self.image_size
            box1_xy = tf.concat([box1_xy_min, box1_xy_max], axis=1)
            box2 = box_true[i]
            box2_xy_min = (box2[:, :2] - 0.5 * box2[:, 2:4]) / self.image_size
            box2_xy_max = (box2[:, :2] + 0.5 * box2[:, 2:4]) / self.image_size
            box2_xy = tf.concat([box2_xy_min, box2_xy_max], axis=1)
            iou = self.compute_iou(box1_xy, box2_xy)
            max_index = tf.argmax(iou, axis=0)
            max_iou = tf.reduce_max(iou, axis=0)

            coo_response_mask[i + max_index] = 1
            coo_not_response_mask[i + 1 - max_index] = 1

            box_true_iou[i + max_index, 4] = max_iou

        coo_response_mask = tf.cast(coo_response_mask, dtype=tf.bool)
        coo_not_response_mask = tf.cast(coo_not_response_mask, dtype=tf.bool)

        box_pred_response = tf.reshape(box_pred[coo_response_mask], [-1, 5])
        box_true_response = tf.reshape(box_true[coo_response_mask], [-1, 5])
        box_true_response_iou = tf.reshape(box_true_iou[coo_response_mask], [-1, 5])

        box_pred_not_response = tf.reshape(box_pred[coo_not_response_mask], [-1, 5])
        box_true_not_response = tf.reshape(box_true[coo_not_response_mask], [-1, 5])
        box_true_not_response[:, 4] = 0

        # 计算response loss
        response_loss = tf.keras.losses.mse(
            y_true=box_true_response_iou[:, 4],
            y_pred=box_pred_response[:, 4]
        )

        # 计算not response loss
        not_response_loss = tf.keras.losses.mse(
            y_true=box_true_not_response[:, 4],
            y_pred=box_pred_not_response[:, 4]
        )

        # 计算local loss(位置误差)
        local_loss = tf.keras.losses.mse(
            y_true=box_true_response[:, :2],
            y_pred=box_pred_response[:, :2]
        ) + tf.keras.losses.mse(
            y_true=box_true_response[:, 2:4],
            y_pred=box_true_response[:, 2:4]
        )

        # 计算class loss
        class_loss = tf.keras.losses.mse(
            y_true=class_true,
            y_pred=class_pred
        )

        loss = self.l_coord * local_loss + response_loss + not_response_loss + self.l_noobj * nooobj_loss + class_loss
        return loss
