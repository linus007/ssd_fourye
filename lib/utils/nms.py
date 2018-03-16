import numpy as np

def nms(dets, threshold):
    scores = dets[:, 4]
    # discard scores with calue of 0.0
    inds_keep = np.where(scores > 0.0)[0]
    scores = scores[inds_keep]
    dets = dets[inds_keep]

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores)[::-1]
    keep = []

    while(order.size > 0):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)

        inter = w * h
        over = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(over < threshold)[0]
        order = order[inds + 1]

    keep = inds_keep[keep]
    return keep
