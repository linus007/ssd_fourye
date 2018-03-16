"""
import _init_paths
from datasets.pascal_voc import pascal_voc
import os.path as osp

def main():
    imdb = pascal_voc("trainval")
    img_paths = imdb.all_image_pathes()
    print("length of images:%d"%len(img_paths))
    print("the first image_path in trainval:{}" \
        .format(img_paths[0]))
    print("the first image_path in trainval:{}" \
        .format(img_paths[len(img_paths) - 1]))
    print("the abs path of the first image:{}"  \
        .format(osp.abspath(img_paths[0])))

    info = imdb.get_gt_infos()

    img_paths = info["image_pathes"]
    print("length of images:%d"%len(img_paths))
    print("the first image_path in trainval:{}" \
        .format(img_paths[0]))
    print("the first image_path in trainval:{}" \
        .format(img_paths[len(img_paths) - 1]))
    print("the abs path of the first image:{}"  \
        .format(osp.abspath(img_paths[0])))
    gt_boxes = info["boxes"]
    gt_classes = info["classes"]

    assert len(gt_boxes) == len(gt_classes)
    print("length of boxes and classes: %d"%len(gt_boxes))
    print("the first 5 boxes and classes:")
    print(gt_boxes[:5])
    print(gt_classes[:5])
    print("the last 5 boxes:")
    print(gt_boxes[-5:])
    print(gt_classes[-5:])
    imdb = pascal_voc("trainval")

    lables = imdb.prepare()[2]
    print("{}".format(lables[2][0, 12, 6, 0]))






if __name__ == "__main__":
    main()
"""
