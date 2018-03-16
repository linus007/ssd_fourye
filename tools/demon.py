import _init_paths

from net.vgg16 import VGG16
from detect.detect import detector
import os
import argparse
from utils.timer import Timer
from datasets.pascal_voc import pascal_voc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modle_file"
        , default="model/save.ckpt-0", type=str)

    parser.add_argument("--img", default="000026.jpg", type=str)
    parser.add_argument("--gpu", default='0', type=str)

    parser.add_argument("--is_img", default="1", type=int)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICE"] = args.gpu

    imdb = pascal_voc("test")

    net = VGG16(imdb, False)

    detect = detector(net, imdb, args.modle_file)
    assert os.path.exists(args.img)
    detect.image_detect(args.img)

if __name__ == "__main__":
    main()
