"""
Set up lib path for ssd
"""
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

lib_dir = osp.join(this_dir, "..", "lib")
add_path(lib_dir)
