from video_analysis.vision import *
from video_analysis.config import *
from video_analysis.data import *
import os
import sys
import numpy as np


def main(fname):
    if len(fname) > 0:
        fname = fname[0]
    else:
        fname = "config.json"

    fg = frame_grabber()
    io = input_output()
    cp = config_parser(fname)

    config = cp.parse()
    fnames = io.construct_filenames(config)
    K = io.construct_intrinsic_matrix(config)
    resize_scale = config["preprocess"]["resize_scale"]
    recolor_threshold = config["preprocess"]["recolor_threshold"]

    for f in fnames:
        print "Processing:",
        chunks = f.split("/")
        cid = chunks[-2]
        tid = chunks[-1]
        print f
        d = np.asarray(config["distortion"][cid])

        fg.read_video(f, camera=cid, test=tid, K=K[int(cid)],
                      dist=d, scale=resize_scale, clip=recolor_threshold)


if __name__ == "__main__":
    main(sys.argv[1:])
