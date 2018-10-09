import json
import cv2
import numpy as np


class input_output(object):
    """docstring for input_output."""

    def __init__(self):
        super(input_output, self).__init__()

    def load_img(path):
        ret, img = cv2.imread(path)
        # assert ret is True:
        #     pass

    def construct_filenames(self, config):
        cameras = config["cameras"]
        tests = config["tests"]

        files = list()
        vlocation = config["raw_video"]
        for c in cameras:
            for speed in tests:
                for amplitude in tests[speed]:
                    fname = "{vlocation}/{cam}/S{speed}A{amplitude}.MP4".format(
                        vlocation=vlocation, speed=speed, amplitude=amplitude,
                        cam=c
                    )
                    files.append(fname)
        return files

    def construct_intrinsic_matrix(self, config):
        cameras = config["cameras"]
        cameras.sort()
        print cameras
        K_list = list()
        for c_id, c in enumerate(cameras):
            K = np.eye(3)
            intrinsics = config["intrinsics"][c]
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]
            skew = intrinsics["skew"]
            K[0, 0] = fx
            K[0, 1] = skew
            K[0, 2] = cx
            K[1, 1] = fy
            K[1, 2] = cy
            K_list.append(K)
        return K_list
