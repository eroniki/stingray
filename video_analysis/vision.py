import cv2
import os
import numpy as np
import json


class vision(object):
    """docstring for vision."""

    def __init__(self, arg):
        super(vision, self).__init__()
        self.arg = arg

    def hough(self, arg):
        pass

    def blob_analysis(self, arg):
        pass

    def triangulate(self, arg):
        pass


class frame_grabber(object):
    """docstring for frame_grabber."""

    def __init__(self):
        super(frame_grabber, self).__init__()

    def check_video_exists(self, loc):
        pass

    def check_folder_exists(self, fname):
        pass

    def save_img(self, frame, camera, id, test):
        pass

    def read_video(self, fname, show=False):
        cap = cv2.VideoCapture(fname)

        while(cap.isOpened()):
            ret, frame = cap.read()
            self.save_img(frame, id)
            if show is True:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def find_all_videos(self, search_path, ext=".MP4",
                        serialize=True, location=None):

        found = list()

        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith(ext):
                    found.append(os.path.join(root, file))

        found.sort()
        if serialize:
            return self.serialize(list=found, location=location)
        else:
            return found

    def serialize(self, list, save=True, location=None):
        d = {"size": len(list), "videos": list}
        if save is True:
            if location is None:
                raise ValueError('Error: location was not specified')
            with open(location, 'w') as outfile:
                json.dump(d, outfile, indent=4)

        return json.dumps(d, indent=4)

    def load_videolist(self, path):
        with open(path) as f:
            data = json.load(f)
        return data
