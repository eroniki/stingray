import cv2
import os
import numpy as np
import json
import re


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


class frame_grabber(vision):
    """docstring for frame_grabber."""

    def __init__(self):
        super(frame_grabber, self).__init__()

    def check_video_exists(self, loc):
        pass

    def check_folder_exists(self, fname):
        pass

    def save_img(self, frame, camera, frame_id, test):
        test_name = os.path.basename(test)
        path = os.path.join("data/frames/", str(camera),
                            test_name, str(frame_id) + ".jpg")
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        cv2.imwrite(path, frame)

    def read_video(self, fname, camera, test, show=False):
        cap = cv2.VideoCapture(fname)
        id = 0
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 300, 300)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                self.save_img(frame, camera=camera, frame_id=id, test=test)
                if show is True:
                    cv2.imshow('image', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
            id += 1
        cap.release()
        cv2.destroyAllWindows()

    def find_all_videos(self, search_path, ext=".MP4",
                        serialize=True, location=None):

        found = list()

        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith(ext):
                    found.append(os.path.join(root, file))

        found.sort(key=self.natural_keys)

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

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split('(\d+)', text)]
