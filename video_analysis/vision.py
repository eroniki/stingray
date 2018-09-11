import cv2
import os
import numpy as np
import json
import re


class vision(object):
    """docstring for vision."""

    def __init__(self):
        super(vision, self).__init__()
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByArea = True
        self.params.minArea = 150
        self.params.maxArea = 500

        self.params.filterByCircularity = True
        self.params.minCircularity = 0.1

        self.params.minThreshold = 60
        self.params.maxThreshold = 175

        self.params.filterByColor = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False

        self.params.thresholdStep = 20
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def load_img(self, path):
        return cv2.imread(path)

    def hough(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                             param1=250, param2=10,
                             dp=2,
                             minRadius=10,
                             maxRadius=15,
                             minDist=30)
        if c is None:
            raise ValueError("No circles found, change Hough params")
        else:
            return c[0]

    def blob_analysis(self, img):
        keypoints = self.detector.detect(img)
        im_with_keypoints = cv2.drawKeypoints(img, keypoints,
                                              np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow("Keypoints", im_with_keypoints)
        # cv2.waitKey(0)
        return keypoints, im_with_keypoints

    def get_rois_from_blobs(self, kps):
        rect = np.zeros([len(kps), 4], dtype=np.float64)
        for k_id, kp in enumerate(kps):
            r = kp.size
            min_bound = (np.asarray(kp.pt) - 1.1 * r).astype(np.int)
            max_bound = (np.asarray(kp.pt) + 1.1 * r).astype(np.int)
            min_bound = np.clip(min_bound, 0, None)

            h, w = max_bound - min_bound
            # TODO: Implement this
            # max_bound = np.clip(max_bound, None,)
            rect[k_id, :] = np.array([kp.pt[0] - w / 2,
                                      kp.pt[1] - h / 2,
                                      w, h])

        return rect

    def find_blobs_containing_circle(self, canvas, rect, circles):
        matches = np.zeros([len(rect), len(circles)], dtype=np.int8)

        for r_id, r in enumerate(rect):
            for c_id, c in enumerate(circles):
                contains = self.check_blob_contains_point(r, c[0:2])
                matches[r_id, c_id] += contains
                xc, yc, rc = c
                if contains:
                    cv2.circle(canvas, (xc, yc), rc, (0, 255, 0), 4)
        return matches, canvas

    def triangulate(self, arg):
        pass

    def resize_img(self, img, f):
        return cv2.resize(img, (0, 0), fx=f, fy=f)

    def recolor_img(self, img, clip, tile_size=8):
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip,
                                tileGridSize=(tile_size, tile_size))
        img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
        return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    def check_blob_contains_point(self, bb, p):
        if p[0] > bb[0] and p[0] < (bb[0] + bb[2]):
            if p[1] > bb[1] and p[1] < (bb[1] + bb[3]):
                return True
            else:
                return False
        else:
            return False


class frame_grabber(vision):
    """docstring for frame_grabber."""

    def __init__(self):
        super(frame_grabber, self).__init__()

    def check_folder_exists(self, folder):
        pass

    def check_file_exist(self, path):
        return os.path.exists(path)

    def preprocess(self, img, scale, clip):
        img = self.resize_img(img, scale)
        img = self.recolor_img(img, clip=1.0)
        return img

    def save_img(self, frame, camera, frame_id, test):
        test_name = os.path.basename(test)
        path = os.path.join("data/frames/", str(camera),
                            test_name, str(frame_id) + ".jpg")
        if not os.path.exists(os.path.dirname(path)):
            try:
                os.makedirs(os.path.dirname(path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise IOError("Couldn't handle the folder situations")
        if os.path.exists(path):
            raise IOError("Image was already processed!")

        cv2.imwrite(path, frame)

    def read_video(self, fname, camera, test, show=False):
        cap = cv2.VideoCapture(fname)
        id = 0
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 300, 300)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                try:
                    frame = self.preprocess(frame, scale=0.5, clip=1.0)
                    self.save_img(frame, camera=camera, frame_id=id, test=test)
                except Exception as e:
                    print e
                    break

                if show is True:
                    cv2.imshow('image', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
            id += 1
        cap.release()
        cv2.destroyAllWindows()

    def find_all_imgs(self, search_path, ext=".jpg",
                      serialize=False, location=None):
        return self.find_all_videos(search_path=search_path,
                                    ext=ext,
                                    serialize=serialize,
                                    location=location)

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
