from video_analysis.vision import *
import os
import matplotlib.pyplot as plt


def prep():
    fg = frame_grabber()

    cameras = ["data/raw_video/1/",
               "data/raw_video/2/",
               "data/raw_video/3/",
               "data/raw_video/4/"]

    video_list_json = ["data/output/videolist_c1.json",
                       "data/output/videolist_c2.json",
                       "data/output/videolist_c3.json",
                       "data/output/videolist_c4.json"]

    video_list = list()

    for c_id, c in enumerate(cameras):
        json = fg.find_all_videos(search_path=c,
                                  location=video_list_json[c_id])
        videos = fg.load_videolist(video_list_json[c_id])
        video_list.append(videos)
        for v_id, video in enumerate(videos["videos"]):
            video_loc = str(video)
            print video_loc
            fg.read_video(video_loc, c_id, test=video, show=False)


def main_new():
    fg = frame_grabber()
    tests = ["S5A20.MP4", "S10A50.MP4", "S20A60.MP4"]

    cameras_ = [0, 1]
    img_path_list_for_all_test = list()
    for t_id, test in enumerate(tests):
        img_path_list = list()
        for c_id, c in enumerate(cameras_):
            print "test id:", t_id, test, "camera_id:", c_id, c
            path = os.path.join('data/frames/', str(c), test + "/")
            img_path_list_ = fg.find_all_imgs(search_path=path,
                                              serialize=False)

            img_path_list.append(img_path_list_)
        img_path_list_for_all_test.append(img_path_list)

    test_id = 0

    img_path_list = img_path_list_for_all_test[test_id]

    n_file = len(img_path_list[0])
    n_c = len(cameras_)
    h = 540
    w = 960
    points = np.zeros((n_file, 26, n_c))
    F = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
    points_all = list()
    for f_id in range(n_file):
        print f_id,
        montage = np.zeros((h, w * n_c, 3), dtype=np.uint8)
        matched = np.zeros((26, n_c))

        points_all_cams = list()
        for c_id, c in enumerate(cameras_):
            print img_path_list[c_id][f_id],
            img = fg.load_img(img_path_list[c_id][f_id])

            montage[:, c_id * w: (c_id + 1) * w, :] = img
            roi = img[160: 380, 90: 870, :]
            matches, circles, kps, \
                canvas_circles, canvas_blobs, canvas_fused = fg.analyze_roi(
                    roi)

            points = list()
            for b_id, row in enumerate(matches):
                if (np.any(row)):
                    idx = np.where(row == 1)[0]
                    for candidate in idx:
                        x = (circles[candidate, 0] + (c_id)
                             * w + 90).astype(np.int16)
                        y = (circles[candidate, 1] + 160).astype(np.int16)
                        r = (circles[candidate, 2]).astype(np.int16)
                        p = np.array([circles[candidate, 0].astype(np.int16),
                                      circles[candidate, 1].astype(np.int16),
                                      1])
                        points.append(p)
                        cv2.circle(montage, (x, y), r, (0, 255, 0), 4)
            print len(points),
            points_all_cams.append(points),
        print
        points_all.append(points_all_cams)
        print len(points_all_cams)

        cv2.imshow("Keypoints Filtered", montage)
        k = cv2.waitKey(10)
        if k == 27:
            break
            cv2.destroyAllWindows()

    print len(points_all),
    qq = np.asarray(points_all, dtype=object)
    print qq.shape, len(qq[0, 0])
    ffname = tests[test_id] + str(cameras_[0]) + \
        "_" + str(cameras_[1]) + ".npy"
    np.save(ffname, qq)

    # print len(points_all), len(points_all[0]), len(points_all[1])


if __name__ == "__main__":
    main_new()
