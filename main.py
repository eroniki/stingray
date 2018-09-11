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


def main():
    fg = frame_grabber()
    tests = ["S5A20.MP4", "S10A50.MP4", "S20A60.MP4"]

    cameras_ = [0, 1]

    for t_id, test in enumerate(tests):
        for c_id, c in enumerate(cameras_):
            print "test id:", t_id, test, "camera_id:", c_id, c
            path = os.path.join('data/frames/', str(c), test + "/")
            img_path_list = fg.find_all_imgs(search_path=path, serialize=False)
            # print img_path_list
            for i_id, img_path in enumerate(img_path_list):
                img = fg.load_img(img_path)
                img = img[160:380, 90:870, :]

                kps, img2 = fg.blob_analysis(img)
                canvas = np.copy(img2)
                canvas2 = img.copy()

                rect = fg.get_rois_from_blobs(kps)

                try:
                    circles = fg.hough(img)
                    # circles = circles[0]
                except ValueError:
                    circles = None

                if circles is not None:
                    for i in range(len(circles)):
                        x, y, r = circles[i]
                        cv2.circle(canvas, (x, y), r, (0, 255, 0), 4)
                    matches, canvas = fg.find_blobs_containing_circle(canvas2,
                                                                      rect,
                                                                      circles)

                cv2.imshow("Keypoints Filtered", canvas2)
                cv2.imshow("Keypoints", canvas)
                k = cv2.waitKey(10)
                if k == 27:
                    break
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
