from video_analysis.vision import *
import os


def grab_frames(json):
    fb = frame_grabber()
    fb.load_videolist(json)


def main():

    fb = frame_grabber()

    cameras = ["data/raw_video/1/",
               "data/raw_video/2/",
               "data/raw_video/3/",
               "data/raw_video/4/"]

    video_list = ["data/output/videolist_c1.json",
                  "data/output/videolist_c2.json",
                  "data/output/videolist_c3.json",
                  "data/output/videolist_c4.json"]

    for i, c in enumerate(cameras):
        json = fb.find_all_videos(search_path=c, location=video_list[i])


if __name__ == "__main__":
    main()
