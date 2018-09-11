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
            fg.read_video(video_loc, c_id, test=video, show=True)


if __name__ == "__main__":
    main()
