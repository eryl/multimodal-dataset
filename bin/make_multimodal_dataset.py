"""Script for extracting video-snippets with dialogue"""

import argparse
import csv
import multiprocessing

from multimodal.dataset.make_video_dataset import make_dataset


def main():
    parser = argparse.ArgumentParser(description="Script for making a multimodal dataset from video")
    parser.add_argument('input', help="Either a CSV containing an index of video files to process, "
                                      "or a single video file with corresponding subtitles", nargs='+')
    parser.add_argument('--nprocesses', help="Number of processes to use for creating datasets", type=int, default=1)
    parser.add_argument('--no-video', help="Don't extract the video stream", action='store_true')
    args = parser.parse_args()

    if '.csv' in args.input[0]:
        with open(args.input[0]) as fp:
            csv_reader = csv.DictReader(fp)
            videos = list(csv_reader)
    else:
        video_files = []
        subtitles = []
        for file in args.input:
            if '.srt' in file:
                subtitles.append(file)
            else:
                video_files.append(file)
        if not(video_files) or len(video_files) != len(subtitles):
            raise ValueError("Input files is not valid. The same number of video files as subtitles must be supplied")

        videos = [dict(mp4=video, srt=subtitle) for video, subtitle in zip(video_files, subtitles)]

    if args.nprocesses > 1:
        pool = multiprocessing.Pool(args.nprocesses)
        for video in videos:
            video_file = video['mp4']
            subtitles_file = video['srt']
            pool.apply_async(make_dataset, (video_file, subtitles_file), dict(skip_video=args.no_video))
        pool.close()
        pool.join()
    else:
        for video in videos:
            video_file = video['mp4']
            subtitles_file = video['srt']
            make_dataset(video_file, subtitles_file, skip_video=args.no_video)


if __name__ == '__main__':
    main()

