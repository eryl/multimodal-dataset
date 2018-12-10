"""Script for extracting video-snippets with dialogue"""

import argparse
import csv
import multiprocessing
from collections import defaultdict

from multimodal.dataset.make_video_dataset import make_dataset


def main():
    parser = argparse.ArgumentParser(description="Script for making a multimodal dataset from video")
    parser.add_argument('input', help="Either a CSV containing an index of video files to process, "
                                      "or a single video file followed by zero or more corresponding subtitles. "
                                      "The first subtitle will be flagged as default.", nargs='+')
    parser.add_argument('--nprocesses', help="Number of processes to use for creating datasets", type=int, default=1)
    parser.add_argument('--skip-video', help="Don't extract the video stream", action='store_true')
    parser.add_argument('--skip-audio', help="Don't extract the audio stream", action='store_true')
    parser.add_argument('--target-width',
                        help="Scale video to have this width at most. Height will be rescaled to keep aspect ratio. "
                             "If both --target-width and --target-height is given, both will be used, potentially "
                             "giving a different aspect-ratio",
                        type=int)
    parser.add_argument('--target-height',
                        help="Scale video to have this height at most. Width will be rescaled to keep the aspect ratio",
                        type=int)
    args = parser.parse_args()

    if '.csv' in args.input[0]:
        with open(args.input[0]) as fp:
            csv_reader = csv.DictReader(fp)
            videos = list(csv_reader)
    else:
        video_files = []
        subtitles = defaultdict(list)
        current_video = None
        for file in args.input:
            if '.srt' in file:
                subtitles[current_video].append(file)
            else:
                video_files.append(file)
                current_video = file
        videos = [dict(mp4=video, srt=subtitles.get(video, None)) for video in video_files]

    if args.nprocesses > 1:
        pool = multiprocessing.Pool(args.nprocesses)
        for video in videos:
            video_file = video['mp4']
            subtitles_files = video['srt']
            pool.apply_async(make_dataset, (video_file, subtitles_files), dict(skip_video=args.skip_video, skip_audio=args.skip_audio, video_size=(args.target_width, args.target_height)))
        pool.close()
        pool.join()
    else:
        for video in videos:
            video_file = video['mp4']
            subtitles_files = video['srt']
            make_dataset(video_file, subtitles_files, skip_video=args.skip_video, skip_audio=args.skip_audio, video_size=(args.target_width, args.target_height))


if __name__ == '__main__':
    main()

