"""Script for extracting video-snippets with dialogue"""

import argparse
import os
import os.path
import re
import glob
import csv
import srt
import numpy as np
import ffmpeg
import subprocess
import time
import h5py
import imageio
import itertools
import html.parser
import multiprocessing

from multimodal.dataset.facet import AudioFacet, VideoFacet, SubtitleFacet


class SubtitleParser(html.parser.HTMLParser):
    def __init__(self):
        super(SubtitleParser, self).__init__()
        self.color = None
        self.data = None

    def handle_starttag(self, tag, attrs):
        if tag == 'font':
            if attrs[0][0] == 'color':
                color = attrs[0][1].strip('#')
                self.color = (int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16))
    def handle_data(self, data):
        self.data = data

    def wipe(self):
        self.data = None
        self.color = None

class VideoConverter(multiprocessing.Process):
    pass

def main():
    parser = argparse.ArgumentParser(description="Script for making a multimodal dataset from video")
    parser.add_argument('input', help="Either a CSV containing an index of video files to process, "
                                      "or a single video file with corresponding subtitles", nargs='+')
    parser.add_argument('--nprocesses', help="Number of processes to use for creating datasets", type=int, default=1)
    parser.add_argument('--no-video', help="Don't extract the video stream", action='store_true')
    args = parser.parse_args()

    if '.csv' in args.input:
        with open(args.input) as fp:
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
        video_queue = multiprocessing.Queue(args.nprocesses*2)
        processes = [VideoConverter(video_queue) for i in range(args.nprocesses)]
    else:
        for video in videos:
            video_file = video['mp4']
            subtitles_file = video['srt']
            uncompress_video(video_file, subtitles_file, skip_video=args.no_video)


def extract_audio(store, video_name):
    out, _ = (ffmpeg
              .input(video_name)
              .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
              .overwrite_output()
              .run(capture_stdout=True)
              )
    data = np.frombuffer(out, np.int16)
    audio_modality = store.create_group('audio')
    audio_facet = AudioFacet.create_facet('audio1', audio_modality, data, 16000)


def extract_subtitles(store, subtitles_name):
    with open(subtitles_name) as fp:
        subtitles = list(srt.parse(fp.read()))
    subtitle_parser = SubtitleParser()

    font_colors = dict()
    color_i = 0
    colors = []
    texts = []
    times = []
    for subtitle in subtitles:
        subtitle_parser.wipe()
        start = subtitle.start.total_seconds()
        end = subtitle.end.total_seconds()
        subtitle_parser.feed(subtitle.content)
        text = subtitle_parser.data
        color_tuple = subtitle_parser.color
        if color_tuple is not None:
            if not color_tuple in font_colors:
                font_colors[color_tuple] = color_i
                color_i += 1
            color = font_colors[color_tuple]
        else:
            color = -1
        times.append([start, end])
        colors.append(color)
        texts.append(text)
    subtitles_modality = store.create_group('subtitles')
    subtitles_color_index = np.array([color for color, i in sorted(font_colors.items(), key=lambda x: x[1])],
                                     np.uint8)
    subtitles_facet = SubtitleFacet.create_facet(os.path.basename(subtitles_name), subtitles_modality, subtitles_color_index, colors, texts, times)


def extract_video(store, video_name, chunksize=2 ** 24):
    video_reader = imageio.get_reader(video_name)
    video_metadata = video_reader.get_meta_data()
    fps = video_metadata['fps']
    nframes = video_metadata['nframes']
    width, height = video_metadata['size']
    video_reader.close()


    if width < height:
        ratio = 256 / width
        width = 256
        height = int(height * ratio)
    else:
        ratio = 256 / height
        height = 256
        width = int(width * ratio)
    video_reader = imageio.get_reader(video_name, size=(width, height))
    frame_size = 8 * width * height * 3
    frames_per_chunk = chunksize // frame_size
    chunk_height = 2 ** 20 / (3 * width * 8)
    if chunk_height > height:
        chunk_height = height

    video_modality = store.create_group('video')
    video_facet = VideoFacet.create_facet('video0',
                                          video_modality,
                                          shape=(nframes, height, width, 3),
                                          dtype=np.uint8,
                                          rate=fps)


    n_chunks = int(np.ceil(nframes / frames_per_chunk))
    frame_iter = iter(video_reader)
    for i in range(n_chunks):
        print("Chunk {}/{}".format(i, n_chunks))
        frames = np.array(list(itertools.islice(frame_iter, frames_per_chunk)))
        video_facet.write(i * frames_per_chunk, frames)


def uncompress_video(video_name, subtitles_name, skip_video=False, chunksize=2 ** 24):
        print("Extracting video {} with subtitles".format(video_name, subtitles_name))
        store_name = '{}.h5'.format(os.path.splitext(video_name)[0])
        with h5py.File(store_name, 'w') as store:
            extract_audio(store, video_name)
            extract_subtitles(store, subtitles_name)
            if not skip_video:
                extract_video(store, video_name, chunksize)



if __name__ == '__main__':
    main()

