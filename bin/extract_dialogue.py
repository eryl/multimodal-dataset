"""Script for extracting video-snippets with dialogue"""

import argparse
import os
import os.path
import re
import glob
import srt
import numpy as np
import ffmpeg
import subprocess
import time
import h5py
import imageio
import itertools
import html.parser


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


def main():
    parser = argparse.ArgumentParser(description="Script for extracrting video snippets with dialogue")
    parser.add_argument('store', help="Path to h5py to store data to")
    parser.add_argument('videos', nargs='+', help="Videos or directories to process")
    args = parser.parse_args()
    videos = []
    not_processed = set(args.videos)
    video_pattern = r'.*_([a-z0-9]+)_[a-z]+\.mp4'
    while not_processed:
        video = not_processed.pop()
        if os.path.isdir(video):
            not_processed.update(os.listdir(video))
        else:
            m = re.match(video_pattern, video)
            if m:
                videos.append((m.group(1), video))
    for pid, video_name in videos:
        subtitles, = glob.glob(os.path.join(os.path.dirname(video_name), '*{}*.srt'.format(pid)))
        extract_dialogue(args.store, video_name, subtitles)


def extract_dialogue(store_path, video_name, subtitle_name, chunksize=2**24):
    with h5py.File(store_path, 'w') as store:
        movie_group = store.create_group(video_name)

        out, _ = (ffmpeg
                  .input(video_name)
                  .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True)
                  )
        sound = np.frombuffer(out, np.int16)
        sound_dataset = movie_group.create_dataset('sound', data=sound, chunks=True, compression='gzip', shuffle=True)
        sound_dataset.attrs['ar'] = 16000
        video_reader = imageio.get_reader(video_name)
        video_metadata = video_reader.get_meta_data()
        fps = video_metadata['fps']
        nframes = video_metadata['nframes']
        width, height = video_metadata['size']
        video_reader.close()

        with open(subtitle_name) as fp:
            subtitles = list(srt.parse(fp.read()))
        subtitle_parser = SubtitleParser()

        parsed_subtitles = []
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
        subtitles_group = movie_group.create_group('subtitles')
        subtitles_color_index = np.array([color for color,i in sorted(font_colors.items(), key=lambda x: x[1])], np.uint8)
        subtitles_color_index_dataset = subtitles_group.create_dataset('color_index', data=subtitles_color_index)
        subtitle_color = subtitles_group.create_dataset('colors', data=np.array(colors, dtype=np.int8))
        subtitle_dataset = subtitles_group.create_dataset('texts', shape=(len(texts),), dtype=h5py.special_dtype(vlen=str))
        subtitle_dataset[:] = texts
        subtitle_times = subtitles_group.create_dataset('times', data=np.array(times, dtype=np.float32))

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
        chunk_height = 2**20 / (3*width*8)
        if chunk_height > height:
            chunk_height = height
        video_dataset = movie_group.create_dataset('video',
                                                   shape=(nframes, height, width, 3),
                                                   dtype=np.uint8,
                                                   chunks=(1, height, width, 1),
                                                   compression='gzip',
                                                   shuffle=True)
        video_dataset.attrs['fps'] = fps
        n_chunks = int(np.ceil(nframes / frames_per_chunk))
        frame_iter = iter(video_reader)
        for color in range(n_chunks):
            print("Chunk {}/{}".format(color, n_chunks))
            frames = np.array(list(itertools.islice(frame_iter, frames_per_chunk)))
            video_dataset[color*frames_per_chunk:(color+1)*frames_per_chunk] = frames




if __name__ == '__main__':
    main()

