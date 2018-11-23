import argparse
import os.path
import re
import glob
from collections import defaultdict
import imageio
import sys
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--output')
    args = parser.parse_args()

    video_data = defaultdict(dict)
    for directory in args.directories:
        print(directory)
        files = []
        files.extend(glob.glob(os.path.join(directory, '**', '*.mp4'), recursive=True))
        files.extend(glob.glob(os.path.join(directory, '**', '*.srt'), recursive=True))
        files.extend(glob.glob(os.path.join(directory, '**', '*.ttml'), recursive=True))

        for file in files:
            m = re.match(r'.*_([a-z0-9]+)_[a-z]+\.(mp4|srt|ttml)', file)
            if m:
                base_name = m.group(1)
                ext = m.group(2)
                video_data[base_name][ext] = file
                if ext == 'mp4':
                    video_reader = imageio.get_reader(file)
                    video_metadata = video_reader.get_meta_data()
                    fps = video_metadata['fps']
                    nframes = video_metadata['nframes']
                    duration = nframes/fps
                    video_reader.close()
                    video_data[base_name]['duration'] = duration

            else:
                print("No match for {}".format(file))

    if args.output is not None:
        fp = open(args.output, 'w')
    else:
        fp = sys.stdout
    csv_writer = csv.DictWriter(fp, fieldnames=['mp4', 'srt', 'ttml', 'duration'])
    csv_writer.writeheader()
    csv_writer.writerows(sorted(video_data.values(), key=lambda x: x['mp4']))




if __name__ == '__main__':
    main()