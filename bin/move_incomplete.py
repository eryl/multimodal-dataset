import argparse
import os.path
import re
import glob
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('target_dir')
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    all_keys = set()
    os.makedirs(args.target_dir, exist_ok=True)
    for directory in args.directories:
        print(directory)
        files = []
        files.extend(glob.glob(os.path.join(directory, '**', '*.mp4'), recursive=True))
        files.extend(glob.glob(os.path.join(directory, '**', '*.srt'), recursive=True))
        files.extend(glob.glob(os.path.join(directory, '**', '*.ttml'), recursive=True))
        print(files)
        triplets = defaultdict(list)
        for file in files:
            m = re.match(r'.*_([a-z0-9]+)_[a-z]+\.(mp4|srt|ttml)', file)
            if m:
                base_name = m.group(1)
                triplets[base_name].append(file)
            else:
                print("No match for {}".format(file))

        all_keys.update(triplets.keys())
        to_move = []
        for id, triplets in triplets.items():
            if len(triplets) < 3:
                for file in triplets:
                    to_move.append((file, os.path.join(args.target_dir, os.path.basename(file))))
        if not args.test:
            for src, dst in to_move:
                os.rename(src, dst)
        else:
            print(to_move)
    print(len(all_keys))




if __name__ == '__main__':
    main()