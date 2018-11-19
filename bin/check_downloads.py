import os.path
import re
import argparse
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Check missing files')
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()
    pattern = r"(\w+_\d*_-_\d*)\..*"
    collected_files = defaultdict(list)
    for directory in args.directories:
        for file in sorted(os.listdir(directory)):
            try:
                m = re.match(r'.*_([a-z0-9]+)_[a-z]+\.(mp4|srt|ttml)', file)
                if m:
                    name = m.group(1)
                    collected_files[name].append(file)
                else:
                    print("Match failed, going for split, ", file)
                    name, *rest = file.split('-')
                    if rest:
                        name += '-' + rest[0].split('_')[1]
            except ValueError as e:
                print(e)
                print("When processing ", file)
    missing_video = []
    missing_subtitles = []
    for name, files in sorted(collected_files.items()):
        if len(files) < 3:
            mp4_present = False
            subtitles_present = False
            for file in files:
                if '.mp4' in file:
                    mp4_present = True
                if '.srt' in file or '.ttml' in file:
                    subtitles_present = True
            if not mp4_present:
                missing_video.append(name)
            if not subtitles_present:
                missing_subtitles.append(name)
    print("{} missing subtitles:".format(len(missing_subtitles)), ','.join(missing_subtitles))




if __name__ == '__main__':
    main()
