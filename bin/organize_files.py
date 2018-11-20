import argparse
import os.path
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    args = parser.parse_args()

    for directory in args.directories:
        for file in os.listdir(directory):
            m = re.match(r'(.*)_-_.*\.(mp4|srt|ttml)', file)
            if m:
                base_name = m.group(1)
                if not os.path.exists(base_name):
                    os.makedirs(base_name)
                os.rename(file, os.path.join(base_name, file))




if __name__ == '__main__':
    main()