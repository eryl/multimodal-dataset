import glob
import argparse
from multimodal.dataset.video import VideoDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('output')
    parser.add_argument('--subrip', help="If this flag is set, write to subrip format, otherwise just output the text without time information", action='store_true')
    args = parser.parse_args()

    h5_files = glob.glob(args.directory + '/**/*.h5', recursive=True)
    with open(args.output, 'w') as output_fp:
        for dataset_path in h5_files:
            with VideoDataset(dataset_path) as dataset:
                subtitles_facet = dataset.get_facet('subtitles')
                if args.subrip:
                    output_fp.write(subtitles_facet.get_subrip_texts())
                else:
                    output_fp.write('\n'.join(subtitles_facet.get_texts()))




if __name__ == '__main__':
    main()