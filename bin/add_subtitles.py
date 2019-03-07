import argparse
import os.path
import glob
from multimodal.dataset.video import VideoDataset

def main():
    parser = argparse.ArgumentParser(description="Add subtitles to datasets")
    parser.add_argument('datasets', help="Dataset paths", nargs='+')
    parser.add_argument('--remove-existing', help="Remove existing subtitles in the datasets, essentially doing a subtitle reset", action='store_true')
    args = parser.parse_args()

    dataset_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            datasets = glob.glob(os.path.join(dataset_path + '/**/' + '*.h5'), recursive=True)
            dataset_paths.extend(datasets)
        else:
            raise ValueError("Not a directory: {}".format(dataset_path))

    subtitle_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            subtitles = glob.glob(os.path.join(dataset_path + '/**/' + '*.srt'), recursive=True)
            subtitle_paths.extend(subtitles)

    datasets = dict()
    for dataset_path in dataset_paths:
        dataset, rest = dataset_path.rsplit('_', maxsplit=1)
        if not dataset:
            dataset = rest
        dataset_name = os.path.basename(dataset)
        datasets[dataset_name] = dataset_path

    subtitles = dict()
    for subtitle_path in subtitle_paths:
        subtitle, rest = subtitle_path.rsplit('_', maxsplit=1)
        if not subtitle:
            subtitle = rest
        subtitle_name = os.path.basename(subtitle)
        subtitles[subtitle_name] = subtitle_path

    no_subtitles = []
    found_pairs = []
    for dataset_name, dataset in datasets.items():
        if dataset_name in subtitles:
            found_pairs.append((dataset, subtitles[dataset_name]))
        else:
            no_subtitles.append(dataset)

    print("Videos with no subtitles", no_subtitles)
    for dataset_path, subtitles_path in found_pairs:
        with VideoDataset(dataset_path, 'r+') as dataset:
            if args.remove_existing:
                dataset.remove_modality('subtitles')
            print("Extracting subtitles ", subtitles_path)
            dataset.add_subtitles(subtitles_path)


if __name__ == '__main__':
    main()