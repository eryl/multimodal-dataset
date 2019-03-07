import argparse
import os.path
import glob
from multimodal.dataset.multimodal import remove_modality

def main():
    parser = argparse.ArgumentParser(description="Remove modalities from datasets")
    parser.add_argument('datasets', help="Dataset paths", nargs='+')
    parser.add_argument('modality', help="The modality to remove")
    args = parser.parse_args()

    dataset_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            datasets = glob.glob(os.path.join(dataset_path + '/**/' + '*.h5'), recursive=True)
            dataset_paths.extend(datasets)
        else:
            raise ValueError("Not a directory: {}".format(dataset_path))
    for dataset_path in dataset_paths:
        remove_modality(dataset_path, args.modality)


if __name__ == '__main__':
    main()