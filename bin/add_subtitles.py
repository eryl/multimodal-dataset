import argparse
import os.path
import glob
from multimodal.dataset.video import VideoDataset

def main():
    parser = argparse.ArgumentParser(description="Add subtitles to datasets")
    parser.add_argument('datasets', help="Dataset paths", nargs='+')
    parser.add_argument('--subtitle-format', help="Glob format to use for matching subtitles", default='*.srt')
    parser.add_argument('--subtitle-name', help="The name to use for identifying the subtitle track in the data set")
    parser.add_argument('--dry-run',
                        help="If set, don't actually do anything just print out what would be done",
                        action='store_true')
    parser.add_argument('--remove-existing', help="Remove existing subtitles in the datasets, essentially doing a subtitle reset", action='store_true')
    parser.add_argument('--minimum-matching-ratio', help="The paths need to overlap with at least this much to even be suggested to the user", type=float, default=0.7)
    args = parser.parse_args()

    dataset_paths = set()
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            datasets = glob.glob(os.path.join(dataset_path + '/**/' + '*.h5'), recursive=True)
            dataset_paths.update(datasets)
        else:
            raise ValueError("Not a directory: {}".format(dataset_path))

    subtitle_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            subtitles = glob.glob(os.path.join(dataset_path + '/**/' + args.subtitle_format), recursive=True)
            subtitle_paths.extend(subtitles)

    accepted_diff_substrings = set()
    unacceptable_diff_substrings = set()
    all_paths = [(path, 'dataset') for path in dataset_paths] + [(path, 'subtitle') for path in subtitle_paths]
    all_paths.sort()
    current_file_index = 0

    reference_pair = all_paths[current_file_index]
    collected_pairs = [reference_pair]
    all_collections = []
    for current_pair in all_paths[1:]:
        reference_path = os.path.basename(reference_pair[0])
        current_path = os.path.basename(current_pair[0])
        reference_pair = current_pair
        j = 0
        min_length = min(len(reference_path), len(current_path))
        for j in range(min_length):
            if reference_path[j] != current_path[j]:
                break
        if j/min_length < args.minimum_matching_ratio:
            all_collections.append(collected_pairs)
            collected_pairs = [current_pair]
            continue
        difference_reference = reference_path[j:]
        difference_current = current_path[j:]
        difference = frozenset([difference_reference, difference_current])
        if not difference in accepted_diff_substrings and not difference in unacceptable_diff_substrings:
            answer = input("Difference between current pairs: {}. Is this an acceptable one? (y/n)".format(difference))
            if answer.lower()[0] == 'y':
                accepted_diff_substrings.add(difference)
            else:
                unacceptable_diff_substrings.add(difference)
        if difference in accepted_diff_substrings:
            collected_pairs.append(current_pair)
        else:
            all_collections.append(collected_pairs)
            collected_pairs = [current_pair]
    all_collections.append(collected_pairs)

    for collected_pairs in all_collections:
        try:
            [dataset_path] = [path for path, type in collected_pairs if type == 'dataset']
        except ValueError:
            print("Missing datasets for {}".format(collected_pairs))
            continue
        subtitles = [path for path, type in collected_pairs if type == 'subtitle']
        if len(subtitles) > 0:
            dataset_paths.remove(dataset_path)
            print("Extracting subtitles ", dataset_path, subtitles)
            if not args.dry_run:
                with VideoDataset(dataset_path, 'r+') as dataset:
                    if args.remove_existing:
                        if args.subtitle_name is None:
                            dataset.remove_modality('subtitles')
                        else:
                            dataset.remove_facet('subtitles', args.subtitle_name)
                    if len(subtitles) > 1:
                        for i, s in enumerate(subtitles):
                            subtitle_name = args.subtitle_name
                            if subtitle_name is not None:
                                subtitle_name += str(i)
                            dataset.add_subtitles(s, name=subtitle_name)
                    else:
                        [s] = subtitles
                        dataset.add_subtitles(s, name=args.subtitle_name)
    print("Datasets without subtitles: ", sorted(dataset_paths))

if __name__ == '__main__':
    main()