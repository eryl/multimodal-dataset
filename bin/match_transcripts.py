import argparse
import os.path
import glob

def main():
    parser = argparse.ArgumentParser(description="Add subtitles to datasets")
    parser.add_argument('transcripts_folder', help="Transcripts folder")
    parser.add_argument('dataset_folder', help="Dataset paths")
    parser.add_argument('--remove-existing', help="Remove existing subtitles in the datasets, essentially doing a subtitle reset", action='store_true')
    args = parser.parse_args()

    dataset_paths = []
    if os.path.isdir(args.dataset_folder):
        datasets = glob.glob(os.path.join(args.dataset_folder + '/**/' + '*.h5'), recursive=True)
        dataset_paths.extend(datasets)
    else:
        raise ValueError("Not a directory: {}".format(args.dataset_folder))

    transcripts_paths = []
    if os.path.isdir(args.transcripts_folder):
        transcripts = glob.glob(os.path.join(args.transcripts_folder + '/**/' + '*.json'), recursive=True)
        transcripts_paths.extend(transcripts)

    datasets = dict()
    for dataset_path in dataset_paths:
        dataset, rest = dataset_path.rsplit('_', maxsplit=1)
        if not dataset:
            dataset = rest
        dataset_name = os.path.basename(dataset)
        datasets[dataset_name] = dataset_path

    transcripts = dict()
    for transcript_path in transcripts_paths:
        transcript, rest = transcript_path.rsplit('_', maxsplit=1)
        if not transcript:
            transcript = rest
        transcript_name = os.path.basename(transcript)
        transcripts[transcript_name] = transcript_path

    matched_datasets = []
    for transcript_name, transcript in transcripts.items():
        if transcript_name in datasets:
            matched_datasets.append(datasets[transcript_name])
    for matched_dataset in matched_datasets:
        print(matched_dataset)

if __name__ == '__main__':
    main()