"""
Goes through a dataset and add dataset to audio tracks with Voice Activity Detection data.
"""
import multiprocessing.dummy as multiprocessing
import os.path
import argparse
import numpy as np
import scipy.io.wavfile as wavefile
import glob

from multimodal.dataset.add_vad_signal import add_voiced_segment_facet

def main():
    parser = argparse.ArgumentParser(description="Script for finding speech in the audio "
                                                 "modality of multimodal datasets")
    parser.add_argument('datasets', help="Datasets to process", nargs='+')
    parser.add_argument('--mode',
                        help="WEBRTC VAD mode, 0 gives most false positives, 3 gives most false negatives",
                        type=int,
                        choices=(0,1,2,3),
                        default=3)
    parser.add_argument('--n-processes', type=int, default=1)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    dataset_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            datasets = glob.glob(os.path.join(dataset_path + '/**/' + '*.h5'), recursive=True)
            dataset_paths.extend(datasets)
        elif os.path.isfile(dataset_path):
            dataset_paths.append(dataset_path)
    print("Dataset paths: ", dataset_paths)

    if args.n_processes > 1:
        with multiprocessing.Pool(args.n_processes) as pool:
            for dataset_path in dataset_paths:
                pool.apply_async(add_voiced_segment_facet, (dataset_path, args.mode), kwds={'overwrite': args.overwrite})
            pool.close()
            pool.join()
    else:
        for dataset_path in dataset_paths:
            add_voiced_segment_facet(dataset_path, args.mode, overwrite=args.overwrite)



def save_waves(voiced_frames, sample_rate, dataset_path):
    for start, end, audio in voiced_frames:
        filename = os.path.splitext(dataset_path)[0] + '{:.03f}-{:.03f}.wav'.format(start/sample_rate, end/sample_rate)
        wavefile.write(filename, 16000, audio)
    filename = os.path.splitext(dataset_path)[0] + '.wav'
    wavefile.write(filename, 16000, np.concatenate([audio for start, end, audio in voiced_frames]))


if __name__ == '__main__':
    main()