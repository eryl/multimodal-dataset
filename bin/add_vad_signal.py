"""
Goes through a dataset and add dataset to audio tracks with Voice Activity Detection data.
"""
import multiprocessing.dummy as multiprocessing
import os.path
import argparse
import numpy as np
import scipy.io.wavfile as wavefile

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
    args = parser.parse_args()
    if args.n_processes > 1:
        with multiprocessing.Pool(args.n_processes) as pool:
            for dataset_path in args.datasets:
                pool.apply_async(add_voiced_segment_facet, (dataset_path, args.mode))
            pool.close()
            pool.join()
    else:
        for dataset_path in args.datasets:
            add_voiced_segment_facet(dataset_path, args.mode)



def save_waves(voiced_frames, sample_rate, dataset_path):
    for start, end, audio in voiced_frames:
        filename = os.path.splitext(dataset_path)[0] + '{:.03f}-{:.03f}.wav'.format(start/sample_rate, end/sample_rate)
        wavefile.write(filename, 16000, audio)
    filename = os.path.splitext(dataset_path)[0] + '.wav'
    wavefile.write(filename, 16000, np.concatenate([audio for start, end, audio in voiced_frames]))


if __name__ == '__main__':
    main()