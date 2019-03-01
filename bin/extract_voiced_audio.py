"""
Example of how to use the SubtitleAudio dataset
"""
import argparse
import os.path
import unicodedata
import re
import numpy as np
import scipy.io.wavfile as wavefile

from multimodal.dataset.video import VideoDataset


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    From Django: https://docs.djangoproject.com/en/2.1/_modules/django/utils/text/#slugify
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def main():
    parser = argparse.ArgumentParser(description="Script for extracting audio segments corresponding to the subtitles "
                                                 "in the given multimodal datasets")
    parser.add_argument('datasets', help="The multimodal video datasets to load. These should be HDF5 files "
                                         "produced by the make_multimodal_dataset.py from video files with subtitles",
                        nargs='+')
    parser.add_argument('output_dir', help="Directory to output files to")
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = '.'
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    for dataset_path in args.datasets:
        with VideoDataset(dataset_path) as dataset:
            sample_rate = dataset.get_samplerate('audio')
            voiced_segments = dataset.get_time_interval_frames('voiced_segments', 'audio')
            n_digits = 7
            digit_format = '{{:0{}d}}_'.format(n_digits)
            for i, (times, voiced_segment) in enumerate(voiced_segments):
                filename = os.path.join(output_dir, os.path.splitext(os.path.basename(dataset_path))[0] + digit_format.format(i) + '.wav')
                wavefile.write(filename, sample_rate, voiced_segment)


if __name__ == '__main__':
    main()