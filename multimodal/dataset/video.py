import h5py
import numpy as np

from multimodal.dataset.multimodal import MultiModalDataset


class SubtitlesAudioDataset(MultiModalDataset):
    """
    Dataset iterating over subtitles and audio in the video dataset.
    """
    pass


class ComplementSubtitlesAudioVideoDataset(MultiModalDataset):
    """
    Dataset iterating over audio and video in the video dataset which are not part of the subtitles.
    """
    def __init__(self, *args, minimum_time=1, **kwargs):
        super(ComplementSubtitlesAudioVideoDataset, self).__init__(*args, **kwargs)
        subtitles = self.modalities['subtitles'].get_facet()
        self.times = subtitles.get_time_complement(minimum_time)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        raise NotImplementedError()


class SubtitlesVideoDataset(MultiModalDataset):
    """
    Dataset iterating over subtitles and audio in the video dataset.
    """

    pass
