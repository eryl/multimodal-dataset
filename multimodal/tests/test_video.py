import unittest
import numpy as np
from multimodal.dataset.video import VideoDataset

class TestVideoDataset(unittest.TestCase):
    def test_stream_wrappers(self):
        with VideoDataset('/data/datasets/ad_fixed_transcriptions/Matron_Medicine_and_Me_Series_2_-_1._Fern_Britton_b0bbplbl_audiodescribed.h5') as dataset:
            subtitle_streams = dataset.get_subtitled_streams(['video', 'audio'], subtitle_id='audiodescription')
            for segment in subtitle_streams:
                print(segment)

    def test_random_stream_wrappers(self):
        rng = np.random.RandomState(1729)
        with VideoDataset('/data/datasets/ad_fixed_transcriptions/Matron_Medicine_and_Me_Series_2_-_1._Fern_Britton_b0bbplbl_audiodescribed.h5') as dataset:
            subtitle_streams = dataset.get_subtitled_streams_randomized(['video', 'audio'], subtitle_id='audiodescription', rng=rng)
            for segment in subtitle_streams:
                print(segment)
