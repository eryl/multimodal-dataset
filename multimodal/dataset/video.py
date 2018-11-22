import h5py
import numpy as np


class VideoDataset(object):
    def __init__(self, store):
        self.store_path = store
        self.store = h5py.File(store)

    def close(self):
        self.store.close()

    def sequential_subtitle_iterator(self, sequence_length, batch_size):
        self.build_subtitle_index()


    def random_subtitle_iterator(self, sequence_length, batch_size, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        pass

    def build_subtitle_index(self):
        """
        Goes through all the videos in the dataset and extract subtitle information.
        :return:
        """
        for name, group_object in self.store.items():
            subtitles_group = group_object['subtitles']
            times = subtitles_group['times']
            



