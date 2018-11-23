import h5py
import numpy as np


class VideoDataset(object):
    def __init__(self, store):
        self.store_path = store
        self.store = h5py.File(store)
        self.index = None

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
        if self.index is None:
            index = {}

            i = 0
            for name, group_object in self.store.items():
                subtitles_group = group_object['subtitles']
                times = subtitles_group['times'][:]
                texts = subtitles_group['texts'][:]
                length = [len(text.split()) for text in texts]
                index[name] = (times, texts, length)
                i += 1
            self.index = index

            
    def num_subtitles(self):
        self.build_subtitle_index()
        return sum(len(times) for times, texts, lengths in self.index.values())

    def subtitle_lengths(self):
        self.build_subtitle_index()
        return [length for times, texts, lengths in self.index.values() for length in lengths]

    def get_dataset(self, index):
        pass


