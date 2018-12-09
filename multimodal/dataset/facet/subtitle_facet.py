import numpy as np
import h5py
from multimodal.dataset.facet.facet_handler import FacetHandler

class SubtitleFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(SubtitleFacet, self).__init__(*args, **kwargs)
        self.texts = self.facetgroup['texts']
        self.times = self.facetgroup['times']

    @classmethod
    def create_facet(cls, name, modality_group, subtitles_color_index, colors, texts, times):
        subtitles_facet = modality_group.create_group(name)
        subtitles_facet.create_dataset('color_index', data=subtitles_color_index)
        subtitles_facet.create_dataset('colors', data=np.array(colors, dtype=np.int8))
        subtitle_texts = subtitles_facet.create_dataset('texts', shape=(len(texts),),
                                                             dtype=h5py.special_dtype(vlen=str))
        subtitle_texts[:] = texts
        subtitles_facet.create_dataset('times', data=np.array(times, dtype=np.float32))
        subtitles_facet.attrs['FacetHandler'] = 'SubtitleFacet'
        return SubtitleFacet(subtitles_facet)

    def get_times_complement(self, minimum_time=0):
        """Returns timestamps of parts which doesn't have subtitles"""
        n_times = len(self.times)
        times_complement = np.zeros((n_times, 2), dtype=self.times.dtype)
        times_complement[1:, 0] = self.times[:, 1]
        times_complement[:-1, 1] = self.times[:, 0]
        long_enough = times_complement[:,1] - times_complement[:,0] > minimum_time
        return times_complement[long_enough]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        if isinstance(item, slice):
            times = self.times[item]
            texts = self.texts[item]
            return zip(times, texts)
        elif isinstance(item, int):
            time = self.times[item]
            text = self.texts[item]
            return time, text
        else:
            raise TypeError("Invalid argument type.")


