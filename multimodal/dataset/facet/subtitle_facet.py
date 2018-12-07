import numpy as np
import h5py
from multimodal.dataset.facet.facet_handler import FacetHandler

class SubtitleFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(SubtitleFacet, self).__init__(*args, **kwargs)

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
