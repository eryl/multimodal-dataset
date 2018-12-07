import numpy as np
from multimodal.dataset.facet.facet_handler import FacetHandler

class VideoFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(VideoFacet, self).__init__(*args, **kwargs)
        self.frames = self.facetgroup['frames']

    @classmethod
    def create_facet(cls, name, modality_group, shape, dtype, rate):
        video_group = modality_group.create_group(name)

        video_group.attrs['FacetHandler'] = 'VideoFacet'
        video_group.attrs['rate'] = rate

        video_data = video_group.create_dataset('frames',
                                                shape=shape,
                                                dtype=dtype,
                                                chunks=(1,) + tuple(shape[1:]),
                                                compression='gzip',
                                                shuffle=True)
        return VideoFacet(video_group)

    def write(self, start_frame, frames):
        # Write chunk of frames to the dataset
        self.frames[start_frame:start_frame+len(frames)] = frames


