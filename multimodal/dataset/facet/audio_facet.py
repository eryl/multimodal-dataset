import numpy as np
from multimodal.dataset.facet.facet_handler import FacetHandler

class AudioFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(AudioFacet, self).__init__(*args, **kwargs)
        self.frames = self.facetgroup['sound']
        self.rate = self.facetgroup.attrs['rate']

    @classmethod
    def create_facets(cls, audio_modality, video_path):
        ## TODO: Add all audio streams as facets
        cls.create_facet('audio1', audio_modality, video_path)

    @classmethod
    def create_facet(cls, name, audio_modality, video_name, rate=16000):
        import ffmpeg
        group = audio_modality.create_group(name)
        out, _ = (ffmpeg
                  .input(video_name)
                  .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True)
                  )
        data = np.frombuffer(out, np.int16)
        audio_facet = group.create_dataset('sound', data=data, chunks=True, compression='gzip', shuffle=True)
        group.attrs['rate'] = rate
        group.attrs['FacetHandler'] = 'AudioFacet'
        return AudioFacet(group)
