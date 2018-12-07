from multimodal.dataset.facet.facet_handler import FacetHandler

class AudioFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(AudioFacet, self).__init__(*args, **kwargs)

    @classmethod
    def create_facet(cls, name, audio_modality, data, rate):
        group = audio_modality.create_group(name)
        audio_facet = group.create_dataset('sound', data=data, chunks=True, compression='gzip', shuffle=True)
        group.attrs['rate'] = rate
        group.attrs['FacetHandler'] = 'AudioFacet'
        return AudioFacet(group)
