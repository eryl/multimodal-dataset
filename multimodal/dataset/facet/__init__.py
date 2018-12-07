from multimodal.dataset.facet.video_facet import VideoFacet
from multimodal.dataset.facet.audio_facet import AudioFacet
from multimodal.dataset.facet.subtitle_facet import SubtitleFacet

def is_facet(h5_group):
    return 'FacetHandler' in h5_group.attrs


def make_facet(facet_group):
    handler_key = facet_group.attrs['FacetHandler']
    if handler_key == 'VideoFacet':
        return VideoFacet(facet_group)
    elif handler_key == 'AudioFacet':
        return AudioFacet(facet_group)
    elif handler_key == 'SubtitleFacet':
        return SubtitleFacet(facet_group)
    else:
        raise NotImplementedError("Could not find facet handler {}".format(handler_key))

