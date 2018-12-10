import os
import os.path
import h5py

from multimodal.dataset.facet import AudioFacet, VideoFacet, SubtitleFacet


def extract_audio(store, video_name):
    print("Extracting audio ", video_name)
    audio_modality = store.create_group('audio')
    AudioFacet.create_facets(audio_modality, video_name)


def extract_subtitles(store, subtitles_name):
    print("Extracting subtitles ", subtitles_name)
    subtitles_modality = store.create_group('subtitles')
    SubtitleFacet.create_facets(subtitles_modality, subtitles_name)


def extract_video(store, video_name):
    print("Extracting video ",video_name)
    video_modality = store.create_group('video')
    VideoFacet.create_facets(video_modality, video_name)


def make_dataset(video_name, subtitles_name, skip_video=False):
        print("Making video dataset using video {} and subtitles {}".format(video_name, subtitles_name))
        store_name = '{}.h5'.format(os.path.splitext(video_name)[0])
        with h5py.File(store_name, 'a') as store:
            extract_audio(store, video_name)
            extract_subtitles(store, subtitles_name)
            if not skip_video:
                extract_video(store, video_name)

