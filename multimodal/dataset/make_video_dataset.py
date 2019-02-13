import os
import os.path
import h5py

from multimodal.dataset.facet import AudioFacet, VideoFacet, SubtitleFacet


def extract_audio(store, video_name):
    print("Extracting audio ", video_name)
    audio_modality = store.create_group('audio')
    AudioFacet.create_facets(audio_modality, video_name)


def extract_subtitles(store, subtitles_files):
    print("Extracting subtitles ", subtitles_files)
    subtitles_modality = store.create_group('subtitles')
    SubtitleFacet.create_facets(subtitles_modality, subtitles_files)


def extract_video(store, video_name, video_size):
    print("Extracting video ",video_name)
    video_modality = store.create_group('video')
    VideoFacet.create_facets(video_modality, video_name, video_size)


def make_dataset(video_name, subtitles_name=None, skip_video=False, skip_audio=False, video_size=(None, None)):
        print("Making video dataset using video {} and subtitles {}".format(video_name, subtitles_name))
        store_name = '{}.h5'.format(os.path.splitext(video_name)[0])
        with h5py.File(store_name, 'w') as store:
            if not skip_audio:
                extract_audio(store, video_name)
            if subtitles_name is not None:
                extract_subtitles(store, subtitles_name)
            if not skip_video:
                extract_video(store, video_name, video_size=video_size)

