import os
import os.path
import h5py

from multimodal.dataset.video import VideoDataset


def make_dataset(video_name, subtitles_names=None, skip_video=False, skip_audio=False, video_size=(None, None)):
        print("Making video dataset using video {} and subtitles {}".format(video_name, subtitles_names))
        store_name = '{}.h5'.format(os.path.splitext(video_name)[0])
        with VideoDataset(store_name, 'w') as dataset:
            if not skip_audio:
                print("Extracting audio ", video_name)
                dataset.add_audio(video_name)
            if subtitles_names is not None:
                print("Extracting subtitles ", subtitles_names)
                dataset.add_multiple_subtitles(subtitles_names)
            if not skip_video:
                print("Extracting video ", video_name)
                dataset.add_video('video0', video_name, video_size)
