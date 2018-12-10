import numpy as np
from multimodal.dataset.multimodal import MultiModalDataset, MultiModalDatasets


class SubtitlesAndStreamsWrapper(object):
    """
    Dataset iterating over subtitles and audio in the video dataset.
    """
    def __init__(self, *args, subtitles, streams, **kwargs):
        super(SubtitlesAndStreamsWrapper, self).__init__(*args, **kwargs)
        self.subtitles = subtitles
        self.streams = streams

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = self.subtitles[item]
        if isinstance(item, slice):
            times, text = zip(*subtitles)
            frames = [stream.get_frames(times) for stream in self.streams]
            return zip(text, *frames)
        elif isinstance(item, int):
            times, text = subtitles
            frames = [stream.get_frames(times) for stream in self.streams]
            return [text] + frames
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class SubtitlesComplementAndStreamsWrapper(SubtitlesAndStreamsWrapper):
    """
    Dataset iterating over audio and video in the video dataset which are not part of the subtitles.
    """
    def __init__(self, *args, minimum_time=1, **kwargs):
        super(SubtitlesComplementAndStreamsWrapper, self).__init__(*args, **kwargs)
        self.times = self.subtitles.get_times_complement(minimum_time)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        if isinstance(item, slice):
            times = self.times[item]
            frames = [stream.get_frames(times) for stream in self.streams]
            return zip(*frames)
        elif isinstance(item, int):
            times = self.times[item]
            frames = [stream.get_frames(times) for stream in self.streams]
            return frames
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class VideoDataset(MultiModalDataset):
    def __init__(self, *args, **kwargs):
        super(VideoDataset, self).__init__(*args, **kwargs)
        self.subtitles = self.modalities['subtitles'].get_facet()
        self.audio = self.modalities['audio'].get_facet()
        self.video = self.modalities['video'].get_facet()

    def get_subtitles_audio_wrapper(self):
        return SubtitlesAndStreamsWrapper(subtitles=self.subtitles, streams=[self.audio])

    def get_subtitles_video_wrapper(self):
        return SubtitlesAndStreamsWrapper(subtitles=self.subtitles, streams=[self.video])

    def get_subtitles_audio_video_wrapper(self):
        return SubtitlesAndStreamsWrapper(subtitles=self.subtitles, streams=[self.audio, self.video])

    def get_subtitles_complement_audio_wrapper(self):
        return SubtitlesComplementAndStreamsWrapper(subtitles=self.subtitles, streams=[self.audio])

    def get_subtitles_complement_video_wrapper(self):
        return SubtitlesComplementAndStreamsWrapper(subtitles=self.subtitles, streams=[self.video])

    def get_subtitles_complement_audio_video_wrapper(self):
        return SubtitlesComplementAndStreamsWrapper(subtitles=self.subtitles, streams=[self.audio, self.video])


class WrapperCollection(object):
    def __init__(self, wrappers):
        self.wrappers = wrappers
        self.wrapper_lengths = [len(wrapper) for wrapper in self.wrappers]
        self.cumsum_lengths = np.cumsum(self.wrapper_lengths)
        self.total_length = sum(self.wrapper_lengths)

    def __len__(self):
        return sum(self.wrapper_lengths)

    def __getitem__(self, item):
        if isinstance(item, slice):
            # Figure out which wrappers the item spans
            raise NotImplementedError()
        elif isinstance(item, int):
            wrapper_id = np.searchsorted(self.cumsum_lengths, item)
            wrapper_item = item - self.cumsum_lengths[wrapper_id]
            return self.wrappers[wrapper_id][wrapper_item]
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class VideoDatasets(object):
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
        self.datasets = [VideoDataset(dataset_path) for dataset_path in dataset_paths]
        self.subtitles = [dataset.modalities['subtitles'].get_facet() for dataset in self.datasets]
        self.audio = [dataset.modalities['audio'].get_facet() for dataset in self.datasets]
        self.video = [dataset.modalities['video'].get_facet() for dataset in self.datasets]

    def get_subtitles_audio_wrapper(self):
        wrappers = [SubtitlesAndStreamsWrapper(subtitles=subtitles, streams=[audio])
                    for subtitles, audio in zip(self.subtitles, self.audio)]
        return WrapperCollection(wrappers)

    def get_subtitles_video_wrapper(self):
        wrappers = [SubtitlesAndStreamsWrapper(subtitles=subtitles, streams=[video])
                    for subtitles, video in zip(self.subtitles, self.video)]
        return WrapperCollection(wrappers)

    def get_subtitles_audio_video_wrapper(self):
        wrappers = [SubtitlesAndStreamsWrapper(subtitles=subtitles, streams=[audio, video])
                    for subtitles, audio, video in zip(self.subtitles, self.audio, self.video)]
        return WrapperCollection(wrappers)

    def get_subtitles_complement_audio_wrapper(self):
        wrappers = [SubtitlesComplementAndStreamsWrapper(subtitles=subtitles, streams=[audio])
                    for subtitles, audio in zip(self.subtitles, self.audio)]
        return WrapperCollection(wrappers)

    def get_subtitles_complement_video_wrapper(self):
        wrappers = [SubtitlesComplementAndStreamsWrapper(subtitles=subtitles, streams=[video])
                    for subtitles, video in zip(self.subtitles, self.video)]
        return WrapperCollection(wrappers)

    def get_subtitles_complement_audio_video_wrapper(self):
        wrappers = [SubtitlesComplementAndStreamsWrapper(subtitles=subtitles, streams=[audio, video])
                    for subtitles, audio, video in zip(self.subtitles, self.audio, self.video)]
        return WrapperCollection(wrappers)


