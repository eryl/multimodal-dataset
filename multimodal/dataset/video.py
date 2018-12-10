from multimodal.dataset.multimodal import MultiModalDataset

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


