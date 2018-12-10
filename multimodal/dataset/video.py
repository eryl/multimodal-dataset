from multimodal.dataset.multimodal import MultiModalDataset


class SubtitlesAudioDataset(MultiModalDataset):
    """
    Dataset iterating over subtitles and audio in the video dataset.
    """
    def __init__(self, *args, **kwargs):
        super(SubtitlesAudioDataset, self).__init__(*args, **kwargs)
        self.subtitles = self.modalities['subtitles'].get_facet()
        self.audio = self.modalities['audio'].get_facet()

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = self.subtitles[item]
        if isinstance(item, slice):
            times, text = zip(*subtitles)
            audio = self.audio.get_frames(times)
            return zip(text, audio)
        elif isinstance(item, int):
            times, text = subtitles
            audio = self.audio.get_frames(times)
            return text, audio
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class ComplementSubtitlesAudioVideoDataset(MultiModalDataset):
    """
    Dataset iterating over audio and video in the video dataset which are not part of the subtitles.
    """
    def __init__(self, *args, minimum_time=1, **kwargs):
        super(ComplementSubtitlesAudioVideoDataset, self).__init__(*args, **kwargs)
        self.subtitles = self.modalities['subtitles'].get_facet()
        self.audio = self.modalities['audio'].get_facet()
        self.video = self.modalities['video'].get_facet()
        self.times = self.subtitles.get_time_complement(minimum_time)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        if isinstance(item, slice):
            times = self.times[item]
            audio = self.audio.get_frames(times)
            video = self.video.get_frames(times)
            return zip(audio, video)
        elif isinstance(item, int):
            times = self.times[item]
            audio = self.audio.get_frames(times)
            video = self.video.get_frames(times)
            return audio, video
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class SubtitlesVideoDataset(MultiModalDataset):
    """
    Dataset iterating over subtitles and video in the video dataset.
    """
    def __init__(self, *args, **kwargs):
        super(SubtitlesVideoDataset, self).__init__(*args, **kwargs)
        self.subtitles = self.modalities['subtitles'].get_facet()
        self.video = self.modalities['video'].get_facet()

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = self.subtitles[item]
        if isinstance(item, slice):
            times, text = zip(*subtitles)
            video = self.video.get_frames(times)
            return zip(text, video)
        elif isinstance(item, int):
            times, text = subtitles
            video = self.video.get_frames(times)
            return text, video
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))