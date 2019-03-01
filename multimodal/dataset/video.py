import numpy as np
from numbers import Integral
from multimodal.dataset.multimodal import MultiModalDataset, MultiModalDatasets


class TimeModality(object):
    def get_frames(self, times):
        return times
    def get_facet(self):
        return self

class SubtitlesAndStreamsWrapper(object):
    """
    Dataset iterating over subtitles and audio in the video dataset.
    """
    def __init__(self, *args, subtitles, streams, max_duration=None, rng=None, **kwargs):
        super(SubtitlesAndStreamsWrapper, self).__init__(*args, **kwargs)
        if rng is None:
            rng = np.random.RandomState()
        self.subtitles = subtitles
        self.streams = streams
        self.max_duration = max_duration
        self.rng = rng

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = self.subtitles[item]
        if isinstance(item, slice):
            times, text = zip(*subtitles)
            if self.max_duration is not None:
                # We should randomly sample shorter time intervals for the times which are to long
                segment_lengths = times[:,1] - times[:,0]
                long_indices = times[:,1] - times[:,0] > self.max_duration
                segment_start = self.rng.random_sample(times.shape[0]) * (segment_lengths - self.max_duration)
                times[long_indices] = np.hstack([segment_start, segment_start+self.max_duration])
            frames = [stream.get_frames_by_second(times) for stream in self.streams]
            return zip(text, *frames)
        elif isinstance(item, Integral):
            times, text = subtitles
            if self.max_duration is not None:
                segment_length = times[1] - times[0]
                if segment_length > self.max_duration:
                    start_time = self.rng.random_sample() * (segment_length - self.max_duration)
                    times[0] = start_time
                    times[1] = start_time + self.max_duration
            frames = [stream.get_frames_by_second(times) for stream in self.streams]
            return (text, frames)
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
            if self.max_duration is not None:
                # We should randomly sample shorter time intervals for the times which are to long
                segment_lengths = times[:,1] - times[:,0]
                long_indices = times[:,1] - times[:,0] > self.max_duration
                segment_start = self.rng.random_sample(times.shape[0]) * (segment_lengths - self.max_duration)
                times[long_indices] = np.hstack([segment_start, segment_start+self.max_duration])
            frames = [stream.get_frames_by_second(times) for stream in self.streams]
            return zip(*frames)
        elif isinstance(item, Integral):
            times = self.times[item]
            if self.max_duration is not None:
                segment_length = times[1] - times[0]
                if segment_length > self.max_duration:
                    start_time = self.rng.random_sample() * (segment_length - self.max_duration)
                    times[0] = start_time
                    times[1] = start_time + self.max_duration
            frames = [stream.get_frames_by_second(times) for stream in self.streams]
            return frames
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class VideoDataset(MultiModalDataset):
    def get_samplerate(self, stream):
        return self.modalities[stream].get_samplerate()

    def get_subtitled_streams(self, stream_facets, max_duration=None, rng=None):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        subtitles = self.modalities['subtitles'].get_facet()
        return SubtitlesAndStreamsWrapper(subtitles=subtitles, streams=streams, max_duration=max_duration, rng=rng)

    def get_subtitled_complement_streams(self, stream_facets, max_duration=None, rng=None):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        subtitles = self.modalities['subtitles'].get_facet()
        return SubtitlesComplementAndStreamsWrapper(subtitles=subtitles,
                                                    streams=streams,
                                                    max_duration=max_duration,
                                                    rng=rng)

    def setup_modalities(self):
        MultiModalDataset.setup_modalities(self)
        ## We add virtual modalities here
        time_modality = TimeModality()
        self.modalities['time'] = time_modality

    def get_streams(self, stream_facets, interval):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]



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
        elif isinstance(item, Integral):
            wrapper_id = np.searchsorted(self.cumsum_lengths, item)
            wrapper_item = item - self.cumsum_lengths[wrapper_id]
            return self.wrappers[wrapper_id][wrapper_item]
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))


class VideoDatasets(object):
    def __init__(self, dataset_paths):
        self.dataset_paths = dataset_paths
        self.datasets = [VideoDataset(dataset_path) for dataset_path in dataset_paths]

    def get_facet_wrapper(self, *args, **kwargs):
        wrappers = [dataset.get_facet_wrapper(*args, **kwargs) for dataset in self.datasets]
        return WrapperCollection(wrappers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for dataset in self.datasets:
            dataset.close()



