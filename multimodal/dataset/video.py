import numpy as np
from numbers import Integral
from multimodal.dataset.multimodal import MultiModalDataset, MultiModalDatasets
from multimodal.dataset.facet.subtitle_facet import SubtitleFacet
from multimodal.dataset.facet.video_facet import VideoFacet
from multimodal.dataset.facet.audio_facet import AudioFacet

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
            frames = [stream.get_frames_by_seconds(times) for stream in self.streams]
            return zip(text, *frames)
        elif isinstance(item, Integral):
            times, text = subtitles
            if self.max_duration is not None:
                segment_length = times[1] - times[0]
                if segment_length > self.max_duration:
                    start_time = self.rng.random_sample() * (segment_length - self.max_duration)
                    times[0] = start_time
                    times[1] = start_time + self.max_duration
            frames = [stream.get_frames_by_seconds(times) for stream in self.streams]
            return (text, frames)
        else:
            raise TypeError("Invalid argument type. {}".format(type(item)))

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


class SubtitlesAndRandomStreamsWrapper(object):
    """
    Wrapper which supports __getitem__ over a subtitle facet and one or more stream facets. This version returns random
    subsets of the streams, instead of the true corresponding time segment.
    """
    def __init__(self, *args, subtitles, streams, synched_streams=True, max_duration=None, rng=None, **kwargs):
        super(SubtitlesAndRandomStreamsWrapper, self).__init__(*args, **kwargs)
        if rng is None:
            rng = np.random.RandomState()
        self.subtitles = subtitles
        self.streams = streams
        self.max_duration = max_duration
        self.rng = rng

        # We want to transpose the times segments so that they belong to the wrong subtitle
        stream_lengths = min(stream.get_length_s() for stream in self.streams)
        self.times = np.array(self.subtitles.get_times())

        if synched_streams:
            stream_times = self.make_random_time_segments(self.times, stream_lengths)
            self.stream_times = [stream_times for stream in self.streams]
        else:
            self.stream_times = [self.make_random_time_segments(self.times, stream_lengths) for stream in self.streams]

    def make_random_time_segments(self, times, stream_lengths):
        # Randomly pick some other time segment from the subtitles, but adjust its length to fit the original
        # length for each segment. This way the segment will contain the information used to for some other
        # subtitle, making it harder to solve.
        transposed_times = np.copy(times)
        self.rng.shuffle(transposed_times)
        for i in range(len(times)):
            original_time = times[i]
            transposed_time = transposed_times[i]
            original_time_duration = original_time[1] - original_time[0]
            tranposed_time_duration = transposed_time[1] - transposed_time[0]
            diff_s = original_time_duration - tranposed_time_duration
            diff_s_half = diff_s / 2
            if diff_s < 0:
                # The transposed segment is longer, lets trim it at both sides
                start, end = transposed_time
                start -= diff_s_half
                end += diff_s - diff_s_half
            else:
                # The transposed segment is shorter, let's expand it in both directions, minding edge cases (the
                # end or start of the stream)
                start, end = transposed_time
                start -= diff_s_half
                end += diff_s - diff_s_half

                if start < 0:
                    # Shift the time from the start to the end
                    end -= start
                    start = 0
                elif end > stream_lengths:
                    start -= end - stream_lengths
                    end = stream_lengths
            new_duration = end - start
            transposed_times[i] = start, end
        return transposed_times

    def __len__(self):
        return len(self.subtitles)

    def __getitem__(self, item):
        subtitles = self.subtitles[item]
        if isinstance(item, slice):
            _, text = zip(*subtitles)
            frames = []
            for stream_times, stream in zip(self.stream_times, self.streams):
                times = stream_times[item]
                if self.max_duration is not None:
                    # We should randomly sample shorter time intervals for the times which are to long
                    segment_lengths = times[:,1] - times[:,0]
                    long_indices = times[:,1] - times[:,0] > self.max_duration
                    segment_start = self.rng.random_sample(times.shape[0]) * (segment_lengths - self.max_duration)
                    times[long_indices] = np.hstack([segment_start, segment_start+self.max_duration])
                frames.append(stream.get_frames_by_seconds(times))
                return zip(text, *frames)
        elif isinstance(item, Integral):
            _, text = subtitles
            frames = []
            for stream_times, stream in zip(self.stream_times, self.streams):
                times = stream_times[item]
                if self.max_duration is not None:
                    segment_length = times[1] - times[0]
                    if segment_length > self.max_duration:
                        start_time = self.rng.random_sample() * (segment_length - self.max_duration)
                        times[0] = start_time
                        times[1] = start_time + self.max_duration
                frames.append(stream.get_frames_by_seconds(times))
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

    def get_subtitled_streams(self, stream_facets, subtitle_id=None, max_duration=None, rng=None):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        subtitles = self.modalities['subtitles'].get_facet(subtitle_id)
        return SubtitlesAndStreamsWrapper(subtitles=subtitles, streams=streams, max_duration=max_duration, rng=rng)

    def get_subtitled_complement_streams(self, stream_facets, subtitle_id=None, max_duration=None, rng=None):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        subtitles = self.modalities['subtitles'].get_facet(subtitle_id)
        return SubtitlesComplementAndStreamsWrapper(subtitles=subtitles,
                                                    streams=streams,
                                                    max_duration=max_duration,
                                                    rng=rng)

    def get_subtitled_streams_randomized(self, stream_facets, subtitle_id = None, max_duration = None, rng = None):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        subtitles = self.modalities['subtitles'].get_facet(subtitle_id)
        return SubtitlesAndRandomStreamsWrapper(subtitles=subtitles,
                                                streams=streams,
                                                max_duration=max_duration,
                                                rng=rng)

    def get_time_interval_frames(self, time_interval_name, stream_facet_name, max_duration=None, rng=None):
        """
        :param time_interval_name:
        :param stream_facet_name:
        :param max_duration:
        :param rng:
        :return:
        """
        facet = self.modalities[stream_facet_name].get_facet()
        return facet.get_time_interval_frames(time_interval_name)

    def setup_modalities(self):
        MultiModalDataset.setup_modalities(self)
        ## We add virtual modalities here
        time_modality = TimeModality()
        self.modalities['time'] = time_modality

    def get_streams(self, stream_facets):
        if isinstance(stream_facets, str):
            stream_facets = [stream_facets]
        streams = [self.modalities[stream_facet].get_facet() for stream_facet in stream_facets]
        return [stream.get_all_frames() for stream in streams]

    def add_multiple_subtitles(self, subtitles_files):
        for subtitles_file in subtitles_files:
            self.add_subtitles(subtitles_file)

    def add_subtitles(self, subtitles_file, name=None):
        subtitles_modality_group = self.store.require_group('subtitles')
        if name is None:
            import os.path
            name = os.path.basename(subtitles_file)
        subtitle_facet = SubtitleFacet.create_facet(name, subtitles_modality_group, subtitles_file)

    def add_video(self, name, video_file, target_size):
        video_modality_group = self.store.require_group('video')
        video_facet = VideoFacet.create_facet(name, video_modality_group, video_file, target_size)

    def add_audio(self, video_file, target_sample_rate=16000):
        # TODO: Add all audio streams as facets
        audio_modality_group = self.store.require_group('audio')
        audio_facet = AudioFacet.create_facet('audio0', audio_modality_group, video_file, target_sample_rate)



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



