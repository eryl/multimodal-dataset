import numpy as np
from multimodal.dataset.facet.facet_handler import FacetHandler

class AudioFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(AudioFacet, self).__init__(*args, **kwargs)
        self.frames = self.facetgroup['sound']
        self.rate = self.facetgroup.attrs['rate']

    @classmethod
    def create_facets(cls, audio_modality, video_path):
        ## TODO: Add all audio streams as facets
        cls.create_facet('audio1', audio_modality, video_path)

    @classmethod
    def create_facet(cls, name, audio_modality, video_name, rate=16000):
        import ffmpeg
        group = audio_modality.require_group(name)
        out, _ = (ffmpeg
                  .input(video_name)
                  .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                  .overwrite_output()
                  .run(capture_stdout=True)
                  )
        data = np.frombuffer(out, np.int16)
        audio_facet = group.create_dataset('sound', data=data, chunks=True, compression='gzip', shuffle=True)
        group.attrs['rate'] = rate
        group.attrs['FacetHandler'] = 'AudioFacet'
        return AudioFacet(group)

    def has_time_intervals(self, name):
        return name in self.facetgroup

    def add_time_intervals(self, name, times, overwrite=False):
        """
        Adds a dataset with times, useful for annotating an audio stream with e.g. voiced parts
        :param name: The name of the times dataset
        :param times: A numpy array of shape (n_pairs, 2)
        """
        if overwrite and name in self.facetgroup:
            del self.facetgroup[name]
        self.facetgroup.create_dataset(name, data=times, chunks=True, compression='gzip', shuffle=True)

    def get_time_intervals(self, name):
        return self.facetgroup[name][:]

    def get_time_interval_frames(self, name):
        """
        Returns an iterator over the time intervals denoted by *name* and the frames corresponding to that interval
        :param name:
        :return:
        """
        times = self.facetgroup[name][:]
        sample_rate = self.rate
        for start, end in times:
            frames = self.frames[start:end]
            yield (start/sample_rate, end/sample_rate), frames

    def get_samplerate(self):
        return self.rate

    def get_length_s(self):
        """
        Return the length in seconds
        :return:
        """
        return self.rate*len(self.frames)


    def get_frames_by_seconds(self, times):
        """
        Return the frames given by times (given in fraction of seconds) as a numpy array
        :return:
        """
        return self.get_frames(np.array(times*self.rate, dtype=np.uint))

    def get_frames(self, times):
        """
        Return the frames given by times (exact frame indices) as a numpy array
        :return:
        """

        try:
            start, end = times
            return self.frames[start: end]
        except ValueError:
            frames = []
            for i in range(len(times)):
                start_frame, end_frame = times[i]
                frames.append(self.frames[start_frame: end_frame])
            return frames

    def get_all_frames(self):
        return self.frames[:]


class MuLawFacet(AudioFacet):
    """
    Dataset wrapper for audio facets which returns mu-law encoded sequences
    """
    def __init__(self, *args, u=255, k=256, **kwargs):
        super().__init__(*args, **kwargs)
        self.u = u
        self.logu = np.log(1 + self.u)
        self.k = k

    def get_frames(self, times):
        """
        Return the frames given by times as a numpy array
        :return:
        """
        frames = super().get_frames(times).astype(np.float32)
        frames += frames.min()
        frames /= frames.max()
        frames *= 2
        frames -= 1
        u_lawed = np.sign(frames) * np.log(1 + frames * np.abs(frames)) / self.logu
        return ((u_lawed + 1) / 2 * self.u).astype(np.uint8)




