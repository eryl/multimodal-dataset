import numpy as np
import imageio
import itertools
from multimodal.dataset.facet.facet_handler import FacetHandler

class VideoFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(VideoFacet, self).__init__(*args, **kwargs)
        self.frames = self.facetgroup['frames']
        self.frame_sizes = self.facetgroup['frame_sizes']
        self.fps = self.facetgroup.attrs['rate']

    def get_frames(self, times):
        """
        Return the frames given by times as a numpy array
        :return:
        """
        try:
            start, end = times
            start_frame = int(start*self.fps)
            end_frame = int(end*self.fps)
            return self.uncompress_frames(start_frame, end_frame)
        except ValueError:
            frames = []
            frames_indices = (times*self.fps).astype(np.uint)
            for i in range(len(frames_indices)):
                start_frame, end_frame = frames_indices[i]
                frames.append(self.uncompress_frames(start_frame, end_frame))
            return frames

    def uncompress_frames(self, start_frame, end_frame):
        """
        Returns the uncompressed frames from a start_frame (inclusive) to end_frame (non-inclusive)
        :param start_frame: First frame to decompress.
        :param end_frame: end of range, this frame is not included in the decompressed volume
        :return: A numpy nd-array with shape (end-start, height, width, channels)
        """
        sizes = self.frame_sizes[start_frame:end_frame]
        if start_frame > 0:
            start_byte = self.frame_sizes[start_frame - 1]
        else:
            start_byte = 0
        end_byte = sizes[-1]
        frame_data = self.frames[start_byte: end_byte]
        # The frame sizes are cumulative sum, we need to subtract the start byte to get their sizes relative to the
        # frame data we extracted
        sizes -= start_byte
        frame_start = 0
        frames = []
        for frame_end in sizes:
            frame = frame_data[frame_start: frame_end]
            frames.append(imageio.imread(frame.tobytes(), 'jpeg'))
            frame_start = frame_end
        return np.array(frames)

    @classmethod
    def create_facets(cls, video_modality, video_path):
        # TODO: Add all video streams from video_path
        cls.create_facet('video1', video_modality, video_path)

    @classmethod
    def create_facet(cls, name, video_modality, video_path, chunksize=256):

        video_reader = imageio.get_reader(video_path)
        video_metadata = video_reader.get_meta_data()
        fps = video_metadata['fps']
        nframes = video_metadata['nframes']
        width, height = video_metadata['size']
        video_reader.close()

        if width < height:
            ratio = 256 / width
            width = 256
            height = int(height * ratio)
        else:
            ratio = 256 / height
            height = 256
            width = int(width * ratio)
        video_reader = imageio.get_reader(video_path, size=(width, height))

        facetgroup = video_modality.create_group(name)
        facetgroup.attrs['FacetHandler'] = 'VideoFacet'
        facetgroup.attrs['rate'] = fps

        frame_sizes = facetgroup.create_dataset('frame_sizes',
                                                shape=(0,),
                                                maxshape=(None,),
                                                chunks=(2**11,),
                                                dtype=np.uint64)
        frames = facetgroup.create_dataset('frames',
                                           shape=(0,),
                                           maxshape=(None,),
                                           dtype=np.uint8,
                                           chunks=(2**16,),
                                           compression='gzip',
                                           shuffle=True)
        n_chunks = int(np.ceil(nframes / chunksize))
        frame_iter = iter(video_reader)
        current_frame_index = 0
        current_frame_size_index = 0
        cumulative_frame_sizes = 0
        for i in range(n_chunks):
            print("Chunk {}/{}".format(i, n_chunks))
            chunk_frames = list(itertools.islice(frame_iter, chunksize))
            frames_arrays = [np.frombuffer(imageio.imsave('<bytes>', im, 'jpeg', flags=100), dtype=np.uint8) for im in chunk_frames]
            chunk_frame_sizes = np.cumsum([len(arr) for arr in frames_arrays]) + cumulative_frame_sizes
            cumulative_frame_sizes = chunk_frame_sizes[-1]
            frames_bytes = np.concatenate(frames_arrays)
            old_size = frames.shape[0]
            new_size = old_size + len(frames_bytes)
            frames.resize((new_size,))
            frame_sizes.resize((frame_sizes.shape[0] + len(chunk_frame_sizes),))
            frames[current_frame_index:current_frame_index+len(frames_bytes)] = frames_bytes
            frame_sizes[current_frame_size_index:current_frame_size_index+len(chunk_frame_sizes)] = chunk_frame_sizes
            current_frame_size_index += len(chunk_frame_sizes)
            current_frame_index += len(frames_bytes)

        return VideoFacet(facetgroup)

    @classmethod
    def create_facet_new(cls, name, video_modality, video_path, chunksize=256):
        import ffmpeg
        video_reader = imageio.get_reader(video_path)
        video_metadata = video_reader.get_meta_data()
        fps = video_metadata['fps']
        nframes = video_metadata['nframes']
        width, height = video_metadata['size']
        video_reader.close()

        if width < height:
            width = 256
            height = -1
        else:
            height = 256
            width = -1

        facetgroup = video_modality.create_group(name)
        facetgroup.attrs['FacetHandler'] = 'VideoFacet'
        facetgroup.attrs['rate'] = fps

        frame_sizes = facetgroup.create_dataset('frame_sizes',
                                                shape=(0,),
                                                maxshape=(None,),
                                                chunks=(2**11,),
                                                dtype=np.uint64)
        frames = facetgroup.create_dataset('frames',
                                           shape=(0,),
                                           maxshape=(None,),
                                           dtype=np.uint8,
                                           chunks=(2**16,),
                                           compression='gzip',
                                           shuffle=True)

        process1 = (
            ffmpeg
                .input(video_path)
                .filter('scale', size='{}:{}'.format(width, height))
                .output('pipe:', format='image2', vcodec='mjpeg', **{'qscale:v':2})
                .run_async(pipe_stdout=True)
        )


        while True:
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = (
                np
                    .frombuffer(in_bytes, np.uint8)
                    .reshape([height, width, 3])
            )
            out_frame = in_frame * 0.3
            process2.stdin.write(
                frame
                    .astype(np.uint8)
                    .tobytes()
            )

        process2.stdin.close()
        process1.wait()
        process2.wait()

        n_chunks = int(np.ceil(nframes / chunksize))
        frame_iter = iter(video_reader)
        current_frame_index = 0
        current_frame_size_index = 0
        cumulative_frame_sizes = 0
        for i in range(n_chunks):
            print("Chunk {}/{}".format(i, n_chunks))
            chunk_frames = list(itertools.islice(frame_iter, chunksize))
            frames_arrays = [np.frombuffer(imageio.imsave('<bytes>', im, 'jpeg', flags=100), dtype=np.uint8) for im in chunk_frames]
            chunk_frame_sizes = np.cumsum([len(arr) for arr in frames_arrays]) + cumulative_frame_sizes
            cumulative_frame_sizes = chunk_frame_sizes[-1]
            frames_bytes = np.concatenate(frames_arrays)
            old_size = frames.shape[0]
            new_size = old_size + len(frames_bytes)
            frames.resize((new_size,))
            frame_sizes.resize((frame_sizes.shape[0] + len(chunk_frame_sizes),))
            frames[current_frame_index:current_frame_index+len(frames_bytes)] = frames_bytes
            frame_sizes[current_frame_size_index:current_frame_size_index+len(chunk_frame_sizes)] = chunk_frame_sizes
            current_frame_size_index += len(chunk_frame_sizes)
            current_frame_index += len(frames_bytes)

        return VideoFacet(facetgroup)


