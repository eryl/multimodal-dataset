import tempfile

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
    def create_facets(cls, video_modality, video_path, video_size):
        # TODO: Add all video streams from video_path
        cls.create_facet('video1', video_modality, video_path, video_size)

    @classmethod
    def create_facet(cls, name, video_modality, video_path, video_size, chunksize=256):
        video_reader = imageio.get_reader(video_path)
        video_metadata = video_reader.get_meta_data()
        fps = video_metadata['fps']
        nframes = video_metadata['nframes']
        width, height = video_metadata['size']
        video_reader.close()
        target_width, target_height = video_size
        print("Original size is ", width, height)
        print("Target size is ", target_width, target_height)
        if target_width is not None and target_height is not None:
            width = target_width
            height = target_height
        elif target_width is not None:
            ratio = target_width / width
            print("Ratio is ", ratio)
            width = target_width
            height = int(height * ratio)
        elif target_height is not None:
            ratio = target_height / height
            height = target_height
            width = int(width * ratio)

        video_reader = imageio.get_reader(video_path, size=(width, height))

        facetgroup = video_modality.require_group(name)
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
    def create_facet_new(cls, name, video_modality, video_path, video_size, chunksize=256):
        import ffmpeg
        import os.path

        video_reader = imageio.get_reader(video_path)
        video_metadata = video_reader.get_meta_data()
        fps = video_metadata['fps']
        nframes = video_metadata['nframes']
        width, height = video_metadata['size']
        video_reader.close()
        target_width, target_height = video_size
        print("Original size is ", width, height)
        print("Target size is ", target_width, target_height)
        if target_width is not None and target_height is not None:
            width = target_width
            height = target_height
        elif target_width is not None:
            ratio = target_width / width
            print("Ratio is ", ratio)
            width = target_width
            height = int(height * ratio)
        elif target_height is not None:
            ratio = target_height / height
            height = target_height
            width = int(width * ratio)

        facetgroup = video_modality.create_group(name)
        facetgroup.attrs['FacetHandler'] = 'VideoFacet'
        facetgroup.attrs['rate'] = fps

        frame_sizes = facetgroup.create_dataset('frame_sizes',
                                                shape=(0,),
                                                maxshape=(None,),
                                                chunks=(2 ** 11,),
                                                dtype=np.uint64)
        frames = facetgroup.create_dataset('frames',
                                           shape=(0,),
                                           maxshape=(None,),
                                           dtype=np.uint8,
                                           chunks=(2 ** 16,),
                                           compression='gzip',
                                           shuffle=True)
        n_chunks = int(np.ceil(nframes / chunksize))

        directory = tempfile.mkdtemp()
        n_digits = int(np.ceil(np.log10(nframes)))
        filename = os.path.join(directory,
                                os.path.splitext(os.path.basename(video_path))[0] + '_%0{}d.jpg'.format(n_digits))

        out, _ = (
            ffmpeg
                .input(video_path)
                .filter('scale', size='{}:{}'.format(width, height))
                .output(filename, format='image2', vcodec='mjpeg', **{'qscale:v':2})
                .run()
        )
        data = []
        sizes = []
        cumsum_sizes = []
        read_bytes = 0
        filename_pattern = os.path.join(directory,
                                 os.path.splitext(os.path.basename(video_path))[0] + '_{{:0{}d}}.jpg'.format(n_digits))
        current_frame_size_index = 0
        current_frame_index = 0
        for i in range(nframes):
            filename = filename_pattern.format(i)
            with open(filename, 'rb') as image:
                bytes = image.read()
                data.append(bytes)
                size = len(bytes)
                sizes.append(size)
                cumsum_sizes.append(read_bytes + size)
                read_bytes += size
            if read_bytes > chunksize:
                chunk_frame_sizes = np.array(sizes) + current_frame_index
                frames_bytes = np.concatenate(data)
                old_size = frames.shape[0]
                new_size = old_size + len(frames_bytes)
                frames.resize((new_size,))
                frame_sizes.resize((frame_sizes.shape[0] + len(chunk_frame_sizes),))
                frames[current_frame_index:current_frame_index + len(frames_bytes)] = frames_bytes
                frame_sizes[current_frame_size_index:current_frame_size_index + len(chunk_frame_sizes)] = chunk_frame_sizes
                current_frame_size_index += len(chunk_frame_sizes)
                current_frame_index += len(frames_bytes)
                read_bytes = 0
                sizes = []
                data = []
                cumsum_sizes = []
        chunk_frame_sizes = np.array(sizes) + current_frame_index
        frames_bytes = np.concatenate(data)
        old_size = frames.shape[0]
        new_size = old_size + len(frames_bytes)
        frames.resize((new_size,))
        frame_sizes.resize((frame_sizes.shape[0] + len(chunk_frame_sizes),))
        frames[current_frame_index:current_frame_index + len(frames_bytes)] = frames_bytes
        frame_sizes[current_frame_size_index:current_frame_size_index + len(chunk_frame_sizes)] = chunk_frame_sizes


        return VideoFacet(facetgroup)


