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

    def get_frames(self, start, end):
        """
        Return the frames between the start and end as a numpy array
        :param start: Start time in seconds
        :param end: End time in seconds
        :return:
        """
        start_frame = int(start*self.fps)
        end_frame = int(end*self.fps)
        sizes = self.frame_sizes[start_frame:end_frame]
        if start_frame > 0:
            start_byte = self.frame_sizes[start_frame-1]
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



