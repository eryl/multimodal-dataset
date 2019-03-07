import collections
import webrtcvad
import numpy as np

from multimodal.dataset.video import VideoDataset


def add_voiced_segment_facet(dataset_path, vad_mode=3, frame_duration_ms=30, overwrite=False):
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)

    with VideoDataset(dataset_path, mode='r+') as dataset:
        audio_facets, = dataset.get_all_facets(['audio'])
        for audio_facet in audio_facets:
            if overwrite or not audio_facet.has_time_intervals('voiced_segments'):
                sample_rate = audio_facet.get_samplerate()
                voiced_frames = list(vad_slice_audio_signal(audio_facet.get_all_frames(), sample_rate, vad, frame_duration_ms))
                voiced_frame_intervals = np.array([[start, end] for start, end, segment in voiced_frames])
                audio_facet.add_time_intervals('voiced_segments', voiced_frame_intervals, overwrite=overwrite)
            else:
                print("Audio facet {} already has voiced segment times".format(audio_facet.group_name()))


def vad_slice_audio_signal(audio_frames, sample_rate, vad, frame_duration_ms=30, padding_duration_ms = 100):
    """
    Apply the VAD classifier to all audio frames and return a list of start, end tuples for each active region detected.
    :param audio_frames: A numpy array of PCM audio samples
    :param sample_rate: The sample rate of the audio sampels
    :return:
    """

    frame_length = frame_duration_ms * sample_rate//1000

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    num_windows = len(audio_frames) // frame_length  # We will throw away the last samples less than 30 ms, this should not be an issue

    voiced_frames = []
    voiced_start_frame = None

    for i in range(num_windows):
        start = i*frame_length
        end = start+frame_length
        audio = audio_frames[start:end]
        is_speech = vad.is_speech(audio.tobytes(), sample_rate)

        if not triggered:
            ring_buffer.append((audio, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True

                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)

                # We calculate where these voiced frames start so we can calculate language offsets correctly
                buffer_length_frames = (len(ring_buffer)-1)*frame_length
                voiced_start_frame = start - buffer_length_frames
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(audio)
            ring_buffer.append((audio, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                concatenated = np.concatenate(voiced_frames)
                voiced_end_frame = voiced_start_frame + len(concatenated)
                yield (voiced_start_frame, voiced_end_frame, concatenated)
                ring_buffer.clear()
                voiced_frames = []
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        concatenated = np.concatenate(voiced_frames)
        voiced_end_frame = voiced_start_frame + len(concatenated)
        yield (voiced_start_frame, voiced_end_frame, concatenated)
