"""
Goes through a dataset and add dataset to audio tracks with Voice Activity Detection data.
"""
import multiprocessing.dummy as multiprocessing
import collections
import time
import os.path
import argparse
import queue
import webrtcvad
import numpy as np
import scipy.io.wavfile as wavefile
from google.cloud import speech


from multimodal.dataset.video import VideoDataset
from multimodal.dataset.add_vad_signal import add_voiced_segment_facet
from multimodal.intervals import merge_intervals, filter_overlapping_intervals, trim_intervals

def main():
    parser = argparse.ArgumentParser(description="Script for finding speech in the audio "
                                                 "modality of multimodal datasets")
    parser.add_argument('dataset', help="Datasets to process", nargs='+')
    parser.add_argument('--mode',
                        help="WEBRTC VAD mode, 0 gives most false positives, 3 gives most false negatives",
                        type=int,
                        choices=(0,1,2,3),
                        default=3)
    parser.add_argument('--language-code', help="What language code to use for transcription", default='en-US')
    parser.add_argument('--minimum-duration', help="A speech segment needs to be at least this long")
    parser.add_argument('--quota', help="Maximum number of requests per minute", type=int, default=300)
    args = parser.parse_args()

    for dataset_path in args.dataset:
        find_non_subtitled_intervals(dataset_path)


def find_non_subtitled_intervals(dataset_path, merge_duration_subtitles_ms=300, merge_duration_voiced_ms=500, trim_duration_ms=500):
    add_voiced_segment_facet(dataset_path, overwrite=True)

    with VideoDataset(dataset_path) as dataset:
        sample_rate = dataset.get_samplerate('audio')
        audio_facet = dataset.get_facet('audio')
        subtitles = dataset.get_facet('subtitles')
        subtitle_times = subtitles.get_times()
        #subtitle_times = subtitles.get_times_filtered(lambda time, text: not text.isupper())
        merge_duration_frames_subtitles = merge_duration_subtitles_ms * sample_rate // 1000
        merge_duration_voiced_frames = merge_duration_voiced_ms * sample_rate // 1000
        trim_duration_frames = trim_duration_ms * sample_rate // 1000

        subtitled_intervals = merge_intervals((sample_rate*subtitle_times).astype(np.uint), merge_duration_frames_subtitles)
        voiced_intervals = audio_facet.get_time_intervals('voiced_segments')
        voiced_non_subtitled_times = filter_overlapping_intervals(voiced_intervals, subtitled_intervals, filter_coverage=0.7)

        voiced_non_subtitled_times_merged = trim_intervals(merge_intervals(voiced_non_subtitled_times, merge_duration_voiced_frames), trim_duration_frames)

        voiced_non_subtitled_audio = audio_facet.get_frames(voiced_non_subtitled_times_merged)

        timed_audio_segments = [(start, end, audio) for (start, end), audio in zip(voiced_non_subtitled_times_merged, voiced_non_subtitled_audio)]
        #transcribe(timed_audio_segments, sample_rate, dataset_path)
        save_waves(timed_audio_segments, sample_rate, dataset_path)


def transcribe_dataset(datasets_paths, vad):
    for dataset_path in datasets_paths:
        with VideoDataset(dataset_path) as dataset:
            sample_rate = dataset.get_samplerate('audio')
            frame_duration = 30
            n_frame_samples = (sample_rate // 1000) * frame_duration
            speech_frames = []
            for times, audio_segment, in dataset.get_subtitled_complement_streams(['time', 'audio']):
                n_frames = len(audio_segment) // n_frame_samples
                start_time, end_time = times
                start_sample = int(start_time * sample_rate)
                for i in range(n_frames):
                    start = i * n_frame_samples
                    end = start + n_frame_samples
                    frame_samples = audio_segment[start:end]
                    if vad.is_speech(frame_samples.tobytes(), sample_rate):
                        speech_frames.append((start_sample + start, start_sample + end, frame_samples))
            if speech_frames:
                speech_segments = []
                current_frames = []
                current_start, current_end, current_sample = speech_frames[0]
                current_frames.append(current_sample)
                for frame_start, frame_end, sample in speech_frames[1:]:
                    if current_end + 1 >= frame_start:
                        current_end = frame_end
                        current_frames.append(sample)
                    else:
                        speech_segments.append((current_start, current_end, np.concatenate(current_frames)))
                        current_frames = [sample]
                        current_start = frame_start
                        current_end = frame_end
                speech_segments.append((current_start, current_end, np.concatenate(current_frames)))

                audio = np.concatenate([segment for start, end, segment in speech_segments])
                filename = os.path.splitext(dataset_path)[0] + '.wav'
                wavefile.write(filename, 16000, audio)

                # transcriptions = transcribe_sync(speech_segments, sample_rate, quota=args.quota, language_code=args.language_code)

                # transcript_file = os.path.splitext(dataset_path)[0] + '.csv'
                # with open(transcript_file, 'w') as fp:
                #    fp.write("#start\tend\tconfidence\ttranscript\n")
                #    for transcript in transcriptions:
                #        fp.write("{}\t{}\t{}\t{}\n".format(*transcript))


def transcribe(speech_segments, sample_rate, dataset_path, quota=300, language_code='en-GB', n_processes=30):
    # We have a maximum qouta of 300 requests per minute, for now we just run few enough processes for it not to matter
    speech_segment_queue = multiprocessing.Queue()
    transcription_queue = multiprocessing.Queue()
    for speech_segment in speech_segments:
        speech_segment_queue.put(speech_segment)

    wait_time = 60*n_processes/quota
    processes = [multiprocessing.Process(target=transcribe_worker,
                                         args=(speech_segment_queue,
                                               transcription_queue,
                                               sample_rate,
                                               language_code,
                                               wait_time))
                 for i in range(n_processes)]
    for p in processes:
        p.start()
    transcriptions = []
    for i in range(len(speech_segments)):
        result = transcription_queue.get(True)
        if result is not None:
            transcriptions.append(result)

    transcriptions.sort()
    transcript_file = os.path.splitext(dataset_path)[0] + '_transcript.csv'
    with open(transcript_file, 'w') as fp:
        fp.write("#start\tend\tconfidence\ttranscript\n")
        for transcript in transcriptions:
            fp.write("{}\t{}\t{}\t{}\n".format(*transcript))


def transcribe_worker(speech_segment_queue, transcription_queue, sample_rate, language_code, wait_time):
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code=language_code,
        sample_rate_hertz=sample_rate,
        enable_word_time_offsets=True,
        model='video')

    try:
        while True:
            t0 = time.time()
            (start, end, speech_segment) = speech_segment_queue.get(True, 1)

            print("Processing segment starting at {}, ending at {}".format(start, end))
            content = speech_segment.tobytes()
            audio = speech.types.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            results = response.results
            if results:
                alternative = response.results[0].alternatives[0]
                transcription_queue.put((start, end, alternative.confidence, alternative.transcript))
            else:
                transcription_queue.put(None)
            t1 = time.time()
            if t1 < t0 + wait_time:
                time.sleep((t0+wait_time) - t1)
    except queue.Empty:
        pass



def save_waves(voiced_frames, sample_rate, dataset_path):
    for start, end, audio in voiced_frames:
        filename = os.path.splitext(dataset_path)[0] + '_{:08.3f}-{:08.3f}.wav'.format(start/sample_rate, end/sample_rate)
        wavefile.write(filename, sample_rate, audio)
    filename = os.path.splitext(dataset_path)[0] + '.wav'
    wavefile.write(filename, sample_rate, np.concatenate([audio for start, end, audio in voiced_frames]))


if __name__ == '__main__':
    main()