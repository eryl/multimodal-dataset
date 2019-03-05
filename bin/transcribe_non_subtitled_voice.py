"""
Goes through a dataset and add dataset to audio tracks with Voice Activity Detection data.
"""
import multiprocessing.dummy as multiprocessing
#import multiprocessing
import collections
import time
import os.path
import argparse
import numpy as np
import scipy.io.wavfile as wavefile
from google.cloud import speech
import queue
import glob

from multimodal.dataset.video import VideoDataset
from multimodal.dataset.add_vad_signal import add_voiced_segment_facet
from multimodal.intervals import merge_intervals, filter_overlapping_intervals, trim_intervals
import json

def main():
    parser = argparse.ArgumentParser(description="Script for finding speech in the audio "
                                                 "modality of multimodal datasets")
    parser.add_argument('datasets', help="Datasets to process", nargs='+')
    parser.add_argument('--mode',
                        help="WEBRTC VAD mode, 0 gives most false positives, 3 gives most false negatives",
                        type=int,
                        choices=(0,1,2,3),
                        default=3)
    parser.add_argument('--language-code', help="What language code to use for transcription", default='en-US')
    parser.add_argument('--minimum-duration', help="A speech segment needs to be at least this long")
    parser.add_argument('--quota', help="Maximum number of requests per minute", type=int, default=150)
    parser.add_argument('--n-processes', type=int, default=1)
    parser.add_argument('--n-transcription-processes', type=int, default=30)
    args = parser.parse_args()

    dataset_paths = []
    for dataset_path in args.datasets:
        if os.path.isdir(dataset_path):
            datasets = glob.glob(os.path.join(dataset_path + '/**/' + '*.h5'), recursive=True)
            dataset_paths.extend(datasets)
        elif os.path.isfile(dataset_path):
            dataset_paths.append(dataset_path)
    print("Dataset paths: ", dataset_paths)
    if args.n_processes > 1:
        transcribe_multiprocessing(dataset_paths, args.n_processes, n_transcription_processes=args.n_transcription_processes, quota=args.quota)
    else:
        transcribe_single_process(dataset_paths, n_transcription_processes=args.n_transcription_processes, quota=args.quota)


def transcribe_single_process(dataset_paths, n_transcription_processes=30, quota=300, timeout=0.3):
    speech_request_queue = multiprocessing.Queue(n_transcription_processes*4)
    transcription_results_queue = multiprocessing.Queue()
    wait_time = 60 * n_transcription_processes / quota
    do_quit = multiprocessing.Event()
    pid = 0
    speech_transcriber_processes = [multiprocessing.Process(target=speech_transcriber_worker,
                                                            args=(speech_request_queue, transcription_results_queue,
                                                                  do_quit, wait_time),
                                                            kwargs=dict(timeout=timeout, language_code='en-GB'))
                                    for i in range(n_transcription_processes)]
    for p in speech_transcriber_processes:
        p.start()
    for dataset_path in dataset_paths:
        transcribe_dataset(dataset_path, pid, do_quit, speech_request_queue, transcription_results_queue, timeout=timeout)
    do_quit.set()
    for p in speech_transcriber_processes:
        p.join()


def transcribe_multiprocessing(dataset_paths, n_processes, n_transcription_processes=30, timeout=0.3, quota=300):
    dataset_queue = multiprocessing.Queue()  # This is a queue of all the datasets to transcribe
    for dataset_path in dataset_paths:
        dataset_queue.put(dataset_path)

    speech_request_queue = multiprocessing.Queue()   # This is a queue of speech segments to transcribe, transcription processes
                                             # takes objects from this queue and uses Google Cloud speech to transcribe
                                             # them into text
    per_process_speech_request_queues = [multiprocessing.Queue(n_transcription_processes) for i in range(n_processes)]  # To not starve any of the

    transcription_results_queue = multiprocessing.Queue()  # transcription processes put the results on this queue. This
                                                           # process then places them in the correct queue for the
                                                           # dataset transcriber process
    per_process_transcription_results_queues = [multiprocessing.Queue() for i in range(n_processes)]
    do_quit = multiprocessing.Event()

    speech_requests_process = multiprocessing.Process(target=speech_transcription_multiplexer, args=(speech_request_queue, per_process_speech_request_queues, do_quit), kwargs=dict(timeout=timeout))
    speech_requests_process.start()
    transcription_demultiplexer = multiprocessing.Process(target=transcription_results_demultiplexer, args=(transcription_results_queue, per_process_transcription_results_queues, do_quit), kwargs=dict(timeout=timeout))
    transcription_demultiplexer.start()

    dataset_transcriber_processes = [multiprocessing.Process(target=dataset_transcriber_worker,
                                                             args=(pid, dataset_queue, request_queue,
                                                                   results_queue, do_quit),
                                                             kwargs=dict(timeout=0.3))
                                     for pid, (results_queue, request_queue)
                                     in enumerate(zip(per_process_transcription_results_queues,
                                                      per_process_speech_request_queues))]
    for p in dataset_transcriber_processes:
        p.start()

    wait_time = 60 * n_transcription_processes / quota
    speech_transcriber_processes = [multiprocessing.Process(target=speech_transcriber_worker,
                                                            args=(speech_request_queue, transcription_results_queue,
                                                                  do_quit, wait_time),
                                                            kwargs=dict(timeout=timeout, language_code='en-GB'))
                                    for i in range(n_transcription_processes)]
    for p in speech_transcriber_processes:
        p.start()

    try:
        # We now wait for the dataset transcription workers to exit
        while dataset_queue.qsize() > 0:
            print("Datasets processed: {}/{}".format(len(dataset_paths)-dataset_queue.qsize(), len(dataset_paths)))
            time.sleep(5)
    except KeyboardInterrupt:
        # Keyboard interrupt
        pass
    finally:
        for p in dataset_transcriber_processes:
            p.join()

    do_quit.set()
    for p in dataset_transcriber_processes:
        p.join()
    for p in speech_transcriber_processes:
        p.join()
    speech_requests_process.join()
    transcription_demultiplexer.join()


def speech_transcription_multiplexer(speech_request_queue, per_process_speech_request_queues, do_quit, timeout=0.3):
    """
    Process which multiplexes speech requests from the dataset workers, so that their speech requests gets fairly
    distributed to the speech transcription workers, avoiding starving dataset processes
    :return:
    """
    while not do_quit.is_set():
        for pq in per_process_speech_request_queues:
            try:
                message = pq.get(True, timeout)
                speech_request_queue.put(message)
            except queue.Empty:
                pass


def transcription_results_demultiplexer(transcription_results_queue, per_process_transcription_result_queue, do_quit, timeout=0.3):
    """
    Demultiplexer
    :return:
    """
    while not do_quit.is_set():
        try:
            pid, message = transcription_results_queue.get(True, timeout)
            per_process_transcription_result_queue[pid].put((pid, message))
        except queue.Empty:
            pass


def dataset_transcriber_worker(pid, dataset_queue, speech_queue, results_queue, do_quit, timeout=0.3):

    while not do_quit.is_set():
        try:
            dataset_path = dataset_queue.get(False)
            transcribe_dataset(dataset_path, pid, do_quit, speech_queue, results_queue, timeout=timeout)
        except queue.Empty:
            break


def transcribe_dataset(dataset_path, pid, do_quit, speech_queue, results_queue, timeout=0.3):
    print("Transcribing dataset {}".format(dataset_path))
    transcripts = []
    num_requests_made = 0
    num_responses_received = 0
    sample_rate = get_samplerate(dataset_path)
    speech_segments = find_non_subtitled_intervals(dataset_path)
    try:
        speech_segment = next(speech_segments)
        while not do_quit.is_set():
            # Fill upp the request queue
            while not do_quit.is_set():
                try:
                    start, end, audio = speech_segment
                    speech_queue.put_nowait((pid, (start, end, audio, sample_rate)))
                    num_requests_made += 1
                    speech_segment = next(speech_segments)
                except queue.Full:
                    break
            # Pump all responses
            while not do_quit.is_set():
                try:
                    response = results_queue.get()
                    transcripts.append(response)
                    num_responses_received += 1
                except queue.Empty:
                    break
    except StopIteration:
        pass
    # No more speech segments, now wait for all the requests to arrive
    while num_requests_made > num_responses_received and not do_quit.is_set():
        try:
            response = results_queue.get(True, timeout)
            transcripts.append(response)
            num_responses_received += 1
            print("<{}>: Segments processed: {:%}".format(pid, num_responses_received / num_requests_made))
        except queue.Empty:
            pass
    filtered_transcripts = []
    for pid, (start, end, result) in transcripts:
        if result is not None:
            confidence, transcript, words = result
            filtered_transcripts.append(
                dict(start=start, end=end, confidence=confidence, transcript=transcript, words=words))
    filtered_transcripts.sort(key=lambda x: x['start'])

    transcript_file = os.path.splitext(dataset_path)[0] + '.json'
    with open(transcript_file, 'w') as fp:
        json.dump(filtered_transcripts, fp, sort_keys=True, indent=4)


def speech_transcriber_worker(speech_queue, transcription_results_queue, do_quit, wait_time, timeout=0.3, language_code='en-GB'):
    client = speech.SpeechClient()

    while not do_quit.is_set():
        try:
            t0 = time.time()
            pid, (start, end, speech_segment, sample_rate) = speech_queue.get(True, timeout)
            config = speech.types.RecognitionConfig(
                encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code=language_code,
                sample_rate_hertz=sample_rate,
                enable_word_time_offsets=True)

            content = speech_segment.tobytes()
            audio = speech.types.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            results = response.results
            if results:
                alternative = response.results[0].alternatives[0]
                confidence = alternative.confidence
                transcript = alternative.transcript
                words = alternative.words
                filtered_words = []
                for word_info in words:
                    start_time = start + word_info.start_time.seconds + word_info.start_time.nanos * 1e-9
                    end_time = start + word_info.end_time.seconds + word_info.end_time.nanos * 1e-9
                    word = word_info.word
                    filtered_words.append((start_time, end_time, word))
                transcription = (confidence, transcript, filtered_words)
            else:
                transcription = None
            transcription_results_queue.put((pid, (start, end, transcription)))
            t1 = time.time()
            #if t1 < t0 + wait_time:
            #    time.sleep((t0 + wait_time) - t1)
            time.sleep(wait_time)
        except queue.Empty:
            pass

def get_samplerate(dataset_path):
    with VideoDataset(dataset_path) as dataset:
        sample_rate = dataset.get_samplerate('audio')
        return sample_rate

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

        for (start, end), audio in zip(voiced_non_subtitled_times_merged, voiced_non_subtitled_audio):
            yield start/sample_rate, end/sample_rate, audio


# def transcribe_dataset(datasets_paths, vad):
#     for dataset_path in datasets_paths:
#         with VideoDataset(dataset_path) as dataset:
#             sample_rate = dataset.get_samplerate('audio')
#             frame_duration = 30
#             n_frame_samples = (sample_rate // 1000) * frame_duration
#             speech_frames = []
#             for times, audio_segment, in dataset.get_subtitled_complement_streams(['time', 'audio']):
#                 n_frames = len(audio_segment) // n_frame_samples
#                 start_time, end_time = times
#                 start_sample = int(start_time * sample_rate)
#                 for i in range(n_frames):
#                     start = i * n_frame_samples
#                     end = start + n_frame_samples
#                     frame_samples = audio_segment[start:end]
#                     if vad.is_speech(frame_samples.tobytes(), sample_rate):
#                         speech_frames.append((start_sample + start, start_sample + end, frame_samples))
#             if speech_frames:
#                 speech_segments = []
#                 current_frames = []
#                 current_start, current_end, current_sample = speech_frames[0]
#                 current_frames.append(current_sample)
#                 for frame_start, frame_end, sample in speech_frames[1:]:
#                     if current_end + 1 >= frame_start:
#                         current_end = frame_end
#                         current_frames.append(sample)
#                     else:
#                         speech_segments.append((current_start, current_end, np.concatenate(current_frames)))
#                         current_frames = [sample]
#                         current_start = frame_start
#                         current_end = frame_end
#                 speech_segments.append((current_start, current_end, np.concatenate(current_frames)))
#
#                 audio = np.concatenate([segment for start, end, segment in speech_segments])
#                 filename = os.path.splitext(dataset_path)[0] + '.wav'
#                 wavefile.write(filename, 16000, audio)
#
                # transcriptions = transcribe_sync(speech_segments, sample_rate, quota=args.quota, language_code=args.language_code)

                # transcript_file = os.path.splitext(dataset_path)[0] + '.csv'
                # with open(transcript_file, 'w') as fp:
                #    fp.write("#start\tend\tconfidence\ttranscript\n")
                #    for transcript in transcriptions:
                #        fp.write("{}\t{}\t{}\t{}\n".format(*transcript))



# def get_voiced_non_subtitled_segments(dataset, merge_duration_subtitles_ms, merge_duration_voiced_ms, trim_duration_ms):
#     sample_rate = dataset.get_samplerate('audio')
#     audio_facet = dataset.get_facet('audio')
#     subtitles = dataset.get_facet('subtitles')
#     subtitle_times = subtitles.get_times()
#     # subtitle_times = subtitles.get_times_filtered(lambda time, text: not text.isupper())
#     merge_duration_frames_subtitles = merge_duration_subtitles_ms * sample_rate // 1000
#     merge_duration_voiced_frames = merge_duration_voiced_ms * sample_rate // 1000
#     trim_duration_frames = trim_duration_ms * sample_rate // 1000
#
#     subtitled_intervals = merge_intervals((sample_rate * subtitle_times).astype(np.uint),
#                                           merge_duration_frames_subtitles)
#     voiced_intervals = audio_facet.get_time_intervals('voiced_segments')
#     voiced_non_subtitled_times = filter_overlapping_intervals(voiced_intervals, subtitled_intervals,
#                                                               filter_coverage=0.7)
#
#     voiced_non_subtitled_times_merged = trim_intervals(
#         merge_intervals(voiced_non_subtitled_times, merge_duration_voiced_frames), trim_duration_frames)
#
#     voiced_non_subtitled_audio = audio_facet.get_frames(voiced_non_subtitled_times_merged)
#
#     timed_audio_segments = [(start, end, audio) for (start, end), audio in
#                             zip(voiced_non_subtitled_times_merged, voiced_non_subtitled_audio)]
#     return timed_audio_segments
#



def transcribe(speech_segments, sample_rate, dataset_path, quota=300, language_code='en-GB', n_processes=30):
    # We have a maximum qouta of 300 requests per minute, for now we just run few enough processes for it not to matter
    speech_segment_queue = multiprocessing.Queue()
    transcription_queue = multiprocessing.Queue()
    for speech_segment in speech_segments:
        speech_segment_queue.put(speech_segment)

    wait_time = 60 * n_processes / quota
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
            (pid, start, end, speech_segment) = speech_segment_queue.get(True, 1)

            print("Processing segment starting at {}, ending at {}".format(start, end))
            content = speech_segment.tobytes()
            audio = speech.types.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            results = response.results
            if results:
                alternative = response.results[0].alternatives[0]
                transcription_queue.put((pid, start, end, alternative.confidence, alternative.transcript))
            else:
                transcription_queue.put((pid, start, end, None))
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