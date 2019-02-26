"""
Goes through a dataset and add dataset to audio tracks with Voice Activity Detection data.
"""
import os.path
import argparse
import webrtcvad
import numpy as np
import scipy.io.wavfile as wavefile
from google.cloud import speech

from multimodal.dataset.video import VideoDataset

def main():
    parser = argparse.ArgumentParser(description="Script for finding speech in the audio "
                                                 "modality of multimodal datasets")
    parser.add_argument('dataset', help="Datasets to process", nargs='+')
    parser.add_argument('--mode',
                        help="WEBRTC VAD mode, 0 gives most false positives, 3 gives most false negatives",
                        type=int,
                        choices=(0,1,2,3),
                        default=0)
    parser.add_argument('--language-code', help="What language code to use for transcription", default='en-GB')
    args = parser.parse_args()
    vad = webrtcvad.Vad(args.mode)

    for dataset_path in args.dataset:
        with VideoDataset(dataset_path) as dataset:
            sample_rate = dataset.get_samplerate('audio')
            frame_duration = 30
            n_frame_samples = (sample_rate//1000)*frame_duration
            speech_frames = []
            for times, audio_segment, in dataset.get_subtitled_complement_streams(['time', 'audio']):
                n_frames = len(audio_segment) // n_frame_samples
                start_time, end_time = times
                start_sample = int(start_time*sample_rate)
                for i in range(n_frames):
                    start = i*n_frame_samples
                    end = start + n_frame_samples
                    frame_samples = audio_segment[start:end]
                    if vad.is_speech(frame_samples.tobytes(), sample_rate):
                        speech_frames.append((start_sample+start, start_sample+end, frame_samples))
            if speech_frames:
                merged_speech_frames = []
                current_frames = []
                current_start, current_end, current_sample = speech_frames[0]
                current_frames.append(current_sample)
                for frame_start, frame_end, sample in speech_frames[1:]:
                    if current_end+1 >= frame_start:
                        current_end = frame_end
                        current_frames.append(sample)
                    else:
                        merged_speech_frames.append((current_start, current_end, np.concatenate(current_frames)))
                        current_frames = [sample]
                        current_start = frame_start
                        current_end = frame_end
                merged_speech_frames.append((current_start, current_end, np.concatenate(current_frames)))

                client = speech.SpeechClient()
                config = speech.types.RecognitionConfig(
                    encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
                    language_code=args.language_code,
                    sample_rate_hertz=sample_rate)
                transcripts_operations = []
                for start, end, speech_segment in merged_speech_frames:
                    content = speech_segment.tobytes()
                    audio = speech.types.RecognitionAudio(content=content)
                    response = client.recognize(config=config, audio=audio)
                    results = response.results
                    if results:
                        alternative = response.results[0].alternatives[0]
                        transcripts_operations.append((start, end, alternative.confidence, alternative.transcript))
                transcriptions = transcripts_operations
                #for start, end, transcripts_operation in transcripts_operations:
                #    op_result = transcripts_operation.result()
                #    for result in op_result.result:
                #        for alternative in result.alternatives:
                #            transcriptions.append((start, end, alternative.confidence, alternative.transcript))

                transcript_file = os.path.splitext(dataset_path)[0] + '.csv'
                with open(transcript_file, 'w') as fp:
                    fp.write("#start\tend\tconfidence\ttranscript\n")
                    for transcript in transcriptions:
                        fp.write("{}\t{}\t{}\t{}\n".format(*transcript))








if __name__ == '__main__':
    main()