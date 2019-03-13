import argparse
import os.path
import glob
import json
from datetime import timedelta
from multimodal.intervals import merge_annotated_intervals
import multimodal.srt as srt

def main():
    parser = argparse.ArgumentParser(description="Converts transcripts to subrip format")
    parser.add_argument('transcripts', help="Transcripts folder or json file", nargs='+')
    parser.add_argument('--min-overlap', help="If the transcript has time offsets for words, they will be merged if the distance between their time intervals is at least this much", type=float, default=0.1)
    args = parser.parse_args()

    transcripts_paths = []
    for path in args.transcripts:
        if os.path.isdir(path):
            transcripts = glob.glob(os.path.join(path + '/**/' + '*.json'), recursive=True)
            transcripts_paths.extend(transcripts)
        elif '.json' in path:
            transcripts_paths.append(path)
        else:
            raise ValueError("Not a directory os transcripts json file: {}".format(path))


    for transcript_path in transcripts_paths:
        # First create a subrip file from the transcript by merging the times for the words if they are present

        with open(transcript_path) as fp:
            transcript = json.load(fp)
            subtitle_entries = []
            i = 1
            for transcription in transcript:
                if 'words' in transcription:
                    # We have words, let's merge the ones which are less than merge_time_ms milliseconds apart
                    merged_intervals = merge_annotated_intervals(transcription['words'], args.min_overlap)

                    for start, end, words in merged_intervals:
                        entry = srt.Subtitle(i, start=timedelta(seconds=float(start)),
                                             end=timedelta(seconds=float(end)),
                                             content=' '.join(words))
                        subtitle_entries.append(entry)
                        i += 1
                else:
                    start = transcription['start']
                    end = transcription['end']
                    content = transcription['transcript']
                    entry = srt.Subtitle(i, start=timedelta(seconds=float(start)),
                                         end=timedelta(seconds=float(end)),
                                         content=content)
                    subtitle_entries.append(entry)
                    i += 1
            subrip_file = srt.compose(subtitle_entries)
            basename = os.path.splitext(transcript_path)[0]
            output_file = basename + '_transcript.srt'
            with open(output_file, 'w') as fp:
                fp.write(subrip_file)



if __name__ == '__main__':
    main()