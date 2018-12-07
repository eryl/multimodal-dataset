import argparse
import sounddevice
import multimodal.dataset.video
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    dataset = multimodal.dataset.video.VideoDataset(args.dataset)
    with sounddevice.OutputStream(device=7, samplerate=24000, channels=1, dtype=np.int16) as stream:
        for i in range(20):
            (audio, text, ar) = dataset.get_subtitle(i)
            print(text)
            stream.write(audio)


if __name__ == '__main__':
    main()

