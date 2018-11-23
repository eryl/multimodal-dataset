import argparse
import sounddevice
import multimodal.dataset.video
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    dataset = multimodal.dataset.video.VideoDataset(args.dataset)
    (audio, text) = dataset.get_subtitle(0)
