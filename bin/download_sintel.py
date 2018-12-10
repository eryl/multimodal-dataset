import requests
import os.path

SINTEL_VIDEO_URL = 'http://peach.themazzone.com/durian/movies/sintel-1024-surround.mp4'
SINTEL_VIDEO_FILE_PATH = 'sintel-1024-surround.mp4'
SINTEL_SUBTITLE_URL = 'https://durian.blender.org/wp-content/content/subtitles/sintel_en.srt'
SINTEL_SUBTITLE_FILE_PATH = 'sintel_en.srt'

def main():
    if not os.path.exists(SINTEL_VIDEO_FILE_PATH):
        print("Downloading sintel video")
        r = requests.get(SINTEL_VIDEO_URL)
        with open(SINTEL_VIDEO_FILE_PATH, 'wb') as video:
            video.write(r.content)
    if not os.path.exists(SINTEL_SUBTITLE_FILE_PATH):
        print("Downloading subtitles")
        r = requests.get(SINTEL_SUBTITLE_URL)
        with open(SINTEL_SUBTITLE_FILE_PATH, 'wb') as subtitle:
            subtitle.write(r.content)


if __name__ == '__main__':
    main()
