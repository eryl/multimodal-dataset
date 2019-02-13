import requests
import os.path

SINTEL_VIDEO_URL = 'http://peach.themazzone.com/durian/movies/sintel-1024-surround.mp4'
SINTEL_VIDEO_FILE_PATH = 'sintel-1024-surround.mp4'
SINTEL_SUBTITLE_URL = 'https://durian.blender.org/wp-content/content/subtitles/sintel_en.srt'
SINTEL_SUBTITLE_FILE_PATH = 'sintel_en.srt'

def download(url, output_path):
    print("Downloading sintel video")
    r = requests.get(url, stream=True)
    n_bytes = int(r.headers['Content-Length'])
    n_written = 0
    with open(output_path, 'wb') as file:
        for chunk in r.iter_content(chunk_size=2**20):
            if chunk:  # filter out keep-alive new chunks
                file.write(chunk)
            n_written += len(chunk)
            print("{:%}".format(n_written/n_bytes))

def main():
    if not os.path.exists(SINTEL_VIDEO_FILE_PATH):
        print('Downloading Sintel Video from {}'.format(SINTEL_VIDEO_URL))
        download(SINTEL_VIDEO_URL, SINTEL_VIDEO_FILE_PATH)
    else:
        print('Sintel video already exists at {}'.format(SINTEL_VIDEO_FILE_PATH))


    if not os.path.exists(SINTEL_SUBTITLE_FILE_PATH):
        print("Downloading enlgish subtitles from {}".format(SINTEL_SUBTITLE_URL))
        r = requests.get(SINTEL_SUBTITLE_URL)
        with open(SINTEL_SUBTITLE_FILE_PATH, 'wb') as subtitle:
            subtitle.write(r.content)
    else:
        print("Subtitles already exists at {}".format(SINTEL_SUBTITLE_FILE_PATH))


if __name__ == '__main__':
    main()
