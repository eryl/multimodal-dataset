# MULTIMODAL PROJECT

This package is for doing machine learning on multimodal data, video with subtitles in particular.

## Requirements
The package relies on the packages:
 - `h5py`
 - `numpy`
 - `imageio`.

For converting movies to datasets, you will need `python-ffmpeg` and `srt`. 
You will also need a ffmpeg binary installed on your system.

To run the demo (see below) you will also need the `requests` package, 
alternatively you can download the Sintel files manually. 
Step 3 in the demo also requires `scipy` to be installed.

## Installation
In the base project directory, run:
```text
$ python setup.py develop
```

This will add symbolic links in your python distro to this package, which will update 
when you pull new changes.

## Demo
A demo of how to use the datasets can be found in the `bin/extract_subtitle_audio.py` script. First you need a 
multimodal video dataset to work with. After installing the package (see above), do the following:

1. Download the dataset: 
   ```text
   $ mkdir -p data && cd data && python ../bin/download_sintel.py
   ```
   This will download the free movie [Sintel](https://durian.blender.org/) along with english subtitles and place them in the `data/` directory.

2. Convert the movie and subtitle to a multimodal dataset, in the directory `data/`:
   ```text
   $ python ../bin/make_multimodal_dataset.py sintel-1024-surround.mp4 sintel_en.srt
   ```
   This will make a HDF5 multimodal dataset from the movie and subtitles, the dataset will have the same name as the video, but with the extension .h5 instead of .mp4

3. Make a directory for the wave files and run the extraction script:
   ```text
   $ mkdir -p sintel_waves && python ../bin/extract_subtitle_audio.py sintel-1024-surround.h5 sintel_waves
   ```
   If you get a numpy float deprecation warning, everything is still allright.

The directory `data/sintel_waves` will now contain wav files named by the subtitle text containing the corresponding audio.

## Batched data
The datasets doesn't support batched data yet, since this will 
require packing and padding of sequences. This is on the todo-list.