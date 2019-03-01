import numpy as np
from multimodal.dataset.facet.facet_handler import FacetHandler
from numbers import Integral

class SubtitleFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(SubtitleFacet, self).__init__(*args, **kwargs)
        self.texts = self.facetgroup['texts']
        self.times = self.facetgroup['times']

    @classmethod
    def create_facets(cls, subtitles_modality, subtitles_paths):
        import os.path
        for subtitles_path in subtitles_paths:
            subtitle_name = os.path.splitext(os.path.basename(subtitles_path))[0]
            cls.create_facet(subtitle_name, subtitles_modality, subtitles_path)

    @classmethod
    def create_facet(cls, name, modality_group, subtitles_file):
        import multimodal.srt as srt
        import html.parser
        import h5py

        class SubtitleParser(html.parser.HTMLParser):
            def __init__(self):
                super(SubtitleParser, self).__init__()
                self.color = None
                self.data = None

            def handle_starttag(self, tag, attrs):
                if tag == 'font':
                    if attrs[0][0] == 'color':
                        color = attrs[0][1].strip('#')
                        if self.color is None:
                            self.color = []
                        self.color.append((int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)))

            def handle_data(self, data):
                if self.data is None:
                    self.data = []
                self.data.append(data)

            def wipe(self):
                self.data = None
                self.color = None

        with open(subtitles_file) as fp:
            subtitles = list(srt.parse(fp.read()))
        subtitle_parser = SubtitleParser()

        font_colors = dict()
        color_i = 0
        all_colors = []
        all_texts = []
        times = []

        for subtitle in subtitles:
            subtitle_parser.wipe()
            start = subtitle.start.total_seconds()
            end = subtitle.end.total_seconds()
            subtitle_parser.feed(subtitle.content)
            texts = subtitle_parser.data
            colors = subtitle_parser.color
            color_ids = []
            for text in texts:
                if colors is not None:
                    for color_tuple in colors:
                        if not color_tuple in font_colors:
                            font_colors[color_tuple] = color_i
                            color_i += 1
                        color_id = font_colors[color_tuple]
                else:
                    color_id = -1
                color_ids.extend([color_id, len(text)])

            all_texts.append(''.join(texts))
            all_colors.extend(color_ids)
            times.append([start, end])
        subtitles_color_index = np.array([color for color, i in sorted(font_colors.items(), key=lambda x: x[1])],
                                         np.uint8)
        subtitles_facet = modality_group.require_group(name)
        subtitles_facet.create_dataset('color_index', data=subtitles_color_index)
        subtitles_facet.create_dataset('colors', data=np.array(all_colors, dtype=np.int32))
        subtitle_texts = subtitles_facet.create_dataset('texts', shape=(len(all_texts),),
                                                             dtype=h5py.special_dtype(vlen=str))
        subtitle_texts[:] = all_texts
        subtitles_facet.create_dataset('times', data=np.array(times, dtype=np.float32))
        subtitles_facet.attrs['FacetHandler'] = 'SubtitleFacet'
        return SubtitleFacet(subtitles_facet)

    def get_times_complement(self, minimum_time=0):
        """Returns timestamps of parts which doesn't have subtitles"""
        n_times = len(self.times)
        times_complement = np.zeros((n_times+1, 2), dtype=self.times.dtype)
        times_complement[1:, 0] = self.times[:, 1]
        times_complement[:-1, 1] = self.times[:, 0]
        long_enough = times_complement[:,1] - times_complement[:,0] > minimum_time
        return times_complement[long_enough]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        if isinstance(item, slice):
            times = self.times[item]
            texts = self.texts[item]
            return zip(times, texts)
        elif isinstance(item, Integral):
            if item >= len(self.times):
                raise IndexError()
            time = self.times[item]
            text = self.texts[item]
            return time, text
        else:
            raise TypeError("Invalid argument type.")


