import numpy as np
from multimodal.dataset.facet.facet_handler import FacetHandler
from numbers import Integral

NULL_COLOR = (-1, -1, -1)

class SubtitleFacet(FacetHandler):
    def __init__(self, *args, **kwargs):
        super(SubtitleFacet, self).__init__(*args, **kwargs)
        self.strings = self.facetgroup['strings'][:]
        self.string_index = self.facetgroup['string_index'][:]
        self.times = self.facetgroup['times'][:]
        self.string_styles = self.facetgroup['string_styles'][:]
        self.string_colors = self.facetgroup['string_colors'][:]

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
                self.colors = []
                self.strings = []
                self.attributes = []
                self.color = NULL_COLOR
                self.underline = False
                self.bold = False
                self.italic = False


            def handle_starttag(self, tag, attrs):
                if tag == 'font':
                    for (attr, value) in attrs:
                        if attr == 'color':
                            color = value.strip('#')
                            self.color = (int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16))
                elif tag == 'u':
                    self.underline = True
                elif tag == 'b':
                    self.bold = True
                elif tag == 'i':
                    self.italic = True

            def handle_data(self, data):
                self.strings.append(data)
                self.colors.append(self.color)
                self.attributes.append(dict(b=self.bold, u=self.underline, i=self.italic))

            def handle_endtag(self, tag):
                if tag == 'font':
                    self.color = NULL_COLOR
                elif tag == 'u':
                    self.underline = False
                elif tag == 'b':
                    self.bold = False
                elif tag == 'i':
                    self.italic = False

            def wipe(self):
                self.colors.clear()
                self.strings.clear()
                self.attributes.clear()
                self.color = NULL_COLOR
                self.underline = False
                self.bold = False
                self.italic = False

        with open(subtitles_file) as fp:
            subtitles = list(srt.parse(fp.read()))
        subtitle_parser = SubtitleParser()

        ## Each subtitle can contain multiple font-tagged elements. Since we want to preserve the font-information, we
        ## need to save the potentially multiple subtitles as independent strings. We solve this by using two layers of
        ## indexing. Each font-tagged string is a separate string in the subtitles dataset, this means that it will have
        ## equal or more rows than the intervals. We use a separate index to mark which strings belong to each subtitle
        ## interval using start and end indices into the strings dataset.

        subtitle_times = []  # These are the subtitle times as start and end intervals
        subtitle_string_index = []  # These are start and end indices into the subtitle_strings dataset.
        # Its shape is (len(subtitle_times), 2)
        subtitle_strings = []
        subtitle_strings_colors = []  # These are the font colors for each of the strings
        subtitle_strings_styles = []  # These are other attributes (underline, italic or bold)
        start_subtitle = 0
        for subtitle in subtitles:
            subtitle_parser.wipe()
            start = subtitle.start.total_seconds()
            end = subtitle.end.total_seconds()
            subtitle_times.append((start, end))

            subtitle_parser.feed(subtitle.content)

            subtitle_strings.extend(subtitle_parser.strings)
            subtitle_strings_colors.extend(subtitle_parser.colors)
            subtitle_strings_styles.extend(subtitle_parser.attributes)
            subtitle_string_index.append((start_subtitle, start_subtitle+len(subtitle_parser.strings)))
            start_subtitle += len(subtitle_parser.strings)

        subtitles_facet = modality_group.require_group(name)

        subtitles_facet.create_dataset('times', data=np.array(subtitle_times, dtype=np.float32))
        subtitles_facet.create_dataset('string_index', data=np.array(subtitle_string_index, dtype=np.uint32))

        subtitle_texts = subtitles_facet.create_dataset('strings',
                                                        shape=(len(subtitle_strings),),
                                                        dtype=h5py.special_dtype(vlen=str))
        subtitle_texts[:] = subtitle_strings

        arr_subtitle_strings_colors = np.array(subtitle_strings_colors, dtype=np.int16)
        subtitles_facet.create_dataset('string_colors', data=arr_subtitle_strings_colors)

        arr_subtitle_string_styles = np.array([(style['b'], style['i'], style['u']) for style in subtitle_strings_styles], dtype=np.bool)
        styles_dataset = subtitles_facet.create_dataset('string_styles', data=arr_subtitle_string_styles)

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
            string_indices = self.string_index[item]
            strings = self.strings[string_indices]
            return zip(times, strings)
        elif isinstance(item, Integral):
            if item >= len(self.times):
                raise IndexError()
            time = self.times[item]
            string_indices = self.string_index[item]
            strings = self.strings[string_indices]
            return time, strings
        else:
            raise TypeError("Invalid argument type.")

    def get_times(self):
        return self.times[:]

    def get_times_filtered(self, filter):
        """
        Only return times for which the filter returns true
        :param filter: A function accepting two arguments, the times and the text for each subtitle
        :return: The times where the filter function returns true
        """
        filtered = []
        for i in range(len(self.times)):
            times = self.times[i]
            start_string, end_string = self.string_index[i]
            strings = [string for string in self.strings[start_string : end_string] if filter(string)]
            if strings:
                filtered.append((times, '\n'.join(strings)))
        return np.array(filtered)

    def get_texts(self):
        return self.strings[:]

    def get_subrip_texts(self):
        """
        Returns the subtitles in SubRip format
        :return: a string with the SubRip formatted subtitles
        """
        import multimodal.srt as srt
        from datetime import timedelta
        subtitle_entries = []
        for i in range(len(self.times)):
            start, end = self.times[i]
            start_string, end_string = self.string_index[i]
            strings = []
            for j in range(start_string, end_string):
                opening_tags = []
                closing_tags = []
                string = self.strings[j]
                string_color = self.string_colors[j]
                (bold, italic, underlined) = self.string_styles[j]
                if np.all(string_color != NULL_COLOR):
                    hex_color_string = ''.join(['{:02x}'.format(c) for c in string_color])
                    opening_tags.append('<font color="#{}">'.format(hex_color_string))
                    closing_tags.append('</font>')
                if bold:
                    opening_tags.append('<b>')
                    closing_tags.append('</b>')
                if italic:
                    opening_tags.append('<i>')
                    closing_tags.append('</i>')
                if underlined:
                    opening_tags.append('<u>')
                    closing_tags.append('</u>')
                formatted_string = ''.join(opening_tags) + string + ''.join(reversed(closing_tags))
                strings.append(formatted_string)
            text = '\n'.join(strings)
            entry = srt.Subtitle(i+1, start=timedelta(seconds=float(start)), end=timedelta(seconds=float(end)), content=text)
            subtitle_entries.append(entry)
        return srt.compose(subtitle_entries)
