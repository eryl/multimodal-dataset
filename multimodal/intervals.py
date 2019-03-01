import numpy as np


def filter_overlapping_intervals(to_be_filtered, filter_intervals, filter_coverage=0.6):
    """
    Removes any interval in *to_be_filtered* which overlaps an interval in *filter_intervals* by a ratio of at least *filter_coverate*
    :param to_be_filtered: An ndarray fo shape (n, 2) with time-sorted intervals
    :param filter_intervals: An ndarray of shape (m, 2) with time-sorted intervals
    :param filter_coverage: The ratio of how much the intervals needs to be covered by the filter interval to be removed
    :return: A ndarray of shape (p < n, 2) with the intervals not covered by any filter intervals.
    """
    current_filter_interval = 0
    current_interval = 0
    filtered_intervals = []
    while current_filter_interval < len(filter_intervals) and current_interval < len(to_be_filtered):
        start, end = to_be_filtered[current_interval]
        interval_length = end - start
        overlap = 0

        # Find all filter intevals which overlaps the current, and add their overlap
        while current_filter_interval < len(filter_intervals):
            filter_start, filter_end = filter_intervals[current_filter_interval]

            if filter_start > end:
                # The current filter is beyond the current interval to be filtered, we're done and the interval did not get filtered
                filtered_intervals.append((start, end))
                break

            elif start < filter_end:
                # The filter overlaps the interval, accumulate the overlap
                # it to the list
                filter_length = filter_end - filter_start
                filter_overlap = (filter_length -
                           (max(0, start - filter_start) +  # This is how much the filter overlaps on the "left" side of the interval
                            max(0, filter_end - end)))      # This is how much the filter overlaps on the "right" side of the interval
                overlap += filter_overlap
                if end < filter_end:
                    if overlap / interval_length < filter_coverage:
                        filtered_intervals.append((start, end))
                    break

            current_filter_interval += 1
        current_interval += 1
    return np.array(filtered_intervals)

def old_filter_overlapping_intervals(to_be_filtered, filter_intervals, filter_coverage=0.6):
    """
    Removes any interval in *to_be_filtered* which overlaps an interval in *filter_intervals* by a ratio of at least *filter_coverate*
    :param to_be_filtered: An ndarray fo shape (n, 2) with time-sorted intervals
    :param filter_intervals: An ndarray of shape (m, 2) with time-sorted intervals
    :param filter_coverage: The ratio of how much the intervals needs to be covered by the filter interval to be removed
    :return: A ndarray of shape (p < n, 2) with the intervals not covered by any filter intervals.
    """
    current_subtitle_interval = 0
    current_voiced_interval = 0
    filtered_intervals = []
    while current_subtitle_interval < len(filter_intervals) and current_voiced_interval < len(to_be_filtered):
        sub_start, sub_end = filter_intervals[current_subtitle_interval]
        sub_length = sub_end - sub_start
        # First find the first voiced interval which overlaps the current subtitled interval
        while current_voiced_interval < len(to_be_filtered):
            voiced_start, voiced_end = to_be_filtered[current_voiced_interval]

            if voiced_end < sub_start:
                # The voiced completely precedes the subtitled interval and the voiced is directly placed in the list
                filtered_intervals.append((voiced_start, voiced_end))
            elif voiced_start < sub_end:
                # The voiced interval overlaps the subtitle, if it overlaps by less than filter_coverage, we add
                # it to the list
                voiced_length = voiced_end - voiced_start
                overlap = voiced_length - (max(0, sub_start - voiced_start) + max(0, voiced_end - sub_end))
                overlap_ratio = overlap / voiced_length
                if overlap_ratio < filter_coverage:
                    filtered_intervals.append((voiced_start, voiced_end))
            else:
                # Now the voiced interval is on the other side of the current subtitled and we should switch to the
                # next subtitled interval
                break
            current_voiced_interval += 1
        current_subtitle_interval += 1
    return np.array(filtered_intervals)


def merge_intervals(intervals, merge_duration):
    """
    Merge intervals if the number of frames between them is less than merge_duration
    :param intervals: A ndarray of shape (n_intervals, 2)
    :param merge_duration: The threshold of how many time steps to use for merging
    :return: A ndarray of shape (n_merged_intervals, 2)
    """
    if len(intervals) < 2:
        return intervals
    merged_intervals = []
    previous_start, previous_end = intervals[0]
    for i in range(1, len(intervals)):
        start, end = intervals[i]
        if start < previous_end + merge_duration:
            previous_end = end
        else:
            merged_intervals.append([previous_start, previous_end])
            previous_start, previous_end = start, end
    merged_intervals.append([previous_start, previous_end])
    return np.array(merged_intervals)


def trim_intervals(intervals, trim_length):
    interval_lengths = intervals[:,1] - intervals[:,0]
    return intervals[interval_lengths > trim_length]


def limit_length(videos, sample_rate):
    frames_per_minute = sample_rate*60
    for start, end, segment in videos:
        if len(segment) >= frames_per_minute:
            # We divide the samples as evenly as possible among the windows
            num_sub_segments = int(np.ceil(len(segment)/frames_per_minute))
            sub_segment_length = int(np.ceil(len(segment)/num_sub_segments))
            for i in range(num_sub_segments):
                sub_start = i*sub_segment_length
                sub_end = sub_start + sub_segment_length
                sub_segment = segment[sub_start:sub_end]
                actual_sub_end = sub_start + len(sub_segment)  # The slice might have gotten fewer elements
                yield (start + sub_start, start + actual_sub_end, sub_segment)
        else:
            yield start, end, segment
