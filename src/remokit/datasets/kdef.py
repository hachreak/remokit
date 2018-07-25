"""Dataset KDEF

Dataset source: www.emotionlab.se/resources/kdef
"""

import os
from .. import detect

_label = {
    'NE': 'neutral',
    'AN': 'angry',
    'DI': 'disgust',
    'AF': 'afraid',
    'HA': 'happy',
    'SA': 'sad',
    'SU': 'surprised',
}


def get_data(files_stream):
    """Get a streaming of label/image to process."""
    for filename in files_stream:
        yield get_label(filename), detect.load_img(filename)


def get_label(filename):
    """Convert a 2 chars label to a full name label."""
    key = os.path.basename(filename)[4:6]
    return _label[key]
