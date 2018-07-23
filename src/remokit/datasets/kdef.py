"""Dataset KDEF

Dataset source: www.emotionlab.se/resources/kdef
"""

import os
from .. import dataset

_label = {
    'NE': 'neutral',
    'AN': 'angry',
    'DI': 'disgust',
    'AF': 'afraid',
    'HA': 'happy',
    'SA': 'sad',
    'SU': 'surprised',
}


def get_data(directory, batch_size=None):
    batch_size = batch_size or 100

    files = dataset.get_files(directory)
    loader = dataset.loader(dataset.stream_batch(files, batch_size))
    for filename, img in loader:
        yield _get_label(filename), img


def _get_label(filename):
    key = os.path.basename(filename)[4:6]
    return _label[key]
