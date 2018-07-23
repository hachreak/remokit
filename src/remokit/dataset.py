
"""Get dataset."""

import dlib
import os
import numpy as np
from remokit import detect
from keras.utils import to_categorical


_category = {
    'neutral': 0,
    'angry': 1,
    'disgust': 2,
    'afraid': 3,
    'happy': 4,
    'sad': 5,
    'surprised': 6,
}


def get_files(directory):
    """Get list of images."""
    for root, dirnames, files in os.walk(directory):
        for name in files:
            print("get file: ", os.path.join(root, name))
            yield os.path.join(root, name)


def first(stream, size):
    index = 0
    for value in stream:
        yield value
        index += 1
        if index >= size:
            raise StopIteration


def batch(block, size):
    for block in np.array_split(block, size):
        yield block


def stream_batch(stream, size):
    """Create batch on the fly."""
    while True:
        batch = []
        try:
            for i in range(size):
                value = next(stream)
                #  import ipdb; ipdb.set_trace()
                batch.append(value)
            yield batch
        except StopIteration:
            if not batch:
                raise StopIteration
            yield batch


def loader(batches):
    """Load images in batch from disk."""
    for block in batches:
        images = []
        # load in a single row
        for filename in block:
            images.append((filename, detect.load_img(filename)))
        # yield one by one
        for (filename, image) in images:
            print("loader: ", filename)
            yield filename, image


def stream_training(stream, detector, predictor, img_x=None, img_y=None):
    img_x = img_x or 100
    img_y = img_y or 100
    for label, img in stream:
        dets = detect.detect(detector, img)
        shapes = detect.shapes(img, predictor, dets)
        for shape in shapes:
            feature = detect.expand2img(detect.shape2matrix(shape))
            yield (
                to_categorical(label2category(label), 7),
                dlib.resize_image(feature, img_x, img_y)
            )


def _batch_train(stream, size=None):
    for batch in stream_batch(stream, size):
        x_train = []
        y_train = []
        for label, img in batch:
            x_train.append(img)
            y_train.append(label)
        x_train = feature2input(np.array(x_train))
        yield x_train, np.array(y_train)


def batch_training(stream, detector, predictor, size,
                   img_x=None, img_y=None):
    return _batch_train(
        stream_training(stream, detector, predictor, img_x, img_y), size
    )


def feature2input(features):
    (img_x, img_y) = features[0].shape
    features = features.reshape(features.shape[0], img_x, img_y, 1)
    features = features.astype('float32')
    features /= 255
    return features


def label2category(label):
    return _category[label]
    #  return [_category[label] for label in labels]

#  def get_data():
#      img = detector.load_img(
#          "/home/hachreak/Downloads/dlib/examples/faces/2007_007763.jpg")
#      d = detector.get_detector()
#      p = detector.get_predictor(
#          "/tmp/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
#      dets = detector.detect(d, img)
#      feats = detector.shapes(img, p, dets)
#      feat = feats[0]
#      mat = detector.shape2matrix(feat)
#      imgf = detector.expand2img(mat)
#      imgf /= 255
#      y_test = np.array([1])
#      y_test = to_categorical(y_test, 7)
#      x_test = np.array([imgf])
#      return (x_test, y_test), (x_test, y_test)
