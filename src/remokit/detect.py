
"""Face detector."""

import dlib
import numpy as np


def get_detector():
    """Get default face detector."""
    return dlib.get_frontal_face_detector()


def get_predictor(predictor_path):
    """Get default shape predictor."""
    return dlib.shape_predictor(predictor_path)


def load_img(filename):
    """Load a image."""
    return dlib.load_rgb_image(filename)


def detect(detector, img):
    """Detect faces inside the image."""
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces
    # The score is bigger for more confident detections.
    # The third argument to run is an optional adjustment to the detection
    # threshold, where a negative value will return more detections and a
    # positive value fewer.
    dets, scores, idx = detector.run(img, 1, 0)
    result = {}
    for i, d in enumerate(dets):
        #  crop_img = img[d.top():d.bottom(), d.left():d.right()]
        result[i] = {
            'score': scores[i],
            'idx': idx[i],
            #  'img': crop_img,
            'detected': d,
        }
    return result


def shapes(img, predictor, detected):
    """Get detected face shapes."""
    ret = []
    for (k, v) in detected.items():
        ret.append(predictor(img, v['detected']))
    return ret


def align(img, shapes, size=150):
    faces = dlib.full_object_detections()
    for shape in shapes:
        faces.append(shape)
    return dlib.get_face_chips(img, faces, size=size)


def shape2matrix(feature):
    """Convert a dlib shape in a numpy matrix."""
    left = feature.rect.left()
    top = feature.rect.top()
    return np.matrix([
        [part.y - top, part.x - left] for part in feature.parts()
    ])


def expand2img(matrix):
    """Expand shape points in a b/n image."""
    maxval = matrix.max(axis=0)
    cols = maxval.item(0) + 1
    rows = maxval.item(1) + 1
    img = np.zeros((cols, rows), dtype='int')
    for row in matrix:
        y = row.item(0)
        x = row.item(1)
        img[y][x] = 255
    return img
