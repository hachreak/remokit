
"""Face detector."""

import dlib


def get_detector():
    return dlib.get_frontal_face_detector()


def get_predictor(predictor_path):
    return dlib.shape_predictor(predictor_path)


def load_img(filename):
    return dlib.load_rgb_image(filename)


def detect(detector, img):
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


def features(img, predictor, detected):
    for (k, v) in detected.items():
        v['features'] = predictor(img, v['detected'])
    return detected
