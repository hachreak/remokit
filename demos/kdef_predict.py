
"""Predict using CNN model01 with KDEF dataset."""

from remokit import dataset
from remokit.datasets.kdef import get_label, get_data
from remokit.dataset import build_batches, ordered_categories, get_files, first
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

shape_predictor = "data/shape_predictor_68_face_landmarks.dat"
directory = "data/KDEF-straight"

img_x, img_y, img_channels = 100, 100, 1
#  num_classes = len(_category)
epochs = 5
batch_size = 5
k = 10
index = 1
index_trained = 0

files = get_files(directory)
files = first(files, 20)

files = list(files)
_, files = dataset.kfold_split(
    files, get_label=get_label, k=k, index=index
)

batches = build_batches(files, get_data, get_label, k, index, batch_size,
                        epochs, img_x, img_y, shape_predictor)

# get CNN model
model = load_model("data/kdef_train_{0}.h5".format(index_trained))

(x_train, y_train) = next(batches)
y_pred = model.predict(x_train)

print('Confusion Matrix')
y_pred = [dataset.category2label(v) for v in y_pred]
y_train = [dataset.category2label(v) for v in y_train]
matrix = confusion_matrix(y_train, y_pred)
print(matrix)

report = classification_report(y_train, y_pred,
                               target_names=ordered_categories())
print(report)
