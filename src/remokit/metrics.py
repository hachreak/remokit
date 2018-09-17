# -*- coding: utf-8 -*-
#
# This file is part of remokit.
# Copyright 2018 Leonardo Rossi <leonardo.rossi@studenti.unipr.it>.
#
# pysenslog is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pysenslog is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pysenslog.  If not, see <http://www.gnu.org/licenses/>.

"""Metrics analysis."""

import os
import numpy as np
import json
import itertools

import matplotlib
from keras import activations

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt  # disable=E402

from vis.visualization import visualize_saliency
from vis.utils import utils

from . import dataset


def extract(get_value, metrics):
    """Extract metrics."""
    return np.array([get_value(m) for m in metrics])


def get_mmm(values):
    """Get some basic info as min, avg, max."""
    return values.min(), values.mean(), values.max(), np.var(values)


def get_valid_metrics(metrics):
    """Get only valid metrics with accuracy over a threshold."""
    return filter(lambda m: m['acc'] >= 0.50, metrics)


def aggregate(metrics):
    """Get all kind of stats."""

    def get_class_metrics(key, metrics):
        report = metrics[0]['report']['classes'][key]
        return {
            k: get_mmm(extract(
                lambda m: m['report']['classes'][key][k], metrics))
            for k in report.keys()
        }

    fmetrics = get_valid_metrics(metrics)

    return {
        "count": {
            "total": len(metrics),
            "invalid": len(metrics) - len(fmetrics),
            "valid": len(fmetrics)
        },
        "acc": get_mmm(extract(lambda m: m['acc'], fmetrics)),
        "loss": get_mmm(extract(lambda m: m['loss'], fmetrics)),
        "report": {
            "average": {
                "f1-score": get_mmm(extract(
                    lambda m: m['report']['average']['f1-score'], fmetrics
                )),
                "precision": get_mmm(extract(
                    lambda m: m['report']['average']['precision'], fmetrics
                )),
                "recall": get_mmm(extract(
                    lambda m: m['report']['average']['recall'], fmetrics
                )),
            },
            "classes": {
                k: get_class_metrics(k, fmetrics)
                for k in fmetrics[0]['report']['classes'].keys()
            }
        },
        "confusion_matrix": np.array([
            m['confusion_matrix'] for m in fmetrics]
        ).mean(axis=0)
    }


def normalize_confusion_matrix(metrics):
    """Normalize confusion matrix."""
    cm = metrics['confusion_matrix']
    metrics['confusion_matrix'] = \
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return metrics


def load(metrics_file):
    """Load raw metrics."""
    with open(metrics_file) as data_file:
        return json.load(data_file)


def show_matrix(matrix, labels):
    """Print a matrix."""
    for i, row in enumerate(matrix):
        to_print = ''.join(['{:8}'.format(round(item, 2)) for item in row])
        print("{0:<15} {1}".format(labels[i], to_print))


def show_mmm(name, key, stats):
    """Show min/mean/max."""
    (min_, mean_, max_, var) = stats[key]
    print("{0}: {1:8} {2:8} {3:8} {4:12}".format(
        name, round(min_, 2), round(mean_, 2), round(max_, 2), round(var, 6)
    ))


def show_prf(stats):
    """Show precision, recall, f1-score."""
    labels = ['precision', 'recall', 'f1-score', 'support']
    to_print = "{0:16} ".format(" ")
    to_print += "".join(["{0:12}".format(k) for k in labels])
    print(to_print)
    for c, value in stats['report']['classes'].items():
        to_print = "{0:12}".format(c)
        for k in labels:
            mean = value[k][1]
            to_print += "{0:12}".format(round(mean, 2))
        print(to_print)


def show(stats):
    """Show all stats."""
    print("Total   : {0}".format(stats['count']['total']))
    print("Valid   : {0}".format(stats['count']['valid']))
    print("Invalid : {0}".format(stats['count']['invalid']))
    print("")
    print("{0:14} {1:8}{2:9}{3:9}{4:10}".format(
        " ", "min", "mean", "max", "variance"
    ))
    show_mmm("Accuracy", "acc", stats)
    show_mmm("Loss    ", "loss", stats)
    print("")
    print("Confusion matrix:")
    show_matrix(stats['confusion_matrix'], dataset.ordered_categories())
    print("")
    show_prf(stats)
    print("")


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_metrics(metrics, filename):
    # create directory if it doesn't exists
    path, _ = os.path.split(filename)
    if not os.path.isdir(path):
        os.makedirs(path)
    # save metrics
    with open(filename, 'w') as outfile:
        json.dump(metrics, outfile, cls=NumpyEncoder)


def append_metrics(metric, filename):
    """Append metrics."""
    try:
        metrics = json.load(open(filename, 'r'))
    except IOError:
        metrics = []
    metrics.append(metric)
    save_metrics(metrics, filename)


def plot_loss(metrics):
    """Plot training/validation accuracy."""
    history = metrics['history']

    # Get training and test accuracy histories
    training_loss = history['loss']
    val_loss = history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize accuracy history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss Score')

    return plt


def plot_accuracy(metrics):
    """Plot training/validation accuracy."""
    history = metrics['history']

    # Get training and test accuracy histories
    training_accuracy = history['acc']
    val_accuracy = history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_accuracy) + 1)

    # Visualize accuracy history
    plt.plot(epoch_count, training_accuracy, 'r--')
    plt.plot(epoch_count, val_accuracy, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Score')

    return plt


def plot_confusion_matrix(metrics, title='Confusion matrix', cmap=None):
    target_names = dataset.ordered_categories()
    cm = metrics['confusion_matrix']
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    xlabel = 'Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'
    plt.xlabel(xlabel.format(accuracy, misclass))

    return plt


def plot_prediction(title, y_pred):
    """Plot predictions."""
    plt.xlabel('Sequence')
    plt.ylabel('Prediction')
    plt.title(title)
    labels = dataset.ordered_categories()
    indices = list(range(0, len(y_pred)))
    for i in range(0, len(y_pred[0])):
        plt.plot(indices, [y[i] for y in y_pred], label=labels[i])
    plt.legend()
    return plt


def extrapolate_points(y_pred, count, degree):
    """extrapolate 'count' points from prediction."""
    orig_count = len(y_pred)
    x = range(0, orig_count)
    y = y_pred
    z = np.polyfit(x, y, degree)
    f = np.poly1d(z)
    points = []
    last = orig_count - 1
    for x1 in np.linspace(0, last, count):
        points.append(f(x1))
    return points


def plot_saliency(model, batches, cmap=None):
    """Visualize saliency."""
    # color map
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    # Swap softmax with linear
    layer_idx = len(model.layers) - 1
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    for x_test, y_test in batches:
        for class_idx in range(0, len(x_test)):
            idx = dataset.categorical2category(y_test[class_idx])

            grads = visualize_saliency(
                model, layer_idx, filter_indices=idx,
                seed_input=x_test[class_idx],
                backprop_modifier='guided'
            )
            # Plot with 'jet' colormap to visualize as a heatmap.
            plt.imshow(grads, cmap='jet')
            plt.colorbar()
            plt.suptitle(dataset.category2label(idx))
            yield plt


def boxplot(title_x, title_y, labels_x, values):
    """Blox plot accuracy min/mean/max."""
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    axes.boxplot(values, vert=True, patch_artist=True)

    axes.yaxis.grid(True)
    axes.set_xticks([y+1 for y in range(len(values))], )
    axes.set_xlabel(title_x)
    axes.set_ylabel(title_y)

    plt.setp(axes, xticks=[y+1 for y in range(len(values))],
             xticklabels=labels_x)

    return plt
