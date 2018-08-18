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

from . import dataset


def extract(get_value, metrics):
    """Extract metrics."""
    return np.array([get_value(m) for m in metrics])


def get_mmm(values):
    """Get some basic info as min, avg, max."""
    return values.min(), values.mean(), values.max()


def aggregate(metrics):
    """Get all kind of stats."""

    def get_class_metrics(key, metrics):
        report = metrics[0]['report']['classes'][key]
        return {
            k: get_mmm(extract(
                lambda m: m['report']['classes'][key][k], metrics))
            for k in report.keys()
        }

    fmetrics = filter(lambda m: m['acc'] >= 0.20, metrics)

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
    (min_, mean_, max_) = stats[key]
    print("{0}: {1:8} {2:8} {3:8}".format(
        name, round(min_, 2), round(mean_, 2), round(max_, 2)
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
            (_, mean, _) = value[k]
            to_print += "{0:12}".format(round(mean, 2))
        print(to_print)


def show(stats):
    """Show all stats."""
    print("Total   : {0}".format(stats['count']['total']))
    print("Valid   : {0}".format(stats['count']['valid']))
    print("Invalid : {0}".format(stats['count']['invalid']))
    print("")
    print("{0:14} {1:8}{2:9}{3:12}".format(" ", "min", "mean", "max"))
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


def plot_accuracy(metrics):
    """Plot training/validation accuracy."""
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

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
