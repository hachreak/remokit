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

"""Show metrics."""

from __future__ import absolute_import

import sys
from remokit import metrics


def show_all(filenames):
    """Show all metrics."""
    for filename in filenames:
        print("##### {0}\n".format(filename))
        metrics.show(metrics.aggregate(metrics.load(filename)))


def plot_accuracy(run_number, filename):
    """Plot accuracy."""
    metrics.plot_accuracy(metrics.load(filename)[run_number]).show()


def boxplot_accuracies(filenames):
    values = []
    names = []
    for i, filename in enumerate(filenames):
        ms = metrics.get_valid_metrics(metrics.load(filename))
        values.append([m['acc'] for m in ms])
        names.append("Exp {0}".format(i))
    metrics.boxplot(
        'Experiments', 'Accuracy', names, values
    ).show()


def plot_loss(run_number, filename):
    """Plot loss."""
    metrics.plot_loss(metrics.load(filename)[run_number]).show()


def plot_confusion_matrix(filename, normalize):
    """Plot confusion matrix."""
    m = metrics.aggregate(metrics.load(filename))
    if normalize:
        m = metrics.normalize_confusion_matrix(m)
    metrics.plot_confusion_matrix(m).show()


if len(sys.argv) == 1:
    print('Usage: {0} [cm] [filename] [normalized]'.format(sys.argv[0]))
    print('Usage: {0} [accuracies] [filename1] [filename2] ...'.format(
        sys.argv[0]))
    print('Usage: {0} [accuracy] [run-number] [filename]'.format(sys.argv[0]))
    print('Usage: {0} [loss] [run-number] [filename]'.format(sys.argv[0]))
    print('       {0} [all] [filename1] [filename2] ..'.format(sys.argv[0]))
    sys.exit(1)

if sys.argv[1] == 'cm':
    filename = sys.argv[2]
    normalize = len(sys.argv) > 3
    plot_confusion_matrix(filename, normalize)
if sys.argv[1] == 'accuracies':
    boxplot_accuracies(sys.argv[2:])
if sys.argv[1] == 'accuracy':
    run_number = int(sys.argv[2])
    filename = sys.argv[3]
    plot_accuracy(run_number, filename)
if sys.argv[1] == 'loss':
    run_number = int(sys.argv[2])
    filename = sys.argv[3]
    plot_loss(run_number, filename)
elif sys.argv[1] == 'all':
    show_all(sys.argv[2:])
