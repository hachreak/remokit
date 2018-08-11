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


if len(sys.argv) == 1:
    print('Usage: {0} [accuracy] [run-number] [filename]'.format(sys.argv[0]))
    print('       {0} [all] [filename]'.format(sys.argv[0]))
    sys.exit(1)

if sys.argv[1] == 'accuracy':
    run_number = int(sys.argv[2])
    filename = sys.argv[3]
    plot_accuracy(run_number, filename)
elif sys.argv[1] == 'all':
    show_all(sys.argv[2:])
