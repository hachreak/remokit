#!/bin/sh

python demos/experiments/experiment_01.py preprocess demos/experiments/01_face_multidataset_model03.json && echo "preprocess OK" >> data/01_face_multidataset.check
python demos/experiments/experiment_01.py run demos/experiments/01_face_multidataset_model03.json && echo "model 03 OK" >> data/01_face_multidataset.check
python demos/experiments/experiment_01.py run demos/experiments/01_face_multidataset_model04v2.json && echo "model 04v2 OK" >> data/01_face_multidataset.check
python demos/experiments/experiment_01.py run demos/experiments/01_face_multidataset_model05.json && echo "model 05 OK" >> data/01_face_multidataset.check
python demos/experiments/experiment_01.py run demos/experiments/01_face_multidataset_model08.json && echo "model 08 OK" >> data/01_face_multidataset.check
