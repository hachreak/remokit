
test_index=0
validation_index=1
seed=0

DEMOS="demos/experiments"
FILES="experiment_03_frankenstein_eyes.json
       experiment_03_frankenstein_eyesmouth.json
       experiment_03_frankenstein_face.json
       experiment_03_frankenstein_mouth.json"

ONEMODEL="experiment_04_onemodel.py"
CONFIG="experiment_03_frankenstein.json"

# preprocess images
for f in $FILES; do
  echo "preprocess $f"
  python ${DEMOS}/${ONEMODEL} preprocess ${DEMOS}/$f
done

# train submodels
for f in $FILES; do
  echo "Train model [$test_index $validation_index $seed] config file $f"
  python ${DEMOS}/${ONEMODEL} run ${DEMOS}/$f $test_index $validation_index $seed
done

# run experiment
python ${DEMOS}/experiment_01.py run ${DEMOS}/$CONFIG
