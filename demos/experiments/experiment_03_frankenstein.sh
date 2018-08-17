
test_index=0
validation_index=1
seed=0

DEMOS="demos/experiments"
FILES="experiment_01_eyes_model04.json experiment_01_eyesmouth_model04.json
       experiment_01_face_model04.json experiment_01_mouth_model04.json"

ONEMODEL="experiment_04_onemodel.py"

# preprocess images
for f in $FILES; do
  echo "Preprocess images for $f"
  python ${DEMOS}/experiment_01.py preprocess ${DEMOS}/$f
done

# train submodels
for f in $FILES; do
  echo "Train model [$test_index $validation_index $seed] config file $f"
  python ${DEMOS}/${ONEMODEL} $test_index $validation_index $seed ${DEMOS}/$f
done

# run experiment
CONFIG=experiment_03_frankenstein.json
python ${DEMOS}/experiment_01.py run ${DEMOS}/$CONFIG
