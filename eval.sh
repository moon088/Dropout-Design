#!/bin/bash

PYTHON=python3
PATH_FILE_SCRIPT=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)/$(basename "${BASH_SOURCE:-$0}")
echo "$PATH_FILE_SCRIPT"

DIR_SCRIPT=$(dirname $PATH_FILE_SCRIPT)
cd $DIR_SCRIPT
echo "$DIR_SCRIPT"

DIR_CONF=$DIR_SCRIPT"/../conf/"

echo "####################"
echo "#  Training Phase  #"
echo "####################"
for FILE in $(ls $DIR_CONF); do
    if [ "${FILE##*.}" = "json" ]; then
        echo $FILE
        $PYTHON "./training.py" -j "$DIR_CONF$FILE"
    fi
done

echo "####################"
echo "# Evaluation Phase #"
echo "####################"
DIR_MODELS=$DIR_SCRIPT"/models/"
$PYTHON "./remove.py"
for FILE in $(ls $DIR_MODELS); do
    if [ "${FILE##*.}" = "h5" ]; then
        echo $DIR_MODELS$FILE
        $PYTHON "./prediction.py" -e "$DIR_MODELS$FILE"
    fi
done

$PYTHON "./accumulation.py"

echo "Finished"
echo -n "Press any key: "
read DATA