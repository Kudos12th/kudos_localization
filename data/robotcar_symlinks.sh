#!/bin/bash

set -x
# change the directory
ROBOTCAR_SDK_ROOT=~/dir/AtLoc/data/robotcar-dataset-sdk

ln -s ${ROBOTCAR_SDK_ROOT}/models/ ~/dir/AtLoc/data/robotcar_camera_models
ln -s ${ROBOTCAR_SDK_ROOT}/python/ ~/dir/AtLoc/data/robotcar_sdk
set +x