#! /bin/bash
cd ../bin/

# ./rgbd_tum ../Vocabulary/ORBvoc.bin ../Examples/RGB-D/rgbd-data.yaml ../data/rgbd-data ../data/rgbd-data/associate.txt
./rgbd_tum ../Vocabulary/ORBvoc.bin ../Examples/RGB-D/TUM2.yaml ~/Downloads/rgbd_dataset_freiburg2_rpy ~/Downloads/rgbd_dataset_freiburg2_rpy/associate.txt
