#!/bin/bash

!wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip

!unzip train2014.zip

ls -1 train2014 > train.csv

