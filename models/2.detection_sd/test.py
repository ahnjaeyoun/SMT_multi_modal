from make_txt import make_txtfile
import os
import json

TRAIN_JSON_DIR= './dataset/labels_json/train/'
TRAIN_TXT_DIR= './dataset/labels/train/'
VAL_JSON_DIR= './dataset/labels_json/val/'
VAL_TXT_DIR= './dataset/labels/val/'


make_txtfile(TRAIN_JSON_DIR, TRAIN_TXT_DIR, 512)
make_txtfile(VAL_JSON_DIR, VAL_TXT_DIR, 512)

