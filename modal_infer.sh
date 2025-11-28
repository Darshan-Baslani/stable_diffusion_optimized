#!/bin/bash
cp /root/weights/*.ckpt /root/naive/data/
python naive/sd/inference.py "$@"
