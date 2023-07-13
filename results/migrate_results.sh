#!/bin/sh

rsync -av --exclude='event*' --exclude='*.ckpt' "$1" .

