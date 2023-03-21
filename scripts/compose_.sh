#!/usr/bin/env bash
postfix="$1"
fn=render"$postfix".mp4
ffmpeg -y -r 30  -i ./render"$postfix"/r_0%03d0.png -crf 10 "$fn"
ffmpeg -i "$fn" -i "$fn" -i "$fn" -i "$fn" -filter_complex "concat=n=4" -crf 10 `basename $(pwd)`"$postfix".mp4

