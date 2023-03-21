#!/usr/bin/env bash
postfix="$1"
fn="$postfix".mp4
ffmpeg -y -r 30  -i "$postfix"/r_%04d0.png -crf 10 "$fn"
ffmpeg -i "$fn" -i "$fn" -i "$fn" -i "$fn" -filter_complex "concat=n=4" -pix_fmt yuv422p10 -c:v prores -profile:v 3 -c:a pcm_s24le `basename $(pwd)``basename $postfix`.mov

