#!/usr/bin/env bash
postfix="$1"
ffmpeg -y -r 30  -i ./render"$postfix"/r_%04d0.png -crf 10 render.mp4
ffmpeg -y -r 30  -i ./render"$postfix"/r_%04d0.png -vf reverse -crf 10 render_rev.mp4
ffmpeg -i render.mp4 -i render_rev.mp4 -i render.mp4 -i render_rev.mp4 -filter_complex "concat=n=4" -crf 10 `basename $(pwd)`"$postfix"_pp.mp4

