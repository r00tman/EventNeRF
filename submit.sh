#!/usr/bin/env bash
script="$1"
scriptbase="`basename "$script"`"
scriptbase="${scriptbase%.sh}"
name="$scriptbase"_`date +%Y-%m-%d_%H-%M-%S`
dir=archive/"$name"
echo will mkdir -p "$dir"
mkdir -p "$dir"
echo will cp -r *.py "$script" configs "$dir"
cp -r *.py "$script" configs scripts "$dir"
cd "$dir"
chmod -R g+w .
echo will sbatch "$scriptbase" in "$dir"
sbatch "$scriptbase".sh
