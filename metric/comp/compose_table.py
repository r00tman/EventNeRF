#!/usr/bin/env python3
import numpy as np
import os
from os import path
import glob

scenes = ['drums', 'lego', 'chair', 'ficus', 'mic', 'hotdog', 'materials']
metrics = ['psnr', 'ssim', 'lpips']

print(r'\begin{tabular}{'+'|'.join(['c']*(len(scenes)*len(metrics)+1))+'}')
for scene in scenes:
    print(r' & \multicolumn{',len(metrics),'}{|c|}{', scene, '}', end='', sep='')
print('\\\\')
print('method ', end='')
for scene in scenes:
    for metric in metrics:
        print('&', metric, end=' ')
print('\\\\')
print(r'\hline')
for method in ['e2vid', 'event']:
    print(method, end='')
    for scene in scenes:
        for metric in metrics:
            print(' & $0.00$', end='')
    print('\\\\')
print(r'\end{tabular}')
