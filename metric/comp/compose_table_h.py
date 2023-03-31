#!/usr/bin/env python3
import numpy as np
import os
from os import path
import glob


def read_metric(scene, method, metric):
    base = path.join(scene, method)
    if metric == 'psnr':
        v = float(open(base+'.log').readlines()[-1].strip())
    elif metric == 'ssim':
        v = float(open(base+'.ssim').readlines()[-1].strip())
    elif metric == 'lpips':
        v = float(open(base+'.lpips.alex').readlines()[-1].strip())

    return v

scenes = ['drums', 'lego', 'chair', 'ficus', 'mic', 'hotdog', 'materials']
metrics = {'psnr': 'PSNR $\\uparrow$', 'ssim': 'SSIM $\\uparrow$', 'lpips': 'LPIPS $\\downarrow$'}
methods = {'e2vid': 'E2VID+NeRF', 'event': 'Our EventNeRF'}

print(r'\begin{tabular}{'+'|'.join(['c']*(len(methods)*len(metrics)+1))+'}')
for method in methods.values():
    print(r' & \multicolumn{',len(metrics),'}{c}{', method, '}', end='', sep='')
print('\\\\')
print('Scene ', end='')
for method in methods:
    for metric in metrics.values():
        print('&', metric, end=' ')
print('\\\\')
print(r'\hline')
method_res = dict()
for scene in scenes:
    print(scene.capitalize(), end='')
    for method in methods:
        othermethod = [x for x in methods if x != method][0]
        for metric in metrics:
            val = read_metric(scene, method, metric)
            otherval = read_metric(scene, othermethod, metric)
            txt = '%.2f'%val
            if metric != 'lpips':
                if val >= otherval:
                    txt = r'\mathbf{'+txt+'}'
            else:
                if val <= otherval:
                    txt = r'\mathbf{'+txt+'}'
            print(f' & ${txt}$', end='')
            method_res.setdefault(method, dict()).setdefault(metric, []).append(val)
    print('\\\\')
print(r'\hline')

print('Average', end='')
for method in methods:
    for metric in metrics:
        val = np.mean(method_res[method][metric])
        txt = '%.2f'%val
        if method == 'event':
            txt = r'\mathbf{'+txt+'}'
        print(f' & ${txt}$', end='')

print()
print(r'\end{tabular}')
