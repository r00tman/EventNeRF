#!/usr/bin/env python3
import numpy as np
import os
from os import path
import glob


def read_metric(scene, method, metric):
    base = path.join(scene, method)
    # print('base --- '+base)
    if metric == 'psnr':
        v = float(open(base+'.log').readlines()[-1].strip())
    elif metric == 'ssim':
        v = float(open(base+'.ssim').readlines()[-1].strip())
    elif metric == 'lpips':
        v = float(open(base+'.lpips.alex').readlines()[-1].strip())

    return v

scenes = ['drums', 'lego', 'chair', 'ficus', 'mic', 'hotdog', 'materials']
metrics = {'psnr': 'PSNR $\\uparrow$', 'ssim': 'SSIM $\\uparrow$', 'lpips': 'LPIPS $\\downarrow$'}
# methods = {'e2vid': 'E2VID+NeRF', 'event': 'Our EventNeRF', 'ngp': 'NGP log 0.05', 'ngp_log010': 'NGP log 0.10'}
methods = {'e2vid': 'E2VID+NeRF', 'event': 'Our EventNeRF', 'ngp_log010': 'Our EventNGP'}

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
        othermethods = [x for x in methods if x != method]
        for metric in metrics:
            val = read_metric(scene, method, metric)
            othervals = [read_metric(scene, othermethod, metric) for othermethod in othermethods]
            txt = '%.2f'%val
            if metric != 'lpips':
                if all(round(val, 2) >= round(otherval, 2) for otherval in othervals):
                    txt = r'\mathbf{'+txt+'}'
            else:
                if all(round(val, 2) <= round(otherval, 2) for otherval in othervals):
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
